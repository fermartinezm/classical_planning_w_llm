from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Find the first and last parentheses
    first_paren = fact.find('(')
    last_paren = fact.rfind(')')
    if first_paren == -1 or last_paren == -1 or first_paren >= last_paren:
        # Not a valid PDDL fact string format
        return []
    return fact[first_paren + 1:last_paren].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the individual costs.
    The cost for a package depends on whether it's on the ground or in a vehicle,
    and the shortest path distance between its current effective location and its goal location.

    # Assumptions
    - Each drive, pick-up, and drop action costs 1.
    - Roads are bidirectional.
    - Vehicle capacity and availability are simplified: the heuristic assumes
      a vehicle is available when needed for pick-up and can carry the package,
      and that vehicles can move freely along roads without conflict.
    - The road network is connected, or unreachable goals result in infinite cost.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build the road network graph from static facts.
    - Compute all-pairs shortest paths between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every locatable object (packages and vehicles).
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined:
       a. Check if the package is already at its goal location on the ground. If yes, this package contributes 0 to the heuristic.
       b. If the package is not at its goal location on the ground, determine its current effective location:
          - If the package is on the ground at some location L: The effective location is L.
          - If the package is inside a vehicle V, and vehicle V is at location L: The effective location is L.
       c. Calculate the minimum actions needed for this package to reach its goal location from its current effective location:
          - If the package is on the ground: It needs a pick-up (1 action), then the vehicle needs to drive from the current location to the goal location (shortest_path actions), then the package needs to be dropped (1 action). Total = 1 + shortest_path + 1.
          - If the package is inside a vehicle: The vehicle needs to drive from its current location to the goal location (shortest_path actions), then the package needs to be dropped (1 action). Total = shortest_path + 1.
       d. Add the calculated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost. If any required shortest path was infinite, the total cost will reflect this (e.g., be infinite).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is typically (at package location)
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location

        # Build the road graph and identify all locations.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    locations.add(l1)
                    locations.add(l2)
                    self.road_graph.setdefault(l1, set()).add(l2)
                    self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional

        self.locations = list(locations) # Store all known locations

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in self.locations:
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft() # Use popleft for BFS queue

                self.shortest_paths[(start_loc, current_loc)] = dist

                # Get neighbors from the graph, handle locations with no roads
                neighbors = self.road_graph.get(current_loc, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            # If the graph is disconnected, some locations might not be reached.
            # The shortest_paths dict will not contain entries for unreachable pairs.
            # We handle this during lookup in __call__ by providing a default inf value.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map locatables (packages, vehicles) to their current location or container.
        current_positions = {} # locatable -> location or vehicle
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3:
                    locatable, location = parts[1], parts[2]
                    current_positions[locatable] = location
            elif predicate == "in":
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_positions[package] = vehicle # Package is inside vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined.
        for package, goal_l in self.goal_locations.items():
            # Check if package is already at goal location on the ground.
            if (f"(at {package} {goal_l})") in state:
                continue # Package is at goal, cost is 0 for this package.

            # Package is not at goal location on the ground.
            # Find its current status and effective location.
            current_pos = current_positions.get(package)

            if current_pos is None:
                # This package is not 'at' any location and not 'in' any vehicle.
                # This indicates an invalid state representation for a package not at goal.
                # Treat as unreachable goal for this package.
                return float('inf')

            # Determine if the package is on the ground or in a vehicle
            package_on_ground = False
            current_l = None # Effective location of the package

            # Check if the package is on the ground by looking for an '(at package ?l)' fact
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and len(parts) == 3 and parts[1] == package:
                    package_on_ground = True
                    current_l = parts[2] # Package is on the ground at current_l
                    break # Found package's ground location

            if package_on_ground:
                # Package is on the ground at current_l. We already know current_l != goal_l.
                # Cost: Pick-up (1) + Drive (shortest_path(current_l, goal_l)) + Drop (1)
                drive_cost = self.shortest_paths.get((current_l, goal_l), float('inf'))
                package_cost = 1 + drive_cost + 1 # Pick-up + Drive + Drop
            else:
                # Package is not on the ground. It must be in a vehicle.
                # current_pos holds the vehicle name.
                vehicle_name = current_pos # current_pos is the vehicle name

                # Find the vehicle's location
                vehicle_l = current_positions.get(vehicle_name)
                if vehicle_l is None:
                    # Vehicle exists but is not at any location. Invalid state.
                    return float('inf')

                # Package is in vehicle_name, which is at vehicle_l.
                # The effective location of the package is vehicle_l.
                # Cost: Drive (shortest_path(vehicle_l, goal_l)) + Drop (1)
                drive_cost = self.shortest_paths.get((vehicle_l, goal_l), float('inf'))
                package_cost = drive_cost + 1 # Drive + Drop

            # If drive_cost was inf, package_cost will be inf or inf+1.
            total_cost += package_cost

        # If total_cost accumulated any infinity, the goal is unreachable from this state.
        if total_cost >= float('inf'): # Use >= in case inf + finite is slightly larger than inf
             return float('inf')

        return total_cost
