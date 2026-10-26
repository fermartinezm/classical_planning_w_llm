# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque
# import sys # Not strictly needed if using float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the required number of actions to move each package
    from its current location to its goal location, summing the costs for all
    packages not yet at their destination. It considers the steps of picking up,
    driving, and dropping, using shortest path distances for driving costs.

    # Assumptions
    - The heuristic assumes that a suitable vehicle is always available when needed
      to pick up a package or transport it.
    - Vehicle capacity constraints are ignored.
    - The cost of driving between two locations is the shortest path distance
      in the road network.
    - Each pick-up and drop action costs 1.
    - Vehicle names start with 'v' and location names start with 'l' (based on examples).

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build a graph representing the road network from static facts.
    - Compute the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every package
       and the current location for every vehicle.
    2. Initialize the total heuristic cost to 0.
    3. Check if the current state is a goal state. If yes, return 0.
    4. For each package that has a goal location defined:
       a. Determine the package's current state: Is it on the ground at a location,
          or inside a vehicle? This is done by looking up the package in the
          `current_locations` map. If the value is a string starting with 'v',
          it's a vehicle; otherwise, it's a location.
       b. If the package is on the ground at `current_l` (and `current_l` is not the goal):
          - Add 1 to the cost (for the pick-up action).
          - Find the shortest path distance between `current_l` and the package's `goal_l`
            in the road network. Add this distance to the cost (for drive actions).
          - Add 1 to the cost (for the drop action).
          - If the shortest path is infinite (locations are disconnected), the state is
            likely unsolvable or requires actions not modeled by the heuristic; return infinity.
       c. If the package is inside a vehicle `current_v`:
          - Find the current location `vehicle_l` of the vehicle `current_v` from the
            `current_locations` map.
          - Find the shortest path distance between `vehicle_l` and the package's `goal_l`.
            Add this distance to the cost (for drive actions).
          - Add 1 to the cost (for the drop action).
          - If the vehicle's location is unknown or the path is infinite, return infinity.
    5. The total heuristic value is the sum of costs calculated for all packages
       not at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road graph, and computing shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location

        # Build the road graph
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph.setdefault(l2, set()).add(l1)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
             return 0

        # Track where locatables (packages, vehicles) are
        current_locations = {} # Maps object name (p1, v1) to its location (l1) or container (v1)
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "at":
                # (at ?x ?l) where ?x is locatable (vehicle or package)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_locations[obj] = loc
            elif predicate == "in":
                # (in ?p ?v) where ?p is package, ?v is vehicle
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_locations[package] = vehicle # Store the vehicle as the location for the package

        total_cost = 0

        for package, goal_location in self.goal_locations.items():
            # If package is already at goal, cost is 0 for this package
            # We check this implicitly by seeing if the package's current state matches the goal fact
            goal_fact = f"(at {package} {goal_location})"
            if goal_fact in state:
                continue # Package is already at its goal location

            # Package is not at goal, calculate cost
            if package not in current_locations:
                 # Package location/container is unknown in this state.
                 # This state is likely invalid or unreachable.
                 return float('inf') # Indicate a very high cost

            current_state_info = current_locations[package]

            # Determine if the package is in a vehicle based on naming convention
            # This is a domain-specific assumption based on example object names.
            is_in_vehicle = current_state_info.startswith('v')

            if not is_in_vehicle:
                # Package is on the ground at current_l
                current_l = current_state_info

                # Cost: pick-up + drive + drop
                # Need to drive from current_l to goal_l
                drive_cost = self.distances.get((current_l, goal_location), float('inf'))

                if drive_cost == float('inf'):
                    # Goal location is unreachable from current location
                    return float('inf') # Indicate a very high cost

                total_cost += 1  # pick-up action
                total_cost += drive_cost # drive actions
                total_cost += 1  # drop action

            else:
                # Package is inside vehicle current_v
                current_v = current_state_info

                # Find the location of the vehicle
                if current_v not in current_locations:
                     # Vehicle exists but its location is not specified? Invalid state.
                     return float('inf') # Indicate a very high cost

                vehicle_l = current_locations[current_v]

                # Cost: drive + drop
                # Need to drive from vehicle_l to goal_l
                drive_cost = self.distances.get((vehicle_l, goal_location), float('inf'))

                if drive_cost == float('inf'):
                    # Goal location is unreachable from vehicle's current location
                    return float('inf') # Indicate a very high cost

                total_cost += drive_cost # drive actions
                total_cost += 1  # drop action

        return total_cost
