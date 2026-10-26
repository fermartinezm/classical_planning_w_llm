import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe return empty list or raise error
        # For this context, assuming valid fact strings
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all other reachable nodes in a graph.

    Args:
        graph: An adjacency list representation of the graph (dict: node -> list of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from the start_node.
        Nodes not reachable will not be in the dictionary or have distance infinity if pre-initialized.
    """
    distances = {node: float('inf') for node in graph} # Initialize with all nodes in graph keys
    if start_node not in graph:
         # If start_node is not in the graph (e.g., isolated location),
         # it can only reach itself with distance 0.
         distances[start_node] = 0
         return distances # No paths to other nodes

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node has neighbors defined in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances.get(neighbor, float('inf')) == float('inf'): # Use get for safety
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    # Add any locations that were not in graph keys but were start_node
    if start_node not in distances:
         distances[start_node] = 0

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move each
    package from its current location (or vehicle) to its goal location,
    independently for each package. It considers pick-up, drive, and drop actions.
    The drive cost is estimated as the shortest path distance in the road network.
    Vehicle capacity and availability are ignored.

    # Assumptions:
    - Each package needs to reach a specific goal location on the ground.
    - Packages can be on the ground at a location or inside a vehicle.
    - Vehicles are always at a specific location.
    - The cost of pick-up, drop, and drive actions is 1.
    - Vehicle capacity and the need for a vehicle to be at the pick-up location
      are ignored for simplicity (non-admissible).
    - The road network is static and provides connections between locations.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Identifies all packages (objects in 'at' goals) and vehicles (objects in 'capacity' init facts).
    - Builds the road network graph from static 'road' facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the cost for each package that is
    not yet at its goal location on the ground:

    1.  Identify the current status of each package that needs to reach a goal:
        - Is it on the ground at a location `L_current`? (fact `(at package L_current)`)
        - Is it inside a vehicle `V`? (fact `(in package V)`)
    2.  Determine the physical location `L_vehicle` for any vehicle `V` holding a package.
    3.  For each package `p` with goal `L_goal`:
        - If `p` is already at `L_goal` on the ground (`(at p L_goal)` is true), the cost for this package is 0.
        - If `p` is on the ground at `L_current` (`L_current != L_goal`):
            - Estimated cost for `p` = 1 (pick-up) + shortest_path(`L_current`, `L_goal`) (drive) + 1 (drop).
        - If `p` is inside vehicle `V`, and `V` is at `L_vehicle`:
            - If `L_vehicle != L_goal`:
                - Estimated cost for `p` = shortest_path(`L_vehicle`, `L_goal`) (drive) + 1 (drop).
            - If `L_vehicle == L_goal`:
                - Estimated cost for `p` = 1 (drop).
        - If any required shortest path is infinite (goal unreachable), the heuristic returns infinity.
    4.  The total heuristic value is the sum of the estimated costs for all packages not yet at their goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package and identify packages
        self.packages = set()
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.packages.add(package)
                self.goal_locations[package] = location

        # Identify vehicles from initial capacity facts
        self.vehicles = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "capacity":
                 self.vehicles.add(parts[1])

        # Build road graph and collect all relevant locations
        self.road_graph = collections.defaultdict(list)
        locations = set()

        # Locations from initial state (where objects are)
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                locations.add(parts[2])

        # Locations from goal state (where packages need to be)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at":
                 locations.add(parts[2])

        # Locations and connections from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Ensure all collected locations are keys in the graph dict, even if isolated
        for loc in locations:
             if loc not in self.road_graph:
                  self.road_graph[loc] = []


        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for loc in locations:
            self.shortest_paths[loc] = bfs(self.road_graph, loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map vehicles to their current locations
        vehicle_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and parts[1] in self.vehicles:
                vehicle_locations[parts[1]] = parts[2]

        # Map packages to their current status (location string or vehicle string)
        current_package_status = {}
        for package in self.packages:
            found = False
            for fact in state:
                parts = get_parts(fact)
                if not parts: continue # Skip malformed facts

                if parts[0] == "at" and parts[1] == package:
                    current_package_status[package] = parts[2] # Package is on the ground at this location
                    found = True
                    break
                elif parts[0] == "in" and parts[1] == package:
                    current_package_status[package] = parts[2] # Package is in this vehicle
                    found = True
                    break
            # If a package is not found in 'at' or 'in' facts, the state is likely invalid.
            # For valid states, all goal packages should be locatable.

        total_cost = 0

        # Calculate cost for each package that is not yet at its goal location on the ground
        for package, goal_location in self.goal_locations.items():
            current_status = current_package_status.get(package)

            # If package is not in the state at all, it's an invalid state for this heuristic
            if current_status is None:
                 # This shouldn't happen in valid PDDL states where objects persist.
                 # Return infinity to indicate an unhandleable or potentially unsolvable state.
                 return float('inf')

            # Check if package is already at goal location on the ground
            # The goal is (at package goal_location).
            if current_status == goal_location and f"(at {package} {goal_location})" in state:
                 continue # Package is at goal, cost is 0 for this package

            # Package is not at goal location on the ground. Calculate cost.
            cost_for_package = 0

            if current_status in self.vehicles: # Package is in a vehicle
                V = current_status # The vehicle name
                L_vehicle = vehicle_locations.get(V) # Location of the vehicle

                if L_vehicle is None:
                    # Vehicle location unknown. This shouldn't happen in valid states.
                    # Return infinity as this state might be unsolvable or indicates an issue.
                    return float('inf') # Or a large number

                if L_vehicle != goal_location:
                    # Cost: drive + drop
                    # Need shortest path from vehicle's current location to package's goal location
                    drive_cost = self.shortest_paths.get(L_vehicle, {}).get(goal_location, float('inf'))
                    if drive_cost == float('inf'):
                        # Goal location is unreachable from the vehicle's current location
                        return float('inf')
                    cost_for_package = drive_cost + 1 # 1 for the drop action
                else: # L_vehicle == goal_location
                    # Cost: drop
                    cost_for_package = 1 # 1 for the drop action

            else: # Package is on the ground at current_status (which is a location)
                # Must be at current_status != goal_location because the goal_location case was skipped
                L_current = current_status # The package's current location on the ground
                # Cost: pick-up + drive + drop
                # Need shortest path from package's current location to package's goal location
                drive_cost = self.shortest_paths.get(L_current, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # Goal location is unreachable from the package's current location
                    return float('inf')
                cost_for_package = 1 + drive_cost + 1 # 1 for pick-up, 1 for drop

            total_cost += cost_for_package

        return total_cost

