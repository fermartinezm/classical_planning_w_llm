from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts (similar to Logistics example)
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

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently. The cost for a package includes picking it up
    (if on the ground), driving it to the destination, and dropping it.

    # Assumptions
    - The cost of a 'drive' action between two locations is the shortest path
      distance (number of road segments) between them.
    - Vehicle capacity constraints are ignored. It is assumed that a suitable vehicle
      is always available for pickup and dropoff actions when needed.
    - Actions (pick-up, drop, drive) have a cost of 1 each.
    - The road network is undirected if both (road l1 l2) and (road l2 l1) exist.
      The implementation builds a bidirectional graph for every road fact.
    - Objects appearing in '(at obj loc)' facts that are not packages with goals
      are assumed to be vehicles.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from the static 'road' facts.
    - Identify all locations present in the problem (from roads, initial state, goals)
      to ensure BFS covers all relevant nodes.
    - Compute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state by verifying if all goal facts are true. If yes, the heuristic is 0.
    2. Initialize total estimated cost to 0.
    3. Iterate through the facts in the current state to identify:
       - The current location or vehicle containment for every package that has a goal location defined.
       - The current location for every vehicle.
    4. For each package `p` that has a goal location `goal_loc`:
       a. Check if the package is already `at goal_loc` on the ground. If yes, its cost contribution is 0.
       b. If the package is not at its goal location, determine its current status:
          - If `p` is currently `at current_loc` (on the ground):
             - The package needs to be picked up, transported, and dropped.
               Estimated cost = 1 (pick-up) + shortest_path_distance(`current_loc`, `goal_loc`) (drive) + 1 (drop).
          - If `p` is currently `in vehicle v`:
             - Find the current location `vehicle_loc` of vehicle `v`.
             - The package needs to be transported to the goal location and dropped.
               Estimated cost = shortest_path_distance(`vehicle_loc`, `goal_loc`) (drive) + 1 (drop).
          - If the package's status is unknown (not 'at' or 'in' in the state), assign a high cost.
    5. Sum the estimated costs for all packages.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and computing
        shortest path distances between all locations.
        """
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location

        # Build the road network graph and collect all locations.
        self.road_graph = {}
        all_locations = set()

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                # Assuming roads are bidirectional based on example instances
                self.road_graph.setdefault(loc2, []).append(loc1)

        # Add locations from initial state and goals that might not be in road facts
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 # Assuming the third argument of 'at' is always a location
                 all_locations.add(loc)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 all_locations.add(loc)

        # Ensure all identified locations are keys in the graph dictionary
        for loc in all_locations:
             self.road_graph.setdefault(loc, [])

        # Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_node in self.road_graph:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start_node to compute distances to all reachable nodes."""
        queue = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0 # Distance to self is 0

        while queue:
            current_node, current_dist = queue.popleft()

            # Store distance from start_node to current_node
            self.distances[(start_node, current_node)] = current_dist

            # Explore neighbors
            # Check if current_node exists in graph keys (handles isolated nodes)
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_dist + 1))
                        # Store reverse distance as well for convenience, assuming undirected graph
                        self.distances[(neighbor, start_node)] = current_dist + 1


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Track current status (location or vehicle) for each package with a goal
        current_package_status = {} # {package_name: ('at' or 'in', location_or_vehicle)}
        # Track current location for each vehicle
        current_vehicle_locations = {} # {vehicle_name: location}

        # Populate status dictionaries from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue

            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if the object is one of the packages we have a goal for
                if obj in self.package_goals:
                     current_package_status[obj] = ('at', loc)
                # Assume other 'at' facts are for vehicles
                # This relies on domain structure where only locatables (vehicles/packages) are 'at' locations
                # and vehicles are not packages with goals.
                # If it's not a package with a goal, assume it's a vehicle.
                elif obj not in current_package_status: # Avoid processing the same object twice if it appears in multiple relevant fact types (unlikely here)
                     current_vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Only track packages we have goals for
                if package in self.package_goals:
                     current_package_status[package] = ('in', vehicle)

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at the goal location (on the ground)
            # This check is necessary because the overall state might not be the goal,
            # but this specific package might be done.
            if (f"(at {package} {goal_location})") in state:
                 continue

            # Package is not at its goal location. Estimate cost to get it there.
            if package not in current_package_status:
                 # Package status is unknown. This indicates an issue with state parsing
                 # or an invalid state. Assign a high cost.
                 total_cost += 1000
                 continue

            status, current_loc_or_vehicle = current_package_status[package]

            if status == 'at':
                # Package is on the ground at current_loc_or_vehicle
                current_loc = current_loc_or_vehicle
                # Cost: pick-up + drive + drop
                # Need to drive from current_loc to goal_location
                drive_cost = self.distances.get((current_loc, goal_location), float('inf'))

                if drive_cost == float('inf'):
                    # If unreachable, assign a very high cost
                    total_cost += 1000
                else:
                    total_cost += 1 # pick-up action
                    total_cost += drive_cost # drive actions
                    total_cost += 1 # drop action

            elif status == 'in':
                # Package is inside a vehicle (current_loc_or_vehicle is the vehicle name)
                vehicle = current_loc_or_vehicle
                # Find the vehicle's location
                vehicle_loc = current_vehicle_locations.get(vehicle)

                if vehicle_loc is None:
                    # Vehicle location unknown. Assign high cost.
                    total_cost += 1000
                    continue

                # Cost: drive + drop
                # Need to drive vehicle from vehicle_loc to goal_location
                drive_cost = self.distances.get((vehicle_loc, goal_location), float('inf'))

                if drive_cost == float('inf'):
                     # If unreachable, assign a very high cost
                     total_cost += 1000
                else:
                    total_cost += drive_cost # drive actions
                    total_cost += 1 # drop action

        return total_cost
