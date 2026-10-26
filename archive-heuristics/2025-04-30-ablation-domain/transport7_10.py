from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in `args`.
    """
    parts = get_parts(fact)
    # Use zip to handle cases where the fact might have more parts than the pattern args
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location independently. It sums the estimated costs for each
    package. The cost for a single package includes loading, unloading, and
    the shortest path distance for a vehicle to transport it.

    # Assumptions
    - The cost of any action (load, unload, drive) is 1.
    - Vehicle capacity is ignored; any vehicle can carry any package.
    - The availability of a vehicle at the package's location for loading is not explicitly modeled in the cost, only the package's journey is considered.
    - Multiple packages cannot share a vehicle trip in the heuristic calculation (costs are summed independently).
    - The shortest path between locations is the number of drive actions required.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations based on `road` facts, including all locations mentioned in initial state and goals.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Extracts vehicle capacity mapping from `capacity-predecessor` facts (although capacity is ignored in the current heuristic calculation, it's good practice to parse relevant static info).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or vehicle status for every package and vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `L_goal`:
        a. Find the current state of `p`:
           - If `(at p L_curr)` is true in the state: `p` is on the ground at `L_curr`.
           - If `(in p v)` is true for some vehicle `v`, find `v`'s location `L_veh` using `(at v L_veh)`: `p` is in vehicle `v` which is at `L_veh`.
        b. If `p` is already at its goal location (`(at p L_goal)` is true):
           - The cost for this package is 0.
        c. If `p` is on the ground at `L_curr` (`L_curr != L_goal`):
           - Estimated cost for `p`: 1 (load) + shortest_distance(`L_curr`, `L_goal`) + 1 (unload).
        d. If `p` is in vehicle `v` which is at `L_veh`:
           - If `L_veh == L_goal`: Estimated cost for `p`: 1 (unload).
           - If `L_veh != L_goal`: Estimated cost for `p`: shortest_distance(`L_veh`, `L_goal`) + 1 (unload).
    4. Sum the estimated costs for all packages.
    5. Return the total sum as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all locations

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
            # Other goal types are ignored for this package-centric heuristic

        # 2. Build the location graph from 'road' facts and collect all locations.
        self.road_graph = {}
        all_locations_set = set()

        # Collect locations from road facts and build graph
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc_from, loc_to = get_parts(fact)
                if loc_from not in self.road_graph:
                    self.road_graph[loc_from] = []
                self.road_graph[loc_from].append(loc_to)
                all_locations_set.add(loc_from)
                all_locations_set.add(loc_to)

        # Collect locations from initial state 'at' facts
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_locations_set.add(loc)
                 # Ensure location is in graph keys even if no roads
                 if loc not in self.road_graph:
                     self.road_graph[loc] = []

        # Collect locations from goal 'at' facts
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 all_locations_set.add(loc)
                 # Ensure location is in graph keys even if no roads
                 if loc not in self.road_graph:
                     self.road_graph[loc] = []

        all_locations = list(all_locations_set) # Use the collected set

        # 3. Compute all-pairs shortest path distances.
        self.distances = {}

        for start_node in all_locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                curr_node, curr_dist = q.popleft()

                # Get neighbors from the graph (use .get for safety)
                for neighbor in self.road_graph.get(curr_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = curr_dist + 1
                        q.append((neighbor, curr_dist + 1))

        # 4. Extract capacity mapping from 'capacity-predecessor' facts.
        # This mapping is not used in the current heuristic calculation,
        # but is extracted as per documentation requirement.
        self.capacity_map = {'c0': 0} # Base capacity mapping
        capacity_successors = {} # Map predecessor to successor
        for fact in static_facts:
             if match(fact, "capacity-predecessor", "*", "*"):
                 _, s1, s2 = get_parts(fact)
                 capacity_successors[s1] = s2

        current_size = 'c0'
        capacity_value = 0
        while current_size in capacity_successors:
             next_size = capacity_successors[current_size]
             capacity_value += 1
             self.capacity_map[next_size] = capacity_value
             current_size = next_size


    def get_distance(self, loc_from, loc_to):
        """
        Get the precomputed shortest distance between two locations.
        Returns float('inf') if not reachable.
        """
        # If either location is not in our collected set (shouldn't happen with robust collection),
        # or if there's no path, distance will be missing or inf.
        return self.distances.get((loc_from, loc_to), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current locations/status of packages and vehicles.
        package_status = {} # package_name -> {'type': 'at', 'location': loc} or {'type': 'in', 'vehicle': veh}
        vehicle_locations = {} # vehicle_name -> location_name

        # Collect vehicle locations first
        for fact in state:
            if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 # Assuming objects starting with 'v' are vehicles based on example
                 if obj.startswith('v'):
                     vehicle_locations[obj] = loc

        # Collect package locations/status
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                # Assuming objects starting with 'p' are packages based on example
                # and checking if they are relevant to goals
                if obj.startswith('p') and obj in self.goal_locations:
                     package_status[obj] = {'type': 'at', 'location': loc}
            elif match(fact, "in", "*", "*"):
                _, pkg, veh = get_parts(fact)
                # Only track packages relevant to goals
                if pkg.startswith('p') and pkg in self.goal_locations:
                    package_status[pkg] = {'type': 'in', 'vehicle': veh}


        total_cost = 0  # Initialize action cost counter.

        # 3. Calculate cost for each package not at its goal.
        for package, goal_location in self.goal_locations.items():
            # Check if package is at goal
            # A package is at goal if it's on the ground at the goal location.
            if package in package_status and package_status[package]['type'] == 'at' and package_status[package]['location'] == goal_location:
                # Package is already at its goal location on the ground. Cost is 0 for this package.
                continue

            # Package is not at goal, calculate its cost
            pkg_cost = 0

            if package not in package_status:
                 # This package is relevant to the goal but its current status
                 # (at or in) is not in the state. This indicates an invalid state
                 # or a package we cannot track. Assign high cost.
                 return float('inf') # Or handle as error

            current_status = package_status[package]

            if current_status['type'] == 'at':
                # Package is on the ground at L_curr, L_curr != L_goal
                current_location = current_status['location']
                # Needs load, drive, unload
                pkg_cost += 1 # load
                pkg_cost += self.get_distance(current_location, goal_location) # drive
                pkg_cost += 1 # unload

            elif current_status['type'] == 'in':
                # Package is in a vehicle v
                vehicle = current_status['vehicle']
                # Find vehicle's location
                if vehicle not in vehicle_locations:
                    # Vehicle location is unknown. Cannot estimate travel cost.
                    # Assign high cost as this state might be problematic or lead to unsolvable path.
                    return float('inf')

                vehicle_location = vehicle_locations[vehicle]

                if vehicle_location == goal_location:
                    # Package is in vehicle at goal location. Needs unload.
                    pkg_cost += 1 # unload
                else:
                    # Package is in vehicle at L_veh, L_veh != L_goal. Needs drive, unload.
                    pkg_cost += self.get_distance(vehicle_location, goal_location) # drive
                    pkg_cost += 1 # unload

            # If any step in the package's journey involves an unreachable location,
            # the distance will be inf, making pkg_cost inf.
            if pkg_cost == float('inf'):
                 return float('inf')

            total_cost += pkg_cost

        return total_cost
