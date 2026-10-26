from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Note: The Heuristic base class is assumed to be provided by the environment.
# If running this code standalone, you would need to define a simple base class
# like the dummy one shown in the thought block above.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, ignoring vehicle capacity and availability constraints
    beyond the need for a vehicle to perform pickup/drop/drive actions. It sums
    the estimated costs for each package independently.

    # Assumptions
    - Any vehicle can carry any package (size constraints are ignored).
    - A suitable vehicle is always available at the required location for pickup/drop/drive.
    - The cost of moving a vehicle between two locations is the shortest path distance
      in the road network.
    - Pickup and drop actions each cost 1.
    - Goal conditions primarily involve packages being at specific locations ((at ?p ?l)).
      Other goal predicates are not considered by this heuristic.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build the road network graph from static 'road' facts and locations mentioned
      in initial state and goals.
    - Precompute all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is a goal state according to the task definition. If yes, return 0.
    2. Initialize total heuristic cost to 0.
    3. Find the current location or containing vehicle for all locatables (packages and vehicles) in the state. Store this in a dictionary `current_locatables_status`.
    4. For each package 'p' that has a goal location 'l_goal' specified in `self.goal_locations`:
        a. If the fact '(at p l_goal)' is present in the current state, this goal for package 'p' is satisfied. Continue to the next package (add 0 cost for this one).
        b. If '(at p l_goal)' is NOT in the state:
            i. Get the current status of package 'p' from `current_locatables_status`. This will be either a location 'l_curr' (if on the ground) or a vehicle 'v' (if inside).
            ii. If the status is a location 'l_curr':
                - The package is on the ground at 'l_curr'.
                - The estimated cost for this package is 1 (pickup) + shortest_path_distance(l_curr, l_goal) (drive) + 1 (drop).
                - If the shortest path distance is unknown (locations are disconnected in the road network), add a large penalty (e.g., 1000).
                - Add this cost to the total.
            iii. If the status is a vehicle 'v':
                - The package is inside vehicle 'v'.
                - Get the current location 'l_v' of vehicle 'v' from `current_locatables_status`.
                - The estimated cost for this package is shortest_path_distance(l_v, l_goal) (drive) + 1 (drop).
                - If the shortest path distance is unknown (locations are disconnected in the road network), add a large penalty (e.g., 1000).
                - Add this cost to the total.
            iv. If the package's status is not found in the state facts (e.g., package not mentioned), add 0 cost for this package (assuming it's not relevant or cannot be moved).
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances in the road network.
        """
        self.task = task # Store task for access to goals and static facts

        # 1. Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.task.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other types of goal predicates if any exist

        # 2. Build the road network graph
        self.locations = set()
        self.roads = {} # Adjacency list: {loc: [neighbor1, neighbor2, ...]}

        # Find all locations from static facts, initial state, and goals
        for fact in self.task.static:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "road" and len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    self.roads.setdefault(l1, []).append(l2)
                # Add locations from other static predicates if necessary (e.g., airport in logistics)
                # In transport, only 'road' and 'at' predicates involve locations directly.
                # 'capacity' and 'capacity-predecessor' involve sizes. 'in' involves locatables.
                # So only 'road' and 'at' facts define locations we care about for the graph.

        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 self.locations.add(loc_name)

        # Add goal locations to ensure they are included in the graph nodes
        for goal_loc in self.goal_locations.values():
             self.locations.add(goal_loc)


        # Ensure all locations found are keys in the roads dictionary
        for loc in list(self.locations): # Iterate over a copy as set might change during iteration
             self.roads.setdefault(loc, [])

        # 3. Precompute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.locations:
            self.shortest_paths[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find shortest distances to all
        reachable locations.
        """
        distances = {start_node: 0}
        queue = [start_node]
        visited = {start_node}

        q_index = 0 # Manual queue index for efficiency
        while q_index < len(queue):
            curr_node = queue[q_index]
            q_index += 1

            # Get neighbors from the road network
            neighbors = self.roads.get(curr_node, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[curr_node] + 1
                    queue.append(neighbor)

        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # 1. Check if the current state is a goal state according to the task definition.
        # This is the most reliable way to ensure h=0 only at the goal.
        if self.task.goal_reached(state):
             return 0

        # 2. Initialize total heuristic cost to 0.
        total_cost = 0

        # 3. Find current locations of all locatables (packages and vehicles)
        current_locatables_status = {} # {obj_name: location_or_vehicle_name}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "at" and len(parts) == 3:
                    obj_name, loc_name = parts[1], parts[2]
                    current_locatables_status[obj_name] = loc_name
                elif parts[0] == "in" and len(parts) == 3:
                    package_name, vehicle_name = parts[1], parts[2]
                    current_locatables_status[package_name] = vehicle_name # Package is in vehicle

        # 4. Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If the fact '(at p l_goal)' is present in the current state,
            # this specific goal for package 'p' is satisfied. We add 0 cost for it.
            # We only calculate cost if this specific goal fact is NOT in the state.
            # The overall goal_reached check at the start handles the h=0 case correctly.
            # This loop calculates the *contribution* of each package towards the total heuristic.
            # If a package is already at its goal, its contribution is 0.

            # Check if the goal fact for this package is NOT in the state.
            if f"(at {package} {goal_location})" not in state:

                # Get the current status of package 'p'
                current_status = current_locatables_status.get(package)

                if current_status is None:
                     # Package not found in state facts.
                     # Add 0 cost for this package.
                     continue

                # Case i: Package is on the ground at l_curr
                if current_status in self.locations: # Check if the status is a location name
                    l_curr = current_status
                    # Need to pickup, drive, drop
                    # Cost = 1 (pickup) + dist(l_curr, l_goal) + 1 (drop)
                    distance = self.shortest_paths.get(l_curr, {}).get(goal_location)
                    if distance is not None:
                        total_cost += distance + 2
                    else:
                        # Goal location is unreachable from current location via roads.
                        # Add a large penalty.
                        total_cost += 1000

                # Case ii: Package is inside a vehicle v
                else: # current_status is a vehicle name
                    vehicle = current_status
                    # Find the location of the vehicle
                    l_v = current_locatables_status.get(vehicle)

                    if l_v is None or l_v not in self.locations:
                        # Vehicle location unknown or not a valid location.
                        # Add a large penalty.
                        total_cost += 1000
                        continue

                    # Need to drive vehicle to goal, then drop package
                    # Cost = dist(l_v, l_goal) + 1 (drop)
                    distance = self.shortest_paths.get(l_v, {}).get(goal_location)
                    if distance is not None:
                        total_cost += distance + 1
                    else:
                        # Goal location unreachable from vehicle's current location.
                        total_cost += 1000 # Large penalty

        # 5. Return the total accumulated cost.
        return total_cost
