from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential malformed facts
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently. The cost for a package includes pick-up (if on ground),
    driving, and drop-off. Driving cost is estimated by the shortest path distance
    in the road network. Capacity constraints are ignored.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The road network is static and traversable (all relevant locations are connected).
    - Vehicles can pick up and drop off packages at any location they are at.
    - Capacity is assumed sufficient for necessary operations (this is a simplification).
    - The cost of each action (drive, pick-up, drop) is 1.
    - Roads are bidirectional.
    - Vehicles are identifiable (e.g., by starting with 'v').

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest paths (distances) in the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of every locatable object (packages and vehicles)
       that is 'at' a location. Also, identify which package is 'in' which vehicle.
    3. For each package that has a goal location defined in the task:
        a. Determine the package's current status: Is it on the ground at some location L, or is it inside a vehicle V?
        b. If it's inside vehicle V, find the current location of vehicle V. This is the package's effective current location.
        c. Let the package's current location be L_current.
        d. Let the package's goal location be L_goal (extracted during initialization).
        e. If L_current is the same as L_goal:
            - If the package is on the ground at L_current, it's already at the goal in the correct state. Cost for this package is 0.
            - If the package is inside a vehicle at L_current, it needs to be dropped. Cost for this package is 1 (drop action).
        f. If L_current is different from L_goal:
            - Estimate the cost to move the package from L_current to L_goal.
            - This involves driving the vehicle carrying the package (or a vehicle that will pick it up) from L_current to L_goal. The estimated cost for driving is the shortest path distance between L_current and L_goal in the road network (pre-calculated).
            - If the package is currently on the ground at L_current, it needs to be picked up first (1 action).
            - The package needs to be dropped at L_goal (1 action).
            - Total cost for this package: (1 if on ground else 0) + distance(L_current, L_goal) + 1.
            - If L_current or L_goal are not in the calculated shortest paths (e.g., unreachable), add a large penalty cost instead.
    4. Sum the costs calculated for each package. This sum is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that are not affected by actions.
        self.static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal fact is (at package location)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                # else: malformed goal fact, ignore or handle error

        # Build the road network graph and collect all locations mentioned in roads.
        self.road_graph = {}
        locations_in_roads = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                 if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    locations_in_roads.add(l1)
                    locations_in_roads.add(l2)
                    self.road_graph.setdefault(l1, set()).add(l2)
                    self.road_graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional

        # Include goal locations in the set of locations to ensure BFS covers them
        # even if they are not mentioned in road facts (though they should be for solvability).
        all_relevant_locations = set(locations_in_roads)
        all_relevant_locations.update(self.goal_locations.values())

        self.locations = list(all_relevant_locations) # Store locations for BFS
        self.shortest_paths = self._compute_all_pairs_shortest_paths()


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of relevant locations
        using BFS from each location. Returns a dictionary mapping
        start_loc -> end_loc -> distance.
        """
        distances = {}
        for start_node in self.locations:
            distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0

            while q:
                current_node, dist = q.popleft()

                # If current_node is not in the road graph (e.g., a goal location with no roads),
                # it has no neighbors to explore.
                if current_node not in self.road_graph:
                     continue

                for neighbor in self.road_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where locatable objects (packages and vehicles) are.
        # This maps object_name -> location_name or vehicle_name (if inside).
        current_status = {}
        # Track vehicle locations specifically, as packages inside vehicles inherit their location.
        # This maps vehicle_name -> location_name.
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_status[obj] = loc
                    # Assuming vehicles are the only locatables that can be 'at' a location
                    # and also contain packages. PDDL types confirm vehicle is locatable.
                    # Check if the object is a vehicle based on type or naming convention.
                    # A simple check like starting with 'v' might work for typical problems.
                    # A more robust way would be to parse types from the domain, but that's complex.
                    # Let's assume 'v' prefix for vehicles based on the example state.
                    if obj.startswith('v'):
                         vehicle_locations[obj] = loc
            elif predicate == "in":
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_status[package] = vehicle # Package is inside this vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # If a package is not in the current state (e.g., wasn't in initial state),
            # it cannot be moved. This might indicate an unsolvable problem or an issue.
            # We only consider packages present in the state.
            if package not in current_status:
                 # This package is not in the state, it cannot be moved.
                 # If it's a goal package, the problem is likely unsolvable from this state.
                 # Add a large penalty.
                 total_cost += 1000 # Large penalty for missing goal package
                 continue

            current_location_or_vehicle = current_status[package]

            # Determine the package's physical location and if it's on the ground.
            is_on_ground = True
            current_location = None

            if current_location_or_vehicle in vehicle_locations:
                 # The package is inside a vehicle. Its location is the vehicle's location.
                 vehicle_name = current_location_or_vehicle
                 current_location = vehicle_locations.get(vehicle_name) # Use .get for safety
                 is_on_ground = False
            elif current_location_or_vehicle in self.locations:
                 # The package is on the ground at a known location.
                 current_location = current_location_or_vehicle
                 is_on_ground = True
            # else: current_location remains None, handled below

            # If current_location could not be determined (e.g., package in unlocated vehicle)
            if current_location is None:
                 # This state is problematic for this package.
                 # Add a large penalty.
                 # print(f"Warning: Package {package} has unknown location/status: {current_location_or_vehicle}")
                 total_cost += 1000 # Large penalty for package in unknown state
                 continue


            # If the package is already at its goal location...
            if current_location == goal_location:
                # ...check if it needs dropping.
                if not is_on_ground: # It's in a vehicle at the goal location
                    total_cost += 1 # Needs one drop action
                # If it's on the ground at the goal, cost is 0 for this package.
            else: # Package is not at its goal location
                # Estimate the cost to move from current_location to goal_location.
                # This involves driving distance + pick-up (if needed) + drop-off.

                # Get shortest path distance.
                # Check if both locations are in our pre-calculated distances and reachable.
                distance = self.shortest_paths.get(current_location, {}).get(goal_location)

                if distance is None:
                    # Goal location is unreachable from current location in the road network.
                    # This problem instance might be unsolvable, or this state is a dead end.
                    # For a non-admissible heuristic, we can return a large value.
                    # Add a large penalty to strongly discourage this state.
                    # print(f"Warning: Goal location {goal_location} unreachable from {current_location} for package {package}.")
                    total_cost += 1000 # Penalize unreachable goals strongly
                    continue # Move to the next package

                # Cost includes travel distance
                total_cost += distance

                # If on the ground, needs pick-up action
                if is_on_ground:
                    total_cost += 1 # Pick-up action

                # Always needs drop-off action at the goal location
                total_cost += 1 # Drop action

        return total_cost
