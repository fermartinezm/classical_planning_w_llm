from collections import defaultdict, deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided in the execution environment
# In a real scenario, this would be imported from the planning framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using a dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Subclasses must implement this method")


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # For this domain, facts are expected to be strings like '(predicate arg1 arg2)'
        # If a fact is not in this format, it might be a malformed state element.
        # Returning an empty list or raising an error could be options.
        # For robustness, let's return an empty list, though valid states should not have this.
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, ignoring vehicle capacity constraints
    and potential conflicts between packages needing the same vehicle or path.
    It sums the estimated costs for each package independently.

    For a package not at its goal:
    - If on the ground: 1 (pick-up) + shortest_path_distance (drives) + 1 (drop).
    - If in a vehicle: shortest_path_distance (drives) + 1 (drop).

    # Assumptions
    - All actions (drive, pick-up, drop) have a cost of 1.
    - Vehicles are always available when needed by a package (ignoring capacity and location).
    - Shortest path distances between locations are precomputed.
    - The goal state is defined by the 'at' predicate for packages.
    - Objects listed in goal 'at' predicates are packages. Objects listed in state 'at' predicates
      that are not goal packages are vehicles.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build a graph of locations based on 'road' facts.
    - Compute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Determine the current location of every package and vehicle from the state.
       A package can be at a location ('at' predicate) or inside a vehicle ('in' predicate).
       A vehicle is always at a location ('at' predicate).
       Store these in dictionaries: `package_location` and `vehicle_location`.
    3. For each package that has a specified goal location:
       a. Get the package's goal location (precomputed during initialization).
       b. Get the package's current location from the state information gathered in step 2.
       c. If the package is already at its goal location (i.e., `(at package goal_location)` is true), the cost for this package is 0. Continue to the next package.
       d. If the package is currently at a location (not inside a vehicle) and not at its goal:
          - The package needs to be picked up (1 action).
          - A vehicle needs to drive from the package's current location to its goal location. The minimum number of drive actions is the shortest path distance between these locations (precomputed).
          - The package needs to be dropped at the goal location (1 action).
          - The total cost for this package is 1 + distance + 1.
       e. If the package is currently inside a vehicle:
          - Find the current location of the vehicle from the state information gathered in step 2.
          - The vehicle needs to drive from its current location to the package's goal location. The minimum number of drive actions is the shortest path distance between these locations (precomputed).
          - The package needs to be dropped at the goal location (1 action).
          - The total cost for this package is distance + 1.
       f. Add the calculated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest path distances.
        """
        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify all package names from the goals
        self.package_names = set()
        for goal in task.goals:
            parts = get_parts(goal)
            # Goal facts are typically '(at package location)'
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.package_names.add(package)

        # Build the road graph from static facts.
        self.road_graph = defaultdict(set)
        locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.road_graph[loc1].add(loc2)
                self.road_graph[loc2].add(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)

        # Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = self._bfs(start_loc, locations)

    def _bfs(self, start_node, all_nodes):
        """
        Perform a Breadth-First Search to find shortest distances from start_node
        to all other nodes in the road graph.
        """
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is a valid key in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Check if neighbor is a valid node we are tracking distances for
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """
        Get the precomputed shortest distance between two locations.
        Returns float('inf') if no path exists.
        """
        if loc1 == loc2:
            return 0
        # Ensure both locations are in our precomputed distances map
        if loc1 in self.distances and loc2 in self.distances[loc1]:
             return self.distances[loc1][loc2]
        # If a location is not in the graph (e.g., isolated), distance is infinite
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Map current locations of packages and vehicles.
        # package_location: package_name -> location_name or vehicle_name
        # vehicle_location: vehicle_name -> location_name
        package_location = {}
        vehicle_location = {}

        # Extract current locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts if any
                continue
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Use the precomputed package names from goals to distinguish
                if obj in self.package_names: # It's a package we care about
                     package_location[obj] = loc
                else: # Assume it's a vehicle
                    vehicle_location[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Only track packages that are relevant to the goals
                if package in self.package_names:
                    package_location[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach its goal
        for package, goal_loc in self.goal_locations.items():
            # If package is not found in the current state facts, something is wrong
            # or it's not relevant to the goal (though goal_locations only includes goal packages).
            # Assuming all goal packages are present in the state.
            if package not in package_location:
                 # This case should ideally not happen in a valid planning problem state
                 # derived from the initial state if the package was initially present.
                 # If it could happen, we might need to add a large penalty or handle it.
                 # For now, we assume the package is always findable in the state.
                 continue

            curr_loc = package_location[package]

            # Check if the package is already at its goal location on the ground
            # The goal is (at package location), not (in package vehicle).
            if curr_loc == goal_loc:
                 continue # Cost is 0 for this package

            # If the package is not at the goal location, it needs actions.
            # Minimum actions for a package not at its goal:
            # 1. Get it into a vehicle (if on ground): 1 (pick-up)
            # 2. Drive vehicle to goal location: distance(vehicle_loc, goal_loc)
            # 3. Get it out of the vehicle: 1 (drop)

            package_cost = 0

            if curr_loc in vehicle_location: # Package is inside a vehicle
                vehicle_name = curr_loc
                # Need the location of the vehicle
                if vehicle_name not in vehicle_location:
                    # This implies a package is 'in' a vehicle, but the vehicle's 'at' location is missing.
                    # This shouldn't happen in a valid state, but handle defensively.
                    # If the vehicle location is unknown, we can't estimate drive cost.
                    # Return infinity or a very large number to prune this path.
                    return float('inf') # Cannot estimate cost if vehicle location is unknown

                vehicle_loc = vehicle_location[vehicle_name]

                # Package is in vehicle at vehicle_loc, needs to get to goal_loc
                # Needs to drive from vehicle_loc to goal_loc and then drop.
                dist = self.get_distance(vehicle_loc, goal_loc)
                if dist == float('inf'): return float('inf') # Unreachable goal location
                package_cost += dist # drives
                package_cost += 1 # drop action

            else: # Package is on the ground at curr_loc
                # Package is at curr_loc, needs to get to goal_loc
                # Needs to be picked up, vehicle drives from curr_loc to goal_loc, then drop.
                dist = self.get_distance(curr_loc, goal_loc)
                if dist == float('inf'): return float('inf') # Unreachable goal location
                package_cost += 1 # pick-up action
                package_cost += dist # drives
                package_cost += 1 # drop action

            total_cost += package_cost

        # The heuristic should be 0 only at the goal state.
        # The loop above calculates cost only for packages NOT at their goal.
        # If all packages are at their goal, total_cost will be 0.
        # If the state is NOT a goal state but all goal packages ARE at their goal,
        # this heuristic correctly returns 0. This is acceptable for greedy search.
        # If the state IS a goal state, task.goal_reached(state) is True, and our
        # loop correctly calculates 0 if all goal packages are at their goal.
        # So, h=0 iff all goal packages are at their goal.

        return total_cost

