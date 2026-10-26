# Assuming the Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

from collections import deque

# Helper function to extract components from a PDDL fact string.
# Handles standard '(predicate arg1 arg2)' format.
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Ensure input is a string and strip whitespace
    fact_str = str(fact).strip()
    # Check if it looks like a PDDL fact (starts with '(' and ends with ')')
    if fact_str.startswith('(') and fact_str.endswith(')'):
        # Remove parentheses and split by whitespace
        return fact_str[1:-1].split()
    else:
        # This case should ideally not be reached for valid facts from state/static/goals.
        # Returning the split string is a fallback.
        return fact_str.split()

# Helper function to build the road graph and compute all-pairs shortest paths.
def build_road_graph_and_distances(static_facts):
    """
    Builds a graph from road facts and computes all-pairs shortest paths
    using BFS.

    Args:
        static_facts: A frozenset of static PDDL facts (as strings).

    Returns:
        A dictionary where distances[l1][l2] is the shortest path distance
        from location l1 to location l2. Returns an empty dictionary if no
        road facts are present.
    """
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'road' and len(parts) == 3:
            l1, l2 = parts[1], parts[2]
            locations.add(l1)
            locations.add(l2)
            graph.setdefault(l1, set()).add(l2)
            graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads

    distances = {}
    for start_loc in locations:
        distances[start_loc] = {}
        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        while queue:
            current_loc, dist = queue.popleft()
            distances[start_loc][current_loc] = dist
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    return distances

# Define the heuristic class, inheriting from Heuristic if available
# class transportHeuristic(Heuristic):
class transportHeuristic: # Using this if Heuristic base class is not provided externally
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    actions required for each package to reach its goal location, considering
    pickup, drop, and travel distance. It ignores vehicle capacity constraints
    and potential conflicts when multiple packages need the same vehicle.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from static facts.
    - Precomputes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current position. This can be a location (on the ground)
       or inside a vehicle.
    2. If the package is on the ground at location L_p and needs to go to L_goal:
       - Estimate cost as 1 (pickup) + shortest_distance(L_p, L_goal) (drive) + 1 (drop).
    3. If the package is inside a vehicle V, and V is at location L_v, and the package
       needs to go to L_goal:
       - Estimate cost as shortest_distance(L_v, L_goal) (drive) + 1 (drop).
    4. Sum the estimated costs for all packages not at their goal.
    5. If any required location is unreachable (e.g., disconnected graph), return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations based on road network.

        Args:
            task: The planning task object containing initial state, goals, operators, and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are expected to be of the form (at package location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other potential goal types if any, though domain description
            # and examples suggest only 'at' goals for packages.

        # Build road graph and compute all-pairs shortest paths
        self.distances = build_road_graph_and_distances(self.static)

        # Identify all locations mentioned in the problem (from distances keys)
        self.all_locations = set(self.distances.keys())

        # Note: This heuristic ignores vehicle capacity and specific vehicle assignment,
        # focusing only on the package's movement needs and location distances.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state (a frozenset of facts).

        Returns:
            An integer representing the estimated cost to the goal, or float('inf')
            if the goal seems unreachable from the current state based on distances.
        """
        state = node.state

        # Map locatable objects (packages, vehicles) to their current position.
        # A position can be a location name (if at a location) or a vehicle name
        # (if inside a vehicle).
        current_positions = {}
        # Also track vehicle locations specifically, as packages inside vehicles
        # need the vehicle's location.
        vehicle_locations = {}

        # Iterate through facts in the current state to find object positions
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_positions[obj] = loc
                # Assume any object starting with 'v' is a vehicle for tracking its location.
                # This is a domain-specific assumption based on examples.
                # A more robust approach would use type information from the task definition.
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                current_positions[pkg] = veh # Package is inside this vehicle

            # Ignore other predicates like 'capacity', 'capacity-predecessor', 'road'
            # as they are not directly used in this simplified distance heuristic.

        total_cost = 0

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            current_pos = current_positions.get(package)

            # If a package from the goal is not found in the current state's positions,
            # it's an unexpected state. Treat as unreachable.
            if current_pos is None:
                 # This should ideally not happen in a valid planning problem state.
                 # Returning infinity signals this path is likely invalid or unsolvable.
                 return float('inf')

            # Check if the package is already at its goal location
            if current_pos == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # Determine if the package is on the ground or in a vehicle
            # We check if the current position string is one of the known location names.
            if current_pos in self.all_locations:
                # Package is on the ground at current_pos (which is a location name)
                package_loc = current_pos
                target_loc = goal_location

                # Cost = pickup (1) + drive (dist) + drop (1)
                # Get the shortest distance from the package's current location to the goal location.
                # Use .get() with a default of None to handle cases where a location might not
                # be in the distances map (e.g., disconnected graph components).
                # If the goal location itself is not in the distance map (e.g., not in static roads),
                # this also correctly results in None.
                drive_cost = self.distances.get(package_loc, {}).get(target_loc)

                if drive_cost is None:
                    # Goal location is unreachable from the package's current location.
                    # This state is likely unsolvable or requires complex steps not
                    # captured by this simple distance heuristic. Return infinity.
                    return float('inf')

                total_cost += 1 + drive_cost + 1 # pickup + drive + drop

            else: # current_pos is not a location name, assume it's a vehicle name
                # Package is inside vehicle current_pos
                vehicle_name = current_pos
                vehicle_loc = vehicle_locations.get(vehicle_name)
                target_loc = goal_location

                # If vehicle location is unknown or not a valid location, treat as unreachable.
                if vehicle_loc is None or vehicle_loc not in self.all_locations:
                    return float('inf')

                # Cost = drive (dist) + drop (1)
                # Get the shortest distance from the vehicle's current location to the goal location.
                drive_cost = self.distances.get(vehicle_loc, {}).get(target_loc)

                if drive_cost is None:
                     # Goal location is unreachable from the vehicle's current location.
                     return float('inf')

                total_cost += drive_cost + 1 # drive + drop

        return total_cost
