import collections
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    return fact[1:-1].split()

# Helper function for Breadth-First Search
def bfs(graph, all_locations, start_node):
    """Computes shortest path distances from start_node in an unweighted graph."""
    distances = {loc: float('inf') for loc in all_locations}

    if start_node not in all_locations:
        # Start node is not a known location, cannot reach anything
        return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # If current_node has outgoing edges
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


# Define the heuristic class
# Inherit from Heuristic base class
# class transportHeuristic(Heuristic):
class transportHeuristic: # Use this line if Heuristic base class is not provided
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing
        the estimated costs for each package that is not yet at its
        goal location. The cost for a package is estimated based on
        whether it is currently at a location or inside a vehicle,
        and the shortest path distance on the road network to its
        goal location. It ignores vehicle capacity and the specific
        vehicle used.

    Assumptions:
        - The state representation is a frozenset of PDDL fact strings.
        - Goal facts are primarily of the form '(at package location)'.
        - Vehicle locations are always explicitly stated with '(at vehicle location)'.
        - Package locations are either '(at package location)' or '(in package vehicle)'.
        - The road network defined by '(road l1 l2)' facts is static.
        - Shortest paths between locations represent the minimum number of drive actions.
        - Capacity constraints are ignored.
        - The heuristic assumes solvable states; unreachable goal locations result in infinite heuristic.
        - The Task object passed to __init__ has 'goals', 'static', and 'initial_state' attributes.

    Heuristic Initialization:
        In the constructor (__init__), the heuristic performs the following steps:
        1. Stores the goal facts.
        2. Identifies all relevant locations from static road facts, initial state, and goal facts.
        3. Builds a directed graph representing the road network from '(road l1 l2)' static facts.
        4. Computes all-pairs shortest paths between all identified locations using Breadth-First Search (BFS).
           These distances are stored in a dictionary mapping (start_location, end_location) tuples to distances.
        5. Parses the goal facts to create a mapping from each package to its goal location.

    Step-By-Step Thinking for Computing Heuristic:
        In the __call__ method, for a given state:
        1. Determines the current location or container for every locatable object (packages and vehicles)
           by examining the '(at obj loc)' and '(in pkg veh)' facts in the state.
        2. Initializes a total heuristic cost to 0.
        3. Iterates through each package that has a specified goal location in the problem definition.
        4. For the current package, it checks if the package is already at its goal location. If yes, the cost for this package is 0.
        5. If the package is not at its goal location:
           a. Retrieves the package's current status (either a location string or a vehicle string) from the state information.
           b. If the status is a vehicle string (meaning the package is inside that vehicle):
              - It finds the current location of the vehicle by looking up the vehicle in the state information (should be an '(at vehicle vehicle_location)' fact).
              - The estimated cost for this package is calculated as the shortest path distance from the vehicle's current location to the package's goal location, plus 1 action for the 'drop' operation.
           c. If the status is a location string (meaning the package is at that location):
              - The estimated cost for this package is calculated as 1 action for the 'pick-up' operation, plus the shortest path distance from the package's current location to its goal location, plus 1 action for the 'drop' operation.
           d. If the required shortest path distance is infinite (indicating the goal location is unreachable from the current location/vehicle location), the state is considered unsolvable, and the heuristic returns infinity.
        6. The estimated cost for the package is added to the total heuristic cost.
        7. After processing all packages with goal locations, the accumulated total heuristic cost is returned.
    """
    def __init__(self, task):
        # Store goal facts
        self.goals = task.goals

        # Collect all locations mentioned in road facts, initial state, and goals
        all_locations = set()
        road_graph = collections.defaultdict(list)

        # From static facts (roads define connections and locations)
        for static_fact_str in task.static:
            parts = get_parts(static_fact_str)
            if parts[0] == 'road':
                l1 = parts[1]
                l2 = parts[2]
                road_graph[l1].append(l2)
                all_locations.add(l1)
                all_locations.add(l2)
            # Capacity-predecessor facts are static but not needed for this heuristic

        # From initial state (locations of initial objects)
        for init_fact_str in task.initial_state:
             parts = get_parts(init_fact_str)
             if parts[0] == 'at':
                  all_locations.add(parts[2]) # location

        # From goals (locations packages need to reach)
        self.goal_locations = {}
        for goal_fact_str in task.goals:
            parts = get_parts(goal_fact_str)
            # Assuming goal facts are always (at package location)
            if parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location
                all_locations.add(location) # Add goal location to the set

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in all_locations:
            distances = bfs(road_graph, all_locations, start_loc)
            for end_loc in all_locations:
                self.shortest_paths[(start_loc, end_loc)] = distances[end_loc]

    def __call__(self, node):
        state = node.state

        # Map locatable objects (packages, vehicles) to their current status (location string or vehicle string)
        current_status = {}
        for fact_str in state:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'at':
                obj = parts[1] # Can be vehicle or package
                loc = parts[2]
                current_status[obj] = loc
            elif predicate == 'in':
                pkg = parts[1]
                veh = parts[2]
                current_status[pkg] = veh # Package is "located" inside the vehicle

        total_heuristic_cost = 0

        # Iterate through packages that have a goal location defined in the problem
        for package, goal_location in self.goal_locations.items():
            pkg_current_status = current_status.get(package)

            # If package is not found in the current state facts, something is wrong.
            # In a valid state derived from the initial state and actions,
            # every object should have a defined location or container.
            # For robustness, we handle this by returning infinity.
            if pkg_current_status is None:
                 return float('inf') # Should not happen in valid states

            # If package is already at goal, cost is 0 for this package
            if pkg_current_status == goal_location:
                continue

            # Determine package's actual current location (if in a vehicle) or use its current status (if at a location)
            current_package_location = None
            # Check if the status string is a key in current_status. Vehicle names are keys (mapping to location), location names are not.
            if pkg_current_status in current_status:
                vehicle = pkg_current_status
                current_package_location = current_status.get(vehicle) # Get vehicle's location

                # If vehicle location is unknown, state is invalid
                if current_package_location is None:
                     return float('inf') # Invalid state

                # Package is in vehicle at current_package_location
                # Need to drive vehicle to goal_location, then drop package
                travel_cost = self.shortest_paths.get((current_package_location, goal_location), float('inf'))
                if travel_cost == float('inf'):
                     # Goal location unreachable from vehicle location
                     return float('inf') # State is likely unsolvable

                total_heuristic_cost += travel_cost + 1 # +1 for drop action

            else: # The status string is not a vehicle name found in current_status keys, so it must be a location name
                current_package_location = pkg_current_status

                # Package is at current_package_location
                # Need to pick up, drive to goal_location, then drop
                travel_cost = self.shortest_paths.get((current_package_location, goal_location), float('inf'))
                if travel_cost == float('inf'):
                     # Goal location unreachable from package location
                     return float('inf') # State is likely unsolvable

                total_heuristic_cost += 1 + travel_cost + 1 # +1 for pick-up, +1 for drop

        return total_heuristic_cost
