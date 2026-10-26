from collections import deque
# Assuming Heuristic base class is available as shown in the examples.
# If not, the class definition might need adjustment depending on the planner's interface.
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic:
# If inheriting from a base class is required, uncomment the line below:
# class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost to move each package to its goal location independently,
    using shortest path distances for travel. This is an additive heuristic
    that ignores capacity constraints and potential synergies from shared trips.

    # Summary
    For each package not at its goal, estimate the minimum actions:
    - If on the ground at location L: pick-up + drive from L to Goal_L + drop
      Estimated cost: 1 (pick) + shortest_path_distance(L, Goal_L) (drive actions) + 1 (drop)
      Total: 2 + shortest_path_distance(L, Goal_L)
    - If in a vehicle V which is at location L_v: drive from L_v to Goal_L + drop
      Estimated cost: shortest_path_distance(L_v, Goal_L) (drive actions) + 1 (drop)
      Total: 1 + shortest_path_distance(L_v, Goal_L)

    The total heuristic is the sum of these estimated costs for all packages
    that are not yet at their goal location.

    # Assumptions
    - Ignores vehicle capacity constraints.
    - Ignores potential synergies from transporting multiple packages together.
    - Assumes a vehicle is always available to pick up a package (the cost includes the drive for *some* vehicle, implicitly).
    - Assumes action costs are 1.

    # Heuristic Initialization
    - Extracts package goal locations from the task's goal conditions.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest path distances on the road network using BFS.

    # Step-by-Step Thinking for Computing Heuristic
    1. In `__init__`:
       a. Store the task's goals and static facts.
       b. Parse the goals to create a mapping from each package to its goal location.
       c. Parse the static 'road' facts to build an adjacency list representation of the road network graph.
       d. For every location in the graph, run a Breadth-First Search (BFS) to compute the shortest distance (in number of drive actions) to all other reachable locations. Store these distances.
    2. In `__call__` (for a given state):
       a. Parse the current state to determine the location of each package (either on the ground or inside a vehicle) and the location of each vehicle.
       b. Initialize a total heuristic cost to 0.
       c. Iterate through each package that has a goal location defined in the task.
       d. For the current package:
          i. Check if the package is already on the ground at its goal location. If yes, add 0 cost for this package and continue to the next.
          ii. If the package is on the ground at a location L (not the goal): Find the shortest distance from L to the package's goal location (Goal_L) using the precomputed distances. If unreachable, the state is likely unsolvable or very far, return infinity. Otherwise, add 2 (for pick-up and drop actions) plus the distance (for drive actions) to the total cost.
          iii. If the package is inside a vehicle V: Find the current location L_v of vehicle V. Find the shortest distance from L_v to the package's goal location (Goal_L). If unreachable, return infinity. Otherwise, add 1 (for the drop action) plus the distance (for drive actions) to the total cost. (Note: A pick-up is not needed as the package is already in a vehicle).
       e. Return the accumulated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goals and static facts,
        and precomputing shortest path distances.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract package goal locations
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Build road network graph and compute shortest paths
        self.distances = self._compute_shortest_paths()

    def _compute_shortest_paths(self):
        """
        Build the road network graph and compute all-pairs shortest paths using BFS.
        Assumes unit cost for each road segment (drive action).
        """
        graph = {}
        locations = set()

        # Build adjacency list graph from 'road' facts
        for fact in self.static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                locations.add(l1)
                locations.add(l2)
                if l1 not in graph:
                    graph[l1] = []
                graph[l1].append(l2)

        # Compute shortest paths from each location using BFS
        distances = {}
        for start_node in locations:
            distances[start_node] = {}
            # Initialize distances to infinity, distance from start to itself is 0
            for loc in locations:
                 distances[start_node][loc] = float('inf')
            distances[start_node][start_node] = 0

            queue = deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()

                # Check if current_node has outgoing roads
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            # Distance is 1 more than the distance to the current node
                            distances[start_node][neighbor] = distances[start_node][current_node] + 1
                            queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Extract current locations of packages and vehicles, and package containment
        obj_locations = {} # Maps locatable object (package or vehicle) to its location if on ground
        package_in_vehicle = {} # Maps package to vehicle if inside

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                obj_locations[obj] = location
            elif predicate == "in":
                package, vehicle = args
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_loc in self.package_goals.items():
            # Check if package is already at goal on the ground
            if package in obj_locations and obj_locations[package] == goal_loc:
                continue # Package is on the ground at the goal, no cost for this package

            # Package is not at goal on the ground, calculate cost
            cost_for_package = 0

            if package in obj_locations: # Package is on the ground at some location (not goal)
                current_loc = obj_locations[package]
                # Cost: pick-up (1) + drive (dist) + drop (1)
                # Need shortest path from package's current location to its goal location
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                if dist == float('inf'):
                    # Goal is unreachable from the package's current location
                    return float('inf')
                cost_for_package = 2 + dist

            elif package in package_in_vehicle: # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find vehicle's location
                if vehicle not in obj_locations:
                     # This state should ideally not happen in a valid problem/state space
                     # A package can only be 'in' a vehicle if the vehicle exists and is somewhere 'at' a location.
                     # Return infinity to penalize such potentially invalid states.
                     return float('inf')

                current_loc = obj_locations[vehicle] # Location of the vehicle
                # Cost: drive (dist) + drop (1)
                # Need shortest path from the vehicle's current location to the package's goal location
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                if dist == float('inf'):
                    # Goal is unreachable from the vehicle's current location
                    return float('inf')

                # If the vehicle is already at the goal location, only drop is needed
                if current_loc == goal_loc:
                     cost_for_package = 1 # just need to drop
                else:
                     cost_for_package = 1 + dist # drive + drop

            else:
                 # Package is not 'at' a location and not 'in' a vehicle.
                 # This state should not happen in a valid problem/state space.
                 # Return infinity to penalize such potentially invalid states.
                 return float('inf')

            total_cost += cost_for_package

        return total_cost
