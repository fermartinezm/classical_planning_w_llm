from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    to its goal location independently. It considers the actions required to
    pick up, drop, and drive the package, using shortest path distances for
    drive actions.

    # Assumptions
    - Each package can be moved independently.
    - Vehicle capacity and availability are not strictly modeled in the action count,
      assuming a suitable vehicle is available when needed.
    - Roads are bidirectional.
    - The cost of each action (pick-up, drop, drive) is 1.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representation of the road network from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state (i.e., all goal facts are present). If yes, the heuristic is 0.
    2. If not a goal state, initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Determine the package's current status: Is it on the ground at a location `L_current`, or is it inside a vehicle `V`?
       b. If it's inside a vehicle `V`, find the current location `L_vehicle` of that vehicle. The package's effective current location for transport planning is `L_vehicle`.
       c. If the package is already at its goal location (i.e., `(at package goal_location)` is in the state), the cost for this package is 0. Continue to the next package.
       d. If the package is on the ground at `L_current` and `L_current` is not the goal location:
          - The minimum actions needed are: pick-up (1) + drop (1) + drive actions.
          - The number of drive actions is the shortest path distance from `L_current` to `goal_location`.
          - Add 1 + 1 + shortest_path(L_current, goal_location) to the total cost.
       e. If the package is inside a vehicle `V` which is at `L_vehicle`, and `L_vehicle` is not the goal location:
          - The minimum actions needed are: drop (1) + drive actions.
          - The number of drive actions is the shortest path distance from `L_vehicle` to `goal_location`.
          - Add 1 + shortest_path(L_vehicle, goal_location) to the total cost.
       f. If the package is inside a vehicle `V` which is already at `goal_location`:
          - The minimum action needed is: drop (1).
          - Add 1 to the total cost.
    4. Return the total calculated cost. If any required shortest path is infinite (goal unreachable), the total cost will be infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions (frozenset of strings)
        static_facts = task.static  # Static facts (frozenset of strings)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and collect all locations.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in locations:
            distances = self._bfs(start_loc)
            for end_loc, dist in distances.items():
                self.shortest_paths[(start_loc, end_loc)] = dist

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {start_node: 0}
        queue = deque([start_node]) # Use deque for efficient queue operations
        visited = {start_node}

        while queue:
            current_node = queue.popleft() # Dequeue from the left

            # Ensure current_node is a valid key in the graph before accessing neighbors
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor) # Enqueue to the right
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Check if the current state is the goal state
        if self.goals.issubset(state):
            return 0

        # Find current location/status for all locatables (packages and vehicles)
        current_locations = {} # Maps locatable -> location string OR vehicle string (if in vehicle)
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args # obj is vehicle or package
                current_locations[obj] = location
            elif predicate == "in":
                package, vehicle = args # package is in vehicle
                current_locations[package] = vehicle

        total_cost = 0

        # Consider each package that has a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is already at goal location on the ground, it's done.
            if f"(at {package} {goal_location})" in state:
                continue

            # Package is not at goal on the ground. Find its current status.
            current_status = current_locations.get(package)

            # This case indicates an issue with state representation or parsing,
            # as every package should be either at a location or in a vehicle.
            # Returning infinity signals an unsolvable or malformed state path.
            if current_status is None:
                 return float('inf')

            # Case 1: Package is on the ground at a location (current_status is a location string)
            # Check if the status is a known location by seeing if it's a key in the road graph
            if current_status in self.road_graph:
                current_package_location = current_status

                # If the package is on the ground but not at the goal, it needs transport.
                # Actions: pick-up, drive(s), drop.
                drive_cost = self.shortest_paths.get((current_package_location, goal_location), float('inf'))

                if drive_cost == float('inf'):
                     # Goal is unreachable for this package
                     return float('inf')

                total_cost += 1 # pick-up action
                total_cost += 1 # drop action
                total_cost += drive_cost # drive actions

            # Case 2: Package is inside a vehicle (current_status is a vehicle name)
            else: # current_status is a vehicle name
                vehicle_name = current_status
                # Find the vehicle's current location
                current_vehicle_location = current_locations.get(vehicle_name)

                # Vehicle location unknown - should not happen in valid state
                if current_vehicle_location is None:
                     return float('inf')

                # If the vehicle is not at the goal location, it needs to drive there.
                # Actions: drive(s), drop.
                # If the vehicle *is* at the goal location, it just needs to drop.
                drive_cost = self.shortest_paths.get((current_vehicle_location, goal_location), float('inf'))

                if drive_cost == float('inf'):
                     # Goal is unreachable for this package
                     return float('inf')

                total_cost += 1 # drop action
                total_cost += drive_cost # drive actions (0 if vehicle is already at goal)

        return total_cost
