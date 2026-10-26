# Add the required import for the base class
from heuristics.heuristic_base import Heuristic
# Add imports used in the code
from fnmatch import fnmatch
from collections import deque

# Utility functions (copied from example heuristics)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It calculates the cost for each package independently
    based on its current location (on ground or in a vehicle) and its goal location.
    The cost includes pick-up/drop actions and the shortest path distance (number
    of drive actions) between relevant locations in the road network.

    # Assumptions
    - The road network is static and defines possible vehicle movements.
    - Packages need to be picked up by a vehicle to be moved between locations.
    - A package inside a vehicle moves with the vehicle.
    - Vehicle capacity is relaxed; the heuristic assumes a suitable vehicle is
      eventually available for each package movement segment (pick-up, drop).
    - The shortest path distance between locations represents the minimum drive actions.
    - All locations mentioned in goals or road facts are part of the road network graph.
    - For solvable problems, all goal locations are reachable from initial package/vehicle locations.

    # Heuristic Initialization
    - Parse goal conditions to map each package to its goal location.
    - Parse static facts (`road` predicates) to build the road network graph.
    - Compute all-pairs shortest path distances between all locations using BFS.
      This precomputation makes the heuristic calculation in __call__ efficient.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic is 0.
    2. Identify the current location of every package and vehicle. Note if a package is inside a vehicle.
       - Create a map `current_locations` for objects (`at` predicate).
       - Create a map `in_vehicle_map` for packages inside vehicles (`in` predicate).
       - Create a map `vehicle_locations` for vehicles (`at` predicate).
       - If a package is `in` a vehicle, its location is considered inside the vehicle, not `at` a location on the ground.
    3. Initialize total heuristic cost to 0.
    4. Iterate through each package `p` that has a defined goal location (`self.goal_locations`).
    5. Determine the current status of package `p`:
       a. If `p` is currently on the ground at its goal location (`p` in `current_locations` and `current_locations[p] == self.goal_locations[p]`): The package is already at its goal. Cost for this package is 0. Continue to the next package.
       b. If `p` is currently on the ground at `current_l` (`p` in `current_locations` and `current_l != self.goal_locations[p]`):
          - The package needs to be picked up, transported by a vehicle from `current_l` to `goal_l`, and dropped at `goal_l`.
          - The minimum number of drive actions is the shortest distance between `current_l` and `goal_l` in the road network.
          - Cost for this package: 1 (pick-up) + shortest_distance(current_l, goal_l) + 1 (drop).
          - If `goal_l` is unreachable from `current_l`, the problem is likely unsolvable via this path; return infinity.
          - Add this cost to the total.
       c. If `p` is currently inside a vehicle `v` (`p` in `in_vehicle_map`):
          - Find the current location of vehicle `v`, say `vehicle_l` (`v` in `vehicle_locations`). (Assume vehicle location is always known).
          - If `vehicle_l == self.goal_locations[p]`: The vehicle is already at the package's goal location. The package only needs to be dropped.
            - Minimum actions: 1 (drop).
            - Add this cost to the total.
          - If `vehicle_l != self.goal_locations[p]`: The vehicle needs to drive from `vehicle_l` to `goal_l`, and then the package needs to be dropped.
            - The minimum number of drive actions is the shortest distance between `vehicle_l` and `goal_l`.
            - Minimum actions: shortest_distance(vehicle_l, goal_l) + 1 (drop).
            - If `goal_l` is unreachable from `vehicle_l`, return infinity.
            - Add this cost to the total.
       d. If a package is neither `at` a location nor `in` a vehicle, this indicates an invalid state. Return infinity.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        and analyzing the road network.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at ?p ?l)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
            # Note: Other goal types like (in ?p ?v) are not handled by this heuristic
            # as the typical goal is package location. If goals include (in ?p ?v),
            # this heuristic might be inaccurate or need extension. Assuming (at ?p ?l) goals.


        # Build the road network graph (adjacency list).
        self.road_graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = set()
                if loc2 not in self.road_graph:
                    self.road_graph[loc2] = set()
                self.road_graph[loc1].add(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph[loc2].add(loc1)

        # Ensure all locations mentioned in goals are included in the set of all locations
        # even if they are isolated in the static road facts (e.g., goal location with no roads).
        # BFS needs a complete list of nodes it might encounter.
        for loc in self.goal_locations.values():
             all_locations.add(loc)
             # Add to graph if not present, though BFS handles missing keys gracefully
             if loc not in self.road_graph:
                 self.road_graph[loc] = set()


        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = self._bfs(start_loc, all_locations)

        # Capacity information is not explicitly used in this relaxed heuristic
        # self.capacity_predecessors = {}
        # for fact in static_facts:
        #     if match(fact, "capacity-predecessor", "*", "*"):
        #         _, s1, s2 = get_parts(fact)
        #         self.capacity_predecessors[s2] = s1 # s1 is smaller than s2

    def _bfs(self, start_node, all_nodes):
        """
        Performs BFS from a start_node to find distances to all other nodes
        in the road graph. Returns a dictionary mapping nodes to their distances.
        Nodes unreachable from start_node will have distance float('inf').
        """
        distances = {node: float('inf') for node in all_nodes}
        # Handle cases where start_node might not be in the graph (e.g., isolated goal location)
        if start_node not in all_nodes:
             # This shouldn't happen if all_nodes is built correctly from roads and goals
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Get neighbors from the graph, handle nodes that might be in all_nodes
            # but not have any roads connected (isolated).
            neighbors = self.road_graph.get(current_node, set())

            for neighbor in neighbors:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the goal is reached, the heuristic is 0.
        # This check is important for correctness, especially if the loop below
        # calculates a non-zero value due to intermediate states that satisfy
        # some package goals but not the overall task goal.
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located or contained.
        current_locations = {} # obj -> location (if at)
        in_vehicle_map = {}    # package -> vehicle (if in)
        vehicle_locations = {} # vehicle -> location (if at)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                # Store vehicle locations separately for easy lookup
                # Assuming vehicles start with 'v' based on domain/instance examples
                if obj.startswith('v'):
                    vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                in_vehicle_map[package] = vehicle
                # If a package is 'in' a vehicle, it's not 'at' a location on the ground
                if package in current_locations:
                    del current_locations[package]


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its goal location on the ground
            if package in current_locations and current_locations[package] == goal_location:
                # Package is at goal, cost for this package is 0
                continue

            # Package is not at its goal location on the ground.
            # It must be either on the ground elsewhere or inside a vehicle.

            if package in current_locations:
                # Package is on the ground at current_l, not at goal
                current_location = current_locations[package]
                # Cost: pick-up + drive + drop
                # Need distance from current_location to goal_location
                drive_cost = self.distances.get(current_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from current location. Problem likely unsolvable.
                    return float('inf')

                total_cost += 1 # pick-up action
                total_cost += drive_cost # drive actions
                total_cost += 1 # drop action

            elif package in in_vehicle_map:
                # Package is inside a vehicle v
                vehicle = in_vehicle_map[package]
                # Find the vehicle's location
                vehicle_location = vehicle_locations.get(vehicle)

                if vehicle_location is None:
                     # This state indicates a vehicle containing a package has no location.
                     # This shouldn't happen in valid states/plans. Treat as unsolvable.
                     # print(f"Error: Vehicle {vehicle} containing {package} has no location in state.")
                     return float('inf') # Indicate invalid/unsolvable state

                if vehicle_location == goal_location:
                    # Vehicle is at the goal location, just need to drop the package
                    total_cost += 1 # drop action
                else:
                    # Vehicle needs to drive to goal_location, then drop the package
                    drive_cost = self.distances.get(vehicle_location, {}).get(goal_location, float('inf'))

                    if drive_cost == float('inf'):
                         # Goal is unreachable from vehicle location. Problem likely unsolvable.
                         return float('inf')

                    total_cost += drive_cost # drive actions
                    total_cost += 1 # drop action

            else:
                 # Package is neither 'at' a location nor 'in' a vehicle.
                 # This state should not be possible for a locatable object in this domain.
                 # Indicates an invalid state representation.
                 # print(f"Error: Package {package} is neither at a location nor in a vehicle.")
                 return float('inf') # Indicate invalid/unsolvable state

        return total_cost
