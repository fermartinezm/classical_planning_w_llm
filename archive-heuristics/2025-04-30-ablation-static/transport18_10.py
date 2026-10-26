from fnmatch import fnmatch
from collections import defaultdict, deque
# Assuming a Heuristic base class exists as shown in the examples
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
# This is just for structural completeness based on the example code.
# In a real scenario, this would be imported from the planning framework.
class Heuristic:
    def __init__(self, task):
        self.task = task # Assuming the base class stores the task
        pass # Base class might do nothing or common setup

    def __call__(self, node):
        raise NotImplementedError("Subclasses must implement this method")

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts gracefully
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Or raise an error, depending on expected input robustness
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions needed to transport each
    package to its goal location, summing the minimum costs for each package
    independently. It considers pick-up, drop, and driving actions.

    # Assumptions
    - Each action (drive, pick-up, drop) costs 1.
    - Roads are bidirectional (if (road l1 l2) exists, assume (road l2 l1) exists).
    - Vehicle capacity constraints are ignored.
    - Vehicle availability is ignored; a vehicle is assumed to be available
      at the required location whenever a package needs to be picked up or
      a package inside a vehicle needs to be driven.
    - Goal conditions only involve packages being at specific locations.

    # Heuristic Initialization
    - Parses the initial state and static facts to identify all packages,
      vehicles, locations, and the road network.
    - Computes all-pairs shortest path distances between all locations
      using Breadth-First Search (BFS).
    - Extracts the goal location for each package from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package
       and the location of every vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Check if the package is already at its goal location. If yes,
          add 0 cost for this package and continue to the next package.
       b. If the package is currently at a location `L` (and `L` is not the goal):
          - The package needs to be picked up (1 action).
          - A vehicle needs to drive from `L` to the package's goal location.
            The minimum number of drive actions is the shortest path distance
            between `L` and the goal location.
          - The package needs to be dropped at the goal location (1 action).
          - Add `1 + shortest_path_distance(L, goal_location) + 1` to the total cost.
          - If the goal location is unreachable from `L`, the state is likely
            unsolvable from this point for this package, return infinity.
       c. If the package is currently inside a vehicle `V`, and `V` is at location `L`:
          - If `L` is the package's goal location:
            - The package needs to be dropped (1 action).
            - Add `1` to the total cost.
          - If `L` is not the package's goal location:
            - Vehicle `V` needs to drive from `L` to the package's goal location.
              The minimum number of drive actions is the shortest path distance
              between `L` and the goal location.
            - The package needs to be dropped at the goal location (1 action).
            - Add `shortest_path_distance(L, goal_location) + 1` to the total cost.
            - If the goal location is unreachable from `L`, return infinity.
       d. If the package's status is unknown or invalid, return infinity.
    4. The total heuristic value is the sum accumulated in step 3.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts,
           building the road graph, and computing shortest paths."""
        super().__init__(task) # Call base class constructor
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to infer types

        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()
        self.capacity_predecessors = {} # s1 -> s2 mapping (smaller to larger)
        self.road_graph = defaultdict(set)

        # Infer object types and build road graph from initial state and static facts
        all_facts = list(self.initial_state) + list(self.static_facts)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "in":
                 package, vehicle = parts[1], parts[2]
                 self.packages.add(package)
                 self.vehicles.add(vehicle)
            elif predicate == "capacity":
                 vehicle, size = parts[1], parts[2]
                 self.vehicles.add(vehicle)
                 self.sizes.add(size)
            elif predicate == "capacity-predecessor":
                 s1, s2 = parts[1], parts[2]
                 self.sizes.add(s1)
                 self.sizes.add(s2)
                 self.capacity_predecessors[s1] = s2 # Map smaller size to larger size
            elif predicate == "road":
                 l1, l2 = parts[1], parts[2]
                 self.locations.add(l1)
                 self.locations.add(l2)
                 self.road_graph[l1].add(l2)
                 self.road_graph[l2].add(l1) # Assuming bidirectional roads

        # Add locations mentioned in 'at' facts and goals to ensure they are included
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "at":
                 self.locations.add(parts[2]) # Second argument of 'at' is a location

        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             if parts[0] == "at":
                 self.locations.add(parts[2]) # Second argument of 'at' goal is a location

        # Compute all-pairs shortest paths between locations
        self.distance = {}
        for start_loc in self.locations:
            self.distance[start_loc] = self._bfs(start_loc, self.locations)

        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure it's an 'at' goal for a package we identified
            if parts and parts[0] == "at" and parts[1] in self.packages:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

    def _bfs(self, start_node, all_nodes):
        """Computes shortest path distances from start_node to all other nodes
           in the road graph using BFS."""
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph keys before iterating neighbors
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the state is the goal state. If so, heuristic is 0.
        # This relies on the Task object having a goal_reached method.
        # If not, we would need to check self.goals <= state.
        # Let's assume self.task is available from the base class.
        if self.task.goal_reached(state):
             return 0

        # Track current status of packages and vehicles
        package_current_location = {} # package -> location (if at)
        package_in_vehicle = {} # package -> vehicle (if in)
        vehicle_locations = {} # vehicle -> location (if at)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_location[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                if package in self.packages and vehicle in self.vehicles:
                    package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that has a goal location
        for package, goal_location in self.goal_locations.items():
            # Find the package's current status
            package_cost = 0

            if package in package_current_location:
                # Package is currently on the ground at package_current_location[package]
                current_location = package_current_location[package]

                # If package is already at goal, cost is 0 for this package
                if current_location == goal_location:
                    continue

                # Package is at a location but not the goal
                # Needs pick-up (1) + drive from current_location to goal_location + drop (1)
                drive_cost = self.distance.get(current_location, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # Goal is unreachable from current location
                    return float('inf') # State is likely unsolvable

                package_cost += 1 # pick-up
                package_cost += drive_cost # drive actions
                package_cost += 1 # drop

            elif package in package_in_vehicle:
                # Package is currently inside a vehicle
                vehicle = package_in_vehicle[package]

                if vehicle not in vehicle_locations:
                     # Vehicle location unknown - invalid state?
                     return float('inf')

                vehicle_current_location = vehicle_locations[vehicle]

                # If the vehicle is already at the goal location, just need to drop
                if vehicle_current_location == goal_location:
                    package_cost += 1 # drop
                else:
                    # Vehicle needs to drive from its current location to the goal location
                    # Then package needs to be dropped.
                    drive_cost = self.distance.get(vehicle_current_location, {}).get(goal_location, float('inf'))
                    if drive_cost == float('inf'):
                        # Goal is unreachable for the vehicle
                        return float('inf') # State is likely unsolvable

                    package_cost += drive_cost # drive actions
                    package_cost += 1 # drop
            else:
                 # Package is neither at a location nor in a vehicle. Invalid state?
                 return float('inf') # State is likely unsolvable

            total_cost += package_cost

        # If the state is not the goal state but total_cost is 0, it means all
        # goal packages are at their goal locations, but other goal conditions
        # might not be met (though domain examples suggest only package locations).
        # Since the prompt allows non-admissible heuristics and prioritizes
        # minimizing expanded nodes for greedy best-first search, returning 0
        # when all package location goals are met is acceptable even if the
        # state isn't the *absolute* goal state according to all goal facts,
        # as long as the search checks the true goal state separately.
        # However, the initial check `if self.task.goal_reached(state): return 0`
        # handles the case where the full goal is met. If we reach here,
        # the state is NOT the goal state, so the heuristic should be > 0
        # unless the problem is trivially solved from here (which the goal_reached
        # check covers). The calculated total_cost should be > 0 if not at goal.
        # If total_cost is 0 here, it implies goal_locations was empty or all
        # packages were already at their goals, which contradicts the initial check.
        # So, total_cost should be > 0 if we reach this point and total_cost != inf.

        return total_cost

