import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions (copied from Logistics example, useful for parsing PDDL facts)
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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs a Breadth-First Search to find shortest distances from a start node
    to all other nodes in a graph.

    Args:
        graph: An adjacency list represented as a dictionary {node: set_of_neighbors}.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary {node: distance} containing the shortest distance from the
        start_node to every reachable node. Unreachable nodes will have distance infinity.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Handle cases where the start_node might not be in the graph (e.g., isolated location)
        # Although in transport, all locations with roads should be connected.
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Ensure current_node is a valid key
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, based on its current location (on the ground or in a vehicle)
    and the shortest path distance to its goal location in the road network.
    It ignores vehicle capacity constraints for efficiency and simplicity.

    # Assumptions:
    - The cost of pick-up, drop, and drive actions is 1.
    - Vehicle capacity is ignored. Any vehicle can theoretically carry any package.
    - The road network is static and provides shortest path distances.

    # Heuristic Initialization
    - Builds the road network graph from `(road l1 l2)` static facts.
    - Computes all-pairs shortest paths using BFS to find distances between any two locations.
    - Extracts goal locations for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  For each package that is not yet at its goal location:
    2.  Determine the package's current location:
        - If `(at package location)` is true, the package is on the ground at `location`.
        - If `(in package vehicle)` is true, find the vehicle's location `(at vehicle location)`. The package's effective location for transport planning is the vehicle's location.
    3.  Find the shortest path distance from the package's current effective location to its goal location using the precomputed distances.
    4.  Estimate the minimum actions required for this specific package journey:
        - If the package is on the ground: It needs a `pick-up` (1 action), the vehicle needs to `drive` from the package's location to the goal location (distance actions), and it needs a `drop` (1 action). Total: `1 + distance + 1 = distance + 2`.
        - If the package is inside a vehicle: The vehicle needs to `drive` from its current location to the goal location (distance actions), and the package needs a `drop` (1 action). Total: `distance + 1`.
    5.  Sum the estimated costs for all packages that are not at their goal location.
    6.  If any package's goal location is unreachable from its current location, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road network graph from static facts.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                # The domain file example shows bidirectional roads, so let's assume that.
                self.road_graph.setdefault(loc2, set()).add(loc1)

        # Ensure all locations mentioned in goals or initial state are in the graph,
        # even if they have no roads connected (they would be isolated).
        # This prevents BFS from failing on missing keys.
        all_locatables = set()
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 if loc not in self.road_graph:
                     self.road_graph[loc] = set()
                 all_locatables.add(obj) # Track all locatables initially

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 if loc not in self.road_graph:
                     self.road_graph[loc] = set()
                 all_locatables.add(obj) # Track all locatables in goal

        # Compute all-pairs shortest paths using BFS from every location.
        self.distances = {}
        for start_loc in self.road_graph:
            self.distances[start_loc] = bfs(self.road_graph, start_loc)

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Identify vehicles and packages from initial state and goals
        self.vehicles = set()
        self.packages = set()
        for fact in task.initial_state:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, _ = get_parts(fact)
                self.vehicles.add(vehicle)
            elif match(fact, "at", "*", "*"):
                 _, obj, _ = get_parts(fact)
                 # If it's not a vehicle we've seen, assume it's a package
                 if obj not in self.vehicles:
                     self.packages.add(obj)
            elif match(fact, "in", "*", "*"):
                 _, package, vehicle = get_parts(fact)
                 self.packages.add(package)
                 self.vehicles.add(vehicle)

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, _ = get_parts(goal)
                 # If it's not a vehicle we've seen, assume it's a package
                 if obj not in self.vehicles:
                     self.packages.add(obj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # {obj: location_or_vehicle}
        current_status = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                current_status[obj] = loc
            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                current_status[package] = vehicle # Package is inside this vehicle

        total_cost = 0  # Initialize action cost counter.

        # Consider only packages that have a goal location specified
        for package, goal_location in self.package_goals.items():
            # If package is not in current_status, it's likely an issue with state representation
            # or a package that doesn't exist, but we'll assume valid states.
            if package not in current_status:
                 # This shouldn't happen in a valid state/task, but handle defensively
                 # Maybe the package was never initialized? Assume it's at goal if not found.
                 continue # Or return infinity if strict

            package_current_status = current_status[package]

            # Check if the package is already at its goal location
            # If package_current_status is a vehicle, it's not on the ground at the goal
            if package_current_status == goal_location:
                 # Need to verify it's on the ground at the goal, not just the vehicle is there
                 if match(f"(at {package} {goal_location})", "at", package, goal_location):
                     continue # Package is at the goal location on the ground

            # Package is not at the goal, calculate cost for this package
            if package_current_status in self.packages or package_current_status in self.vehicles:
                 # package_current_status is a vehicle (package is inside)
                 vehicle = package_current_status
                 if vehicle not in current_status:
                      # Vehicle not found in state? Problematic state. Assume unreachable.
                      return float('inf')
                 vehicle_location = current_status[vehicle]

                 # Cost: drive vehicle from its location to goal + drop
                 if vehicle_location not in self.distances or goal_location not in self.distances[vehicle_location]:
                      # Goal location unreachable from vehicle's current location
                      return float('inf')

                 drive_cost = self.distances[vehicle_location][goal_location]
                 if drive_cost == float('inf'):
                     return float('inf')

                 total_cost += drive_cost + 1 # drive + drop

            else:
                 # package_current_status is a location (package is on the ground)
                 package_location = package_current_status

                 # Cost: pick-up + drive from package location to goal + drop
                 if package_location not in self.distances or goal_location not in self.distances[package_location]:
                      # Goal location unreachable from package's current location
                      return float('inf')

                 drive_cost = self.distances[package_location][goal_location]
                 if drive_cost == float('inf'):
                     return float('inf')

                 total_cost += 1 + drive_cost + 1 # pick-up + drive + drop


        return total_cost

