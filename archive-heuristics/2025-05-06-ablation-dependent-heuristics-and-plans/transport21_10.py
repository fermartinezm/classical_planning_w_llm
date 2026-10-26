# Assuming Heuristic base class is available as in the example
# from heuristics.heuristic_base import Heuristic

from collections import deque
from fnmatch import fnmatch

# Dummy Heuristic base class for self-contained code block
# REMOVE THIS DUMMY CLASS WHEN INTEGRATING
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError
# END OF DUMMY CLASS


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of pattern arguments does not exceed the number of fact parts.
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for all packages that are not yet at their destination.

    # Assumptions
    - The cost of moving a package involves:
        - If on the ground: Pick-up (1 action), driving the vehicle carrying it
          to the goal location (shortest path distance in drive actions), and Drop (1 action).
        - If inside a vehicle: Driving the vehicle to the goal location
          (shortest path distance in drive actions), and Drop (1 action).
    - Vehicle capacity constraints are ignored. Any vehicle can carry any package.
    - Any vehicle can be used to transport any package. The heuristic doesn't
      assign specific vehicles to packages.
    - The cost of a 'drive' action is 1, and the minimum number of drive actions
      between two locations is the shortest path distance in the road network.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from the static `(road ?l1 ?l2)` facts.
    - Compute the shortest path distance (number of drive actions) between all
      pairs of locations using Breadth-First Search (BFS). Store these distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. For each package `p` that has a goal location `l_goal`:
        a. Check if `p` is already at `l_goal` on the ground in the current state. If `(at p l_goal)` is true, the cost for this package is 0.
        b. If `p` is not at `l_goal`, determine its current status:
            i. Is `p` currently at some location `l_current`? (i.e., `(at p l_current)` is true for `l_current != l_goal`)
            ii. Is `p` currently inside some vehicle `v`? (i.e., `(in p v)` is true)
        c. If `p` is at `l_current` (`l_current != l_goal`):
            - The estimated cost for this package is 1 (pick-up) + shortest_path_distance(`l_current`, `l_goal`) + 1 (drop).
            - Add this cost to the total heuristic.
        d. If `p` is inside vehicle `v`:
            - Find the current location `l_v` of vehicle `v` (i.e., find `(at v l_v)`).
            - The estimated cost for this package is shortest_path_distance(`l_v`, `l_goal`) + 1 (drop).
            - Add this cost to the total heuristic.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances in the road network.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional based on example instance
                self.road_graph.setdefault(l2, set()).add(l1)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS.
        # Store distances in a dictionary: (from_loc, to_loc) -> distance
        self.shortest_paths = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find shortest paths to all
        other reachable nodes in the road graph.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Handle isolated nodes not present as keys in road_graph
            if current_node not in self.road_graph:
                 continue

            for neighbor in self.road_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        # Store results in self.shortest_paths
        for end_node in self.locations:
            self.shortest_paths[(start_node, end_node)] = distances[end_node]

    def get_shortest_path_distance(self, loc1, loc2):
        """
        Retrieves the precomputed shortest path distance between two locations.
        Returns float('inf') if no path exists (or if locations are not in the graph).
        """
        # The BFS populates distances only for nodes reachable from start_node.
        # If loc1 or loc2 were not in the initial self.locations set, they won't be keys
        # in self.shortest_paths. The .get() method handles this by returning the default (inf).
        return self.shortest_paths.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state
        total_cost = 0

        # Build a quick lookup for object locations/status in the current state
        # Maps object name -> location string OR vehicle name string if 'in'
        current_status_lookup = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                current_status_lookup[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1:]
                current_status_lookup[package] = vehicle # Store the vehicle name

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at its goal location on the ground
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal location on the ground.
            # Find its current status (at a different location or in a vehicle)
            package_current_status_type = None # "at" or "in"
            package_current_location = None # Location if "at", or vehicle's location if "in"

            current_pos_or_vehicle = current_status_lookup.get(package)

            if current_pos_or_vehicle is None:
                 # This package is not mentioned in any 'at' or 'in' fact in the state.
                 # This indicates an invalid state or problem definition.
                 # Treat as unreachable.
                 return float('inf')

            # Check if the status is a location (meaning the package is 'at' that location)
            if current_pos_or_vehicle in self.locations:
                 package_current_status_type = "at"
                 package_current_location = current_pos_or_vehicle
            else:
                 # Assume the status is a vehicle name (meaning the package is 'in' that vehicle)
                 vehicle_name = current_pos_or_vehicle
                 package_current_status_type = "in"

                 # Find the location of the vehicle
                 vehicle_location = current_status_lookup.get(vehicle_name)

                 if vehicle_location is None or vehicle_location not in self.locations:
                     # Vehicle location unknown or invalid, goal likely unreachable
                     return float('inf')

                 package_current_location = vehicle_location


            # Calculate cost based on current status type and location
            if package_current_status_type == "at":
                # Package is on the ground at package_current_location
                # Cost: Pick-up (1) + Drive (shortest path from current_location to goal) + Drop (1)
                drive_dist = self.get_shortest_path_distance(package_current_location, goal_location)
                if drive_dist == float('inf'):
                    return float('inf') # Goal is unreachable
                total_cost += 1 + drive_dist + 1

            elif package_current_status_type == "in":
                # Package is inside a vehicle, which is at package_current_location (vehicle's location)
                # Cost: Drive (shortest path from vehicle_location to goal) + Drop (1)
                drive_dist = self.get_shortest_path_distance(package_current_location, goal_location)
                if drive_dist == float('inf'):
                    return float('inf') # Goal is unreachable
                total_cost += drive_dist + 1

            # If package_current_location == goal_location when package_current_status_type == "at",
            # this case is already handled by the initial check `f"(at {package} {goal_location})" in state`.
            # If package_current_location == goal_location when package_current_status_type == "in",
            # the drive_dist will be 0, and the cost is 0 + 1 = 1 (just need to drop). This is correct.


        return total_cost
