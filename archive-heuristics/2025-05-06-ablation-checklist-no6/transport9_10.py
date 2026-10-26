from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in 'heuristics.heuristic_base'
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each package to its goal
    location independently, ignoring vehicle capacity and availability constraints.
    The estimate for a package is:
    - 0 if already at the goal.
    - 2 + shortest_road_distance(current_location, goal_location) if on the ground.
      (1 pick-up + N drive + 1 drop)
    - 1 + shortest_road_distance(vehicle_location, goal_location) if inside a vehicle.
      (N drive + 1 drop)

    Precomputes shortest path distances between all locations based on road facts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
            # Ignore other goal types if any, as the heuristic focuses on package delivery.

        # Build the road graph from static facts.
        self.road_graph = {}
        locations_in_roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations_in_roads.add(l1)
                locations_in_roads.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional based on example instance files.
                self.road_graph.setdefault(l2, set()).add(l1)

        # Collect all relevant locations: those in roads and those in goals.
        self.all_locations = list(locations_in_roads.union(set(self.goal_locations.values())))

        # Ensure all relevant locations are keys in the graph dictionary, even if isolated.
        for loc in self.all_locations:
             self.road_graph.setdefault(loc, set())


        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find shortest distances
        to all other nodes in the road graph.
        Returns a dictionary mapping location -> distance.
        """
        distances = {node: float('inf') for node in self.all_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # current_node is guaranteed to be in self.road_graph keys because of pre-population
            for neighbor in self.road_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def get_shortest_distance(self, loc1, loc2):
        """
        Retrieves the precomputed shortest distance between two locations.
        Returns float('inf') if unreachable or locations are not in the graph.
        """
        if loc1 == loc2:
            return 0
        # Ensure both locations are part of the graph we computed distances for
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf') # Should not happen if all_locations is built correctly
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        # Maps locatable object name (package or vehicle) to its state info:
        # - If (at obj loc), maps obj -> loc
        # - If (in package vehicle), maps package -> vehicle
        current_state_info = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                current_state_info[obj] = loc
            elif predicate == "in":
                package, vehicle = args
                current_state_info[package] = vehicle # Store the vehicle name

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location.
            # The goal is (at package goal_location).
            goal_fact = f"(at {package} {goal_location})"
            if goal_fact in state:
                # Package is already at its goal.
                continue

            # Package is not at its goal. Estimate cost.
            package_current_info = current_state_info.get(package)

            if package_current_info is None:
                 # Package location/state is unknown. Treat as unreachable.
                 total_cost += float('inf')
                 continue

            # Case 1: Package is on the ground at current_l (which is not the goal_l)
            # package_current_info is a location name.
            if package_current_info in self.all_locations: # Check if it's a location name we know about
                current_l = package_current_info
                # Cost = pick-up (1) + drive (distance) + drop (1)
                distance = self.get_shortest_distance(current_l, goal_location)
                if distance == float('inf'):
                    total_cost += float('inf') # Goal unreachable for this package
                else:
                    total_cost += 1 + distance + 1 # pick + drive + drop

            # Case 2: Package is inside a vehicle v
            # package_current_info is a vehicle name.
            # We need the vehicle's location.
            else: # Assume it must be a vehicle name if not a location
                 vehicle_name = package_current_info
                 vehicle_current_l = current_state_info.get(vehicle_name) # Get vehicle's location

                 if vehicle_current_l is None or vehicle_current_l not in self.all_locations:
                     # Vehicle location unknown or invalid. Unreachable.
                     total_cost += float('inf')
                 else:
                     # Cost = drive (distance) + drop (1)
                     distance = self.get_shortest_distance(vehicle_current_l, goal_location)
                     if distance == float('inf'):
                         total_cost += float('inf') # Goal unreachable for this package
                     else:
                         total_cost += distance + 1 # drive + drop

        # The heuristic must be 0 only for goal states.
        # Our calculation sums costs for packages *not* at their goal.
        # If total_cost is 0, it means all packages in self.goal_locations were found
        # to be at their goal location in the state.
        # Assuming task.goals only contains (at ?p ?l) facts for packages in self.goal_locations,
        # this implies the state satisfies all goal conditions related to package locations.
        # If task.goals contains other conditions, this heuristic might return 0 for a non-goal state.
        # However, based on typical transport problems and the examples, the goal is solely
        # about package locations. Thus, total_cost == 0 iff the state is a goal state.

        return total_cost
