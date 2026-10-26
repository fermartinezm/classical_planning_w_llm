import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    if len(parts) != len(args) and '*' not in args[-1]:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It calculates the shortest
    path distance for driving and adds costs for pick-up and drop actions.
    It ignores vehicle capacity and availability constraints.

    # Assumptions:
    - Any package not at its goal location needs to be transported.
    - Transport involves picking up the package, driving it to the destination, and dropping it.
    - If a package is already in a vehicle, it only needs driving and dropping.
    - Vehicle capacity and availability are ignored (relaxed).
    - The cost of driving between two locations is the shortest path distance in the road network.
    - Pick-up and drop actions each cost 1.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Build the road network graph from static `road` facts.
    - Precompute shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all packages that are not at their goal location.
    2. For each such package:
       a. Determine its current location. This could be a ground location (`at`) or inside a vehicle (`in`). If inside a vehicle, the package's effective location is the vehicle's location.
       b. Determine its goal location.
       c. Calculate the shortest path distance between the package's current effective location and its goal location using the precomputed distances.
       d. Add the cost for this package:
          - If the package is on the ground: 1 (pick-up) + distance (drive) + 1 (drop).
          - If the package is inside a vehicle: distance (drive) + 1 (drop).
    3. Sum the costs for all packages not at their goal location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Build the road network graph.
        self.road_graph = collections.defaultdict(set)
        all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].add(loc2)
                self.road_graph[loc2].add(loc1) # Assuming roads are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add locations from initial state and goals that might not be in road facts
        # (e.g., a single isolated location)
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_locations.add(loc)
        for goal in task.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 all_locations.add(loc)


        # Precompute shortest path distances between all pairs of locations using BFS.
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = self._bfs(start_loc, all_locations)

    def _bfs(self, start_node, all_nodes):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the road graph.
        """
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            for neighbor in self.road_graph.get(current_loc, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations of packages and vehicles.
        package_locations = {} # Stores either a location string or a vehicle string
        vehicle_locations = {} # Stores location string for vehicles

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                # We need to distinguish packages from vehicles.
                # A simple way is to check if the object is a key in package_goals.
                if obj in self.package_goals:
                    package_locations[obj] = loc
                # Assume anything else with an 'at' predicate is a vehicle
                # This might be brittle if other locatables exist, but works for transport domain
                # A more robust way would be to parse types from the domain file
                elif obj.startswith('v'): # Simple check based on example object names
                     vehicle_locations[obj] = loc

            elif predicate == "in":
                package, vehicle = args
                package_locations[package] = vehicle # Store the vehicle name temporarily

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # If package is not in the current state at all, it's likely an error or not relevant
            if package not in package_locations:
                 continue # Or handle as an error

            current_loc_or_vehicle = package_locations[package]

            # Determine the package's actual physical location
            if current_loc_or_vehicle in vehicle_locations:
                # Package is inside a vehicle, its location is the vehicle's location
                package_current_location = vehicle_locations[current_loc_or_vehicle]
                is_in_vehicle = True
            else:
                # Package is on the ground at the recorded location
                package_current_location = current_loc_or_vehicle
                is_in_vehicle = False

            # If the package is already at its goal, cost is 0 for this package.
            if package_current_location == goal_location:
                continue

            # Calculate cost for this package
            # Cost = Pick-up (if on ground) + Drive + Drop
            pick_up_cost = 1 if not is_in_vehicle else 0
            drop_cost = 1 # Always need to drop at the goal

            # Get the drive distance. Handle cases where locations might not be in graph
            # (e.g., initial state has objects at locations not connected by roads, though unlikely)
            drive_distance = self.distances.get(package_current_location, {}).get(goal_location, float('inf'))

            # If goal is unreachable, return infinity heuristic
            if drive_distance == float('inf'):
                 return float('inf')

            total_cost += pick_up_cost + drive_distance + drop_cost

        return total_cost

