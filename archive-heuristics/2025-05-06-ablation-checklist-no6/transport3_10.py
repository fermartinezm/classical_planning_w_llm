# Required imports
from collections import deque, defaultdict
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and provides the Heuristic base class
# If running this code standalone, you might need to define a dummy Heuristic class
# or adjust the import path.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Dummy base class for standalone testing if needed
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Subclass must implement abstract method")


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or return empty list
        return []
    return fact[1:-1].split()

# Helper function for pattern matching facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of args for a match
    if len(parts) != len(args):
        return False
    # Use zip to iterate up to the length of the shorter sequence (parts or args)
    # fnmatch(part, arg) returns True if part matches arg (where arg can be '*')
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions (pick-up, drop, drive) required
    to move each package to its goal location, ignoring vehicle capacity
    and availability constraints. It uses shortest path distances between
    locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        identifying vehicles, building the location graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at ?p ?l)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
            # Ignore other types of goal facts if any

        # Identify vehicles. Look for objects appearing in (capacity ?v ?s) facts
        # in static facts and initial state.
        self.vehicles = set()
        facts_to_check_vehicles = set(static_facts) | set(initial_state) # Combine static and initial facts
        for fact in facts_to_check_vehicles:
             if match(fact, "capacity", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # Ensure it's (capacity vehicle size)
                    vehicle_name = parts[1]
                    self.vehicles.add(vehicle_name)

        # Build the location graph from road facts.
        self.location_graph = defaultdict(set)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (road loc1 loc2)
                    l1, l2 = parts[1:]
                    self.location_graph[l1].add(l2)
                    self.location_graph[l2].add(l1) # Assuming roads are bidirectional
                    locations.add(l1)
                    locations.add(l2)

        self.locations = list(locations) # Store list of all locations

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Perform BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary mapping location to distance.
        """
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute paths
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Track current location for all locatables (packages and vehicles).
        # Map object name -> its location (if at) or the vehicle name (if in)
        current_positions = {}
        # Map vehicle name -> its current location
        vehicle_locations = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                current_positions[obj] = location
                if obj in self.vehicles:
                    vehicle_locations[obj] = location
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                current_positions[package] = vehicle # Store the vehicle name
            # Ignore other predicates like capacity, road, capacity-predecessor in state

        total_cost = 0

        # Consider only packages that have a goal location specified.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # This is the only state where the package goal is satisfied for this package.
            if (f"(at {package} {goal_location})") in state:
                 continue # Package is already at the goal, cost is 0 for this package

            # Package is not at the goal. Estimate cost.

            current_status = current_positions.get(package)

            if current_status is None:
                 # This package exists in goals but not in the current state's 'at' or 'in' facts.
                 # This indicates an invalid state representation or a package that disappeared.
                 # Assign a high cost to discourage paths leading to such states.
                 # print(f"Warning: Package {package} from goals not found in state.")
                 total_cost += 1000 # Large cost
                 continue

            effective_current_loc = None
            is_in_vehicle = False

            if current_status in self.locations:
                # Package is on the ground at current_status location
                effective_current_loc = current_status
                is_in_vehicle = False
            elif current_status in self.vehicles:
                # Package is inside the vehicle named by current_status
                vehicle_name = current_status
                is_in_vehicle = True
                # Find the vehicle's location
                vehicle_location = vehicle_locations.get(vehicle_name)

                if vehicle_location is None or vehicle_location not in self.locations:
                    # Vehicle location unknown or invalid. Cannot estimate cost for this package.
                    # print(f"Warning: Vehicle {vehicle_name} containing package {package} has unknown or invalid location.")
                    total_cost += 1000 # Large cost
                    continue

                effective_current_loc = vehicle_location
            else:
                 # current_status is neither a known location nor a known vehicle name. Invalid state?
                 # print(f"Warning: Package {package} has invalid status '{current_status}'.")
                 total_cost += 1000 # Large cost
                 continue


            # Calculate distance from effective_current_loc to goal_location
            # Check if locations are in the precomputed paths dictionary
            # _bfs returns inf for unreachable nodes, including nodes not in the graph.
            dist = self.shortest_paths.get(effective_current_loc, {}).get(goal_location, float('inf'))

            if dist == float('inf'):
                 # Goal is unreachable via road network from effective location.
                 total_cost += 1000 # Large cost for unreachable goals
                 continue

            # Estimate actions needed for this package
            package_cost = 0
            if is_in_vehicle:
                # Package is in a vehicle at effective_current_loc
                # Needs drive (dist) + drop (1)
                package_cost = dist + 1
            else:
                # Package is on the ground at effective_current_loc
                # Needs pick-up (1) + drive (dist) + drop (1)
                package_cost = 1 + dist + 1 # = 2 + dist

            total_cost += package_cost

        # If total_cost is 0, it means all packages with location goals were already at their goals.
        # This heuristic is 0 iff all packages in self.goal_locations are at their goal locations.
        # This matches the requirement that h=0 only for goal states, assuming goal states
        # are defined solely by package locations.

        return total_cost
