from fnmatch import fnmatch
from collections import deque
# Assuming heuristics.heuristic_base exists and defines a Heuristic base class
# If not, a simple base class might be needed, but the prompt implies it exists.
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move each package
    to its goal location, ignoring vehicle capacity constraints and assuming any vehicle
    can be used. It sums the estimated costs for each package not yet at its goal.
    The cost for a package is estimated based on whether it needs to be picked up,
    driven, and dropped. The drive cost is the shortest path distance in the road network.

    # Assumptions
    - The road network is static and defines a graph between locations.
    - Any vehicle can pick up and drop any package (ignoring specific capacity requirements for simplicity, as this is a non-admissible heuristic).
    - The cost of pick-up, drop, and drive actions is 1.
    - The heuristic sums costs per package, ignoring potential synergies (e.g., multiple packages in one vehicle trip) or conflicts (e.g., multiple packages needing the same vehicle).
    - Goal conditions only involve packages being at specific locations, i.e., facts of the form `(at ?p ?l)`.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds the road network graph from static `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Pre-scan the current state to build a quick lookup for the current location of every locatable object (packages and vehicles). A package's location can be a physical location (if on the ground) or a vehicle name (if inside a vehicle). A vehicle's location is always a physical location.
    3. For each goal fact `(at package goal_location)` specified in the task goals:
        a. Check if this goal fact is already true in the current state. If it is, the cost for this specific goal is 0, and we move to the next goal fact.
        b. If the goal fact is not true, find the package's current location using the pre-scanned information.
        c. Determine the package's effective physical location `current_l`. If the package is on the ground at `l`, `current_l` is `l`. If the package is in a vehicle `v`, find the vehicle's location `(at v l)` using the pre-scanned information and set `current_l` to `l`. If the package's location cannot be determined (e.g., not 'at' or 'in' anywhere, or vehicle location unknown), the goal is unreachable, and the heuristic should return infinity.
        d. Calculate the shortest path distance `d` between `current_l` and `goal_l` using the precomputed distances. If no path exists, the distance is infinite. If the distance is infinite, the goal is unreachable from this state, and the heuristic should return infinity.
        e. Estimate the cost to achieve this specific goal fact:
           - If the package was found on the ground at `current_l`: It needs a pick-up (1 action), a drive from `current_l` to `goal_l` (`d` actions), and a drop at `goal_l` (1 action). Estimated cost = 1 + `d` + 1 = 2 + `d`.
           - If the package was found inside a vehicle at `current_l`: It needs a drive from `current_l` to `goal_l` (`d` actions) and a drop at `goal_l` (1 action). Estimated cost = `d` + 1.
        f. Add the estimated cost for this goal fact to the total heuristic cost.
    4. After processing all goal facts, the total accumulated cost is the heuristic value for the state. If at any point an unreachable goal was detected, infinity would have already been returned.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the road network.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road network graph.
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.graph:
                    self.graph[l1] = []
                if l2 not in self.graph:
                    self.graph[l2] = []
                # Add edges for both directions as roads are bidirectional in examples
                self.graph[l1].append(l2)
                self.graph[l2].append(l1)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS from each location.
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all other nodes.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check if current_node exists in the graph (it should if it's in self.locations)
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Pre-scan state to quickly find package and vehicle locations
        current_locations = {} # Maps locatable (package or vehicle) -> location or vehicle
        for fact in state:
             predicate, *args = get_parts(fact)
             if predicate == "at" and len(args) == 2:
                 obj, loc = args
                 current_locations[obj] = loc
             elif predicate == "in" and len(args) == 2:
                 package, vehicle = args
                 current_locations[package] = vehicle # Package is inside a vehicle

        # Iterate through goal facts and sum costs for those not met
        for goal_fact in self.goals:
            # We only handle (at ?p ?l) goals for packages.
            predicate, *args = get_parts(goal_fact)
            if predicate != "at" or len(args) != 2:
                 # Ignore other goal types or malformed goals for this heuristic
                 # In a real scenario, you might want to handle these or return inf
                 continue

            package, goal_location = args

            # If the goal fact is already true, no cost for this goal.
            if goal_fact in state:
                continue

            # Goal fact (at package goal_location) is not true.
            # Find where the package currently is.
            if package not in current_locations:
                 # Package location unknown - cannot reach goal
                 return float('inf')

            current_loc_or_vehicle = current_locations[package]

            # Determine the effective current location (where the package is physically).
            if current_loc_or_vehicle in self.locations: # Package is on the ground at a location
                current_location = current_loc_or_vehicle
                # Cost: pick-up (1) + drive (distance) + drop (1)
                dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal unreachable for this package
                cost_for_goal = 1 + dist + 1 # pick + drive + drop

            else: # Package is inside a vehicle (current_loc_or_vehicle is a vehicle object name)
                vehicle = current_loc_or_vehicle
                # Find vehicle's location
                if vehicle not in current_locations:
                    # Vehicle location unknown - cannot reach goal
                    return float('inf')
                vehicle_current_location = current_locations[vehicle] # Get vehicle's location

                # Cost: drive (distance) + drop (1)
                dist = self.distances.get(vehicle_current_location, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal unreachable for this package
                cost_for_goal = dist + 1 # drive + drop

            total_cost += cost_for_goal

        return total_cost
