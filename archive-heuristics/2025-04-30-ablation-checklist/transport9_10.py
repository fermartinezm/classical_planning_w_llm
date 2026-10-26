import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions from example heuristics
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the costs for all packages. It relaxes the
    problem by ignoring vehicle capacity constraints and assuming vehicles are
    always available when needed for a package.

    # Assumptions
    - The primary goal is to move packages to specific locations.
    - Vehicle capacity constraints are ignored.
    - Any vehicle can be used to transport any package.
    - The cost of driving between locations is the shortest path distance in the
      road network.
    - Pick-up and drop-off actions each cost 1.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build the road network graph from the static `road` facts.
    - Compute all-pairs shortest paths on the road network using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that is not yet at its goal location:

    1. Determine the package's current status: Is it on the ground at a location,
       or is it inside a vehicle?
    2. If the package is inside a vehicle, determine the vehicle's current location.
       This vehicle's location is the effective starting point for the package's
       next movement phase.
    3. If the package is on the ground at location L_current and needs to reach
       location L_goal:
       - It needs to be picked up (1 action).
       - It needs to be transported from L_current to L_goal (cost is shortest
         path distance).
       - It needs to be dropped off at L_goal (1 action).
       - Total cost for this package: 1 + distance(L_current, L_goal) + 1.
    4. If the package is inside a vehicle at location L_vehicle and needs to reach
       location L_goal:
       - It needs to be transported from L_vehicle to L_goal (cost is shortest
         path distance).
       - It needs to be dropped off at L_goal (1 action).
       - Total cost for this package: distance(L_vehicle, L_goal) + 1.
    5. If the package is already at its goal location (either on the ground or
       inside a vehicle *at* the goal location and needing only a drop):
       - Cost for this package is 0. (The case where it's in a vehicle at the goal
         but needs dropping is handled by the logic below, resulting in distance 0 + 1 drop,
         but if the *goal* is just `(at p l)`, being in a vehicle at `l` is not the goal state,
         so it *does* need the drop. The heuristic should count the drop.)
         Let's refine: If `(at p goal_l)` is the goal, and the package is currently
         `(at p current_l)`, the cost is 2 + dist. If the package is `(in p v)` and
         `(at v current_l_v)`, the cost is 1 + dist. If `current_l == goal_l` or
         `current_l_v == goal_l` respectively, the distance is 0. So the formulas
         1 + 0 + 1 = 2 and 0 + 1 = 1 still apply if the package is *not* yet `(at p goal_l)`.
         If the package *is* `(at p goal_l)`, the cost is 0.

    6. The total heuristic value is the sum of the costs calculated for each
       package not currently satisfying its goal `(at p goal_l)`.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # We only care about (at ?p ?l) goals for packages
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                package, location = parts[1], parts[2]
                # Ensure it's a package type - simple check based on common naming or task objects
                # For robustness, we'd ideally check types from the domain definition,
                # but for this problem, assuming objects starting with 'p' are packages is reasonable
                # or check against task.objects if available (it's not in the provided Task class)
                # Let's rely on the goal structure matching packages.
                self.package_goals[package] = location

        # Build the road network graph and find all locations.
        self.road_graph = collections.defaultdict(set)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].add(l2)
                # Assuming roads are bidirectional unless specified otherwise.
                # The example domain file shows bidirectional roads explicitly.
                self.road_graph[l2].add(l1)
                locations.add(l1)
                locations.add(l2)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node to find shortest
        distances to all other locations in the road network.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            u = queue.popleft()
            current_dist = distances[u]

            if u in self.road_graph: # Check if node has neighbors
                for v in self.road_graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = current_dist + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map locatables (packages and vehicles) to their current location fact.
        # This helps quickly find where something is.
        current_locatable_facts = {}
        # Map vehicles to their ground location
        vehicle_ground_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                current_locatable_facts[obj] = fact
                # Assuming objects starting with 'v' are vehicles
                if obj.startswith('v'): # Simple check for vehicle type
                     vehicle_ground_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                 parts = get_parts(fact)
                 package, vehicle = parts[1], parts[2]
                 current_locatable_facts[package] = fact # Store the (in p v) fact

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined.
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at the goal, cost is 0 for this package

            # Find the package's current status (at a location or in a vehicle)
            package_status_fact = current_locatable_facts.get(package)

            if package_status_fact is None:
                 # This shouldn't happen in a valid state if the package exists,
                 # but handle defensively. Assume package is lost or invalid state.
                 # Returning infinity might be appropriate, but let's assume valid states.
                 # For simplicity, we'll assume the package is findable.
                 # If it's not in current_locatable_facts, it's likely an error
                 # or the package isn't in the initial state/goals correctly.
                 # In a real system, you might log a warning or return infinity.
                 # For this heuristic, let's assume it's always found.
                 pass # Should not reach here in typical valid problems

            parts = get_parts(package_status_fact)
            predicate = parts[0]

            current_location_for_distance = None
            cost_to_pickup = 0 # Cost to get the package into a vehicle

            if predicate == "at":
                # Package is on the ground at parts[2]
                current_location_for_distance = parts[2]
                cost_to_pickup = 1 # Needs a pick-up action

            elif predicate == "in":
                # Package is inside a vehicle parts[2]
                vehicle = parts[2]
                # Find the vehicle's ground location
                current_location_for_distance = vehicle_ground_locations.get(vehicle)
                cost_to_pickup = 0 # Already in a vehicle, no pick-up needed

            # If we couldn't determine a valid starting location (e.g., vehicle location unknown)
            if current_location_for_distance is None:
                 # This indicates an inconsistent state representation or problem definition.
                 # Return a high value to discourage this path.
                 return float('inf')


            # Calculate the distance from the package's effective starting location
            # to the goal location.
            distance_to_goal = self.distances.get(current_location_for_distance, {}).get(goal_location, float('inf'))

            # If the goal location is unreachable from the current location, this path is bad.
            if distance_to_goal == float('inf'):
                return float('inf')

            # Calculate cost for this package:
            # 1 (pick-up if on ground) + distance (drive) + 1 (drop-off)
            # If already in vehicle, pick-up cost is 0.
            # If already at goal_location_for_distance (e.g., in vehicle at goal), distance is 0.
            # If already (at p goal_l), we handled it at the beginning (cost 0).
            # Otherwise, it needs transport and a drop.
            cost_to_dropoff = 1 # Always needs a drop-off to be (at p goal_l)

            cost_for_package = cost_to_pickup + distance_to_goal + cost_to_dropoff

            total_cost += cost_for_package

        return total_cost

