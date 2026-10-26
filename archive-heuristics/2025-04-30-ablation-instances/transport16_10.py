# Add necessary imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (can be defined outside the class or inside if preferred, but outside is cleaner)
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
    # Use zip to match parts with args up to the length of the shorter sequence.
    # This matches the behavior in the example heuristics.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently, ignoring vehicle capacity constraints and
    potential synergies (like carrying multiple packages in one trip).

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Packages can be on the ground or inside a vehicle.
    - The shortest path between any two connected locations is the minimum number of drive actions.
    - Vehicle capacity is ignored.
    - Multiple packages can be transported simultaneously by the same vehicle, but the heuristic counts costs per package independently.
    - All locations mentioned in init/goal states are part of the road network or reachable from it.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations connected by roads and precomputes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1. Initialize total heuristic cost to 0.
    2. For each package `p` that has a goal location `L_goal`:
       a. Check if the package `p` is already at its goal location `L_goal` on the ground in the current state (i.e., `(at p L_goal)` is true). If yes, the cost for this package is 0, continue to the next package.
       b. If the package is not at its goal on the ground, determine its current effective location `L_package_current`.
          - If `(at p L_current)` is true in the state, `L_package_current` is `L_current`.
          - If `(in p V)` is true in the state, find the location of vehicle `V` using `(at V L_v)`. `L_package_current` is `L_v`.
          - If the package's status is not found (e.g., not 'at' any location and not 'in' any vehicle), this indicates an issue; treat as infinite cost.
       c. Calculate the cost contribution for package `p` based on its current effective location `L_package_current` and its goal location `L_goal`:
          - If `L_package_current` is unknown or unreachable from `L_goal`, the cost is infinite.
          - If the package is on the ground at `L_package_current` (`(at p L_package_current)` is true in the state):
            - Cost = 1 (pick-up) + shortest_path_distance(`L_package_current`, `L_goal`) + 1 (drop).
          - If the package is inside a vehicle `V` which is at `L_package_current` (`(in p V)` and `(at V L_package_current)` are true in the state):
            - If `L_package_current == L_goal`: Cost = 1 (drop).
            - If `L_package_current != L_goal`: Cost = shortest_path_distance(`L_package_current`, `L_goal`) + 1 (drop).
       d. Add the calculated cost contribution for package `p` to the total heuristic cost.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes shortest path distances between all locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at obj loc) predicate
                package, location = args
                # Assuming goal is always (at package location)
                self.goal_locations[package] = location

        # Build the road graph and compute shortest paths.
        self.location_graph = {}
        all_locations = set()

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Roads are bidirectional

        # Add locations from goals/initial state that might not be in road facts
        # This ensures BFS is attempted even if a location is isolated
        for loc in self.goal_locations.values():
             all_locations.add(loc)
        # Locations from the initial state are not directly available here,
        # but they should ideally be covered by road facts or goals in valid problems.
        # If an initial location is truly isolated and not a goal, BFS from it
        # will correctly show other locations as unreachable.

        self.shortest_paths = self._compute_all_pairs_shortest_paths(list(all_locations))

    def _compute_all_pairs_shortest_paths(self, locations):
        """
        Computes shortest path distances from each location to all reachable locations
        using BFS.
        """
        distances = {}
        for start_loc in locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """
        Performs BFS starting from start_loc to find distances to all reachable locations.
        """
        dist = {start_loc: 0}
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = dist[current_loc]

            # Check if current_loc exists in the graph keys before accessing neighbors
            # An isolated location might be in 'locations' list but not in location_graph keys
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor not in dist:
                        dist[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return dist

    def get_distance(self, loc1, loc2):
        """
        Returns the precomputed shortest distance between two locations.
        Returns infinity if no path exists or if locations are not in the precomputed map.
        """
        if loc1 == loc2:
            return 0
        # Check if both locations were part of the graph/locations considered during BFS
        # and if loc2 was reached from loc1 during BFS.
        # Use .get() with a default empty dict to avoid KeyError if loc1 wasn't a BFS start point
        if loc1 not in self.shortest_paths or loc2 not in self.shortest_paths.get(loc1, {}):
             return float('inf') # Indicates unreachable
        return self.shortest_paths[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map locatables (packages and vehicles) to their current location or vehicle.
        current_locations = {} # Maps locatable -> location or vehicle
        # Also track which packages are on the ground vs in a vehicle
        package_is_on_ground = {} # Maps package -> True/False

        # Populate current_locations and package_is_on_ground from the state
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                locatable, location = args
                current_locations[locatable] = location
                # If this locatable is a package we care about (i.e., in goals)
                if locatable in self.goal_locations:
                     package_is_on_ground[locatable] = True

            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                current_locations[package] = vehicle # Package is inside this vehicle
                if package in self.goal_locations:
                     package_is_on_ground[package] = False


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            # This check is slightly redundant with the logic below but serves as a quick exit
            if package_is_on_ground.get(package, False) and current_locations.get(package) == goal_location:
                 continue # Package is already at the goal location on the ground, cost is 0 for this package

            # Package is not at the goal location on the ground. Find its current status and effective location.
            package_current_status_obj = current_locations.get(package) # This is either a location string or a vehicle string

            if package_current_status_obj is None:
                 # Package location/status is unknown. Invalid state?
                 total_cost += float('inf')
                 continue

            package_effective_location = None
            is_on_ground = package_is_on_ground.get(package, False) # Default to False if package status wasn't recorded (shouldn't happen for goal packages)

            if is_on_ground:
                 # package_current_status_obj is the location string
                 package_effective_location = package_current_status_obj
            else: # Package is in a vehicle
                 # package_current_status_obj is the vehicle name
                 vehicle_name = package_current_status_obj
                 vehicle_location = current_locations.get(vehicle_name) # Find vehicle's location
                 if vehicle_location is not None:
                      package_effective_location = vehicle_location
                 else:
                      # Vehicle location not found. Invalid state?
                      total_cost += float('inf')
                      continue # Skip to next package


            # If package_effective_location is still None, something went wrong
            if package_effective_location is None:
                 total_cost += float('inf')
                 continue

            # Calculate cost contribution for this package
            if package_effective_location == goal_location:
                # Package is at the goal location, but not on the ground (must be in a vehicle)
                # Needs 1 drop action
                total_cost += 1
            else:
                # Package is not at the goal location
                distance = self.get_distance(package_effective_location, goal_location)

                if distance == float('inf'):
                     # Goal location is unreachable from the package's current location
                     total_cost += float('inf')
                     continue

                # Cost includes transport (drive actions) and drop
                cost_for_package = distance + 1 # drive actions + drop action

                # If the package is currently on the ground, it also needs a pick-up action
                if is_on_ground:
                    cost_for_package += 1 # pick-up action

                total_cost += cost_for_package

        return total_cost
