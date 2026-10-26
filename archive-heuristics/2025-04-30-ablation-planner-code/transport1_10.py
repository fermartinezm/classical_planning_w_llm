from collections import defaultdict, deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions outside the class as seen in examples
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 locationA)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# The class name must be transportHeuristic
# Inherit from Heuristic if the base class is provided in the environment.
class transportHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    all packages to their goal locations. It considers the current state of
    each package (on the ground or in a vehicle) and the shortest path
    distance in the road network.

    # Assumptions
    - Roads are bidirectional.
    - Vehicle capacity constraints are relaxed (ignored).
    - Vehicle availability is relaxed (assumed a vehicle is available where needed for pickup,
      and the cost of getting a vehicle to a package's location is implicitly covered
      by the drive cost associated with moving the package).
    - All locations mentioned in the problem (initial state, goals, roads) are part of a single connected component
      or relevant locations are reachable from each other. Unreachable goals result in infinite heuristic.

    # Heuristic Initialization
    - Extract the road network from static facts to build a graph.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Extract the goal location for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status (on ground or in vehicle) and location for each package.
    2. For each package that has a specified goal location:
       a. If the package is already on the ground at its goal location, the cost for this package is 0.
       b. If the package is in a vehicle that is currently at the package's goal location, the cost for this package is 1 (for the 'drop' action).
       c. If the package is on the ground at a location different from its goal:
          The estimated cost is 1 (for 'pick-up') + the shortest distance (number of 'drive' actions) from the package's current location to its goal location + 1 (for 'drop').
       d. If the package is in a vehicle at a location different from its goal:
          The estimated cost is the shortest distance (number of 'drive' actions) from the vehicle's current location to the package's goal location + 1 (for 'drop').
       e. If the goal location is unreachable from the package's current location (or vehicle's location), the heuristic value is considered infinite.
    3. The total heuristic value for the state is the sum of the estimated costs for all packages with goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting road network, computing distances,
        and storing package goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build road network graph and collect all locations
        self.road_graph = defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = self._bfs(start_loc, locations)

        # Extract package goal locations from the goal conditions
        self.package_goals = {}
        goal_literals = []
        if isinstance(self.goals, str):
            goal_literals.append(self.goals)
        elif isinstance(self.goals, list) and self.goals[0] == 'and':
            goal_literals.extend(self.goals[1:])
        # Assuming other goal structures are not relevant for package locations

        for literal in goal_literals:
             if match(literal, "at", "?p", "?l"):
                p, l = get_parts(literal)[1:]
                # Assume any object appearing as the first argument in an (at ?p ?l) goal is a package
                self.package_goals[p] = l

    def _bfs(self, start_loc, all_locations):
        """Perform BFS from start_loc to find distances to all other locations."""
        distances = {loc: float('inf') for loc in all_locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc exists as a key in road_graph before iterating
            # This handles cases where a location exists but has no roads connected
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Parse current state to find locations of packages and vehicles
        package_status = {} # Maps package -> ('at', loc) or ('in', vehicle)
        vehicle_location = {} # Maps vehicle -> loc

        # Identify all packages that have a goal (these are the ones we track)
        packages_to_track = set(self.package_goals.keys())

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1:]
                if obj in packages_to_track:
                    package_status[obj] = ('at', loc)
                # Assume any other object with an 'at' predicate is a vehicle for simplicity
                # This is a heuristic assumption based on domain structure.
                # It might include other locatables if they exist, but they won't affect
                # the heuristic calculation which only considers packages in package_goals.
                # We only need vehicle locations if a package is 'in' that vehicle.
                # Let's store all 'at' locations for non-tracked packages.
                elif obj not in package_status: # Avoid potential conflict if a package somehow had an 'at' and 'in' fact (invalid state)
                     vehicle_location[obj] = loc # Could be a vehicle or other locatable

            elif predicate == "in":
                p, v = parts[1:]
                if p in packages_to_track:
                    package_status[p] = ('in', v)
                # Vehicle location will be picked up from 'at' facts

        total_cost = 0

        # Calculate cost for each package that has a goal location
        for package, goal_l in self.package_goals.items():
            # If package is not in package_status, it means it's not 'at' a location
            # and not 'in' a vehicle in the current state. This indicates an invalid state.
            if package not in package_status:
                 # This shouldn't happen in a valid state.
                 # Return infinity to penalize invalid states heavily in search.
                 return float('inf')

            status_type, current_where = package_status[package]

            if status_type == 'at':
                current_l = current_where
                # Check if package is already at goal on the ground
                if current_l == goal_l:
                    continue # Package is done

                # Package is on the ground, not at goal. Needs pick + drive + drop.
                # Need distance from current_l to goal_l.
                if current_l not in self.distances or goal_l not in self.distances[current_l]:
                     # Goal location or current location not in road network, or unreachable.
                     return float('inf') # Unreachable goal location

                dist = self.distances[current_l][goal_l]
                if dist == float('inf'):
                    return float('inf') # Unreachable goal location

                # Cost = pick (1) + drive (dist) + drop (1)
                total_cost += 1 + dist + 1

            elif status_type == 'in':
                vehicle = current_where
                # Need the location of the vehicle
                if vehicle not in vehicle_location:
                    # Vehicle location is unknown. Invalid state?
                    # This could happen if a vehicle exists but is not 'at' any location (invalid state).
                    return float('inf') # Vehicle location not found

                current_l = vehicle_location[vehicle]

                # Check if package is in vehicle at the goal location
                if current_l == goal_l:
                    total_cost += 1 # Need to drop (1)
                    continue # Package is in vehicle at goal

                # Package is in a vehicle, not at goal location. Needs drive + drop.
                # Need distance from current_l (vehicle location) to goal_l.
                if current_l not in self.distances or goal_l not in self.distances[current_l]:
                     return float('inf') # Unreachable goal location

                dist = self.distances[current_l][goal_l]
                if dist == float('inf'):
                    return float('inf') # Unreachable goal location

                # Cost = drive (dist) + drop (1)
                total_cost += dist + 1

        return total_cost
