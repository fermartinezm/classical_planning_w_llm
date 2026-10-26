from fnmatch import fnmatch
from collections import defaultdict, deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # The example uses zip, which stops when the shortest iterable is exhausted.
    # This implies the number of parts in the fact must exactly match the number of args in the pattern.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It considers the actions needed for each package
    individually: pick-up, transport (driving), and drop. The transport cost
    is estimated by the shortest path distance in the road network.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - The heuristic calculates the sum of minimum actions required for each
      package independently, ignoring potential synergies (like one vehicle
      carrying multiple packages) or conflicts (like capacity limits or
      vehicle availability).
    - The road network is undirected (road l1 l2 implies road l2 l1).
    - All locations mentioned in road facts or package goals are assumed to be
      part of the road network. Unreachable goals are assigned a large penalty.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build the road network graph from the static facts.
    - Compute the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the task's goal state. If yes, the heuristic value is 0.
    2. If not the goal state, initialize the total heuristic cost to 0.
    3. Create a mapping of current locations/containers for all locatable objects
       (packages and vehicles) by iterating through the state facts.
    4. For each package that has a specified goal location in the task:
    5. Find the package's current position (either a location if on the ground, or a vehicle name if inside a vehicle).
    6. If the package is currently on the ground at location Lp:
       - If Lp is the goal location Lg, the cost for this package is 0.
       - If Lp is not the goal location Lg:
         - Add 1 to the cost for the 'pick-up' action.
         - Find the shortest path distance from Lp to Lg using the precomputed distances. Add this distance to the cost (representing 'drive' actions). If Lg is unreachable from Lp, add a large penalty instead of the distance.
         - Add 1 to the cost for the 'drop' action.
    7. If the package is currently inside a vehicle V:
       - Find the current location Lv of the vehicle V.
       - If Lv is the goal location Lg:
         - Add 1 to the cost for the 'drop' action.
       - If Lv is not the goal location Lg:
         - Find the shortest path distance from Lv to Lg using the precomputed distances. Add this distance to the cost (representing 'drive' actions). If Lg is unreachable from Lv, add a large penalty instead of the distance.
         - Add 1 to the cost for the 'drop' action.
    8. The total heuristic value is the sum of costs calculated for each package.
    9. If the calculated total cost is 0 (meaning all packages with 'at' goals are satisfied according to the heuristic), but the state is not the overall task goal (checked in step 1), return 1 to ensure non-goal states have a positive heuristic value. Otherwise, return the calculated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            # Only consider 'at' goals for packages
            if match(goal, "at", "*", "*"):
                 predicate, package, location = get_parts(goal)
                 self.package_goals[package] = location

        # 2. Build the road network graph and collect all locations
        self.road_network = defaultdict(set)
        locations_from_roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                predicate, l1, l2 = get_parts(fact)
                self.road_network[l1].add(l2)
                self.road_network[l2].add(l1) # Roads are bidirectional
                locations_from_roads.add(l1)
                locations_from_roads.add(l2)

        # Include locations mentioned in goals even if not in road facts
        # This ensures we have distance entries for goal locations, even if isolated.
        # BFS from a location not in the graph will result in infinite distances to others.
        all_relevant_locations = set(locations_from_roads)
        all_relevant_locations.update(self.package_goals.values())

        self.locations = list(all_relevant_locations)

        # 3. Compute all-pairs shortest paths
        self.distance = {}
        for start_loc in self.locations:
            self.distance[start_loc] = self._bfs(start_loc)

        # Define a large cost for unreachable goals
        self.unreachable_cost = 1000 # Arbitrary large number

    def _bfs(self, start_location):
        """
        Perform BFS from a start location to find distances to all other locations.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_location not in self.locations:
             # Start location is not in the graph of known locations
             return distances

        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            # Check if current_loc has neighbors in the graph
            if current_loc in self.road_network:
                for neighbor in self.road_network[current_loc]:
                    # Ensure neighbor is also a known location
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state
        task = node.task # Access the task object to check goal_reached

        # Check if the state is the goal state first
        if task.goal_reached(state):
             return 0

        # Map current location/container for all locatables (packages and vehicles)
        current_positions = {} # Maps object -> location or vehicle
        for fact in state:
            # Use the match utility for robustness
            if match(fact, "at", "*", "*"):
                predicate, obj, location = get_parts(fact)
                current_positions[obj] = location
            elif match(fact, "in", "*", "*"):
                predicate, package, vehicle = get_parts(fact)
                current_positions[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # If package is not mentioned in the current state facts, we can't track it.
            # This shouldn't happen in valid PDDL states, but handle defensively.
            if package not in current_positions:
                 # Assign a large penalty if a package with a goal is missing from state info
                 total_cost += self.unreachable_cost
                 continue

            current_pos = current_positions[package]

            # Case 1: Package is on the ground
            # Check if the current position is one of the known locations
            if current_pos in self.locations:
                current_package_location = current_pos

                # If already at goal location, cost is 0 for this package
                if current_package_location == goal_location:
                    continue # Cost is 0 for this package

                # If not at goal: needs pick-up, transport, drop
                # Cost: pick-up (1) + drive (distance) + drop (1)
                if current_package_location in self.distance and goal_location in self.distance[current_package_location]:
                     drive_cost = self.distance[current_package_location][goal_location]

                     if drive_cost == float('inf'):
                         # Goal is unreachable from current location
                         total_cost += self.unreachable_cost
                     else:
                         total_cost += 1 # pick-up
                         total_cost += drive_cost # drive actions
                         total_cost += 1 # drop
                else:
                     # Goal location or current location not in the precomputed distance map
                     total_cost += self.unreachable_cost

            # Case 2: Package is inside a vehicle
            else: # current_pos is a vehicle name (e.g., 'v1')
                vehicle_name = current_pos
                # Find the location of the vehicle
                if vehicle_name in current_positions and current_positions[vehicle_name] in self.locations:
                    current_vehicle_location = current_positions[vehicle_name]

                    # If vehicle is already at package's goal location
                    if current_vehicle_location == goal_location:
                         # Cost: drop (1)
                         total_cost += 1
                    else: # Vehicle is not at package's goal location
                         # Needs transport from vehicle's current location to package's goal, then drop
                         # Cost: drive (distance) + drop (1)
                         if current_vehicle_location in self.distance and goal_location in self.distance[current_vehicle_location]:
                             drive_cost = self.distance[current_vehicle_location][goal_location]
                             if drive_cost == float('inf'):
                                 total_cost += self.unreachable_cost
                             else:
                                 total_cost += drive_cost # drive actions
                                 total_cost += 1 # drop
                         else:
                             # Goal location or vehicle location not in the precomputed distance map
                             total_cost += self.unreachable_cost
                else:
                    # Vehicle location unknown or not a valid location
                    total_cost += self.unreachable_cost

        # If the state is not the goal state (checked at the beginning),
        # the heuristic must be > 0. If the calculated total_cost is 0,
        # it means all packages with 'at' goals are satisfied according to
        # our calculation, but the overall task goal is not met.
        # In this case, return 1 to ensure h > 0 for non-goal states.
        # Otherwise, return the calculated total_cost.
        if total_cost == 0:
             return 1
        else:
             return total_cost
