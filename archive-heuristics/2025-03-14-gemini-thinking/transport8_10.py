from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(road l1 l2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their goal locations.
    It calculates the shortest path (in terms of number of drive actions) for each package from its current location to its goal location, considering pick-up and drop actions.

    # Assumptions
    - Vehicles are always available at the package's initial location when needed.
    - Vehicle capacity is sufficient and not explicitly considered in the heuristic calculation.
    - The heuristic focuses on minimizing drive, pick-up, and drop actions.

    # Heuristic Initialization
    - Precomputes the shortest path distances between all pairs of locations based on the `road` predicates in the static facts using Breadth-First Search (BFS).
    - Extracts the goal locations for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package.
    2. Determine the goal location of the package.
    3. Check if the package is currently in a vehicle.
    4. If the package is in a vehicle:
        a. Find the current location of the vehicle.
        b. Calculate the shortest path distance (number of drive actions) from the vehicle's location to the package's goal location.
        c. Add the shortest path distance and 1 (for the drop action) to the heuristic estimate for this package.
    5. If the package is not in a vehicle:
        a. Find the current location of the package.
        b. Calculate the shortest path distance from the package's current location to its goal location.
        c. Add 1 (for the pick-up action), the shortest path distance, and 1 (for the drop action) to the heuristic estimate for this package.
    6. Sum up the heuristic estimates for all packages to get the total heuristic value for the state.
    7. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Precompute shortest paths between locations.
        - Extract goal locations for packages.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract road network to compute shortest paths
        roads = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                roads[l1].append(l2)
                roads[l2].append(l1)
                locations.add(l1)
                locations.add(l2)

        self.shortest_paths = collections.defaultdict(dict)
        for start_loc in locations:
            for end_loc in locations:
                if start_loc == end_loc:
                    self.shortest_paths[start_loc][end_loc] = 0
                    continue

                queue = collections.deque([(start_loc, 0)])
                visited = {start_loc}
                path_found = False
                while queue:
                    current_loc, distance = queue.popleft()
                    if current_loc == end_loc:
                        self.shortest_paths[start_loc][end_loc] = distance
                        path_found = True
                        break
                    for neighbor in roads[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, distance + 1))
                if not path_found:
                    self.shortest_paths[start_loc][end_loc] = float('inf') # No path


        # Extract goal package locations
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:]
                self.package_goals[package] = location

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        package_current_locations = {}
        package_in_vehicle = {}
        vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "?p", "?l") and get_parts(fact)[1] in self.package_goals: # is package
                package_current_locations[get_parts(fact)[1]] = get_parts(fact)[2]
            elif match(fact, "in", "?p", "?v"): # package in vehicle
                package_in_vehicle[get_parts(fact)[1]] = get_parts(fact)[2]
            elif match(fact, "at", "?v", "?l") and get_parts(fact)[1] not in self.package_goals: # is vehicle
                vehicle_locations[get_parts(fact)[1]] = get_parts(fact)[2]

        for package, goal_location in self.package_goals.items():
            current_location = package_current_locations.get(package, None)
            vehicle = package_in_vehicle.get(package, None)

            if current_location != goal_location:
                if vehicle:
                    vehicle_loc = vehicle_locations.get(vehicle)
                    if vehicle_loc:
                        path_len = self.shortest_paths[vehicle_loc][goal_location]
                        if path_len != float('inf'):
                            heuristic_value += path_len + 1 # drive + drop
                        else:
                            return float('inf') # No path, unsolvable?
                    else:
                        return float('inf') # Vehicle location unknown?
                else:
                    if current_location:
                        path_len = self.shortest_paths[current_location][goal_location]
                        if path_len != float('inf'):
                            heuristic_value += 1 + path_len + 1 # pickup + drive + drop
                        else:
                            return float('inf') # No path, unsolvable?
                    else:
                        return float('inf') # Package location unknown?


        return heuristic_value
