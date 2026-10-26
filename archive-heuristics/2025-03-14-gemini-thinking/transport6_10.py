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
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their goal locations.
    It calculates the cost for each package individually and sums them up. The cost for each package is estimated as the number of drive actions to reach the package, one pick-up action, the number of drive actions to reach the goal location, and one drop action.

    # Assumptions
    - Vehicles can always carry any package if capacity constraints are met (simplified assumption for heuristic).
    - The heuristic focuses on minimizing drive, pick-up, and drop actions, ignoring capacity management actions as they are implicitly handled by pick-up and drop.
    - The heuristic assumes that for each package, there is always a path from the vehicle's initial location to the package's location and from the package's location to the goal location.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build a road network graph from the static facts to efficiently calculate shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package and its goal location.
    2. Find a suitable vehicle (in this simplified heuristic, we can assume any vehicle can be used and pick the first one available).
    3. Find the current location of the chosen vehicle.
    4. Calculate the shortest path (in terms of number of road actions) from the vehicle's current location to the package's current location using Breadth-First Search (BFS) on the road network. Let's call this path length 'distance1'.
    5. Calculate the shortest path from the package's current location to the package's goal location using BFS. Let's call this path length 'distance2'.
    6. The estimated cost for this package is 'distance1' (drive to package) + 1 (pick-up) + 'distance2' (drive to goal) + 1 (drop).
    7. Sum up the estimated costs for all packages to get the total heuristic value.
    8. If a package is already at its goal location, its contribution to the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.
        Extract goal locations and build the road network from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "?p", "?l"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "?l1", "?l2"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional in examples

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        current_locations_packages = {}
        current_locations_vehicles = {}

        for fact in state:
            if match(fact, "at", "?p", "?l"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                obj_type = None
                for obj_def in node.task.task_domain.types.items(): # Determine object type from domain definition
                    if obj in node.task.task_domain.objects.get(obj_def[0], []):
                        obj_type = obj_def[0]
                        break
                if obj_type == 'package':
                    current_locations_packages[obj] = location
                elif obj_type == 'vehicle':
                    current_locations_vehicles[obj] = location


        heuristic_value = 0
        packages_to_move = [p for p in self.goal_locations if current_locations_packages.get(p) != self.goal_locations[p]]

        for package in packages_to_move:
            start_location_package = current_locations_packages.get(package)
            goal_location_package = self.goal_locations[package]

            if start_location_package == goal_location_package:
                continue # Package already at goal

            # Simplified vehicle selection: just pick the first vehicle.
            vehicle = next(iter(current_locations_vehicles), None) # Get any vehicle
            if vehicle is None: # No vehicle available, this should not happen in typical transport problems, but handle for robustness
                return float('inf')

            start_location_vehicle = current_locations_vehicles.get(vehicle)

            # BFS to find shortest path length
            def get_shortest_path_len(start_loc, end_loc):
                if start_loc == end_loc:
                    return 0
                queue = collections.deque([(start_loc, 0)]) # location, distance
                visited = {start_loc}
                while queue:
                    current_loc, distance = queue.popleft()
                    if current_loc == end_loc:
                        return distance
                    for neighbor in self.road_network[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, distance + 1))
                return float('inf') # No path found

            distance1 = get_shortest_path_len(start_location_vehicle, start_location_package)
            distance2 = get_shortest_path_len(start_location_package, goal_location_package)

            if distance1 == float('inf') or distance2 == float('inf'):
                return float('inf') # No path, unsolvable or heuristic cannot estimate

            heuristic_value += distance1 + 1 + distance2 + 1 # drive to package + pick-up + drive to goal + drop

        return heuristic_value
