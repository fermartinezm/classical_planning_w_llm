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
    It calculates the shortest path in terms of road segments for each package to its destination, considering pick-up and drop actions.

    # Assumptions:
    - Vehicles are always available at the package's initial location when needed.
    - Capacity constraints are ignored for simplicity and efficiency.
    - The heuristic focuses on minimizing drive actions and pick-up/drop actions for each package individually.

    # Heuristic Initialization
    - Extracts the road network from static facts to calculate shortest paths between locations.
    - Extracts the goal locations for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package.
    2. Determine the goal location of the package.
    3. If the package is not at its goal location:
        a. If the package is currently 'at' a location (not 'in' a vehicle):
            - Add 1 action for picking up the package.
            - Calculate the shortest path (number of road segments) from the current location to the goal location using BFS on the road network. Add this path length to the heuristic.
            - Add 1 action for dropping off the package at the goal location.
        b. If the package is currently 'in' a vehicle:
            - Determine the vehicle's current location.
            - Calculate the shortest path from the vehicle's current location to the package's goal location. Add this path length to the heuristic.
            - Add 1 action for dropping off the package at the goal location.
    4. Sum up the estimated actions for all packages to get the total heuristic value.
    5. If a package is already at its goal location, it contributes 0 to the heuristic.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts road network from static facts.
        - Extracts goal locations for packages.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build road network as an adjacency list
        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

        # Extract goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package_name = get_parts(goal)[1]
                location_name = get_parts(goal)[2]
                package_type = None
                for obj_type_def in task.name.split('\n'): # very brittle way to get object types, but works for this example
                    if ":objects" in obj_type_def:
                        for obj_def in obj_type_def.split(':')[1].strip().split():
                            if package_name in obj_def and 'package' in obj_def:
                                package_type = 'package'
                                break
                        if package_type == 'package':
                            break
                if package_type == 'package': # only consider goal locations for packages
                    self.package_goals[package_name] = location_name


    def shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations in the road network using BFS.

        Returns path length or a large number if no path exists.
        """
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            if current_location == goal_location:
                return distance

            for neighbor in self.road_network[current_location]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf') # No path found


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        package_locations = {}
        vehicle_locations = {}
        packages_in_vehicles = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                loc_name = get_parts(fact)[2]

                is_package = False
                is_vehicle = False
                for obj_type_def in node.task.name.split('\n'): # brittle way to get object types, but works for example
                    if ":objects" in obj_type_def:
                        for obj_def in obj_type_def.split(':')[1].strip().split():
                            if obj_name in obj_def and 'package' in obj_def:
                                is_package = True
                            if obj_name in obj_def and 'vehicle' in obj_def:
                                is_vehicle = True
                        break

                if is_package:
                    package_locations[obj_name] = loc_name
                elif is_vehicle:
                    vehicle_locations[obj_name] = loc_name
            elif match(fact, "in", "*", "*"):
                package_name = get_parts(fact)[1]
                vehicle_name = get_parts(fact)[2]
                packages_in_vehicles[package_name] = vehicle_name


        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package, None)
            vehicle_carrying_package = packages_in_vehicles.get(package, None)

            if vehicle_carrying_package:
                vehicle_loc = vehicle_locations.get(vehicle_carrying_package)
                if vehicle_loc:
                    path_len = self.shortest_path_length(vehicle_loc, goal_location)
                    if path_len == float('inf'):
                        return float('inf') # unsolvable if no path
                    heuristic_value += path_len + 1 # drive + drop
                else:
                    return float('inf') # should not happen in valid states, but handle for robustness

            elif current_location:
                if current_location != goal_location:
                    path_len = self.shortest_path_length(current_location, goal_location)
                    if path_len == float('inf'):
                        return float('inf') # unsolvable if no path
                    heuristic_value += 2 + path_len # pickup + drive + drop
            else:
                return float('inf') # package location unknown, should not happen in valid states


        return heuristic_value
