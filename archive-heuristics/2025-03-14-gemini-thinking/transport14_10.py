from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their respective goal locations.
    It considers the necessary drive, pick-up, and drop actions based on the current locations of packages and vehicles and the road network.

    # Assumptions
    - Vehicles can carry only one package at a time (simplified capacity assumption for heuristic).
    - The heuristic focuses on the number of actions, not on capacity constraints or optimal vehicle assignment.
    - It assumes that roads are bidirectional.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds a road network graph from the static facts to calculate shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the package's current location. It can be 'at' a location or 'in' a vehicle.
    2. If the package is 'at' a location:
        a. Find the current location of any available vehicle.
        b. Calculate the shortest path (number of road actions) from the vehicle's location to the package's current location using Breadth-First Search (BFS) on the road network.
        c. Add the path length to the heuristic cost (representing 'drive' actions).
        d. Add 1 to the heuristic cost for the 'pick-up' action.
        e. Calculate the shortest path from the package's current location to its goal location using BFS.
        f. Add the path length to the heuristic cost (representing 'drive' actions).
        g. Add 1 to the heuristic cost for the 'drop' action.
    3. If the package is 'in' a vehicle:
        a. Find the vehicle's current location.
        b. Calculate the shortest path from the vehicle's location to the package's goal location using BFS.
        c. Add the path length to the heuristic cost (representing 'drive' actions).
        d. Add 1 to the heuristic cost for the 'drop' action.
    4. Sum up the costs for all packages not yet at their goal locations to get the total heuristic estimate.
    5. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts goal locations for each package.
        - Builds the road network from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                package_name = parts[1]
                location_name = parts[2]
                self.goal_locations[package_name] = location_name

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        - Iterates through each package.
        - Calculates the estimated cost to move each package to its goal location.
        - Sums up the costs for all packages.
        """
        state = node.state
        heuristic_value = 0

        current_package_locations = {}
        current_vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_name = parts[1]
                location_name = parts[2]
                if match(fact, "at", "*", "-", "vehicle"): #check if it is a vehicle
                    current_vehicle_locations[obj_name] = location_name
                elif match(fact, "at", "*", "-", "package"): #check if it is a package
                    current_package_locations[obj_name] = location_name
            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                package_name = parts[1]
                vehicle_name = parts[2]
                current_package_locations[package_name] = vehicle_name # store vehicle name as location if package is in vehicle


        for package, goal_location in self.goal_locations.items():
            current_location_or_vehicle = current_package_locations.get(package)

            if current_location_or_vehicle != goal_location:
                if current_location_or_vehicle in current_vehicle_locations.values(): # package is in a vehicle
                    vehicle_holding_package = [v for v, loc in current_vehicle_locations.items() if loc == current_location_or_vehicle][0] # find vehicle name
                    vehicle_location = current_location_or_vehicle # vehicle location
                    path_len_to_goal = self._get_shortest_path_len(vehicle_location, goal_location)
                    heuristic_value += path_len_to_goal + 1 # drive + drop

                elif current_location_or_vehicle is not None: # package is at a location
                    package_current_location = current_location_or_vehicle
                    vehicle_location = None
                    if current_vehicle_locations:
                        vehicle_name, vehicle_location = next(iter(current_vehicle_locations.items())) # just pick the first vehicle for simplicity
                        path_len_to_package = self._get_shortest_path_len(vehicle_location, package_current_location)
                        heuristic_value += path_len_to_package + 1 # drive + pickup

                    path_len_to_goal = self._get_shortest_path_len(package_current_location, goal_location)
                    heuristic_value += path_len_to_goal + 1 # drive + drop
                else:
                    # Package location not found in state, should not happen in typical problems, but handle for robustness
                    heuristic_value += 100 # assign a high cost if package location is unknown

        return heuristic_value

    def _get_shortest_path_len(self, start_location, end_location):
        """
        Calculate the shortest path length between two locations using BFS on the road network.
        Returns the path length or a large number if no path exists.
        """
        if start_location == end_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # location, distance
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            if current_location == end_location:
                return distance

            for neighbor in self.road_network[current_location]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return 100 # No path found, return a large number to discourage this path
