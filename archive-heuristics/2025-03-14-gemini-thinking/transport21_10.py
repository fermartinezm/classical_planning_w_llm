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
    It considers the actions: drive, pick-up, and drop. The heuristic is based on shortest paths on the road network and the necessary pick-up and drop actions for each package.

    # Assumptions
    - Vehicles always have sufficient capacity to carry any single package.
    - The road network is connected enough to reach all goal locations if they are reachable in the problem instance.
    - We are minimizing the number of actions, and each action (drive, pick-up, drop) has a cost of 1.

    # Heuristic Initialization
    - Extract the road network from the static facts to calculate shortest paths between locations.
    - Identify the goal locations for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package. It can be 'at' a location or 'in' a vehicle.
    2. Determine the goal location of the package.
    3. If the package is 'at' a location and not at the goal location:
        a. Estimate the cost to pick up the package: 1 (pick-up action).
        b. Calculate the shortest path (number of drive actions) from the current location to the goal location using the road network.
        c. Estimate the cost to drop the package at the goal location: 1 (drop action).
        d. Total estimated cost for this package is 2 + shortest path length.
    4. If the package is 'in' a vehicle:
        a. Determine the current location of the vehicle.
        b. Calculate the shortest path (number of drive actions) from the vehicle's current location to the package's goal location.
        c. Estimate the cost to drop the package at the goal location: 1 (drop action).
        d. Total estimated cost for this package is 1 + shortest path length.
    5. Sum up the estimated costs for all packages that are not at their goal locations.
    6. If all packages are at their goal locations, the heuristic value is 0.

    Shortest path calculation is done using Breadth-First Search (BFS) on the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Road network from static facts.
        - Goal locations for each package.
        """
        self.goals = task.goals
        static_facts = task.static

        self.roads = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads[l1].append(l2)
                self.roads[l2].append(l1) # Roads are bidirectional in the examples

        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package_name = get_parts(goal)[1]
                location_name = get_parts(goal)[2]
                self.package_goals[package_name] = location_name

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0

        current_package_locations = {}
        vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                location_name = get_parts(fact)[2]
                if match(fact, "at", "?p - package", "*"):
                    current_package_locations[obj_name] = location_name
                elif match(fact, "at", "?v - vehicle", "*"):
                    vehicle_locations[obj_name] = location_name
            elif match(fact, "in", "*", "*"):
                package_name = get_parts(fact)[1]
                vehicle_name = get_parts(fact)[2]
                current_package_locations[package_name] = vehicle_name # Store vehicle name as location for 'in'

        for package, goal_location in self.package_goals.items():
            current_location_or_vehicle = current_package_locations.get(package)

            if current_location_or_vehicle != goal_location:
                if current_location_or_vehicle in vehicle_locations.values() or current_location_or_vehicle is None: # Package is at some location
                    current_location = current_package_locations.get(package)
                    if current_location is None: #package not in state, should not happen, but for robustness
                        continue
                    if current_location != goal_location:
                        shortest_path_len = self.shortest_path(current_location, goal_location)
                        heuristic_value += 2 + shortest_path_len # pick-up + drive + drop (simplified to 2 + drive)
                else: # Package is in a vehicle
                    vehicle_name = current_location_or_vehicle
                    vehicle_location = vehicle_locations.get(vehicle_name)
                    if vehicle_location is not None:
                        shortest_path_len = self.shortest_path(vehicle_location, goal_location)
                        heuristic_value += 1 + shortest_path_len # drive + drop (simplified to 1 + drive)
                    else:
                        heuristic_value += 1000 # Vehicle location not found, assign a high cost


        return heuristic_value

    def shortest_path(self, start_location, goal_location):
        """Calculate the shortest path length between two locations using BFS."""
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)])
        visited = {start_location}

        while queue:
            current_location, path_len = queue.popleft()

            for neighbor in self.roads.get(current_location, []):
                if neighbor not in visited:
                    if neighbor == goal_location:
                        return path_len + 1
                    visited.add(neighbor)
                    queue.append((neighbor, path_len + 1))
        return float('inf') # Goal not reachable, should not happen in valid problems, but handle for robustness

