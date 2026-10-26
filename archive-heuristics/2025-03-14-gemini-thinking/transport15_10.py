from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l8)".
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
    It calculates the shortest path (in terms of number of roads) for each package from its current location to its goal location and sums up the estimated actions: pick-up, drive along the shortest path, and drop.

    # Assumptions
    - Vehicles are always available at the package's initial location when needed.
    - Vehicle capacity is sufficient for all packages.
    - The heuristic focuses on the movement of packages and ignores capacity management actions in the cost estimation.
    - It assumes that for each package, we need to perform a pick-up action at the starting location, drive along the shortest path to the goal location, and perform a drop action at the goal location.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds an adjacency list representation of the road network from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package. If the package is in a vehicle, consider the vehicle's location as the package's current location for path planning.
    2. Find the shortest path (in terms of number of roads) from the package's current location to its goal location using Breadth-First Search (BFS) on the road network.
    3. Estimate the number of actions for this package as: 1 (pick-up) + shortest path length (drive actions) + 1 (drop).
    4. Sum up the estimated actions for all packages that are not at their goal locations.
    5. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        self.roads = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.roads[l1].append(l2)
                self.roads[l2].append(l1) # Roads are bidirectional in provided examples

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        heuristic_value = 0

        package_current_locations = {}
        vehicle_current_locations = {}
        package_in_vehicle = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                locatable = parts[1]
                location = parts[2]
                if match(fact, "at", "* - package", "*"):
                    package_current_locations[locatable] = location
                elif match(fact, "at", "* - vehicle", "*"):
                    vehicle_current_locations[locatable] = location
            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                package = parts[1]
                vehicle = parts[2]
                package_in_vehicle[package] = vehicle

        for package, goal_location in self.goal_locations.items():
            current_location = package_current_locations.get(package)
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                current_location = vehicle_current_locations.get(vehicle)

            if current_location != goal_location:
                path_length = self.get_shortest_path_length(current_location, goal_location)
                if path_length is None:
                    # No path found, should not happen in solvable instances, but handle for robustness.
                    # Return a large value to discourage this path.
                    return float('inf')
                heuristic_value += 2 + path_length # pick-up + path_length drives + drop

        return heuristic_value

    def get_shortest_path_length(self, start_location, goal_location):
        """
        Find the shortest path length between two locations using BFS.
        Returns the path length or None if no path exists.
        """
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            for neighbor in self.roads[current_location]:
                if neighbor not in visited:
                    if neighbor == goal_location:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None # No path found
