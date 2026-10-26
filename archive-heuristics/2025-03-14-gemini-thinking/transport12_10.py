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
    It calculates the shortest path (in terms of number of roads) between the current location of each package and its goal location,
    and sums up the estimated costs for picking up, driving, and dropping each package.

    # Assumptions
    - For each package, we assume we need to perform a pick-up action, drive action(s) to the goal location, and a drop action.
    - The heuristic focuses on the transportation of packages and ignores vehicle capacity constraints and size considerations for simplicity and efficiency.
    - It assumes that there is always a vehicle available to transport each package, and does not explicitly account for moving vehicles to packages.
    - The shortest path is calculated based on the `road` predicates provided in the static facts.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds a graph representing the road network from the static `road` predicates to efficiently calculate shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package.
    2. Determine the goal location of the package.
    3. If the package is not at its goal location:
        a. If the package is not currently in a vehicle:
            i. Estimate the cost as: 1 (pick-up) + shortest path length from current location to goal location (drive) + 1 (drop).
        b. If the package is currently in a vehicle:
            ii. Estimate the cost as: shortest path length from the vehicle's location to the goal location (drive) + 1 (drop).
    4. Sum up the estimated costs for all packages to get the total heuristic value.
    5. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.
        Extracts goal locations and builds the road network graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                self.package_goals[parts[1]] = parts[2]

        # Build road network graph for shortest path calculations
        self.road_graph = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Roads are bidirectional

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        Estimates the number of actions needed to reach the goal state.
        """
        state = node.state
        heuristic_value = 0

        package_locations = {}
        vehicle_locations = {}
        package_in_vehicle = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                entity = parts[1]
                location = parts[2]
                if entity in self.package_goals: # Assuming package names are in goal keys
                    package_locations[entity] = location
                else: # Assume it's a vehicle
                    vehicle_locations[entity] = location
            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                package = parts[1]
                vehicle = parts[2]
                package_in_vehicle[package] = vehicle

        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package, None)

            if current_location != goal_location:
                if package in package_in_vehicle:
                    vehicle = package_in_vehicle[package]
                    vehicle_location = vehicle_locations.get(vehicle, None)
                    if vehicle_location:
                        path_len = self._shortest_path_length(vehicle_location, goal_location)
                        heuristic_value += path_len + 1 # drive + drop
                    else:
                        heuristic_value += 2 # Assume 1 drive and 1 drop if vehicle location unknown (should not happen in valid states)
                else:
                    path_len = self._shortest_path_length(current_location, goal_location)
                    heuristic_value += 1 + path_len + 1 # pick-up + drive + drop

        return heuristic_value

    def _shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS on the road graph.
        Returns the path length, or a large number (infinity approximation) if no path exists.
        """
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            if current_location == goal_location:
                return distance

            for neighbor in self.road_graph[current_location]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf') # No path found, return infinity approximation
