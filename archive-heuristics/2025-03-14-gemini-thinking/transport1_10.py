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
    It calculates the cost for each package independently and sums them up.
    The cost for each package includes picking it up (if necessary), driving a vehicle to the goal location, and dropping it.
    The driving cost is estimated using the shortest path in terms of number of roads.

    # Assumptions
    - A vehicle is always available at the package's current location when needed for pick-up.
    - Vehicle capacity and size constraints are not explicitly considered in the heuristic estimation,
      but the pick-up and drop actions are counted.
    - The heuristic assumes that for each package, we need to perform a pick-up, drive, and drop sequence if it's not at its goal.

    # Heuristic Initialization
    - Extracts the road network from the static facts to calculate shortest paths between locations.
    - Extracts the goal locations for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Check if the package is already at its goal location. If yes, the cost for this package is 0.
    2. If not at the goal location, estimate the cost as follows:
        a. If the package is not currently in a vehicle, assume a 'pick-up' action is needed (cost +1).
        b. Calculate the shortest path (in terms of number of roads) from the package's current location to its goal location using BFS on the road network. Let the path length be 'path_len'. Assume 'path_len' 'drive' actions are needed (cost + path_len).
        c. Assume a 'drop' action is needed at the goal location (cost +1).
    3. Sum up the estimated costs for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts road network from static facts.
        - Extracts goal locations for packages.
        """
        self.goals = task.goals
        static_facts = task.static

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package_name, goal_location = get_parts(goal)
                self.package_goals[package_name] = goal_location

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package, if it's not at its goal location, estimate the number of actions:
        pick-up (if not in vehicle) + drive (shortest path) + drop.
        Sum these costs for all packages.
        """
        state = node.state
        heuristic_value = 0

        package_locations = {}
        package_in_vehicle = {}
        vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, location = get_parts(fact)
                if any(match(obj, type_name, "*") for type_name in ["p*", "package"]): # Heuristic needs to work even if object names are not exactly "p*"
                    package_locations[obj] = location
                elif any(match(obj, type_name, "*") for type_name in ["v*", "vehicle"]): # Heuristic needs to work even if object names are not exactly "v*"
                    vehicle_locations[obj] = location
            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                package_in_vehicle[package] = vehicle

        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package, None)
            if current_location is None: # Should not happen in valid states, but for robustness
                continue

            if current_location != goal_location:
                cost_for_package = 0
                if package not in package_in_vehicle:
                    cost_for_package += 1 # pick-up

                start_location = current_location
                end_location = goal_location

                if start_location != end_location:
                    path_len = self._get_shortest_path_len(start_location, end_location)
                    if path_len is None: # No path, instance might be unsolvable, or heuristic is not handling this case well. Return infinity or a large value.
                        return float('inf') # Indicate unsolvability or very high cost
                    cost_for_package += path_len # drive actions

                cost_for_package += 1 # drop
                heuristic_value += cost_for_package

        return heuristic_value

    def _get_shortest_path_len(self, start_location, end_location):
        """
        Calculate the shortest path length (number of roads) between two locations using BFS.
        Returns path length or None if no path exists.
        """
        if start_location == end_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            for neighbor in self.road_network[current_location]:
                if neighbor not in visited:
                    if neighbor == end_location:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None # No path found
