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
    It calculates the number of pick-up, drive, and drop actions needed for each package, assuming the vehicle is always available at the package's location.
    The heuristic focuses on the transportation aspect and simplifies capacity and vehicle availability.

    # Assumptions:
    - Vehicles are always available at the starting location of each package.
    - Vehicle capacity is not a limiting factor (simplified heuristic).
    - The heuristic estimates the cost for each package independently and sums them up.
    - Shortest path in terms of road segments is used as the drive cost.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Pre-computes the road network from the static facts to efficiently find shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Identify the current location of the package and its goal location.
    2. Estimate the cost to pick up the package (1 action).
    3. Calculate the shortest path (in terms of number of road segments) from the current location to the goal location using BFS on the road network. The length of this path is the estimated number of drive actions.
    4. Estimate the cost to drop the package at the goal location (1 action).
    5. Sum up these costs for all packages that are not at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts goal locations for each package.
        - Builds the road network from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.package_goals[package] = location

        # Build road network for shortest path calculation
        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional in examples

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not in its goal location, estimate the actions needed:
        pick-up + drive (shortest path) + drop.
        """
        state = node.state
        heuristic_value = 0

        current_package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                # Check if it is a package
                if any(match(obj, p) for p in self.package_goals.keys()): # Simple check if object name matches package name
                    current_package_locations[obj] = location

        for package, goal_location in self.package_goals.items():
            current_location = current_package_locations.get(package)

            if current_location != goal_location:
                # Estimate cost for pick-up, drive, and drop
                cost = 0
                cost += 1 # pick-up action

                # Calculate shortest path using BFS
                path_length = self.shortest_path_length(current_location, goal_location)
                if path_length is None:
                    # No path exists, which should not happen in solvable instances, but handle for robustness.
                    return float('inf') # Indicate unsolvable path for this package, or a very high cost.
                cost += path_length # drive actions

                cost += 1 # drop action
                heuristic_value += cost

        return heuristic_value

    def shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS on the road network.
        Returns the path length (number of roads), or None if no path exists.
        """
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            for neighbor in self.road_network[current_location]:
                if neighbor not in visited:
                    if neighbor == goal_location:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return None # No path found
