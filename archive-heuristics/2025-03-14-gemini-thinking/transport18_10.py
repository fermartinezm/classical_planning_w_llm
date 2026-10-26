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
    This heuristic estimates the number of drive actions required to move all packages to their goal locations.
    It calculates the shortest path (in terms of road actions) for each package from its current location to its goal location and sums these path lengths.
    Pick-up and drop actions are not explicitly counted in this simplified heuristic, focusing on the transportation aspect.

    # Assumptions
    - The primary cost in the Transport domain is assumed to be the number of drive actions.
    - Pick-up and drop actions are considered less significant in terms of cost estimation for heuristic purposes.
    - A road network exists that connects all necessary locations, and paths can be found between package starting locations and goal locations.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds a road network graph from the static facts representing road connections.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package from the current state.
    2. Determine the goal location of the package from the task goals.
    3. Calculate the shortest path (number of road segments) from the current location to the goal location using Breadth-First Search (BFS) on the road network.
    4. Sum up the shortest path lengths for all packages that are not at their goal locations.
    5. The final sum represents the estimated number of drive actions needed to reach the goal state.
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
            if match(goal, "at", "?p - package", "?l - location"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "?l1 - location", "?l2 - location"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not in its goal location, calculate the shortest path
        to its goal and sum these path lengths.
        """
        state = node.state
        heuristic_value = 0

        package_locations = {}
        for fact in state:
            if match(fact, "at", "?p - package", "?l - location"):
                parts = get_parts(fact)
                package = parts[1]
                location = parts[2]
                package_locations[package] = location

        for package, goal_location in self.goal_locations.items():
            current_location = package_locations.get(package)
            if current_location and current_location != goal_location:
                path_length = self.shortest_path_length(current_location, goal_location)
                heuristic_value += path_length

        return heuristic_value

    def shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS.

        Returns the length of the shortest path, or a large value if no path exists.
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

        return float('inf') # No path found, return infinity (or a large value)

