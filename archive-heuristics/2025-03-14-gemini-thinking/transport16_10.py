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
    This heuristic estimates the number of actions required to transport all packages to their goal locations.
    It calculates the shortest path in terms of road segments for each package from its current location to its goal location and sums these distances.
    This heuristic is admissible if we only consider drive actions and assume pick-up and drop are free, which is not the case, so it's not admissible in general, but should be informative.

    # Assumptions:
    - The cost is primarily determined by the driving distance.
    - Pick-up and drop actions are implicitly considered by assuming that a vehicle is always available and capable.
    - Capacity constraints are not explicitly considered in this simplified heuristic.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build a road network graph from the static facts to calculate shortest paths.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package.
    2. Determine the goal location of the package.
    3. Find the shortest path (number of road segments) between the current location and the goal location using Breadth-First Search (BFS) on the road network.
    4. Sum up the shortest path distances for all packages that are not at their goal locations.
    5. The total sum is the estimated number of actions. This heuristic approximates the number of 'drive' actions needed, ignoring 'pick-up' and 'drop' actions for simplicity and efficiency.
    """

    def __init__(self, task):
        """
        Initialize the transportHeuristic by extracting:
        - Goal locations for each package.
        - Road network information from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3:
                    predicate, obj, location = parts
                    if predicate == 'at':
                        is_package = False
                        for type_def in task.task_def.domain.types:
                            if type_def.name == 'package' and obj in task.task_def.domain.objects.get(type_def.name, []):
                                is_package = True
                                break
                        if is_package:
                            self.package_goals[obj] = location

        # Build road network graph
        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                u, v = get_parts(fact)[1], get_parts(fact)[2]
                self.road_network[u].append(v)
                self.road_network[v].append(u) # Roads are bidirectional in examples

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        The heuristic value is the sum of shortest path distances for each package to its goal location.
        """
        state = node.state
        heuristic_value = 0

        package_current_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    predicate, obj, location = parts
                    if predicate == 'at':
                        is_package = False
                        for type_def in node.task.task_def.domain.types:
                            if type_def.name == 'package' and obj in node.task.task_def.domain.objects.get(type_def.name, []):
                                is_package = True
                                break
                        if is_package:
                            package_current_locations[obj] = location

        for package, goal_location in self.package_goals.items():
            current_location = package_current_locations.get(package)
            if current_location != goal_location:
                shortest_path_distance = self.get_shortest_path_distance(current_location, goal_location)
                heuristic_value += shortest_path_distance

        return heuristic_value

    def get_shortest_path_distance(self, start_location, goal_location):
        """
        Calculate the shortest path distance between two locations using BFS on the road network.
        Returns the number of road segments in the shortest path.
        If no path exists, returns infinity (represented as a large number).
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

        return float('inf') # No path found, return infinity (or a very large number)
