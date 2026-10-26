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
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their goal locations.
    It considers the necessary pick-up, drop, and drive actions for each package, based on shortest path distances on the road network.

    # Assumptions
    - The heuristic assumes that for each package, the optimal strategy involves picking it up at its current location,
      driving to the goal location, and dropping it there.
    - It simplifies capacity constraints and assumes any vehicle can carry any package if at the same location.
    - It calculates the shortest path in terms of number of roads, approximating the drive actions needed.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Extracts the road network from the static facts to calculate shortest paths.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the package's current location and its goal location.
    2. If the package is not at its goal location, estimate the cost as follows:
        a. Cost for picking up the package at its current location (1 action).
        b. Calculate the shortest path (number of roads) from the package's current location to its goal location using BFS on the road network. The length of this path represents the estimated number of drive actions.
        c. Cost for dropping the package at its goal location (1 action).
    3. Sum up the estimated costs for all packages not at their goal locations.
    4. The total sum is the heuristic estimate for the current state.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts goal locations for each package.
        - Extracts road network information from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                obj_type = task.type_hierarchy.get(parts[1], [])
                if "package" in obj_type:
                    self.goal_locations[parts[1]] = parts[2]

        self.roads = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                self.roads[parts[1]].append(parts[2])
                self.roads[parts[2]].append(parts[1]) # Roads are bidirectional in provided examples


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not in its goal location, estimate the number of actions:
        pick-up + shortest path drives + drop.
        """
        state = node.state
        heuristic_value = 0

        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_type = node.task.type_hierarchy.get(parts[1], [])
                if "package" in obj_type:
                    package_locations[parts[1]] = parts[2]

        for package, goal_location in self.goal_locations.items():
            current_location = package_locations.get(package)
            if current_location != goal_location:
                heuristic_value += 2 # pick-up and drop actions

                # Shortest path calculation using BFS
                queue = collections.deque([(current_location, 0)]) # (location, distance)
                visited = {current_location}
                path_found = False
                shortest_path_len = 0

                while queue:
                    loc, dist = queue.popleft()
                    if loc == goal_location:
                        shortest_path_len = dist
                        path_found = True
                        break

                    for neighbor in self.roads[loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

                if path_found:
                    heuristic_value += shortest_path_len
                else:
                    # If no path is found, it means the goal is unreachable in this simplified heuristic view.
                    # We should return a large value to discourage this path, but for a greedy heuristic,
                    # a finite value is still needed. We can assume a large path cost if no road connection.
                    heuristic_value += 10 # Assign a penalty if no path is found, indicating longer distance.


        return heuristic_value
