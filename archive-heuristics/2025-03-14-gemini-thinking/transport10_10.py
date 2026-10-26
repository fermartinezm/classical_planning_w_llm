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
    It calculates the shortest path (in terms of road segments) for each package from its current location to its goal location and sums up the estimated actions: pick-up, drive, and drop.

    # Assumptions:
    - Vehicles are always available at the starting location of packages.
    - Capacity constraints are not considered in the heuristic estimation.
    - The heuristic focuses on the number of road segments and pick-up/drop actions, assuming each road segment traversal costs one 'drive' action, and each pick-up and drop costs one action.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a road network graph from the static 'road' predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package.
    2. Determine the goal location of the package.
    3. Find the shortest path (number of road segments) from the current location to the goal location using Breadth-First Search (BFS) on the road network.
       - If no path exists, consider the heuristic value as infinity (or a very large number, practically a large enough estimate). In this implementation, we assume paths always exist for solvable instances.
    4. Estimate the number of actions for this package:
       - 1 'pick-up' action (if the package is not already in a vehicle - we simplify and always assume pick-up is needed at the starting location).
       - Number of 'drive' actions equal to the length of the shortest path (number of road segments).
       - 1 'drop' action at the goal location.
    5. Sum up the estimated actions for all packages to get the total heuristic value.
    6. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the transportHeuristic by extracting:
        - Goal locations for each package.
        - Road network from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                obj_type = task.type_dict.get(parts[1], None)
                if obj_type == 'package':
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
                self.road_network[l2].append(l1) # Roads are bidirectional in examples

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0

        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_type = node.task.type_dict.get(parts[1], None)
                if obj_type == 'package':
                    package_locations[parts[1]] = parts[2]

        for package, goal_location in self.goal_locations.items():
            current_location = package_locations.get(package, None)
            if current_location is None: # Package might be in vehicle, assume its location is vehicle's location for simplicity, or handle based on 'in' predicate if needed for better accuracy
                continue # In 'transport' domain, packages are always 'at' or 'in'. If not 'at', then 'in'. For heuristic simplicity, we consider 'at' only for now.

            if current_location != goal_location:
                # BFS to find shortest path
                queue = collections.deque([(current_location, 0)]) # (location, distance)
                visited = {current_location}
                path_len = -1

                while queue:
                    loc, dist = queue.popleft()
                    if loc == goal_location:
                        path_len = dist
                        break

                    for neighbor in self.road_network[loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

                if path_len != -1: # Path found
                    heuristic_value += 1 # pick-up
                    heuristic_value += path_len # drive actions
                    heuristic_value += 1 # drop

        return heuristic_value
