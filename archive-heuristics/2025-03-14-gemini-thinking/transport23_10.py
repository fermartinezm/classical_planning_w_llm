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
    This heuristic estimates the minimum number of actions required to move all packages to their goal locations.
    It calculates the shortest path in terms of road segments between the current location and the goal location for each package and sums up the estimated costs.
    The cost for each package is estimated as the length of the shortest path plus 2 (for pick-up and drop actions).

    # Assumptions
    - Vehicles are always available at the starting locations of packages.
    - The cost is primarily determined by the number of drive actions and pick-up/drop actions.
    - Capacity constraints are implicitly considered by assuming that pick-up and drop actions are always possible when needed (if preconditions are met in the plan).

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds a road network graph from the static facts representing road connections.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package from the current state.
    2. Determine the goal location of the package from the task goals.
    3. If the current location is the same as the goal location, the cost for this package is 0.
    4. If the locations are different, find the shortest path (in terms of number of road segments) between the current location and the goal location using Breadth-First Search (BFS) on the road network.
    5. If a path is found, estimate the cost for this package as the length of the shortest path + 2 (to account for one pick-up and one drop action).
    6. If no path is found, it implies that the goal is unreachable for this package from its current location given the road network. In this heuristic, we can return a very large number (infinity is not directly representable, but a large enough number will suffice for greedy best-first search to avoid this path if possible). In this implementation, we will return a large number like 10000, assuming typical path lengths are much smaller.
    7. Sum up the estimated costs for all packages to get the total heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Specifically, extract goal package locations and build the road network.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if parts[1] not in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']: # Assuming vehicle names start with 'v'
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
        """
        Estimate the heuristic value for a given state.
        """
        state = node.state
        current_locations = {}
        packages = set()

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                locatable_name = parts[1]
                location_name = parts[2]
                current_locations[locatable_name] = location_name
                # Identify packages from 'at' facts, assuming package names start with 'p'
                if locatable_name.startswith('p'):
                    packages.add(locatable_name)

        heuristic_value = 0
        for package in packages:
            if package not in self.goal_locations: # if package is not in goal, skip
                continue
            goal_location = self.goal_locations[package]
            current_location = current_locations.get(package)

            if current_location == goal_location:
                continue # Package already at goal

            if current_location is None: # Package location not found in state, something is wrong, or assume it's not yet placed.
                return 10000 # Return a large value to penalize

            if current_location != goal_location:
                path_length = self.shortest_path_length(current_location, goal_location)
                if path_length == -1: # No path found
                    return 10000 # Return a large value if no path exists
                heuristic_value += path_length + 2 # +2 for pick-up and drop

        return heuristic_value

    def shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations in the road network using BFS.
        Returns path length or -1 if no path exists.
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
        return -1 # No path found
