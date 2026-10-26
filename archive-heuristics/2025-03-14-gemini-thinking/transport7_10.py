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
    It calculates the shortest path (in terms of number of roads) between the current location of each package and its goal location, and adds a fixed cost for picking up and dropping each package.

    # Assumptions:
    - Vehicles are always available at the package's initial location when needed.
    - Vehicle capacity is not considered in the heuristic estimation.
    - Roads are bidirectional and have a uniform cost of 1 for traversal.
    - The heuristic focuses on minimizing the number of drive, pick-up, and drop actions.

    # Heuristic Initialization:
    - Extracts the goal locations for each package from the task goals.
    - Builds a road network graph from the static facts to calculate shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic:
    For each package that is not at its goal location:
    1. Determine the current location of the package from the current state.
    2. Determine the goal location of the package from the task goals.
    3. Calculate the shortest path length (number of roads) between the current location and the goal location using Breadth-First Search (BFS) on the road network.
    4. Estimate the cost for transporting this package as: shortest path length + 2 (for pick-up and drop actions).
    5. Sum up the estimated costs for all packages that are not at their goal locations.
    6. The total sum is the heuristic value for the given state.
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
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                self.goal_locations[parts[1]] = parts[2]

        self.roads = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.roads[l1].append(l2)
                self.roads[l2].append(l1) # Roads are bidirectional

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not in its goal location, estimate the cost to move it to the goal.
        The total heuristic value is the sum of these individual package costs.
        """
        state = node.state
        heuristic_value = 0

        current_package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] not in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']: # Consider only packages, not vehicles
                    current_package_locations[parts[1]] = parts[2]

        for package, goal_location in self.goal_locations.items():
            if package not in current_package_locations: # Package might be in vehicle, assume current location is needed for pick up.
                continue # Cannot estimate if package location is unknown. Should not happen in valid states.

            current_location = current_package_locations[package]

            if current_location != goal_location:
                shortest_path_len = self.get_shortest_path_length(current_location, goal_location)
                if shortest_path_len is None:
                    return float('inf') # Indicate unsolvable if no path exists. Though roads should ensure connectivity in typical problems.
                heuristic_value += shortest_path_len + 2 # pick-up and drop actions

        return heuristic_value

    def get_shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS.
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
