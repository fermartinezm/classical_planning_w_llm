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
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It calculates the shortest path in terms of road segments for each package from its current location to its goal location and sums up the estimated actions (pick-up, drive, drop) for each package.

    # Assumptions:
    - For each package that is not at its goal location, we assume it needs to be picked up, transported via the shortest path, and dropped at the goal location.
    - We ignore vehicle capacity and availability, assuming there is always a vehicle available and with enough capacity.
    - We only consider the 'drive', 'pick-up', and 'drop' actions and estimate the number of these actions.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build a road network graph from the static facts 'road' predicates.
    - Precompute shortest path distances between all pairs of locations in the road network using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. For each goal condition of the form '(at package location)' in the goal description:
        a. Identify the package and its goal location.
        b. Get the current location of the package from the current state.
        c. If the package is not at its goal location:
            i. Find the shortest path distance (number of road segments) from the current location to the goal location using the precomputed shortest paths. If no path exists, consider distance as infinity (or a very large number, practically handled by BFS returning -1).
            ii. Estimate the number of actions for this package as 2 (for pick-up and drop) + shortest path distance.
            iii. Add this estimated number of actions to the total heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic by extracting goal conditions,
        building the road network, and precomputing shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if parts[1] not in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']: # only consider package goals
                    package = parts[1]
                    location = parts[2]
                    self.goal_locations[package] = location

        self.road_network = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # roads are bidirectional
                locations.add(l1)
                locations.add(l2)
        self.locations = list(locations)
        self.location_to_index = {loc: i for i, loc in enumerate(self.locations)}
        self.index_to_location = {i: loc for i, loc in enumerate(self.locations)}

        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._compute_shortest_paths_from(start_loc)


    def _compute_shortest_paths_from(self, start_loc):
        """Compute shortest path distances from a start location to all other locations using BFS."""
        distances = {loc: -1 for loc in self.locations} # -1 represents infinity
        distances[start_loc] = 0
        queue = collections.deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            for neighbor in self.road_network[current_loc]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Estimate the heuristic value for a given state.
        """
        state = node.state
        current_package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                current_package_locations[obj] = location

        heuristic_value = 0
        for package, goal_location in self.goal_locations.items():
            current_location = current_package_locations.get(package)
            if current_location != goal_location:
                shortest_path_distance = self.shortest_paths.get(current_location, {}).get(goal_location, -1)
                if shortest_path_distance != -1:
                    heuristic_value += 2 + shortest_path_distance # pick-up, drop, and drive actions
                else:
                    heuristic_value += 1000 # Assign a large cost if no path exists, effectively discouraging this path

        return heuristic_value
