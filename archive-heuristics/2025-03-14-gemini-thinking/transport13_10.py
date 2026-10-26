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
    It calculates the shortest path (in terms of road segments) for each package from its current location to its goal location and adds a fixed cost for pick-up and drop actions.

    # Assumptions
    - The heuristic assumes that for each package, we need to perform a pick-up action at the starting location, drive along the shortest path, and perform a drop action at the goal location.
    - It simplifies the capacity constraints and vehicle availability, focusing primarily on the road network and package locations.
    - It does not consider optimizing vehicle usage or capacity planning.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Pre-computes the road network as an adjacency list from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package. If the package is 'in' a vehicle, consider the vehicle's location as the package's current location. If the package is 'at' a location, use that location.
    2. Determine the goal location of the package from the task goals.
    3. Find the shortest path (number of road segments) between the current location and the goal location using Breadth-First Search (BFS) on the road network.
    4. Estimate the cost for this package as the length of the shortest path + 2 (for one pick-up and one drop action). If the package is initially in a vehicle, add an additional cost of 1 for dropping the package from the vehicle first.
    5. Sum up the estimated costs for all packages to get the total heuristic value.
    6. If all packages are at their goal locations, the heuristic value is 0.
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
            if match(goal, "at", "?p", "?l"):
                package = get_parts(goal)[1]
                location = get_parts(goal)[2]
                self.goal_locations[package] = location

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "?l1", "?l2"):
                l1 = get_parts(fact)[1]
                l2 = get_parts(fact)[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not in its goal location, estimate the cost to move it to the goal.
        Sum these costs to get the total heuristic value.
        """
        state = node.state
        heuristic_value = 0

        current_locations = {}
        for fact in state:
            if match(fact, "at", "?obj", "?loc"):
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                current_locations[obj] = loc
            elif match(fact, "in", "?pkg", "?veh"):
                package = get_parts(fact)[1]
                vehicle = get_parts(fact)[2]
                current_locations[package + "_in_vehicle"] = vehicle # Mark package as in vehicle

        for package, goal_location in self.goal_locations.items():
            goal_fact = f'(at {package} {goal_location})'
            if goal_fact in state:
                continue # Package already at goal

            current_location = None
            in_vehicle_cost = 0
            if package + "_in_vehicle" in current_locations:
                vehicle = current_locations[package + "_in_vehicle"]
                vehicle_location_fact = next((fact for fact in state if match(fact, "at", vehicle, "*")), None)
                if vehicle_location_fact:
                    current_location = get_parts(vehicle_location_fact)[2]
                    in_vehicle_cost = 1 # Cost for dropping from vehicle first
            elif package in current_locations:
                current_location = current_locations[package]

            if current_location and current_location != goal_location:
                path_length = self._shortest_path_length(current_location, goal_location)
                if path_length is not None:
                    heuristic_value += path_length + 1 + in_vehicle_cost # +1 for pick-up, + in_vehicle_cost for initial drop if needed
                else:
                    return float('inf') # No path, unsolvable? Or very high cost.

        return heuristic_value

    def _shortest_path_length(self, start_location, goal_location):
        """
        Compute the shortest path length between two locations using BFS on the road network.
        Returns the path length or None if no path exists.
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
