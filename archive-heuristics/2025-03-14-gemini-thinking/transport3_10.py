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
    It calculates the shortest path (in terms of drive actions) between the current location of each package and its goal location,
    and adds estimated pick-up and drop actions.

    # Assumptions
    - The heuristic assumes that for each package, we need to perform a pick-up action at the initial location,
      drive actions along the shortest path to the goal location, and a drop action at the goal location.
    - It simplifies capacity constraints and assumes that vehicles are always available and have sufficient capacity.
    - It only considers the 'road' network for movement and does not account for vehicle availability or complex capacity management.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds an adjacency list representation of the road network from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package. This could be a location or inside a vehicle.
    2. Determine the goal location of the package.
    3. If the package is at a location:
        a. Find the shortest path (number of roads) from the current location to the goal location using Breadth-First Search (BFS) on the road network.
        b. Estimate the cost as: 1 (pick-up) + shortest path length (drive actions) + 1 (drop).
    4. If the package is in a vehicle:
        a. Find the location of the vehicle.
        b. Find the shortest path from the vehicle's location to the goal location using BFS.
        c. Estimate the cost as: shortest path length (drive actions) + 1 (drop).
    5. Sum the estimated costs for all packages to get the total heuristic value.
    6. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the transportHeuristic by extracting goal conditions and static road network.
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
                self.road_network[l2].append(l1) # Roads are bidirectional in the examples

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        heuristic_value = 0

        current_package_locations = {}
        current_vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "?x - locatable", "?l - location"):
                parts = get_parts(fact)
                locatable = parts[1]
                location = parts[2]
                if match(fact, "at", "?p - package", "*"):
                    current_package_locations[locatable] = location
                elif match(fact, "at", "?v - vehicle", "*"):
                    current_vehicle_locations[locatable] = location
            elif match(fact, "in", "?p - package", "?v - vehicle"):
                parts = get_parts(fact)
                package = parts[1]
                vehicle = parts[2]
                current_package_locations[package] = vehicle # Package is 'in' vehicle

        for package, goal_location in self.goal_locations.items():
            current_location = current_package_locations.get(package)

            if current_location != goal_location:
                if current_location in self.road_network: # Package is at a location
                    start_location = current_location
                    end_location = goal_location
                    path_len = self._get_shortest_path_len(start_location, end_location)
                    if path_len is not None:
                        heuristic_value += 2 + path_len # pick-up + drive + drop
                    else:
                        return float('inf') # No path, unsolvable? Or very high cost.

                elif current_location in current_vehicle_locations: # Package is in a vehicle
                    vehicle = current_location
                    start_location = current_vehicle_locations[vehicle]
                    end_location = goal_location
                    path_len = self._get_shortest_path_len(start_location, end_location)
                    if path_len is not None:
                        heuristic_value += 1 + path_len # drive + drop
                    else:
                        return float('inf') # No path, unsolvable? Or very high cost.
                else:
                    start_location = current_location
                    end_location = goal_location
                    path_len = self._get_shortest_path_len(start_location, end_location)
                    if path_len is not None:
                        heuristic_value += 2 + path_len # pick-up + drive + drop
                    else:
                        return float('inf')


        return heuristic_value

    def _get_shortest_path_len(self, start_location, end_location):
        """
        Compute the shortest path length between two locations using BFS on the road network.
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
