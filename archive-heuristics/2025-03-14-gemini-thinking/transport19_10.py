from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 l1)".
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
    It considers the need to pick up, drop, and drive vehicles to move packages.
    The heuristic uses a simplified approach focusing on the number of actions and shortest paths in the road network.

    # Assumptions
    - Vehicles are always available at the starting location of packages.
    - Vehicle capacity is not explicitly considered in the heuristic calculation, assuming vehicles can carry packages when needed.
    - The heuristic focuses on minimizing the number of actions (pick-up, drop, drive) and does not guarantee admissibility.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a road network from the static facts representing `road` predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Check if the package is currently 'in' a vehicle. If yes, account for a 'drop' action to take it out.
    2. Account for a 'pick-up' action to load the package into a vehicle at its current location.
    3. Calculate the shortest path (number of roads) from the package's current location to its goal location using Breadth-First Search (BFS) on the road network. The length of this path represents the number of 'drive' actions needed.
    4. Account for a 'drop' action at the goal location to unload the package.
    5. Sum up the estimated actions for all packages to get the total heuristic value.
    If a package is already at its goal location, it contributes 0 to the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts goal locations for each package.
        - Builds the road network from static 'road' facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if parts[1] not in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']: # Assuming vehicle names start with 'v'
                    package = parts[1]
                    location = parts[2]
                    self.package_goals[package] = location

        # Build road network (adjacency list)
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

        For each package not at its goal location, estimate the number of actions:
        drop (if in vehicle) + pick-up + drive (shortest path) + drop.
        Sum these estimates for all packages.
        """
        state = node.state
        heuristic_value = 0

        # Get current locations of packages and vehicles
        package_locations = {}
        vehicle_locations = {}
        packages_in_vehicles = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                if obj.startswith('p'): # Assuming package names start with 'p'
                    package_locations[obj] = location
                elif obj.startswith('v'): # Assuming vehicle names start with 'v'
                    vehicle_locations[obj] = location
            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                package = parts[1]
                vehicle = parts[2]
                packages_in_vehicles[package] = vehicle

        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package, None)

            if current_location == goal_location:
                continue # Package already at goal

            package_cost = 0

            if package in packages_in_vehicles:
                package_cost += 1 # drop action needed

            package_cost += 1 # pick-up action needed

            # Calculate shortest path using BFS
            start_location = current_location
            end_location = goal_location

            if start_location != end_location:
                queue = collections.deque([(start_location, 0)]) # location, distance
                visited = {start_location}
                path_len = -1

                while queue:
                    current_loc, distance = queue.popleft()

                    if current_loc == end_location:
                        path_len = distance
                        break

                    for neighbor in self.road_network[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, distance + 1))

                if path_len != -1:
                    package_cost += path_len # drive actions
                else:
                    # No path found, this should ideally not happen in solvable instances,
                    # but for robustness, we can assign a high cost or handle as needed.
                    package_cost += 100 # High cost for no path, or handle differently

            package_cost += 1 # drop action at goal

            heuristic_value += package_cost

        return heuristic_value
