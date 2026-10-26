from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

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
    It primarily focuses on the number of drive actions needed based on shortest paths between locations,
    and adds a fixed cost for pick-up and drop actions for each package that is not at its goal location.

    # Assumptions:
    - The heuristic assumes that vehicles are always available at the package's location when needed for pick-up.
    - It simplifies the problem by focusing on drive actions and adding a constant cost for pick-up and drop, ignoring capacity constraints and more complex action sequencing.
    - It assumes that the road network is connected enough to reach all goal locations.

    # Heuristic Initialization
    - Precomputes the shortest path distances between all pairs of locations based on the `road` predicates provided in the static facts.
    - Extracts goal locations for each package from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location in the current state:
    1. Determine the current location of the package.
    2. Determine the goal location of the package.
    3. Find the shortest path distance (number of roads) between the current location and the goal location using the precomputed shortest path distances.
    4. Add this shortest path distance to the heuristic estimate. This represents the estimated number of 'drive' actions.
    5. Add a constant cost (e.g., 2 actions) to account for the 'pick-up' and 'drop' actions required for each package.
    6. Sum up these estimated costs for all packages to get the total heuristic value for the state.
    7. If all packages are at their goal locations, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.
        Precompute shortest paths between locations and extract goal locations for packages.
        """
        self.goals = task.goals
        static_facts = task.static

        locations = set()
        roads = []

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                locations.add(l1)
                locations.add(l2)
                roads.append((l1, l2))

        self.locations = list(locations)
        self.shortest_paths = self._compute_shortest_paths(self.locations, roads)

        self.package_goals = {}
        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                package_name = get_parts(goal_fact)[1]
                goal_location = get_parts(goal_fact)[2]
                self.package_goals[package_name] = goal_location

    def _compute_shortest_paths(self, locations, roads):
        """
        Compute all-pairs shortest paths between locations using Floyd-Warshall algorithm.
        """
        dist = {l1: {l2: float('inf') for l2 in locations} for l1 in locations}
        for l in locations:
            dist[l][l] = 0
        for l1, l2 in roads:
            dist[l1][l2] = 1
            dist[l2][l1] = 1 # Roads are bidirectional

        for k, i, j in itertools.product(locations, locations, locations):
            if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        return dist

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_type = ""
                for obj_def in node.task.task_def.domain.types:
                    if obj_def.name == 'locatable':
                        obj_type = 'locatable'
                        break
                    elif obj_def.name == 'package':
                        obj_type = 'package'
                        break
                if obj_type == 'package':
                    package_name = get_parts(fact)[1]
                    location = get_parts(fact)[2]
                    package_locations[package_name] = location

        for package_name, goal_location in self.package_goals.items():
            current_location = package_locations.get(package_name)
            if current_location != goal_location:
                if current_location is None: # Package is in vehicle, need to find vehicle location
                    for fact in state:
                        if match(fact, "in", package_name, "*"):
                            vehicle_name = get_parts(fact)[2]
                            for vehicle_fact in state:
                                if match(vehicle_fact, "at", vehicle_name, "*"):
                                    current_location_vehicle = get_parts(vehicle_fact)[2]
                                    if current_location_vehicle is not None and goal_location in self.shortest_paths and current_location_vehicle in self.shortest_paths:
                                        path_len = self.shortest_paths.get(current_location_vehicle, {}).get(goal_location, float('inf'))
                                        if path_len != float('inf'):
                                            heuristic_value += path_len + 1 # At least one drop action needed
                                        else:
                                            heuristic_value += 100 # Penalize if no path found, or a large constant
                                    else:
                                        heuristic_value += 100 # Penalize if no path found, or a large constant
                                    break
                            break # if package is in vehicle, we consider vehicle location for path.
                    if current_location is None: # Package is not at goal and not in vehicle, so it is at some location.
                        for fact in state:
                            if match(fact, "at", package_name, "*"):
                                current_location = get_parts(fact)[2]
                                if current_location is not None and goal_location in self.shortest_paths and current_location in self.shortest_paths:
                                    path_len = self.shortest_paths.get(current_location, {}).get(goal_location, float('inf'))
                                    if path_len != float('inf'):
                                        heuristic_value += path_len + 2 # pick-up and drop actions
                                    else:
                                        heuristic_value += 100 # Penalize if no path found, or a large constant
                                    break
                elif current_location is not None and goal_location in self.shortest_paths and current_location in self.shortest_paths:
                    path_len = self.shortest_paths.get(current_location, {}).get(goal_location, float('inf'))
                    if path_len != float('inf'):
                        heuristic_value += path_len + 2 # pick-up and drop actions
                    else:
                        heuristic_value += 100 # Penalize if no path found, or a large constant
                else:
                    heuristic_value += 100 # Penalize if no path found, or a large constant

        return heuristic_value
