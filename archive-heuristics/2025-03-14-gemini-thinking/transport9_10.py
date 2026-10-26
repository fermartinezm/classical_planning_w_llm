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
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their goal locations.
    It calculates the shortest path (in terms of drive actions) between locations and adds the necessary pick-up and drop actions.

    # Assumptions
    - There are enough vehicles and capacity to transport all packages.
    - The heuristic focuses on moving packages to their destination locations and does not explicitly consider vehicle capacity constraints in detail,
      but assumes that pick-up and drop actions are always possible when preconditions are met in a plan.
    - The cost of each drive, pick-up, and drop action is assumed to be 1.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Extracts the road network from the static facts to calculate shortest paths.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package. It can be at a location or inside a vehicle.
    2. Determine the goal location of the package.
    3. If the package is at a location:
        a. Calculate the shortest path (number of drive actions) from the current location to the goal location using BFS on the road network.
        b. Add 2 to the heuristic cost for pick-up and drop actions, plus the length of the shortest path (number of drive actions).
    4. If the package is in a vehicle:
        a. Determine the current location of the vehicle.
        b. Calculate the shortest path from the vehicle's location to the goal location.
        c. Add 1 to the heuristic cost for the drop action, plus the length of the shortest path (number of drive actions).
    5. Sum up the costs for all packages to get the total heuristic value.
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
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

    def get_shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS on the road network.
        Returns the path length or infinity if no path exists.
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

        return float('inf') # No path found

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        package_current_locations = {}
        vehicle_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                locatable = parts[1]
                location = parts[2]
                if match(fact, "at", "?p", "*") and any(match(obj_type_fact, ":objects", "*", "- package") and get_parts(obj_type_fact)[1] == locatable for obj_type_fact in node.task.task_def.domain.types):
                    package_current_locations[locatable] = location
                elif match(fact, "at", "?v", "*") and any(match(obj_type_fact, ":objects", "*", "- vehicle") and get_parts(obj_type_fact)[1] == locatable for obj_type_fact in node.task.task_def.domain.types):
                    vehicle_locations[locatable] = location
            elif match(fact, "in", "*", "*"):
                parts = get_parts(fact)
                package = parts[1]
                vehicle = parts[2]
                package_current_locations[package] = vehicle # Package is 'in' vehicle

        for package, goal_location in self.goal_locations.items():
            current_location_or_vehicle = package_current_locations.get(package)

            if current_location_or_vehicle != goal_location:
                if current_location_or_vehicle in vehicle_locations.values() or current_location_or_vehicle is None: # Package is at a location
                    current_location = package_current_locations.get(package)
                    if current_location is None:
                        # Package is not in the state, assume it is at its initial location.
                        # This is a simplification and might not be accurate in all cases.
                        # For a more robust heuristic, we should track initial package locations.
                        # For now, we assume it starts at some location and needs pick-up, drive, drop.
                        # In a real planner, initial state is always given.
                        initial_location = None # We don't have access to initial state here easily.
                        # In this heuristic, if package is not in state and not at goal, we assume it needs to be moved.
                        heuristic_value += 2 + self.get_shortest_path_length(current_location, goal_location) if current_location else 2 + self.get_shortest_path_length(list(self.road_network.keys())[0] if self.road_network else None, goal_location) if goal_location else 0

                    else:
                        path_len = self.get_shortest_path_length(current_location, goal_location)
                        heuristic_value += 2 + path_len if path_len != float('inf') else 2 + 1000 # Add a large penalty if no path
                else: # Package is in a vehicle
                    vehicle = current_location_or_vehicle
                    vehicle_location = vehicle_locations.get(vehicle)
                    if vehicle_location:
                        path_len = self.get_shortest_path_length(vehicle_location, goal_location)
                        heuristic_value += 1 + path_len if path_len != float('inf') else 1 + 1000 # Add a large penalty if no path
                    else:
                        heuristic_value += 1 + 1000 # Vehicle location not found, add large penalty

        return heuristic_value
