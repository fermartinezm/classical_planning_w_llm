from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport packages
    to their goal locations. It counts necessary pick-up and drop actions for
    packages not at their goal and adds the cost of the longest required vehicle drive.
    It ignores vehicle capacity constraints and vehicle availability at pick-up locations.

    # Assumptions
    - Packages need to be moved from their current location (ground or in vehicle)
      to a specific goal location on the ground.
    - The cost of a drive action is 1 per road segment.
    - Capacity constraints are ignored.
    - Vehicle availability at pick-up locations is ignored for drive cost calculation.
    - Location names and vehicle names are distinct strings.
    - All goal locations are reachable from all other locations via roads.

    # Heuristic Initialization
    - Builds a graph of locations based on `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Extracts goal locations for each package from the task goals.
    - Identifies all known location names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize `transitions_needed = 0` and `max_drive_distance = 0`.
    2. Parse the current state to quickly find the location of all locatables (`at` facts) and which package is in which vehicle (`in` facts).
    3. For each package `p` that has a goal location `loc_p_goal`:
       a. Check if `(at p loc_p_goal)` is true in the current state. If yes, this package is done, continue to the next package.
       b. Find the current state of package `p` (either on the ground at `loc_p_current` or in a vehicle `v`).
       c. If package `p` is on the ground at `loc_p_current` (where `loc_p_current != loc_p_goal`):
          - Increment `transitions_needed` by 2 (for the required pick-up and drop actions).
          - Calculate the shortest path distance from `loc_p_current` to `loc_p_goal`. Update `max_drive_distance` if this distance is greater.
       d. If package `p` is inside a vehicle `v`:
          - Increment `transitions_needed` by 1 (for the required drop action).
          - Find the current location `loc_v_current` of vehicle `v`.
          - Calculate the shortest path distance from `loc_v_current` to `loc_p_goal`. Update `max_drive_distance` if this distance is greater.
    4. The heuristic value is `transitions_needed + max_drive_distance`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Graph of locations and shortest paths.
        - Goal locations for each package.
        - Set of all location names.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from road facts and collect all location names
        self.graph = {}
        self.locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.graph:
                    self.graph[l1] = []
                self.graph[l1].append(l2)

        # Ensure all locations mentioned in goals are included, even if not in roads (unlikely but safe)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                  _, _, location = get_parts(goal)
                  self.locations.add(location)

        # Compute all-pairs shortest paths using BFS
        self.distances = {loc: {other_loc: float('inf') for other_loc in self.locations} for loc in self.locations}
        for start_node in self.locations:
            self.distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)])

            while queue:
                current_loc, current_dist = queue.popleft()

                # Check if current_loc has outgoing roads in the graph
                if current_loc in self.graph:
                    for neighbor in self.graph[current_loc]:
                        if self.distances[start_node][neighbor] == float('inf'):
                            self.distances[start_node][neighbor] = current_dist + 1
                            queue.append((neighbor, current_dist + 1))

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            if match(goal, "at", "*", "*"):
                 _, package, location = get_parts(goal)
                 self.goal_locations[package] = location
            # Ignore other goal types if any

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Track current location of locatables (packages and vehicles)
        current_locatables_location = {} # {object_name: location_name} for objects on the ground
        packages_in_vehicle = {} # {package_name: vehicle_name} for packages inside vehicles

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                locatable, location = parts[1], parts[2]
                current_locatables_location[locatable] = location
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                packages_in_vehicle[package] = vehicle

        transitions_needed = 0
        max_drive_distance = 0

        # Iterate through packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at its goal location on the ground
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at goal

            # Package is not at goal. Find its current state.
            current_location_info = None # Will store either location string or vehicle string

            if package in current_locatables_location:
                 current_location_info = current_locatables_location[package] # Package is on the ground
            elif package in packages_in_vehicle:
                 current_location_info = packages_in_vehicle[package] # Package is in a vehicle

            # If package location/vehicle not found, something is wrong with the state or goal definition
            if current_location_info is None:
                 # This package exists in the goal but not in the current state facts.
                 # This shouldn't happen in a valid planning state representation.
                 # print(f"Warning: Package {package} not found in 'at' or 'in' predicates in state.")
                 continue # Cannot estimate cost for this package

            # Case 1: Package is on the ground at a non-goal location
            # Check if current_location_info is a location string by seeing if it's in our known locations set
            if current_location_info in self.locations:
                current_location = current_location_info
                # Needs pick-up and drop
                transitions_needed += 2
                # Needs drive from current location to goal location
                if current_location in self.distances and goal_location in self.distances[current_location]:
                     dist = self.distances[current_location][goal_location]
                     if dist != float('inf'):
                        max_drive_distance = max(max_drive_distance, dist)
                     # else: goal is unreachable, heuristic should be large (handled by initial max_drive_distance=0 and finite counts)
                # else: current_location or goal_location not in graph? Should not happen if locations set is built correctly.

            # Case 2: Package is inside a vehicle
            # Check if current_location_info is a vehicle string by seeing if it's NOT in our known locations set
            # This assumes locations and vehicle names are disjoint sets of strings.
            elif current_location_info not in self.locations: # It must be a vehicle name
                vehicle = current_location_info
                # Needs drop
                transitions_needed += 1
                # Find vehicle's current location
                current_vehicle_location = None
                if vehicle in current_locatables_location:
                    current_vehicle_location = current_locatables_location[vehicle]

                if current_vehicle_location is not None:
                    # Needs drive from vehicle's current location to package's goal location
                    if current_vehicle_location in self.distances and goal_location in self.distances[current_vehicle_location]:
                         dist = self.distances[current_vehicle_location][goal_location]
                         if dist != float('inf'):
                            max_drive_distance = max(max_drive_distance, dist)
                         # else: goal is unreachable
                # else: vehicle location not found? Should not happen in valid state.

        # The heuristic is the sum of necessary transitions (pickups, drops) and the longest required drive.
        heuristic_value = transitions_needed + max_drive_distance

        return heuristic_value
