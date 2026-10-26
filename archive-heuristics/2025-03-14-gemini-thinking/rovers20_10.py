from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It focuses on achieving each goal fact individually and sums up the estimated costs.
    The heuristic considers actions like navigation, sampling (soil and rock), calibration, image taking, and communication.

    # Assumptions:
    - The heuristic assumes that each goal fact needs to be achieved independently and sums up the costs.
    - It simplifies the problem by considering shortest paths for navigation and necessary actions for data acquisition and communication.
    - It does not explicitly handle resource contention or complex action ordering, providing a lower-bound estimate in many cases.

    # Heuristic Initialization
    - Extracts static information from the task, such as:
        - `can_traverse` relations for each rover.
        - `visible` waypoint pairs.
        - `visible_from` objective-waypoint pairs.
        - `calibration_target` camera-objective pairs.
        - `supports` camera-mode relations.
        - `equipped_for_soil_analysis`, `equipped_for_rock_analysis`, `equipped_for_imaging` for each rover.
        - `store_of` relations for each store and rover.
        - `at_lander` location.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact, the heuristic estimates the minimum number of actions as follows:

    1. For `(communicated_image_data ?o ?m)` goals:
        a. Check if `(communicated_image_data ?o ?m)` is already achieved in the current state. If yes, cost is 0.
        b. If not, estimate the cost to achieve it:
            i.  If `(have_image ?r ?o ?m)` is not achieved:
                - Cost for taking image: 1 (take_image action).
                - If camera is not calibrated for the rover: +1 (calibrate action).
                - Find a waypoint `?wp` visible from objective `?o` and reachable by a rover.
                - Estimate navigation cost to `?wp` from the rover's current location (shortest path in terms of navigation actions).
            ii. Cost for communication: 1 (communicate_image_data action).
            iii. Find a waypoint `?wp_comm` visible from the lander's location and reachable by the rover.
            iv. Estimate navigation cost to `?wp_comm` from the rover's current location (shortest path).
            v. Total cost for `(communicated_image_data ?o ?m)` is the sum of costs from steps i and ii (and navigation costs).

    2. For `(communicated_soil_data ?w)` goals:
        a. Check if `(communicated_soil_data ?w)` is already achieved. If yes, cost is 0.
        b. If not, estimate the cost:
            i.  If `(have_soil_analysis ?r ?w)` is not achieved:
                - Cost for soil sampling: 1 (sample_soil action).
                - If rover's store is full: +1 (drop action).
                - Find a rover equipped for soil analysis and a store for it.
                - Estimate navigation cost to waypoint `?w` from the rover's current location.
            ii. Cost for communication: 1 (communicate_soil_data action).
            iii. Find a waypoint `?wp_comm` visible from the lander's location and reachable by the rover.
            iv. Estimate navigation cost to `?wp_comm` from the rover's current location.
            v. Total cost for `(communicated_soil_data ?w)` is the sum of costs from steps i and ii (and navigation costs).

    3. For `(communicated_rock_data ?w)` goals:
        a. Similar to `(communicated_soil_data ?w)` but for rock samples and `sample_rock` action.

    The heuristic sums up the estimated costs for all goal facts.
    Navigation costs are estimated using Breadth-First Search (BFS) on the waypoint graph defined by `can_traverse` and `visible` predicates.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse = collections.defaultdict(list) # rover -> [(from_waypoint, to_waypoint)]
        self.visible = collections.defaultdict(list) # waypoint -> [visible_waypoint]
        self.visible_from_objective = collections.defaultdict(list) # objective -> [visible_waypoint]
        self.calibration_targets = collections.defaultdict(list) # camera -> [objective]
        self.supports_mode = collections.defaultdict(list) # camera -> [mode]
        self.equipped_for_soil_analysis = set() # rovers
        self.equipped_for_rock_analysis = set() # rovers
        self.equipped_for_imaging = set() # rovers
        self.store_of = {} # store -> rover
        self.at_lander_location = None # lander location

        for fact in static_facts:
            if match(fact, 'can_traverse', '?r', '?wp1', '?wp2'):
                parts = get_parts(fact)
                self.can_traverse[parts[1]].append((parts[2], parts[3]))
            elif match(fact, 'visible', '?wp1', '?wp2'):
                parts = get_parts(fact)
                self.visible[parts[1]].append(parts[2])
            elif match(fact, 'visible_from', '?o', '?wp'):
                parts = get_parts(fact)
                self.visible_from_objective[parts[1]].append(parts[2])
            elif match(fact, 'calibration_target', '?cam', '?obj'):
                parts = get_parts(fact)
                self.calibration_targets[parts[1]].append(parts[2])
            elif match(fact, 'supports', '?cam', '?mode'):
                parts = get_parts(fact)
                self.supports_mode[parts[1]].append(parts[2])
            elif match(fact, 'equipped_for_soil_analysis', '?r'):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_rock_analysis', '?r'):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_imaging', '?r'):
                self.equipped_for_imaging.add(get_parts(fact)[1])
            elif match(fact, 'store_of', '?s', '?r'):
                parts = get_parts(fact)
                self.store_of[parts[1]] = parts[2]
            elif match(fact, 'at_lander', '?l', '?wp'):
                self.at_lander_location = get_parts(fact)[2]

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        current_rover_locations = {}
        current_store_empty = {}
        current_camera_calibrated = {}
        current_have_soil_analysis = {}
        current_have_rock_analysis = {}
        current_have_image = {}

        for fact in state:
            if match(fact, 'at', '?r', '?wp'):
                parts = get_parts(fact)
                current_rover_locations[parts[1]] = parts[2]
            elif match(fact, 'empty', '?s'):
                current_store_empty[get_parts(fact)[1]] = True
            elif match(fact, 'calibrated', '?c', '?r'):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                if rover not in current_camera_calibrated:
                    current_camera_calibrated[rover] = []
                current_camera_calibrated[rover].append(camera)
            elif match(fact, 'have_soil_analysis', '?r', '?wp'):
                rover = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if rover not in current_have_soil_analysis:
                    current_have_soil_analysis[rover] = []
                current_have_soil_analysis[rover].append(waypoint)
            elif match(fact, 'have_rock_analysis', '?r', '?wp'):
                rover = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if rover not in current_have_rock_analysis:
                    current_have_rock_analysis[rover] = []
                current_have_rock_analysis[rover].append(waypoint)
            elif match(fact, 'have_image', '?r', '?o', '?m'):
                rover = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                mode = get_parts(fact)[3]
                if rover not in current_have_image:
                    current_have_image[rover] = collections.defaultdict(list)
                current_have_image[rover][objective].append(mode)


        for goal in self.goals:
            if goal in state:
                continue

            parts = get_parts(goal)
            goal_predicate = parts[0]

            if goal_predicate == 'communicated_image_data':
                objective_goal = parts[1]
                mode_goal = parts[2]
                image_communicated = False
                for fact in state:
                    if fact == goal:
                        image_communicated = True
                        break
                if image_communicated:
                    continue

                min_cost_image = float('inf')
                min_cost_comm = float('inf')

                for rover in self.equipped_for_imaging:
                    if rover not in current_rover_locations:
                        continue # Rover not in initial state, ignore it

                    # Cost to take image
                    cost_take_image = 0
                    if rover not in current_camera_calibrated or any(cam not in current_camera_calibrated[rover] for cam in self.calibration_targets if objective_goal in self.calibration_targets[cam]):
                        cost_take_image += 1 # calibrate
                    cost_take_image += 1 # take_image

                    best_nav_cost_image = float('inf')
                    start_waypoint = current_rover_locations[rover]
                    for visible_waypoint in self.visible_from_objective[objective_goal]:
                        nav_cost = self._get_navigation_cost(start_waypoint, visible_waypoint, rover)
                        if nav_cost != float('inf'):
                            best_nav_cost_image = min(best_nav_cost_image, nav_cost)
                    if best_nav_cost_image == float('inf'):
                        continue # No path to a waypoint visible from objective

                    if rover in current_have_image and objective_goal in current_have_image[rover] and mode_goal in current_have_image[rover][objective_goal]:
                        cost_take_image = 0 # Already have image

                    current_image_cost = cost_take_image + (best_nav_cost_image if best_nav_cost_image != float('inf') else 0)
                    min_cost_image = min(min_cost_image, current_image_cost)


                    # Cost to communicate image
                    cost_comm_image = 1 # communicate_image_data
                    best_nav_cost_comm = float('inf')
                    start_waypoint = current_rover_locations[rover]
                    for visible_waypoint in self.visible[start_waypoint]:
                        if visible_waypoint == self.at_lander_location:
                            nav_cost = self._get_navigation_cost(start_waypoint, visible_waypoint, rover)
                            if nav_cost != float('inf'):
                                best_nav_cost_comm = min(best_nav_cost_comm, nav_cost)
                    if best_nav_cost_comm == float('inf'):
                        continue # No path to waypoint visible to lander

                    min_cost_comm = min(min_cost_comm, cost_comm_image + (best_nav_cost_comm if best_nav_cost_comm != float('inf') else 0))


                if min_cost_image != float('inf') and min_cost_comm != float('inf'):
                    heuristic_value += min_cost_image + min_cost_comm
                elif min_cost_image != float('inf'):
                    heuristic_value += min_cost_image
                elif min_cost_comm != float('inf'):
                    heuristic_value += min_cost_comm
                else:
                    heuristic_value += 100 # Large penalty if goal seems unreachable


            elif goal_predicate == 'communicated_soil_data':
                waypoint_goal = parts[1]
                soil_communicated = False
                for fact in state:
                    if fact == goal:
                        soil_communicated = True
                        break
                if soil_communicated:
                    continue

                min_cost_sample = float('inf')
                min_cost_comm = float('inf')

                for rover in self.equipped_for_soil_analysis:
                    if rover not in current_rover_locations:
                        continue

                    # Cost to sample soil
                    cost_sample_soil = 0
                    store = next((s for s, r in self.store_of.items() if r == rover), None)
                    if store and store in current_store_empty and not current_store_empty[store]:
                        cost_sample_soil += 1 # drop
                    cost_sample_soil += 1 # sample_soil

                    best_nav_cost_sample = float('inf')
                    start_waypoint = current_rover_locations[rover]
                    nav_cost = self._get_navigation_cost(start_waypoint, waypoint_goal, rover)
                    if nav_cost != float('inf'):
                        best_nav_cost_sample = min(best_nav_cost_sample, nav_cost)
                    else:
                        continue # No path to sample waypoint

                    if rover in current_have_soil_analysis and waypoint_goal in current_have_soil_analysis[rover]:
                        cost_sample_soil = 0 # Already have soil analysis

                    current_soil_cost = cost_sample_soil + (best_nav_cost_sample if best_nav_cost_sample != float('inf') else 0)
                    min_cost_sample = min(min_cost_sample, current_soil_cost)


                    # Cost to communicate soil data
                    cost_comm_soil = 1 # communicate_soil_data
                    best_nav_cost_comm = float('inf')
                    start_waypoint = current_rover_locations[rover]
                    for visible_waypoint in self.visible[start_waypoint]:
                        if visible_waypoint == self.at_lander_location:
                            nav_cost = self._get_navigation_cost(start_waypoint, visible_waypoint, rover)
                            if nav_cost != float('inf'):
                                best_nav_cost_comm = min(best_nav_cost_comm, nav_cost)
                    if best_nav_cost_comm == float('inf'):
                        continue # No path to waypoint visible to lander

                    min_cost_comm = min(min_cost_comm, cost_comm_soil + (best_nav_cost_comm if best_nav_cost_comm != float('inf') else 0))


                if min_cost_sample != float('inf') and min_cost_comm != float('inf'):
                    heuristic_value += min_cost_sample + min_cost_comm
                elif min_cost_sample != float('inf'):
                    heuristic_value += min_cost_sample
                elif min_cost_comm != float('inf'):
                    heuristic_value += min_cost_comm
                else:
                    heuristic_value += 100 # Large penalty if goal seems unreachable


            elif goal_predicate == 'communicated_rock_data': # Similar to soil data
                waypoint_goal = parts[1]
                rock_communicated = False
                for fact in state:
                    if fact == goal:
                        rock_communicated = True
                        break
                if rock_communicated:
                    continue

                min_cost_sample = float('inf')
                min_cost_comm = float('inf')

                for rover in self.equipped_for_rock_analysis:
                    if rover not in current_rover_locations:
                        continue

                    # Cost to sample rock
                    cost_sample_rock = 0
                    store = next((s for s, r in self.store_of.items() if r == rover), None)
                    if store and store in current_store_empty and not current_store_empty[store]:
                        cost_sample_rock += 1 # drop
                    cost_sample_rock += 1 # sample_rock

                    best_nav_cost_sample = float('inf')
                    start_waypoint = current_rover_locations[rover]
                    nav_cost = self._get_navigation_cost(start_waypoint, waypoint_goal, rover)
                    if nav_cost != float('inf'):
                        best_nav_cost_sample = min(best_nav_cost_sample, nav_cost)
                    else:
                        continue # No path to sample waypoint

                    if rover in current_have_rock_analysis and waypoint_goal in current_have_rock_analysis[rover]:
                        cost_sample_rock = 0 # Already have rock analysis

                    current_rock_cost = cost_sample_rock + (best_nav_cost_sample if best_nav_cost_sample != float('inf') else 0)
                    min_cost_sample = min(min_cost_sample, current_rock_cost)


                    # Cost to communicate rock data
                    cost_comm_rock = 1 # communicate_rock_data
                    best_nav_cost_comm = float('inf')
                    start_waypoint = current_rover_locations[rover]
                    for visible_waypoint in self.visible[start_waypoint]:
                        if visible_waypoint == self.at_lander_location:
                            nav_cost = self._get_navigation_cost(start_waypoint, visible_waypoint, rover)
                            if nav_cost != float('inf'):
                                best_nav_cost_comm = min(best_nav_cost_comm, nav_cost)
                    if best_nav_cost_comm == float('inf'):
                        continue # No path to waypoint visible to lander

                    min_cost_comm = min(min_cost_comm, cost_comm_rock + (best_nav_cost_comm if best_nav_cost_comm != float('inf') else 0))


                if min_cost_sample != float('inf') and min_cost_comm != float('inf'):
                    heuristic_value += min_cost_sample + min_cost_comm
                elif min_cost_sample != float('inf'):
                    heuristic_value += min_cost_sample
                elif min_cost_comm != float('inf'):
                    heuristic_value += min_cost_comm
                else:
                    heuristic_value += 100 # Large penalty if goal seems unreachable


        return heuristic_value if heuristic_value < 100 else 99 # Cap heuristic value to avoid overestimation


    def _get_navigation_cost(self, start_waypoint, goal_waypoint, rover):
        """Calculates the shortest path (number of navigate actions) between two waypoints for a given rover using BFS."""
        if start_waypoint == goal_waypoint:
            return 0

        queue = collections.deque([(start_waypoint, 0)]) # (waypoint, cost)
        visited = {start_waypoint}

        while queue:
            current_waypoint, cost = queue.popleft()

            for wp1, wp2 in self.can_traverse[rover]:
                if wp1 == current_waypoint:
                    neighbor = wp2
                elif wp2 == current_waypoint:
                    neighbor = wp1
                else:
                    continue # Not a neighbor

                if neighbor not in visited and neighbor in self.visible[current_waypoint]: # Check visibility
                    if neighbor == goal_waypoint:
                        return cost + 1
                    visited.add(neighbor)
                    queue.append((neighbor, cost + 1))
        return float('inf') # No path found
