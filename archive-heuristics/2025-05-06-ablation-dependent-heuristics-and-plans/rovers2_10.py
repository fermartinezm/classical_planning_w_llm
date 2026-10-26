from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency list.

    Args:
        graph: Adjacency list {node: {neighbor1, neighbor2, ...}}
        start_node: The starting node for BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary {node: distance} containing shortest distances from start_node.
        Distance is float('inf') if a node is unreachable.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         return distances # Start node is not a valid waypoint

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_wp = queue.popleft()

        # Check if current_wp exists as a key in the graph dictionary
        if current_wp in graph:
            for neighbor in graph[current_wp]:
                # Ensure neighbor is a valid waypoint
                if neighbor in all_nodes and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_wp] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each uncommunicated goal (soil data, rock data, image data) independently,
    and sums these minimum costs. It considers navigation, sampling/imaging,
    dropping samples, calibrating cameras, and communication actions.

    # Assumptions
    - All goal conditions are of the form (communicated_...).
    - Soil/rock samples are only available at initial locations and are consumed.
      If an initial sample is gone and no rover holds the data, the goal is impossible.
    - Calibration targets and objective visibility are static.
    - Navigation costs 1 action per step between adjacent waypoints a rover can traverse.
    - Sampling, dropping, calibrating, taking image, and communicating each cost 1 action.
    - The heuristic calculates the minimum cost for each unachieved goal independently,
      assuming the best available rover/camera and path is used for that specific goal.
      It does not account for resource contention or synergies between goals.

    # Heuristic Initialization
    - Parses static facts to build rover-specific navigation graphs,
      precompute shortest path distances between all pairs of waypoints for each rover,
      identify communication waypoints (visible from lander), and precompute minimum
      distances from any waypoint to any communication waypoint for each rover.
    - Extracts rover capabilities, camera information, store ownership, lander location,
      objective visibility, and initial sample locations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost `h = 0`.
    2.  Parse the current state to get the location of each rover, store status,
        held data (soil, rock, image), current sample locations, calibrated cameras,
        and communicated data.
    3.  For each goal condition in `task.goals`:
        a.  If the goal is already achieved (e.g., `(communicated_soil_data w)` is in the state), continue to the next goal.
        b.  If the goal is `(communicated_soil_data w)`:
            -   Calculate the minimum cost to achieve this goal over all soil-equipped rovers `R`.
            -   Cost for rover R:
                -   If R does not have `(have_soil_analysis R w)`:
                    -   If `(at_soil_sample w)` was not in the initial state, or is no longer in the current state and no rover holds the data for `w`, this goal is impossible via sampling (skip this rover).
                    -   Otherwise, cost includes: `dist(current_loc(R), w)` (navigate to sample) + (1 if R's store is full) + 1 (sample). The rover is now at `w`.
                -   If R has `(have_soil_analysis R w)`, cost is 0 for sampling/prep. The rover is at its current location.
                -   Cost includes: `min_dist_to_comm[R][current_loc_after_prep]` (navigate to comm point) + 1 (communicate).
            -   Find the minimum cost over all suitable rovers. If no rover can achieve it, the goal is impossible (return infinity).
            -   Add this minimum cost to `h`.
        c.  If the goal is `(communicated_rock_data w)`: Follow similar logic as for soil data, using rock-equipped rovers, rock samples, and rock analysis data.
        d.  If the goal is `(communicated_image_data o m)`:
            -   Calculate the minimum cost to achieve this goal over all imaging-equipped rovers `R` with a camera `C` supporting mode `m`.
            -   Cost for rover R and camera C:
                -   If R does not have `(have_image R o m)`:
                    -   Find waypoints `img_wps` visible from objective `o`. If none, impossible (skip rover/camera).
                    -   Find waypoints `cal_wps` visible from calibration target of C. If none, impossible (skip rover/camera).
                    -   If `(C, R)` is not calibrated:
                        -   Cost includes: `min_dist(current_loc(R), cal_wp)` over `cal_wps` (navigate to cal) + 1 (calibrate). Rover is now at `best_cal_wp`.
                        -   Cost includes: `min_dist(best_cal_wp, img_wp)` over `img_wps` (navigate to image). Rover is now at `best_img_wp`.
                    -   If `(C, R)` is calibrated:
                        -   Cost includes: `min_dist(current_loc(R), img_wp)` over `img_wps` (navigate to image). Rover is now at `best_img_wp`.
                    -   Cost includes: 1 (take_image). Rover is still at `best_img_wp`.
                -   If R has `(have_image R o m)`, cost is 0 for imaging/prep. The rover is at its current location.
                -   Cost includes: `min_dist_to_comm[R][current_loc_after_prep]` (navigate to comm point) + 1 (communicate).
            -   Find the minimum cost over all suitable rover/camera pairs. If no pair can achieve it, the goal is impossible (return infinity).
            -   Add this minimum cost to `h`.
    4.  Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for sample locations

        # --- Precompute Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.modes = set()
        self.cameras = set()
        self.stores = set()
        self.landers = set()

        # Collect all objects by type from initial state and static facts
        # A more robust parser would get objects from the problem definition itself.
        # For this heuristic, we can infer types from predicates found in the examples.
        all_facts = set(initial_state) | set(static_facts) | set(self.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'at_lander' and len(parts) == 3:
                 self.landers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'can_traverse' and len(parts) == 4:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.waypoints.add(parts[3])
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(parts) == 2:
                 self.rovers.add(parts[1])
            elif pred in ['empty', 'full'] and len(parts) == 2:
                 self.stores.add(parts[1])
            elif pred in ['have_rock_analysis', 'have_soil_analysis'] and len(parts) == 3:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred in ['calibrated'] and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
            elif pred in ['supports'] and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.modes.add(parts[2])
            elif pred in ['visible'] and len(parts) == 3:
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred in ['have_image'] and len(parts) == 4:
                 self.rovers.add(parts[1])
                 self.objectives.add(parts[2])
                 self.modes.add(parts[3])
            elif pred in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2:
                 self.waypoints.add(parts[1])
            elif pred in ['communicated_image_data'] and len(parts) == 3:
                 self.objectives.add(parts[1])
                 self.modes.add(parts[2])
            elif pred in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2:
                 self.waypoints.add(parts[1])
            elif pred in ['visible_from'] and len(parts) == 3:
                 self.objectives.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred in ['store_of'] and len(parts) == 3:
                 self.stores.add(parts[1])
                 self.rovers.add(parts[2])
            elif pred in ['calibration_target'] and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.objectives.add(parts[2])
            elif pred in ['on_board'] and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])


        self.rover_capabilities = {r: {'soil': False, 'rock': False, 'imaging': False} for r in self.rovers}
        self.rover_stores = {} # {rover: store}
        self.camera_info = {c: {'on_board': None, 'supports': set(), 'calibration_target': None} for c in self.cameras}
        self.lander_location = None
        self.comm_wps = set() # Waypoints visible from lander
        self.rover_graphs = {r: {} for r in self.rovers} # {rover: {wp: {neighbor_wps}}}
        self.objective_visibility = {} # {objective: {visible_wps}}
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['soil'] = True
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['rock'] = True
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities[get_parts(fact)[1]]['imaging'] = True
            elif match(fact, "store_of", "*", "*"):
                s, r = get_parts(fact)[1:]
                self.rover_stores[r] = s
            elif match(fact, "on_board", "*", "*"):
                c, r = get_parts(fact)[1:]
                if c in self.camera_info: self.camera_info[c]['on_board'] = r
            elif match(fact, "supports", "*", "*"):
                c, m = get_parts(fact)[1:]
                if c in self.camera_info: self.camera_info[c]['supports'].add(m)
            elif match(fact, "calibration_target", "*", "*"):
                c, o = get_parts(fact)[1:]
                if c in self.camera_info: self.camera_info[c]['calibration_target'] = o
            elif match(fact, "at_lander", "*", "*"):
                l, wp = get_parts(fact)[1:]
                self.lander_location = wp
            elif match(fact, "visible", "*", self.lander_location):
                 # Waypoints visible from lander location are communication points
                 wp1, wp2 = get_parts(fact)[1:]
                 if wp2 == self.lander_location:
                     self.comm_wps.add(wp1)
            elif match(fact, "can_traverse", "*", "*", "*"):
                r, w1, w2 = get_parts(fact)[1:]
                if r in self.rover_graphs:
                    self.rover_graphs[r].setdefault(w1, set()).add(w2)
            elif match(fact, "visible_from", "*", "*"):
                o, wp = get_parts(fact)[1:]
                self.objective_visibility.setdefault(o, set()).add(wp)

        # Ensure all waypoints mentioned in can_traverse are in the graph keys
        for r in self.rovers:
             for w1 in self.waypoints:
                 self.rover_graphs[r].setdefault(w1, set())


        # Precompute all-pairs shortest paths for each rover
        self.rover_distances = {} # {rover: {start_wp: {end_wp: distance}}}
        for r in self.rovers:
            self.rover_distances[r] = {}
            for start_wp in self.waypoints:
                self.rover_distances[r][start_wp] = bfs(self.rover_graphs[r], start_wp, self.waypoints)

        # Precompute minimum distance from any waypoint to any communication waypoint for each rover
        self.min_dist_to_comm = {} # {rover: {start_wp: min_distance}}
        for r in self.rovers:
            self.min_dist_to_comm[r] = {}
            for start_wp in self.waypoints:
                min_dist = float('inf')
                # Ensure comm_wps is not empty before iterating
                if self.comm_wps:
                    for comm_wp in self.comm_wps:
                        dist = self.rover_distances[r][start_wp].get(comm_wp, float('inf'))
                        min_dist = min(min_dist, dist)
                self.min_dist_to_comm[r][start_wp] = min_dist


        # Initial sample locations
        for fact in initial_state:
            if match(fact, "at_soil_sample", "*"):
                self.initial_soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.initial_rock_samples.add(get_parts(fact)[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        rover_locations = {} # {rover: wp}
        store_status = {} # {store: 'empty' or 'full'}
        rover_data = {r: {'soil': set(), 'rock': set(), 'image': {}} for r in self.rovers} # {rover: {'soil': {wps}, 'rock': {wps}, 'image': {objective: {modes}}}}
        waypoint_samples = {'soil': set(), 'rock': set()} # {type: {wps}}
        rover_calibrated_cameras = {r: set() for r in self.rovers} # {rover: {cameras}}
        communicated_data = {'soil': set(), 'rock': set(), 'image': set()} # {type: {wps or (o, m)}}

        for fact in state:
            if match(fact, "at", "*", "*"):
                r, wp = get_parts(fact)[1:]
                rover_locations[r] = wp
            elif match(fact, "empty", "*"):
                s = get_parts(fact)[1]
                store_status[s] = 'empty'
            elif match(fact, "full", "*"):
                s = get_parts(fact)[1]
                store_status[s] = 'full'
            elif match(fact, "have_soil_analysis", "*", "*"):
                r, wp = get_parts(fact)[1:]
                rover_data[r]['soil'].add(wp)
            elif match(fact, "have_rock_analysis", "*", "*"):
                r, wp = get_parts(fact)[1:]
                rover_data[r]['rock'].add(wp)
            elif match(fact, "have_image", "*", "*", "*"):
                r, o, m = get_parts(fact)[1:]
                rover_data[r]['image'].setdefault(o, set()).add(m)
            elif match(fact, "at_soil_sample", "*"):
                waypoint_samples['soil'].add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                waypoint_samples['rock'].add(get_parts(fact)[1])
            elif match(fact, "calibrated", "*", "*"):
                c, r = get_parts(fact)[1:]
                rover_calibrated_cameras[r].add(c)
            elif match(fact, "communicated_soil_data", "*"):
                communicated_data['soil'].add(get_parts(fact)[1])
            elif match(fact, "communicated_rock_data", "*"):
                communicated_data['rock'].add(get_parts(fact)[1])
            elif match(fact, "communicated_image_data", "*", "*"):
                o, m = get_parts(fact)[1:]
                communicated_data['image'].add((o, m))

        total_h = 0

        # --- Estimate cost for each unachieved goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            if pred == "communicated_soil_data":
                w = parts[1]
                if w in communicated_data['soil']:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                for r in self.rovers:
                    if not self.rover_capabilities[r]['soil']:
                        continue # Rover cannot perform soil analysis

                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue # Rover location unknown (shouldn't happen in valid states)

                    data_held = (w in rover_data.get(r, {}).get('soil', set()))

                    cost_r = 0
                    loc_after_prep = current_loc # Default location after prep

                    if not data_held:
                        # Need to sample
                        if w not in self.initial_soil_samples:
                             # Sample was not initially available, cannot sample now unless data is held by someone else (checked by data_held)
                             continue # Cannot sample this waypoint

                        # Check if sample is still there or if any rover holds the data
                        can_sample_now = w in waypoint_samples['soil']
                        any_rover_holds_data = any(w in rover_data.get(other_r, {}).get('soil', set()) for other_r in self.rovers)

                        if not can_sample_now and not any_rover_holds_data:
                             # Sample is gone and no one has the data, goal is impossible via sampling
                             continue # Cannot sample this waypoint

                        # If we reach here, either the sample is still there, or some rover holds the data.
                        # If data_held is False, we *must* sample.
                        if not data_held:
                            nav_cost_to_sample = self.rover_distances[r][current_loc].get(w, float('inf'))
                            if nav_cost_to_sample == float('inf'):
                                continue # Cannot reach sample location

                            cost_r += nav_cost_to_sample # Navigate to sample
                            loc_after_prep = w # Rover is at sample location after navigating
                            store = self.rover_stores.get(r)
                            if store and store_status.get(store) == 'full':
                                cost_r += 1 # Drop
                            cost_r += 1 # Sample
                        # else: data_held is True, no sample/drop needed, loc_after_prep remains current_loc


                    # Need to communicate
                    nav_cost_to_comm = self.min_dist_to_comm[r].get(loc_after_prep, float('inf'))
                    if nav_cost_to_comm == float('inf'):
                         continue # Cannot reach a communication point

                    cost_r += nav_cost_to_comm # Navigate to comm point
                    cost_r += 1 # Communicate

                    min_goal_cost = min(min_goal_cost, cost_r)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal is impossible
                total_h += min_goal_cost

            elif pred == "communicated_rock_data":
                w = parts[1]
                if w in communicated_data['rock']:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                for r in self.rovers:
                    if not self.rover_capabilities[r]['rock']:
                        continue # Rover cannot perform rock analysis

                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue

                    data_held = (w in rover_data.get(r, {}).get('rock', set()))

                    cost_r = 0
                    loc_after_prep = current_loc

                    if not data_held:
                        # Need to sample
                        if w not in self.initial_rock_samples:
                             continue # Cannot sample this waypoint

                        can_sample_now = w in waypoint_samples['rock']
                        any_rover_holds_data = any(w in rover_data.get(other_r, {}).get('rock', set()) for other_r in self.rovers)

                        if not can_sample_now and not any_rover_holds_data:
                             continue # Cannot sample this waypoint

                        if not data_held:
                            nav_cost_to_sample = self.rover_distances[r][current_loc].get(w, float('inf'))
                            if nav_cost_to_sample == float('inf'):
                                continue # Cannot reach sample location

                            cost_r += nav_cost_to_sample # Navigate to sample
                            loc_after_prep = w # Rover is at sample location after navigating
                            store = self.rover_stores.get(r)
                            if store and store_status.get(store) == 'full':
                                cost_r += 1 # Drop
                            cost_r += 1 # Sample
                        # else: data_held is True, no sample/drop needed, loc_after_prep remains current_loc

                    # Need to communicate
                    nav_cost_to_comm = self.min_dist_to_comm[r].get(loc_after_prep, float('inf'))
                    if nav_cost_to_comm == float('inf'):
                         continue # Cannot reach a communication point

                    cost_r += nav_cost_to_comm # Navigate to comm point
                    cost_r += 1 # Communicate

                    min_goal_cost = min(min_goal_cost, cost_r)

                if min_goal_cost == float('inf'):
                    return float('inf') # Goal is impossible
                total_h += min_goal_cost

            elif pred == "communicated_image_data":
                o, m = parts[1:]
                if (o, m) in communicated_data['image']:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                for r in self.rovers:
                    if not self.rover_capabilities[r]['imaging']:
                        continue # Rover cannot perform imaging

                    current_loc = rover_locations.get(r)
                    if current_loc is None: continue

                    data_held = (m in rover_data.get(r, {}).get('image', {}).get(o, set()))

                    if data_held:
                        # Option 1: Use held data
                        loc_after_prep = current_loc
                        nav_cost_to_comm = self.min_dist_to_comm[r].get(loc_after_prep, float('inf'))
                        if nav_cost_to_comm != float('inf'):
                            cost_use_held = nav_cost_to_comm + 1 # NavComm + Communicate
                            min_goal_cost = min(min_goal_cost, cost_use_held)
                    else:
                        # Option 2: Take a new image
                        min_cost_take_new_image_r = float('inf')

                        for c in self.cameras:
                            if self.camera_info.get(c, {}).get('on_board') != r: continue
                            if m not in self.camera_info.get(c, {}).get('supports', set()): continue

                            cal_target_c = self.camera_info.get(c, {}).get('calibration_target')
                            if not cal_target_c: continue
                            cal_wps_c = self.objective_visibility.get(cal_target_c, set())
                            if not cal_wps_c: continue

                            img_wps = self.objective_visibility.get(o, set())
                            if not img_wps: continue

                            is_calibrated = c in rover_calibrated_cameras.get(r, set())

                            # Find best path: current_loc -> (cal_wp -> calibrate)? -> img_wp -> take_image -> comm_wp -> communicate
                            min_path_cost_rc = float('inf')

                            for cal_wp in cal_wps_c:
                                nav_cost_to_cal = self.rover_distances[r][current_loc].get(cal_wp, float('inf'))
                                if nav_cost_to_cal == float('inf'): continue

                                for img_wp in img_wps:
                                    nav_cost_cal_to_img = self.rover_distances[r][cal_wp].get(img_wp, float('inf'))
                                    if nav_cost_cal_to_img == float('inf'): continue

                                    nav_cost_img_to_comm = self.min_dist_to_comm[r].get(img_wp, float('inf'))
                                    if nav_cost_img_to_comm == float('inf'): continue

                                    # Cost if calibration is needed
                                    cost_if_cal_needed = nav_cost_to_cal + 1 + nav_cost_cal_to_img + 1 + nav_cost_img_to_comm + 1 # NavCal + Cal + NavImg + TakeImg + NavComm + Comm

                                    # Cost if already calibrated
                                    nav_cost_to_img_direct = self.rover_distances[r][current_loc].get(img_wp, float('inf'))
                                    if nav_cost_to_img_direct == float('inf'): continue
                                    nav_cost_img_to_comm_direct = self.min_dist_to_comm[r].get(img_wp, float('inf'))
                                    if nav_cost_img_to_comm_direct == float('inf'): continue
                                    cost_if_calibrated = nav_cost_to_img_direct + 1 + nav_cost_img_to_comm_direct + 1 # NavImg + TakeImg + NavComm + Comm

                                    path_cost = cost_if_calibrated if is_calibrated else cost_if_cal_needed

                                    min_path_cost_rc = min(min_path_cost_rc, path_cost)

                            min_cost_take_new_image_r = min(min_cost_take_new_image_r, min_path_cost_rc)

                        if min_cost_take_new_image_r != float('inf'):
                             min_goal_cost = min(min_goal_cost, min_cost_take_new_image_r)


                if min_goal_cost == float('inf'):
                    return float('inf') # Goal is impossible
                total_h += min_goal_cost


        return total_h

