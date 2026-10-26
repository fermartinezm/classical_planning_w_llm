from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest paths
def bfs(adj_list, start_wp, all_waypoints):
    """Computes shortest path distances from start_wp to all reachable waypoints."""
    distances = {wp: float('inf') for wp in all_waypoints}
    distances[start_wp] = 0
    queue = deque([start_wp])
    while queue:
        current_wp = queue.popleft()
        current_dist = distances[current_wp]
        for neighbor in adj_list.get(current_wp, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all uncommunicated goals.
    It sums the minimum estimated cost for each individual unachieved goal.
    The cost for a goal (communicating data) is estimated based on the actions needed:
    1. Obtain the data (sample soil/rock, take image).
    2. Navigate to a communication waypoint.
    3. Communicate the data.
    Navigation costs are estimated using shortest paths on the waypoint graph.

    # Assumptions
    - All goals are reachable in principle (e.g., required equipment exists, samples exist initially, objectives are visible).
    - Navigation is possible between waypoints if a 'visible' edge exists (assuming 'can_traverse' is symmetric or consistent across rovers and matches 'visible'). Shortest path is used for navigation cost.
    - The lander location and communication waypoints (visible from lander) are fixed.
    - Calibration is needed before taking an image and uncalibrates the camera. Calibration requires navigating to a waypoint visible from the calibration target.
    - Dropping a sample is needed if the store is full before sampling.
    - The set of objects (rovers, waypoints, etc.) is inferred from their appearance in initial facts and goals.

    # Heuristic Initialization
    The constructor precomputes static information from the task definition:
    - Identifies objects (rovers, waypoints, etc.) by their appearance in initial facts and goals.
    - The location of the lander.
    - The set of waypoints visible from the lander (communication points).
    - The navigation graph based on 'visible' predicates.
    - All-pairs shortest paths between waypoints using BFS.
    - Which rovers are equipped for soil, rock, and imaging.
    - Which store belongs to which rover.
    - Which waypoints are visible from each objective (for imaging).
    - Information about cameras (calibration target, supported modes, on-board rover).
    - Which waypoints are visible from each calibration target.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize total heuristic cost to 0.
    2. Extract the current locations of rovers, the state of rover stores (full/empty), the calibration status of cameras, and which data/images the rovers currently possess. Also, identify current locations of soil/rock samples.
    3. Iterate through each goal condition defined in the task.
    4. If a goal condition is already satisfied in the current state, its cost is 0, and we continue to the next goal.
    5. If a goal condition is NOT satisfied, estimate the minimum cost to achieve it from the current state:
        - **For `(communicated_soil_data ?w)`:**
            - Find equipped rovers.
            - For each equipped rover `r`:
                - If `r` already has `(have_soil_analysis r w)`: Cost is `dist(current_pos(r), comm_wp)` + 1 (communicate). Minimize over communication waypoints.
                - If `r` does NOT have the sample but `(at_soil_sample w)` is true in the state: Cost is `dist(current_pos(r), w)` + (1 if store is full) + 1 (sample) + `dist(w, comm_wp)` + 1 (communicate). Minimize over communication waypoints.
            - The minimum cost for this goal is the minimum over all equipped rovers. If no path is possible, cost is infinity.
        - **For `(communicated_rock_data ?w)`:** Similar logic as soil data, using rock equipment and rock samples.
        - **For `(communicated_image_data ?o ?m)`:**
            - Find equipped rovers with cameras supporting mode `m`.
            - For each such rover `r`:
                - Find cameras `i` on `r` supporting mode `m`.
                - If `r` already has `(have_image r o m)`: Cost is `dist(current_pos(r), comm_wp)` + 1 (communicate). Minimize over communication waypoints.
                - If `r` does NOT have the image:
                    - Find waypoints `p` visible from objective `o`.
                    - For each suitable waypoint `p`:
                        - Cost to reach `p`: `dist(current_pos(r), p)`.
                        - Cost to calibrate (if needed) and take image at `p` using any suitable camera:
                            - For each suitable camera `i`:
                                - If `(calibrated i r)` is NOT in state: Calibration cost is `dist(p, cal_wp)` + 1 (calibrate) + `dist(cal_wp, p)`. Minimize over calibration waypoints `cal_wp` visible from camera `i`'s target.
                                - If `(calibrated i r)` IS in state: Calibration cost is 0.
                                - Add 1 (take_image action).
                            - The minimum image cost at `p` is the minimum over suitable cameras.
                        - Cost to communicate from `p`: `dist(p, comm_wp)` + 1 (communicate). Minimize over communication waypoints.
                        - Total cost for this path (rover, image_wp): Sum of navigation, image cost, navigation, communicate costs.
                    - The minimum cost to get and communicate the image for this rover is the minimum over all suitable imaging waypoints `p`.
            - The minimum cost for this goal is the minimum over all equipped rovers. If no path is possible, cost is infinity.
    6. Add the minimum estimated cost for the unachieved goal to the total heuristic cost.
    7. If the minimum cost for any goal was infinity, the total heuristic is infinity.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Precompute Static Information ---

        # Infer objects by looking at predicates they appear in initial state and goals
        all_objects = set()
        for fact in task.initial_state | task.goals:
             all_objects.update(get_parts(fact)[1:])
        for fact in task.static:
             all_objects.update(get_parts(fact)[1:])

        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        # A more robust approach would parse the :objects section and type hierarchy.
        # This relies on common predicate structures in this specific domain.
        for fact in task.static | task.initial_state | task.goals:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred in ["at", "equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"] and len(parts) > 1:
                 self.rovers.add(parts[1])
            if pred in ["at", "at_lander", "visible", "can_traverse", "at_soil_sample", "at_rock_sample", "visible_from"] and len(parts) > 2:
                 self.waypoints.add(parts[2])
            if pred in ["empty", "full"] and len(parts) > 1:
                 self.stores.add(parts[1])
            if pred == "store_of" and len(parts) > 1:
                 self.stores.add(parts[1])
            if pred in ["calibrated", "supports", "on_board", "calibration_target"] and len(parts) > 1:
                 self.cameras.add(parts[1])
            if pred == "supports" and len(parts) > 2:
                 self.modes.add(parts[2])
            if pred in ["have_image", "communicated_image_data"] and len(parts) > 3:
                 self.modes.add(parts[3])
            if pred == "at_lander" and len(parts) > 1:
                 self.landers.add(parts[1])
            if pred in ["calibration_target", "visible_from"] and len(parts) > 1:
                 self.objectives.add(parts[1])
            if pred in ["have_image", "communicated_image_data"] and len(parts) > 2:
                 self.objectives.add(parts[2])


        # 1. Navigation Graph
        self.navigation_graph = {}
        for fact in task.static:
            if match(fact, "visible", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                if wp1 in self.waypoints and wp2 in self.waypoints:
                    self.navigation_graph.setdefault(wp1, []).append(wp2)
                    self.navigation_graph.setdefault(wp2, []).append(wp1) # Assuming symmetric visible implies symmetric can_traverse

        for wp in self.navigation_graph:
             self.navigation_graph[wp] = list(set(self.navigation_graph[wp]))

        # 2. Shortest Paths between all waypoints
        self.shortest_paths = {}
        for start_wp in self.waypoints:
            distances = bfs(self.navigation_graph, start_wp, self.waypoints)
            for end_wp, dist in distances.items():
                self.shortest_paths[(start_wp, end_wp)] = dist

        # 3. Lander Location and Communication Waypoints
        self.lander_location = None
        for fact in task.initial_state:
             if match(fact, "at_lander", "*", "*"):
                 self.lander_location = get_parts(fact)[2]
                 break
        self.comm_waypoints = set()
        if self.lander_location:
             for fact in task.static:
                 if match(fact, "visible", "*", self.lander_location):
                     wp = get_parts(fact)[1]
                     if wp in self.waypoints: self.comm_waypoints.add(wp)
                 if match(fact, "visible", self.lander_location, "*"):
                     wp = get_parts(fact)[2]
                     if wp in self.waypoints: self.comm_waypoints.add(wp)

        # 4. Rover Capabilities and Stores
        self.equipped_rovers = {'soil': set(), 'rock': set(), 'imaging': set()}
        for fact in task.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_parts(fact)[1]
                if rover in self.rovers: self.equipped_rovers['soil'].add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_parts(fact)[1]
                if rover in self.rovers: self.equipped_rovers['rock'].add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_parts(fact)[1]
                if rover in self.rovers: self.equipped_rovers['imaging'].add(rover)

        self.rover_stores = {}
        for fact in task.static:
            if match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1:3]
                if store in self.stores and rover in self.rovers:
                    self.rover_stores[store] = rover

        # 5. Imaging Information
        self.objective_visible_from = {}
        for fact in task.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:3]
                if obj in self.objectives and wp in self.waypoints:
                    self.objective_visible_from.setdefault(obj, set()).add(wp)

        self.camera_info = {} # camera -> {'target': obj, 'modes': {mode}, 'rover': rover}
        for cam in self.cameras:
            self.camera_info[cam] = {'target': None, 'modes': set(), 'rover': None}

        for fact in task.static:
            if match(fact, "calibration_target", "*", "*"):
                cam, target = get_parts(fact)[1:3]
                if cam in self.cameras and target in self.objectives:
                    self.camera_info[cam]['target'] = target
            elif match(fact, "supports", "*", "*"):
                cam, mode = get_parts(fact)[1:3]
                if cam in self.cameras and mode in self.modes:
                    self.camera_info[cam]['modes'].add(mode)
            elif match(fact, "on_board", "*", "*"):
                cam, rover = get_parts(fact)[1:3]
                if cam in self.cameras and rover in self.rovers:
                    self.camera_info[cam]['rover'] = rover

        self.calibration_target_visible_from = {}
        for cam_info in self.camera_info.values():
            target = cam_info.get('target')
            if target and target in self.objective_visible_from:
                 self.calibration_target_visible_from[target] = self.objective_visible_from[target]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Extract Current State Information ---
        rover_locations = {}
        store_full = {}
        camera_calibrated = {}
        have_soil = set()
        have_rock = set()
        have_image = set()
        at_soil_sample_state = set()
        at_rock_sample_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at" and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                rover_locations[parts[1]] = parts[2]
            elif pred == "full" and len(parts) == 2 and parts[1] in self.stores:
                 store_full[parts[1]] = True
            elif pred == "calibrated" and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                 camera_calibrated[(parts[1], parts[2])] = True
            elif pred == "have_soil_analysis" and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                 have_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis" and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                 have_rock.add((parts[1], parts[2]))
            elif pred == "have_image" and len(parts) == 4 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                 have_image.add((parts[1], parts[2], parts[3]))
            elif pred == "at_soil_sample" and len(parts) == 2 and parts[1] in self.waypoints:
                 at_soil_sample_state.add(parts[1])
            elif pred == "at_rock_sample" and len(parts) == 2 and parts[1] in self.waypoints:
                 at_rock_sample_state.add(parts[1])


        # --- Calculate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            min_goal_cost = float('inf')

            if pred == "communicated_soil_data" and len(parts) == 2:
                w = parts[1]
                if w not in self.waypoints: continue # Invalid goal waypoint

                # Need to communicate soil data from w
                for r in self.equipped_rovers['soil']:
                    current_pos_r = rover_locations.get(r)
                    if current_pos_r is None: continue # Rover location unknown

                    cost_for_rover = float('inf')

                    if (r, w) in have_soil:
                        # Already have sample, just need to communicate
                        min_comm_nav = min((self.shortest_paths.get((current_pos_r, comm_wp), float('inf')) for comm_wp in self.comm_waypoints), default=float('inf'))
                        if min_comm_nav != float('inf'):
                            cost_for_rover = min_comm_nav + 1 # + communicate action
                    elif w in at_soil_sample_state: # Check if sample is still available in *current* state
                        # Need to sample and communicate
                        store_r = next((s for s, owner in self.rover_stores.items() if owner == r), None)
                        if store_r is None: continue # Rover store unknown

                        nav_to_sample = self.shortest_paths.get((current_pos_r, w), float('inf'))
                        if nav_to_sample == float('inf'): continue # Cannot reach sample location

                        drop_cost = 1 if store_full.get(store_r, False) else 0 # Check if store is full in current state

                        # After sampling at w, need to navigate to comm waypoint
                        min_comm_nav_from_sample = min((self.shortest_paths.get((w, comm_wp), float('inf')) for comm_wp in self.comm_waypoints), default=float('inf'))
                        if min_comm_nav_from_sample == float('inf'): continue # Cannot reach comm location from sample location

                        cost_for_rover = nav_to_sample + drop_cost + 1 + min_comm_nav_from_sample + 1 # + sample + communicate

                    min_goal_cost = min(min_goal_cost, cost_for_rover)

            elif pred == "communicated_rock_data" and len(parts) == 2:
                w = parts[1]
                if w not in self.waypoints: continue # Invalid goal waypoint

                # Need to communicate rock data from w
                for r in self.equipped_rovers['rock']:
                    current_pos_r = rover_locations.get(r)
                    if current_pos_r is None: continue

                    cost_for_rover = float('inf')

                    if (r, w) in have_rock:
                        min_comm_nav = min((self.shortest_paths.get((current_pos_r, comm_wp), float('inf')) for comm_wp in self.comm_waypoints), default=float('inf'))
                        if min_comm_nav != float('inf'):
                            cost_for_rover = min_comm_nav + 1
                    elif w in at_rock_sample_state: # Check if sample is still available in *current* state
                        store_r = next((s for s, owner in self.rover_stores.items() if owner == r), None)
                        if store_r is None: continue

                        nav_to_sample = self.shortest_paths.get((current_pos_r, w), float('inf'))
                        if nav_to_sample == float('inf'): continue

                        drop_cost = 1 if store_full.get(store_r, False) else 0

                        min_comm_nav_from_sample = min((self.shortest_paths.get((w, comm_wp), float('inf')) for comm_wp in self.comm_waypoints), default=float('inf'))
                        if min_comm_nav_from_sample == float('inf'): continue

                        cost_for_rover = nav_to_sample + drop_cost + 1 + min_comm_nav_from_sample + 1

                    min_goal_cost = min(min_goal_cost, cost_for_rover)

            elif pred == "communicated_image_data" and len(parts) == 3:
                o, m = parts[1:3]
                if o not in self.objectives or m not in self.modes: continue # Invalid goal objective/mode

                # Need to communicate image data for o in mode m
                for r in self.equipped_rovers['imaging']:
                    current_pos_r = rover_locations.get(r)
                    if current_pos_r is None: continue

                    # Find cameras on this rover that support this mode and are on board
                    suitable_cameras = [cam for cam, info in self.camera_info.items()
                                        if info.get('rover') == r and m in info.get('modes', set())]

                    if not suitable_cameras: continue # This rover cannot take this image in this mode

                    cost_for_rover = float('inf')

                    if (r, o, m) in have_image:
                        # Already have image, just need to communicate
                        min_comm_nav = min((self.shortest_paths.get((current_pos_r, comm_wp), float('inf')) for comm_wp in self.comm_waypoints), default=float('inf'))
                        if min_comm_nav != float('inf'):
                            cost_for_rover = min_comm_nav + 1 # + communicate action
                    else:
                        # Need to take image and communicate
                        # Find waypoints visible from objective o
                        visible_from_o = self.objective_visible_from.get(o, set())
                        if not visible_from_o: continue # Cannot image this objective

                        min_take_image_comm_cost = float('inf')

                        # Iterate through possible waypoints to take the image from
                        for p_opt in visible_from_o:
                            if p_opt not in self.waypoints: continue # Should not happen if data is consistent

                            nav_to_image_wp = self.shortest_paths.get((current_pos_r, p_opt), float('inf'))
                            if nav_to_image_wp == float('inf'): continue # Cannot reach imaging location

                            # Cost to calibrate and take image at p_opt using any suitable camera
                            min_image_cost_at_p = float('inf')
                            for cam in suitable_cameras:
                                cal_target = self.camera_info[cam].get('target')
                                if cal_target is None: continue # Camera has no calibration target

                                visible_from_cal_target = self.calibration_target_visible_from.get(cal_target, set())
                                if not visible_from_cal_target: continue # Cannot calibrate this camera

                                # Cost to calibrate camera cam for rover r at p_opt
                                # Check if already calibrated in current state
                                if (cam, r) in camera_calibrated:
                                     # Already calibrated, cost is 0 before taking image
                                     calibration_cost = 0
                                else:
                                     # Need to calibrate
                                     # Find the optimal w_opt that minimizes p_opt -> w -> p_opt
                                     best_w_opt = None
                                     min_cal_round_trip = float('inf')
                                     for w in visible_from_cal_target:
                                         if w not in self.waypoints: continue # Should not happen

                                         nav_to_w = self.shortest_paths.get((p_opt, w), float('inf'))
                                         nav_back_to_p = self.shortest_paths.get((w, p_opt), float('inf'))
                                         if nav_to_w != float('inf') and nav_back_to_p != float('inf'):
                                             if nav_to_w + nav_back_to_p < min_cal_round_trip:
                                                 min_cal_round_trip = nav_to_w + nav_back_to_p
                                                 best_w_opt = w

                                     if best_w_opt is None: continue # Cannot perform calibration round trip

                                     calibration_cost = min_cal_round_trip + 1 # + calibrate action

                                # Cost to take image action
                                take_image_action_cost = 1

                                # Total cost to get the image at p_opt using cam
                                cost_to_get_image_at_p = calibration_cost + take_image_action_cost
                                min_image_cost_at_p = min(min_image_cost_at_p, cost_to_get_image_at_p)

                            if min_image_cost_at_p == float('inf'): continue # Cannot get image from this waypoint p_opt with any suitable camera

                            # Cost to communicate from p_opt
                            min_comm_nav_from_image_wp = min((self.shortest_paths.get((p_opt, comm_wp), float('inf')) for comm_wp in self.comm_waypoints), default=float('inf'))
                            if min_comm_nav_from_image_wp == float('inf'): continue # Cannot reach comm location from image location

                            # Total cost for this rover/image_waypoint path
                            cost_path = nav_to_image_wp + min_image_cost_at_p + min_comm_nav_from_image_wp + 1 # + communicate

                            min_take_image_comm_cost = min(min_take_image_comm_cost, cost_path)

                        if min_take_image_comm_cost != float('inf'):
                             cost_for_rover = min_take_image_comm_cost

                    min_goal_cost = min(min_goal_cost, cost_for_rover)

            # Add the minimum cost for this goal to the total
            if min_goal_cost == float('inf'):
                # If any goal is unreachable, the total heuristic is infinity
                return float('inf')
            total_cost += min_goal_cost

        return total_cost
