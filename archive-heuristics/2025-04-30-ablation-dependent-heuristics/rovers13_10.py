import collections
import re

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple ('predicate', 'arg1', 'arg2').
    """
    # Remove leading/trailing parentheses and split by spaces
    parts = fact_str.strip()[1:-1].split()
    return tuple(parts)

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing the estimated costs
    for each unsatisfied goal fact. For each goal fact (e.g., communicated_soil_data),
    it calculates the minimum cost for any suitable rover to achieve that specific goal,
    considering necessary steps like navigation, sampling/imaging, calibration, and communication.
    It uses pre-calculated shortest path distances between waypoints.

    Assumptions:
    - The heuristic calculates the cost for each goal fact independently, ignoring potential
      positive or negative interactions between achieving different goals (e.g., one rover
      sampling might consume the sample needed by another, taking an image uncalibrates
      the camera which might affect other image goals for the same camera).
    - Store capacity is only considered simply: if a rover's store is full and it needs
      to sample, a cost of 1 is added for the 'drop' action. It doesn't model managing
      multiple samples or dropping samples that are not needed for a goal.
    - The heuristic assumes necessary static conditions (like 'at_soil_sample' being true
      at a waypoint if soil data from that waypoint is a goal, or 'visible_from' holding
      for imaging/calibration) are met if a path exists.
    - 'can_traverse' is assumed symmetric for navigation graph calculation.
    - Unreachable goal components (e.g., target waypoint unreachable) add a large penalty
      (UNREACHABLE_PENALTY) to the heuristic value.

    Heuristic Initialization:
    1. Parses static facts from the task to identify objects (rovers, waypoints, cameras, etc.)
       and static relationships (at_lander, equipped, store_of, visible, can_traverse,
       on_board, supports, calibration_target, visible_from).
    2. Builds a navigation graph (adjacency list) for each rover based on 'can_traverse' facts.
       Assumes 'can_traverse' is symmetric.
    3. Performs Breadth-First Search (BFS) from every waypoint for each rover to pre-calculate
       all-pairs shortest path distances. Stores these distances.
    4. Identifies communication waypoints (visible from any lander location).
    5. Identifies image waypoints for each objective (visible from the objective).
    6. Identifies calibration waypoints for each calibration target (visible from the target).

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Initialize the total heuristic value `h` to 0.
    3. Parse the dynamic facts from the current state (rover locations, samples held, images held,
       store status, camera calibration status, remaining samples at waypoints).
    4. Iterate through each goal fact specified in the task:
       a. If the goal fact is already true in the current state, add 0 to `h`.
       b. If the goal fact is `(communicated_soil_data ?w)` and is not true:
          i. Determine if sampling is needed (i.e., if no rover currently holds `(have_soil_analysis ?r ?w)`).
          ii. Calculate the minimum cost over all rovers equipped for soil analysis:
              - Find the rover's current location `r_at`.
              - If sampling is needed:
                  - Calculate navigation cost from `r_at` to `?w`. If unreachable, this rover cannot achieve the goal this way; consider the next rover.
                  - Add navigation cost + 1 (sample action).
                  - If the rover's store is full, add 1 (drop action).
                  - The rover is now effectively at `?w`.
              - If sampling is not needed (some rover already has the sample):
                  - The rover is at `r_at` with the sample.
              - Calculate navigation cost from the rover's current effective location (either `?w` after sampling or `r_at` if already held) to the closest communication waypoint visible from the lander. If unreachable, this rover cannot achieve the goal this way; consider the next rover.
              - Add navigation cost + 1 (communicate action).
              - This total is the cost for this specific rover to achieve this goal.
          iii. Take the minimum cost over all suitable rovers.
          iv. If no suitable rover can achieve the goal (minimum cost is infinity), add `UNREACHABLE_PENALTY` to `h`. Otherwise, add the minimum cost to `h`.
   c. If the goal fact is `(communicated_rock_data ?w)` and is not true: Follow similar logic as for soil data.
   d. If the goal fact is `(communicated_image_data ?o ?m)` and is not true:
      i. Determine if taking the image is needed (i.e., if no rover currently holds `(have_image ?r ?o ?m)`).
      ii. Calculate the minimum cost over all rovers equipped for imaging that have a camera supporting mode `?m`:
          - Find the rover's current location `r_at`.
          - If taking the image is needed:
              - Find the set of waypoints `img_wps` visible from objective `?o`. If empty, this rover cannot achieve the goal this way; consider the next rover.
              - Find the camera `?i` on the rover supporting `?m`. Find its calibration target `?t` and the set of waypoints `cal_wps` visible from `?t`.
              - Calculate the cost for two options:
                  - Option 1 (Calibrate -> Image -> Communicate): Calculate navigation cost from `r_at` to the best `cal_wp`, plus 1 (calibrate), plus navigation cost from that `cal_wp` to the best `img_wp`, plus 1 (take_image), plus navigation cost from that `img_wp` to the best `comm_wp`, plus 1 (communicate). If any navigation segment is impossible or required waypoints are missing, this option's cost is infinity.
                  - Option 2 (Image -> Communicate): Only available if camera `?i` is currently calibrated. Calculate navigation cost from `r_at` to the best `img_wp`, plus 1 (take_image), plus navigation cost from that `img_wp` to the best `comm_wp`, plus 1 (communicate). If any navigation segment is impossible or required waypoints are missing, this option's cost is infinity.
              - The cost to get the image and communicate for this rover is the minimum of Option 1 and Option 2 (or just Option 1 if not calibrated).
          - If taking the image is not needed (some rover already has the image):
              - Calculate navigation cost from `r_at` to the closest communication waypoint, plus 1 (communicate).
          - This total is the cost for this specific rover/camera to achieve this goal.
      iii. Take the minimum cost over all suitable rovers/cameras.
      iv. If no suitable rover/camera can achieve the goal (minimum cost is infinity), add `UNREACHABLE_PENALTY` to `h`. Otherwise, add the minimum cost to `h`.
5. Return the total heuristic value `h`.
"""
    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating static information.
        """
        self.goals = task.goals
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.modes = set()
        self.cameras = set()
        self.stores = set()
        self.landers = set()

        self.lander_at = set()
        self.rover_equipment = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.visible_wps = collections.defaultdict(set) # wp -> {visible_wp, ...} (undirected)
        self.rover_adj = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> wp -> {neighbor_wp, ...}
        self.camera_info = collections.defaultdict(dict) # camera -> {rover: r, modes: {m, ...}, target: obj}
        self.visible_from_obj = collections.defaultdict(set) # obj -> {wp, ...}
        self.cal_targets = set() # objectives that are calibration targets

        # Populate all object sets from task.facts
        for fact_str in task.facts:
             fact = parse_fact(fact_str)
             pred = fact[0]
             if pred in ['at', 'can_traverse']:
                 if len(fact) > 1: self.rovers.add(fact[1])
                 if len(fact) > 2: self.waypoints.update(fact[2:])
             elif pred == 'at_lander':
                 if len(fact) > 1: self.landers.add(fact[1])
                 if len(fact) > 2: self.waypoints.add(fact[2])
             elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                 if len(fact) > 1: self.rovers.add(fact[1])
             elif pred in ['empty', 'full']:
                 if len(fact) > 1: self.stores.add(fact[1])
             elif pred in ['have_rock_analysis', 'have_soil_analysis']:
                 if len(fact) > 1: self.rovers.add(fact[1])
                 if len(fact) > 2: self.waypoints.add(fact[2])
             elif pred in ['calibrated', 'on_board']:
                 if len(fact) > 1: self.cameras.add(fact[1])
                 if len(fact) > 2: self.rovers.add(fact[2])
             elif pred == 'supports':
                 if len(fact) > 1: self.cameras.add(fact[1])
                 if len(fact) > 2: self.modes.add(fact[2])
             elif pred == 'visible':
                 if len(fact) > 1: self.waypoints.update(fact[1:])
             elif pred == 'have_image':
                 if len(fact) > 1: self.rovers.add(fact[1])
                 if len(fact) > 2: self.objectives.add(fact[2])
                 if len(fact) > 3: self.modes.add(fact[3])
             elif pred in ['communicated_soil_data', 'communicated_rock_data', 'at_soil_sample', 'at_rock_sample']:
                 if len(fact) > 1: self.waypoints.add(fact[1])
             elif pred == 'communicated_image_data':
                 if len(fact) > 1: self.objectives.add(fact[1])
                 if len(fact) > 2: self.modes.add(fact[2])
             elif pred == 'visible_from':
                 if len(fact) > 1: self.objectives.add(fact[1])
                 if len(fact) > 2: self.waypoints.add(fact[2])
             elif pred == 'store_of':
                 if len(fact) > 1: self.stores.add(fact[1])
                 if len(fact) > 2: self.rovers.add(fact[2])
             elif pred == 'calibration_target':
                 if len(fact) > 1: self.cameras.add(fact[1])
                 if len(fact) > 2: self.objectives.add(fact[2])


        # Parse static facts again to build specific structures
        for fact_str in task.static:
            fact = parse_fact(fact_str)
            pred = fact[0]
            if pred == 'at_lander':
                lander, wp = fact[1:]
                self.lander_at.add(wp)
            elif pred == 'equipped_for_soil_analysis':
                rover = fact[1]
                self.rover_equipment[rover].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = fact[1]
                self.rover_equipment[rover].add('rock')
            elif pred == 'equipped_for_imaging':
                rover = fact[1]
                self.rover_equipment[rover].add('imaging')
            elif pred == 'store_of':
                store, rover = fact[1:]
                self.rover_stores[rover] = store
            elif pred == 'visible':
                w1, w2 = fact[1:]
                self.visible_wps[w1].add(w2)
                self.visible_wps[w2].add(w1) # Assuming visible is symmetric
            elif pred == 'can_traverse':
                rover, w1, w2 = fact[1:]
                self.rover_adj[rover][w1].add(w2)
                # Assuming can_traverse is symmetric for this heuristic's graph
                self.rover_adj[rover][w2].add(w1)
            elif pred == 'on_board':
                camera, rover = fact[1:]
                self.camera_info[camera]['rover'] = rover
            elif pred == 'supports':
                camera, mode = fact[1:]
                self.camera_info[camera].setdefault('modes', set()).add(mode)
            elif pred == 'calibration_target':
                camera, objective = fact[1:]
                self.camera_info[camera]['target'] = objective
                self.cal_targets.add(objective)
            elif pred == 'visible_from':
                objective, wp = fact[1:]
                self.visible_from_obj[objective].add(wp)

        # Ensure all waypoints are in the adjacency list keys for BFS even if they have no edges
        for wp in self.waypoints:
            self.visible_wps.setdefault(wp, set())
            for r in self.rovers:
                self.rover_adj[r].setdefault(wp, set())


        # Pre-calculate shortest paths for each rover
        self.rover_nav_graphs = {}
        for rover in self.rovers:
            self.rover_nav_graphs[rover] = {}
            all_wps = list(self.waypoints) # Use a list to iterate
            for start_wp in all_wps:
                 self.rover_nav_graphs[rover][start_wp] = self._bfs(self.rover_adj[rover], start_wp)

        # Identify communication waypoints
        self.comm_wps = set()
        for lander_wp in self.lander_at:
            self.comm_wps.update(self.visible_wps.get(lander_wp, set()))

        # Identify calibration waypoints by target objective
        self.cal_wps_by_target = {}
        for camera in self.cameras:
            target = self.camera_info[camera].get('target')
            if target:
                self.cal_wps_by_target[target] = self.visible_from_obj.get(target, set())

        # Identify image waypoints by objective
        self.img_wps_by_objective = self.visible_from_obj # Same as visible_from_obj

        self.UNREACHABLE_PENALTY = 1000 # Large penalty for unreachable goal components


    def _bfs(self, graph, start_node):
        """Helper to perform BFS and return distances."""
        distances = {node: float('inf') for node in self.waypoints} # Use all waypoints
        if start_node not in self.waypoints:
             # Start node might not be a known waypoint if parsing is incomplete
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()
            # Ensure current_node is in graph keys before accessing neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor in self.waypoints and neighbor not in visited: # Ensure neighbor is a known waypoint
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def min_dist(self, rover, start_wp, target_wps):
        """Helper to find min distance from start_wp to any in target_wps for a rover."""
        if not target_wps or start_wp not in self.rover_nav_graphs.get(rover, {}):
            return float('inf')
        min_d = float('inf')
        distances_from_start = self.rover_nav_graphs[rover][start_wp]
        for target_wp in target_wps:
            min_d = min(min_d, distances_from_start.get(target_wp, float('inf')))
        return min_d

    def find_closest_wp(self, rover, start_wp, target_wps):
        """Helper to find the closest waypoint in target_wps from start_wp for a rover."""
        if not target_wps or start_wp not in self.rover_nav_graphs.get(rover, {}):
            return None
        min_d = float('inf')
        closest_wp = None
        distances_from_start = self.rover_nav_graphs[rover][start_wp]
        for target_wp in target_wps:
            d = distances_from_start.get(target_wp, float('inf'))
            if d < min_d:
                min_d = d
                closest_wp = target_wp
        return closest_wp # Returns None if target_wps is empty or unreachable

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Check if goal is reached
        if self.goals <= state:
            return 0

        h = 0
        infinity = float('inf')

        # Parse dynamic state facts
        rover_at = {} # rover -> wp
        have_soil = set() # (rover, wp)
        have_rock = set() # (rover, wp)
        have_image = set() # (rover, obj, mode)
        empty_stores = set() # store
        full_stores = set() # store
        calibrated_cameras = set() # (camera, rover)
        soil_samples_at = set() # wp
        rock_samples_at = set() # wp

        for fact_str in state:
            fact = parse_fact(fact_str)
            pred = fact[0]
            if pred == 'at' and len(fact) == 3: rover_at[fact[1]] = fact[2]
            elif pred == 'have_soil_analysis' and len(fact) == 3: have_soil.add((fact[1], fact[2]))
            elif pred == 'have_rock_analysis' and len(fact) == 3: have_rock.add((fact[1], fact[2]))
            elif pred == 'have_image' and len(fact) == 4: have_image.add((fact[1], fact[2], fact[3]))
            elif pred == 'empty' and len(fact) == 2: empty_stores.add(fact[1])
            elif pred == 'full' and len(fact) == 2: full_stores.add(fact[1])
            elif pred == 'calibrated' and len(fact) == 3: calibrated_cameras.add((fact[1], fact[2]))
            elif pred == 'at_soil_sample' and len(fact) == 2: soil_samples_at.add(fact[1])
            elif pred == 'at_rock_sample' and len(fact) == 2: rock_samples_at.add(fact[1])


        # Iterate through unsatisfied goals
        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already satisfied

            goal = parse_fact(goal_str)
            pred = goal[0]

            if pred == 'communicated_soil_data' and len(goal) == 2:
                w = goal[1]

                min_rover_total_cost = infinity
                for r in self.rovers:
                    if 'soil' not in self.rover_equipment.get(r, set()):
                        continue # Rover not equipped

                    r_at = rover_at.get(r)
                    if r_at is None: continue # Rover location unknown

                    current_rover_cost = 0
                    current_wp = r_at # Start from rover's current location

                    if (r, w) not in have_soil:
                        # Need to sample
                        dist_to_sample = self.rover_nav_graphs[r][current_wp].get(w, infinity)
                        if dist_to_sample == infinity:
                            continue # Cannot reach sample waypoint

                        cost_get_sample = dist_to_sample + 1 # navigate + sample
                        store = self.rover_stores.get(r)
                        if store and store in full_stores:
                             cost_get_sample += 1 # drop needed

                        current_rover_cost += cost_get_sample
                        wp_after_sample = w # Rover is now at w

                        # Need to navigate from w to comm_wp, communicate
                        # Find the specific comm_wp that is closest from w
                        comm_wp_chosen = self.find_closest_wp(r, wp_after_sample, self.comm_wps)
                        if comm_wp_chosen is None: continue # Cannot reach any communication waypoint

                        dist_to_comm = self.rover_nav_graphs[r][wp_after_sample].get(comm_wp_chosen, infinity)
                        current_rover_cost += dist_to_comm + 1 # navigate + communicate

                    else: # Already have sample (by this rover)
                        # Need to navigate from r_at to comm_wp, communicate
                        comm_wp_chosen = self.find_closest_wp(r, current_wp, self.comm_wps)
                        if comm_wp_chosen is None: continue # Cannot reach any communication waypoint

                        dist_to_comm = self.rover_nav_graphs[r][current_wp].get(comm_wp_chosen, infinity)
                        current_rover_cost += dist_to_comm + 1 # navigate + communicate

                    min_rover_total_cost = min(min_rover_total_cost, current_rover_cost)

                if min_rover_total_cost == infinity:
                    h += self.UNREACHABLE_PENALTY
                else:
                    h += min_rover_total_cost

            elif pred == 'communicated_rock_data' and len(goal) == 2:
                w = goal[1]

                min_rover_total_cost = infinity
                for r in self.rovers:
                    if 'rock' not in self.rover_equipment.get(r, set()):
                        continue # Rover not equipped

                    r_at = rover_at.get(r)
                    if r_at is None: continue

                    current_rover_cost = 0
                    current_wp = r_at

                    if (r, w) not in have_rock:
                        # Need to sample
                        dist_to_sample = self.rover_nav_graphs[r][current_wp].get(w, infinity)
                        if dist_to_sample == infinity:
                            continue # Cannot reach sample waypoint

                        cost_get_sample = dist_to_sample + 1 # navigate + sample
                        store = self.rover_stores.get(r)
                        if store and store in full_stores:
                             cost_get_sample += 1 # drop needed

                        current_rover_cost += cost_get_sample
                        wp_after_sample = w # Rover is now at w

                        # Need to navigate from w to comm_wp, communicate
                        comm_wp_chosen = self.find_closest_wp(r, wp_after_sample, self.comm_wps)
                        if comm_wp_chosen is None: continue # Cannot reach any communication waypoint

                        dist_to_comm = self.rover_nav_graphs[r][wp_after_sample].get(comm_wp_chosen, infinity)
                        current_rover_cost += dist_to_comm + 1 # navigate + communicate

                    else: # Already have sample (by this rover)
                        # Need to navigate from r_at to comm_wp, communicate
                        comm_wp_chosen = self.find_closest_wp(r, current_wp, self.comm_wps)
                        if comm_wp_chosen is None: continue # Cannot reach any communication waypoint

                        dist_to_comm = self.rover_nav_graphs[r][current_wp].get(comm_wp_chosen, infinity)
                        current_rover_cost += dist_to_comm + 1 # navigate + communicate

                    min_rover_total_cost = min(min_rover_total_cost, current_rover_cost)

                if min_rover_total_cost == infinity:
                    h += self.UNREACHABLE_PENALTY
                else:
                    h += min_rover_total_cost

            elif pred == 'communicated_image_data' and len(goal) == 3:
                o, m = goal[1:]

                min_rover_total_cost = infinity
                for r in self.rovers:
                    if 'imaging' not in self.rover_equipment.get(r, set()):
                        continue # Rover not equipped

                    # Find camera on this rover supporting the mode
                    suitable_camera = None
                    for cam in self.cameras:
                        info = self.camera_info.get(cam, {})
                        if info.get('rover') == r and m in info.get('modes', set()):
                            suitable_camera = cam
                            break
                    if not suitable_camera:
                        continue # No suitable camera on this rover

                    i = suitable_camera
                    r_at = rover_at.get(r)
                    if r_at is None: continue

                    current_wp = r_at

                    if (r, o, m) in have_image:
                        # Already have image, just need to communicate
                        comm_wp_chosen = self.find_closest_wp(r, current_wp, self.comm_wps)
                        if comm_wp_chosen is None: continue # Cannot reach any communication waypoint

                        dist_to_comm = self.rover_nav_graphs[r][current_wp].get(comm_wp_chosen, infinity)
                        current_rover_cost = dist_to_comm + 1 # navigate + communicate
                        min_rover_total_cost = min(min_rover_total_cost, current_rover_cost)

                    else: # Need to take image and communicate
                        img_wps = self.img_wps_by_objective.get(o, set())
                        if not img_wps:
                            continue # Cannot image this objective from anywhere

                        cal_target = self.camera_info[i].get('target')
                        cal_wps = self.cal_wps_by_target.get(cal_target, set()) if cal_target else set()

                        cost_option1 = infinity # Calibrate -> Image -> Communicate
                        cost_option2 = infinity # Image -> Communicate (if calibrated)

                        # Calculate Option 1 (Calibrate -> Image -> Communicate)
                        if cal_wps:
                            # Find best cal_wp and img_wp pair sequence
                            min_nav_cal_img = infinity
                            best_cal_wp = None
                            best_img_wp = None
                            for cal_wp in cal_wps:
                                dist_r_to_cal = self.rover_nav_graphs[r][current_wp].get(cal_wp, infinity)
                                if dist_r_to_cal == infinity: continue
                                # Find the specific img_wp that is closest from cal_wp
                                img_wp_chosen_from_cal = self.find_closest_wp(r, cal_wp, img_wps)
                                if img_wp_chosen_from_cal is not None:
                                    dist_cal_to_img_chosen = self.rover_nav_graphs[r][cal_wp].get(img_wp_chosen_from_cal, infinity)
                                    total_nav = dist_r_to_cal + dist_cal_to_img_chosen
                                    if total_nav < min_nav_cal_img:
                                        min_nav_cal_img = total_nav
                                        best_cal_wp = cal_wp
                                        best_img_wp = img_wp_chosen_from_cal

                            if best_cal_wp and best_img_wp:
                                # Need to navigate from best_img_wp to comm_wp, communicate
                                comm_wp_chosen = self.find_closest_wp(r, best_img_wp, self.comm_wps)
                                if comm_wp_chosen is not None:
                                     dist_img_to_comm = self.rover_nav_graphs[r][best_img_wp].get(comm_wp_chosen, infinity)
                                     cost_option1 = min_nav_cal_img + 1 + 1 + dist_img_to_comm + 1 # nav(r->cal) + cal + nav(cal->img) + img + nav(img->comm) + comm


                        # Calculate Option 2 (Image -> Communicate) - only if camera is calibrated
                        if (i, r) in calibrated_cameras:
                            # Find the specific img_wp that is closest from r_at
                            img_wp_chosen_from_r = self.find_closest_wp(r, current_wp, img_wps)
                            if img_wp_chosen_from_r is not None:
                                dist_r_to_img_chosen = self.rover_nav_graphs[r][current_wp].get(img_wp_chosen_from_r, infinity)
                                if dist_r_to_img_chosen != infinity:
                                    # Need to navigate from img_wp_chosen_from_r to comm_wp, communicate
                                    comm_wp_chosen = self.find_closest_wp(r, img_wp_chosen_from_r, self.comm_wps)
                                    if comm_wp_chosen is not None:
                                        dist_img_to_comm = self.rover_nav_graphs[r][img_wp_chosen_from_r].get(comm_wp_chosen, infinity)
                                        cost_option2 = dist_r_to_img_chosen + 1 + dist_img_to_comm + 1 # nav(r->img) + img + nav(img->comm) + comm

                        # Choose the best option based on calibration status
                        if (i, r) in calibrated_cameras:
                             current_rover_cost = min(cost_option1, cost_option2)
                        else:
                             current_rover_cost = cost_option1

                        if current_rover_cost == infinity: continue # Cannot achieve this goal with this rover/camera

                        min_rover_total_cost = min(min_rover_total_cost, current_rover_cost)

                if min_rover_total_cost == infinity:
                    h += self.UNREACHABLE_PENALTY
                else:
                    h += min_rover_total_cost

            # Note: Other goal types like 'have_soil_analysis', 'have_image', etc. are not
            # typically top-level goals in this domain based on the examples.
            # If they were, they would need specific cost calculations here.
            # Assuming only 'communicated_...' goals based on domain and examples.
            # If an unexpected goal type appears and is not satisfied, it will be ignored
            # by this heuristic, which might be acceptable for a non-admissible heuristic
            # focused on the primary goal types.

        return h
