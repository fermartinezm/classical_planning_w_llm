from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import collections # For BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

def match_fact(fact, *args):
    """
    Check if a single PDDL fact string matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed), e.g., "at", "*", "loc".
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def match_in_state(state, *args):
    """
    Check if any fact in the given state matches a pattern.

    - `state`: A set or frozenset of fact strings.
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if any fact in the state matches the pattern, else `False`.
    """
    for fact in state:
        if match_fact(fact, *args):
            return True
    return False

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    independently based on navigation distances and required actions
    (sample/image, calibrate, communicate).

    Assumptions:
    - Navigation cost between adjacent visible waypoints is 1.
    - Sample, Drop, Calibrate, Take Image, Communicate actions cost 1.
    - Ignores store capacity beyond checking if a drop is needed before sampling.
    - Ignores recalibration requirement after taking an image.
    - Assumes soil/rock samples only exist at locations specified in the initial state.
    - Assumes a suitable rover, camera, calibration target, and visible waypoints exist
      if the goal is an image communication.
    - Uses precomputed shortest path distances for navigation costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # --- Precomputation ---

        # 1. Identify lander location
        self.lander_loc = None
        for fact in self.static_facts:
            if match_fact(fact, "at_lander", "*", "*"):
                self.lander_loc = get_parts(fact)[2]
                break
        # Assuming there is always exactly one lander and its location is static.

        # 2. Identify communication waypoints (visible from lander location)
        self.comm_wps = set()
        if self.lander_loc:
             for fact in self.static_facts:
                 # A waypoint X is a communication point if the lander location is visible from X
                 if match_fact(fact, "visible", "*", self.lander_loc):
                     self.comm_wps.add(get_parts(fact)[1])

        # 3. Identify objects and their static properties
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()

        # Infer objects and types from static facts and initial state (best effort without parser)
        all_facts = self.static_facts | task.initial_state | task.goals
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at': # (at ?x - rover ?y - waypoint) or (at_lander ?x - lander ?y - waypoint)
                 if len(parts) == 3: # (at rover waypoint) - initial state fact
                     self.rovers.add(parts[1])
                     self.waypoints.add(parts[2])
                 # at_lander handled below
            elif pred == 'at_lander': # (at_lander ?x - lander ?y - waypoint) - static fact
                 # lander already found
                 self.waypoints.add(parts[2])
            elif pred == 'can_traverse': # (can_traverse ?r - rover ?x - waypoint ?y - waypoint) - static fact
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.waypoints.add(parts[3])
            elif pred in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'): # (?r - rover) - static fact
                 self.rovers.add(parts[1])
            elif pred in ('empty', 'full'): # (?s - store) - initial state fact
                 self.stores.add(parts[1])
            elif pred in ('have_rock_analysis', 'have_soil_analysis'): # (?r - rover ?w - waypoint) - initial state fact
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'calibrated': # (?c - camera ?r - rover) - initial state fact
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
            elif pred == 'supports': # (?c - camera ?m - mode) - static fact
                 self.cameras.add(parts[1])
                 self.modes.add(parts[2])
            elif pred == 'visible': # (?w - waypoint ?p - waypoint) - static fact
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'have_image': # (?r - rover ?o - objective ?m - mode) - initial state fact
                 self.rovers.add(parts[1])
                 self.objectives.add(parts[2])
                 self.modes.add(parts[3])
            elif pred in ('communicated_soil_data', 'communicated_rock_data'): # (?w - waypoint) - goal fact
                 self.waypoints.add(parts[1])
            elif pred == 'communicated_image_data': # (?o - objective ?m - mode) - goal fact
                 self.objectives.add(parts[1])
                 self.modes.add(parts[2])
            elif pred in ('at_soil_sample', 'at_rock_sample'): # (?w - waypoint) - initial state fact
                 self.waypoints.add(parts[1])
            elif pred == 'visible_from': # (?o - objective ?w - waypoint) - static fact
                 self.objectives.add(parts[1])
                 self.waypoints.add(parts[2])
            elif pred == 'store_of': # (?s - store ?r - rover) - static fact
                 self.stores.add(parts[1])
                 self.rovers.add(parts[2])
            elif pred == 'calibration_target': # (?i - camera ?o - objective) - static fact
                 self.cameras.add(parts[1])
                 self.objectives.add(parts[2])
            elif pred == 'on_board': # (?i - camera ?r - rover) - static fact
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])


        self.initial_soil_samples = {get_parts(fact)[1] for fact in task.initial_state if match_fact(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in task.initial_state if match_fact(fact, "at_rock_sample", "*")}

        self.rover_capabilities = {}
        for r in self.rovers:
            self.rover_capabilities[r] = {
                'soil': match_in_state(self.static_facts, 'equipped_for_soil_analysis', r),
                'rock': match_in_state(self.static_facts, 'equipped_for_rock_analysis', r),
                'imaging': match_in_state(self.static_facts, 'equipped_for_imaging', r),
                'cameras': {}, # {camera: {mode1, mode2, ...}}
                'store': None # store_name
            }
            for fact in self.static_facts:
                if match_fact(fact, 'on_board', '*', r):
                    camera = get_parts(fact)[1]
                    self.rover_capabilities[r]['cameras'][camera] = set()
                    for s_fact in self.static_facts:
                        if match_fact(s_fact, 'supports', camera, '*'):
                            self.rover_capabilities[r]['cameras'][camera].add(get_parts(s_fact)[2])
                elif match_fact(fact, 'store_of', '*', r):
                     self.rover_capabilities[r]['store'] = get_parts(fact)[1]


        self.objective_visible_from = {} # {objective: {waypoint1, waypoint2, ...}}
        for o in self.objectives:
            self.objective_visible_from[o] = {get_parts(fact)[2] for fact in self.static_facts if match_fact(fact, 'visible_from', o, '*')}

        self.camera_calibration_target = {} # {camera: objective}
        for c in self.cameras:
            for fact in self.static_facts:
                if match_fact(fact, 'calibration_target', c, '*'):
                    self.camera_calibration_target[c] = get_parts(fact)[2]
                    break # Assume one target per camera

        self.calibration_target_visible_from = {} # {objective: {waypoint1, waypoint2, ...}}
        # Calibration targets are objectives, so reuse objective_visible_from logic
        for o in self.objectives: # Iterate through all objectives, some might be calibration targets
             self.calibration_target_visible_from[o] = {get_parts(fact)[2] for fact in self.static_facts if match_fact(fact, 'visible_from', o, '*')}


        # 4. Precompute shortest paths for each rover
        self.dist = {} # {rover: {start_wp: {end_wp: distance}}}
        for r in self.rovers:
            graph = {wp: set() for wp in self.waypoints}
            for fact in self.static_facts:
                if match_fact(fact, "can_traverse", r, "*", "*"):
                    _, rover_name, wp_from, wp_to = get_parts(fact)
                    # Check if wp_to is visible from wp_from for navigation
                    if match_in_state(self.static_facts, "visible", wp_from, wp_to):
                         graph[wp_from].add(wp_to)

            self.dist[r] = {}
            for start_wp in self.waypoints:
                self.dist[r][start_wp] = {wp: float('inf') for wp in self.waypoints}
                self.dist[r][start_wp][start_wp] = 0
                queue = collections.deque([start_wp]) # Use deque for efficient BFS queue

                while queue:
                    u = queue.popleft()
                    for v in graph.get(u, set()):
                        if self.dist[r][start_wp][v] == float('inf'):
                            self.dist[r][start_wp][v] = self.dist[r][start_wp][u] + 1
                            queue.append(v)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Get current rover locations
        current_rover_locations = {} # {rover: waypoint}
        for fact in state:
            if match_fact(fact, "at", "*", "*"):
                current_rover_locations[get_parts(fact)[1]] = get_parts(fact)[2]

        # Get current store statuses
        stores_full = {get_parts(fact)[1] for fact in state if match_fact(fact, "full", "*")}

        # Get current data/image statuses (needed to check if sample/image is already obtained)
        have_soil_analysis = { (get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match_fact(fact, "have_soil_analysis", "*", "*") } # {(rover, waypoint), ...}
        have_rock_analysis = { (get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match_fact(fact, "have_rock_analysis", "*", "*") } # {(rover, waypoint), ...}
        have_image = { (get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match_fact(fact, "have_image", "*", "*", "*") } # {(rover, objective, mode), ...}


        # Iterate through each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            g_parts = get_parts(goal)
            pred = g_parts[0]

            if pred == 'communicated_soil_data':
                w = g_parts[1]

                # Check if any rover currently has the sample data for this waypoint
                rover_has_sample = next(((r, wp) for (r, wp) in have_soil_analysis if wp == w), None)

                # If no rover has the sample, check if it was available initially.
                # If it wasn't available initially, this goal is impossible.
                if rover_has_sample is None and w not in self.initial_soil_samples:
                     # print(f"Warning: Goal {goal} refers to sample {w} not in initial state and not held by any rover.")
                     continue # Cannot achieve this goal

                min_h_g = float('inf')
                # Find the best rover to achieve this goal
                for r in self.rovers:
                    # If the sample is already held by a *different* rover, this rover can't communicate it directly.
                    # A more complex heuristic might model transferring samples, but let's assume the rover that samples communicates.
                    # If the sample is held by *another* rover, this rover cannot achieve the goal directly.
                    # If no rover holds the sample, this rover must sample it (if equipped).

                    if rover_has_sample is not None and rover_has_sample[0] != r:
                         # Sample is held by another rover, this rover cannot fulfill this goal directly.
                         continue # Skip this rover for this goal

                    # If sample is not held by *any* rover, this rover must be soil-equipped to get it.
                    if rover_has_sample is None and not self.rover_capabilities[r]['soil']:
                        continue # Rover cannot sample soil

                    r_loc = current_rover_locations.get(r)
                    if r_loc is None:
                         # Rover location unknown, cannot estimate path cost.
                         # This state might be invalid or indicate a problem.
                         # For heuristic, skip this rover.
                         continue

                    # Cost to get the sample: Navigate + Sample (+ Drop if needed)
                    if rover_has_sample is not None and rover_has_sample[0] == r:
                        # This rover already has the sample
                        cost_get_sample = 0
                        current_loc_for_comm = r_loc # Start communication from current location
                    else:
                        # Need to navigate to w and sample
                        dist_to_sample = self.dist[r][r_loc].get(w, float('inf')) # Use .get for safety
                        if dist_to_sample == float('inf'):
                            continue # Cannot reach sample location

                        cost_get_sample = dist_to_sample + 1 # Navigate + Sample action

                        # Check if store is full, requiring a drop action first
                        store = self.rover_capabilities[r]['store']
                        if store and store in stores_full:
                             # Assuming drop can happen anywhere, cost is 1 action before sampling
                             cost_get_sample += 1

                        current_loc_for_comm = w # Will be at waypoint w after sampling


                    # Cost to communicate: Navigate to comm point + Communicate
                    min_dist_to_comm = float('inf')
                    if not self.comm_wps: # No communication points available
                         min_dist_to_comm = float('inf')
                    else:
                        for comm_wp in self.comm_wps:
                            # Ensure the current location for communication is a valid waypoint
                            if current_loc_for_comm in self.dist[r] and comm_wp in self.dist[r][current_loc_for_comm]:
                                min_dist_to_comm = min(min_dist_to_comm, self.dist[r][current_loc_for_comm][comm_wp])
                            # else: This comm_wp is unreachable from current_loc_for_comm for this rover

                    if min_dist_to_comm == float('inf'):
                        continue # Cannot reach any comm point from sample location

                    cost_comm = min_dist_to_comm + 1 # Navigate + Communicate action

                    h_g = cost_get_sample + cost_comm
                    min_h_g = min(min_h_g, h_g)

                if min_h_g != float('inf'):
                    h += min_h_g
                # else: This goal is unreachable by any equipped rover from its current location.
                # Ignore for GBFS.

            elif pred == 'communicated_rock_data':
                w = g_parts[1]

                # Check if any rover currently has the sample data for this waypoint
                rover_has_sample = next(((r, wp) for (r, wp) in have_rock_analysis if wp == w), None)

                # If no rover has the sample, check if it was available initially.
                if rover_has_sample is None and w not in self.initial_rock_samples:
                     # print(f"Warning: Goal {goal} refers to sample {w} not in initial state and not held by any rover.")
                     continue # Cannot achieve this goal

                min_h_g = float('inf')
                # Find the best rover
                for r in self.rovers:
                    # If sample is held by another rover, this rover cannot fulfill goal directly
                    if rover_has_sample is not None and rover_has_sample[0] != r:
                         continue # Skip this rover

                    # If sample is not held by any rover, this rover must be rock-equipped
                    if rover_has_sample is None and not self.rover_capabilities[r]['rock']:
                        continue # Rover cannot sample rock

                    r_loc = current_rover_locations.get(r)
                    if r_loc is None: continue

                    # Cost to get the sample: Navigate + Sample (+ Drop if needed)
                    if rover_has_sample is not None and rover_has_sample[0] == r:
                        cost_get_sample = 0 # Already have it
                        current_loc_for_comm = r_loc # Start communication from current location
                    else:
                        # Need to navigate to w and sample
                        dist_to_sample = self.dist[r][r_loc].get(w, float('inf'))
                        if dist_to_sample == float('inf'):
                            continue # Cannot reach sample location

                        cost_get_sample = dist_to_sample + 1 # Navigate + Sample action

                        # Check if store is full, requiring a drop action first
                        store = self.rover_capabilities[r]['store']
                        if store and store in stores_full:
                             cost_get_sample += 1

                        current_loc_for_comm = w # Will be at waypoint w after sampling


                    # Cost to communicate: Navigate to comm point + Communicate
                    min_dist_to_comm = float('inf')
                    if not self.comm_wps: # No communication points available
                         min_dist_to_comm = float('inf')
                    else:
                        for comm_wp in self.comm_wps:
                             if current_loc_for_comm in self.dist[r] and comm_wp in self.dist[r][current_loc_for_comm]:
                                min_dist_to_comm = min(min_dist_to_comm, self.dist[r][current_loc_for_comm][comm_wp])

                    if min_dist_to_comm == float('inf'):
                        continue # Cannot reach any comm point from sample location

                    cost_comm = min_dist_to_comm + 1 # Navigate + Communicate action

                    h_g = cost_get_sample + cost_comm
                    min_h_g = min(min_h_g, h_g)

                if min_h_g != float('inf'):
                    h += min_h_g
                # else: Ignore unreachable goal

            elif pred == 'communicated_image_data':
                o = g_parts[1]
                m = g_parts[2]

                # Check if any rover currently has the image
                rover_has_image = next(((r, obj, mode) for (r, obj, mode) in have_image if obj == o and mode == m), None)

                min_h_g = float('inf')

                # Find the best rover/camera/waypoint combination
                for r in self.rovers:
                    # If image is held by another rover, this rover cannot fulfill goal directly
                    if rover_has_image is not None and rover_has_image[0] != r:
                         continue # Skip this rover

                    # If image is not held by any rover, this rover must be imaging-equipped
                    if rover_has_image is None and not self.rover_capabilities[r]['imaging']:
                        continue # Rover cannot take images

                    r_loc = current_rover_locations.get(r)
                    if r_loc is None: continue

                    # Cost to get the image: Calibrate + Take Image
                    if rover_has_image is not None and rover_has_image[0] == r:
                        # This rover already has the image
                        cost_get_image = 0
                        current_loc_for_comm = r_loc # Start communication from current location
                    else:
                        # Need to get the image
                        min_cost_get_image_path = float('inf')
                        best_img_wp = None # Keep track of the waypoint where image is taken

                        for camera, supported_modes in self.rover_capabilities[r]['cameras'].items():
                            if m not in supported_modes:
                                continue # Camera does not support mode

                            # Find calibration target for this camera
                            t = self.camera_calibration_target.get(camera)
                            if t is None:
                                continue # Camera has no calibration target

                            # Find waypoints visible from which the target is visible
                            cal_wps = self.calibration_target_visible_from.get(t, set())
                            if not cal_wps:
                                continue # No waypoint to calibrate from

                            # Find waypoints visible from which the objective is visible
                            img_wps = self.objective_visible_from.get(o, set())
                            if not img_wps:
                                continue # No waypoint to image from

                            # Find best path: r_loc -> w_cal -> p
                            for w_cal in cal_wps:
                                dist_to_cal = self.dist[r][r_loc].get(w_cal, float('inf'))
                                if dist_to_cal == float('inf'):
                                    continue # Cannot reach calibration waypoint

                                # Calibration cost is 1 action
                                cost_calibrate_step = dist_to_cal + 1 # Navigate + Calibrate

                                for p in img_wps:
                                    dist_cal_to_img = self.dist[r][w_cal].get(p, float('inf'))
                                    if dist_cal_to_img == float('inf'):
                                        continue # Cannot reach image waypoint from calibration waypoint

                                    # Take Image cost is 1 action
                                    cost_take_image_step = dist_cal_to_img + 1 # Navigate + Take Image

                                    current_path_cost = cost_calibrate_step + cost_take_image_step

                                    if current_path_cost < min_cost_get_image_path:
                                         min_cost_get_image_path = current_path_cost
                                         best_img_wp = p # Store the waypoint where the image would be taken

                        if min_cost_get_image_path == float('inf'):
                            continue # Cannot get the image with this rover/camera combination

                        cost_get_image = min_cost_get_image_path
                        current_loc_for_comm = best_img_wp # Will be at the image waypoint after taking the image


                    # Cost to communicate: Navigate to comm point + Communicate
                    min_dist_to_comm = float('inf')
                    if not self.comm_wps: # No communication points available
                         min_dist_to_comm = float('inf')
                    else:
                        for comm_wp in self.comm_wps:
                             if current_loc_for_comm in self.dist[r] and comm_wp in self.dist[r][current_loc_for_comm]:
                                min_dist_to_comm = min(min_dist_to_comm, self.dist[r][current_loc_for_comm][comm_wp])

                    if min_dist_to_comm == float('inf'):
                        continue # Cannot reach any comm point from image location

                    cost_comm = min_dist_to_comm + 1 # Navigate + Communicate action

                    h_g = cost_get_image + cost_comm
                    min_h_g = min(min_h_g, h_g)

                if min_h_g != float('inf'):
                    h += min_h_g
                # else: Ignore unreachable goal

        return h
