import collections
import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts is at least the number of args
    if len(parts) < len(args):
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(graph, start):
    """Computes shortest path distances from start in an unweighted graph."""
    dist = {node: math.inf for node in graph}
    if start not in graph:
        # If the start node is not in the graph keys (e.g., an object location
        # not involved in any visible/traverse facts), it's unreachable from
        # other waypoints in the graph. Distances remain infinity.
        return dist

    dist[start] = 0
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        if u in graph: # Ensure node exists in graph keys (redundant check but safe)
            for v in graph[u]:
                if dist[v] == math.inf:
                    dist[v] = dist[u] + 1
                    queue.append(v)
    return dist


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It breaks down the problem by individual goal predicates
    (communicating soil data, rock data, or image data) and estimates the
    minimum actions needed for each unsatisfied goal independently, summing
    these estimates. It accounts for necessary steps like sampling, imaging,
    calibrating, and communicating, including estimated travel costs using
    precomputed shortest paths between waypoints.

    # Assumptions
    - Action costs are uniform (cost 1).
    - Navigation between visible and traversable waypoints has a cost equal
      to the shortest path distance in the unweighted graph. The graph is built
      considering if *any* rover can traverse a visible link.
    - Any equipped and capable rover can perform the necessary tasks for a
      goal item (sampling, imaging, communicating). The heuristic estimates
      the minimum travel cost assuming the best-positioned capable rover and
      the optimal sequence of waypoints (sample/image/calibrate -> communicate).
    - Store management is simplified: a drop action (cost 1) is added if *any*
      equipped rover currently has a full store when sampling is needed for
      an unsatisfied goal. It doesn't track specific stores or multiple drops.
    - Camera calibration is simplified: a calibrate action (cost 1) is added
      for an image goal if the optimal path found to take the image requires
      calibration for the chosen camera/rover combination. It doesn't track
      calibration state across multiple image goals or rovers perfectly.
    - The heuristic sums costs for goals independently, ignoring potential
      synergies (e.g., one trip collecting multiple samples or communicating
      multiple data items).

    # Heuristic Initialization
    The constructor precomputes static information to speed up heuristic
    computation during search:
    - Collects all relevant waypoints.
    - Builds the navigation graph based on `visible` and `can_traverse`
      predicates for any rover.
    - Computes all-pairs shortest paths between waypoints using BFS.
    - Identifies the lander's waypoint and the set of communication waypoints
      (visible from the lander).
    - Maps objectives to waypoints from which they are visible (`visible_from`).
    - Maps calibration targets to waypoints from which they are visible (`visible_from`).
    - Extracts rover equipment, camera assignments, supported modes, and
      calibration targets.
    - Maps stores to their respective rovers.
    - Precomputes potential image waypoints for each (objective, mode) pair,
      considering which waypoints are visible from the objective and which
      modes are supported by imaging-equipped rovers.
    - Precomputes potential calibration waypoints for each (camera, rover)
      pair, considering which waypoints are visible from the camera's
      calibration target and which rovers have that camera and are imaging equipped.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic `h` is computed as follows:
    1. Initialize `h = 0`.
    2. Parse the current state to get the location of each rover, which
       samples/images each rover has, the status of each store, and the
       calibration status of each camera on each rover. Also, identify
       remaining soil/rock samples on the ground (though initial samples are static).
    3. Iterate through each goal predicate defined in the task.
    4. If a goal predicate is already true in the current state, add 0 to `h`.
    5. If a goal predicate is `(communicated_soil_data ?w)` and it's not true:
       - Add 1 for the `communicate_soil_data` action.
       - Check if `(have_soil_analysis ?r ?w)` is true for any rover `r`.
       - If NO (sample not collected):
         - Add 1 for the `sample_soil` action.
         - Add 1 for a `drop` action if any soil-equipped rover's store is full.
         - Find the minimum total travel cost for any soil-equipped rover
           for the sequence: rover's current location -> sample waypoint `w`
           -> communication waypoint. Add this cost to `h`.
       - If YES (some rover `r_has_sample` has the sample):
         - Find the minimum travel cost for `r_has_sample` from its current
           location to any communication waypoint. Add this cost to `h`.
       - If any required travel is impossible (no path), return infinity.
    6. If a goal predicate is `(communicated_rock_data ?w)` and it's not true:
       - Follow similar logic as for soil data, using rock-specific predicates
         and equipment.
    7. If a goal predicate is `(communicated_image_data ?o ?m)` and it's not true:
       - Add 1 for the `take_image` action.
       - Add 1 for the `communicate_image_data` action.
       - Find all suitable (rover, camera) pairs that can potentially take
         this image (imaging equipped rover, camera on board, camera supports mode `m`).
       - If no suitable pair exists, return infinity.
       - Find the minimum total travel cost among all suitable (rover, camera)
         pairs for the sequence: rover's current location -> calibration waypoint
         (if needed) -> image waypoint -> communication waypoint.
         - If calibration is needed for a specific (rover, camera) and that
           camera/rover is not currently calibrated, the path includes travel
           to a calibration waypoint.
         - The minimum cost path is calculated considering the best calibration
           waypoint (if needed) and the best image waypoint for that rover/camera.
       - Add this minimum total travel cost to `h`.
       - If the optimal path found required calibration, add 1 for the
         `calibrate` action.
       - If any required travel is impossible, return infinity.
       - If the image is already held by a rover, the cost is just 1 (communicate)
         plus the travel cost for that rover from its current location to a
         communication waypoint.
    8. Return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Precomputation ---

        # Collect all waypoints mentioned in static facts or initial state
        all_waypoints = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                all_waypoints.add(parts[2])
            if parts[0] == 'at_lander' and len(parts) == 3:
                 all_waypoints.add(parts[2])
            if parts[0] in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2:
                 all_waypoints.add(parts[1])

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['visible', 'can_traverse'] and len(parts) == 3:
                all_waypoints.add(parts[1])
                all_waypoints.add(parts[2])
            if parts[0] == 'visible_from' and len(parts) == 3:
                 all_waypoints.add(parts[2])
            if parts[0] == 'at_lander' and len(parts) == 3:
                 all_waypoints.add(parts[2])

        self.all_waypoints = all_waypoints

        # Build navigation graph: edge u -> v if visible(u, v) and can_traverse(any_rover, u, v)
        nav_graph = {wp: set() for wp in self.all_waypoints}
        can_traverse_pairs = set() # (from_wp, to_wp) -> True if any rover can traverse
        # First, find which pairs are traversable by *any* rover
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'can_traverse' and len(parts) == 4:
                # Ignore the specific rover for graph connectivity
                can_traverse_pairs.add((parts[2], parts[3]))

        # Now build the graph using visible and the aggregated traversability
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'visible' and len(parts) == 3:
                wp1, wp2 = parts[1], parts[2]
                if (wp1, wp2) in can_traverse_pairs:
                     nav_graph[wp1].add(wp2)

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_wp in self.all_waypoints:
            self.dist[start_wp] = bfs(nav_graph, start_wp)

        # Extract other static information
        self.lander_wp = None
        self.comm_wps = set()
        self.rover_equipment = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_cameras = collections.defaultdict(set) # rover -> {cam}
        self.camera_modes = collections.defaultdict(set) # cam -> {mode}
        self.camera_cal_target = {} # cam -> target
        self.store_rover = {} # store -> rover
        self.obj_visible_from = collections.defaultdict(set) # obj -> {wp}
        self.target_visible_from = collections.defaultdict(set) # target -> {wp}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander' and len(parts) == 3:
                self.lander_wp = parts[2]
            elif parts[0].startswith('equipped_for_') and len(parts) == 2:
                equip_type = parts[0].split('_')[2] # soil, rock, imaging
                self.rover_equipment[parts[1]].add(equip_type)
            elif parts[0] == 'store_of' and len(parts) == 3:
                self.store_rover[parts[1]] = parts[2]
            elif parts[0] == 'on_board' and len(parts) == 3:
                self.rover_cameras[parts[2]].add(parts[1])
            elif parts[0] == 'supports' and len(parts) == 3:
                self.camera_modes[parts[1]].add(parts[2])
            elif parts[0] == 'calibration_target' and len(parts) == 3:
                self.camera_cal_target[parts[1]] = parts[2]
            elif parts[0] == 'visible_from' and len(parts) == 3:
                 obj_or_target, wp = parts[1], parts[2]
                 # Distinguish between objectives and calibration targets
                 # Check if obj_or_target is a calibration target for any camera
                 is_cal_target = any(obj_or_target == target for target in self.camera_cal_target.values())
                 if is_cal_target:
                     self.target_visible_from[obj_or_target].add(wp)
                 else:
                     self.obj_visible_from[obj_or_target].add(wp)

        # Identify communication waypoints (visible from lander)
        if self.lander_wp:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == 'visible' and len(parts) == 3 and parts[2] == self.lander_wp:
                     self.comm_wps.add(parts[1])

        # Precompute potential image waypoints for each (objective, mode)
        self.obj_img_wps = collections.defaultdict(set) # (obj, mode) -> {wp}
        # Iterate through all objectives and modes
        all_objectives = set(self.obj_visible_from.keys())
        all_modes = set(mode for modes in self.camera_modes.values() for mode in modes)

        for obj in all_objectives:
            for mode in all_modes:
                # A waypoint is a potential image spot for (obj, mode) if obj is visible from it
                # AND there exists *some* imaging-equipped rover with a camera supporting this mode.
                # We only need to check if *any* imaging rover/camera supports the mode.
                is_mode_supported_by_imaging_rover = any(
                    mode in self.camera_modes.get(cam, set())
                    for r in self.rover_equipment if 'imaging' in self.rover_equipment[r]
                    for cam in self.rover_cameras.get(r, set())
                )
                if is_mode_supported_by_imaging_rover:
                    self.obj_img_wps[(obj, mode)].update(self.obj_visible_from.get(obj, set()))


        # Precompute potential calibration waypoints for each (camera, rover)
        self.cam_cal_wps = collections.defaultdict(set) # (cam, rover) -> {wp}
        for r in self.rover_equipment:
            if 'imaging' in self.rover_equipment[r]:
                for cam in self.rover_cameras.get(r, set()):
                    if cam in self.camera_cal_target:
                        target = self.camera_cal_target[cam]
                        if target in self.target_visible_from:
                            self.cam_cal_wps[(cam, r)].update(self.target_visible_from[target])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        rover_curr_wp = {}
        rover_has_soil = {} # (r, w) -> True
        rover_has_rock = {} # (r, w) -> True
        rover_has_image = {} # (r, o, m) -> True
        store_is_empty = {} # s -> True
        camera_is_calibrated = {} # (c, r) -> True
        # soil_samples_left = set() # w -> True # Not needed for heuristic calculation based on goals

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover_curr_wp[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis' and len(parts) == 3:
                rover_has_soil[(parts[1], parts[2])] = True
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3:
                rover_has_rock[(parts[1], parts[2])] = True
            elif parts[0] == 'have_image' and len(parts) == 4:
                rover_has_image[(parts[1], parts[2], parts[3])] = True
            elif parts[0] == 'empty' and len(parts) == 2 and parts[1].startswith('rover') and parts[1].endswith('store'):
                store_is_empty[parts[1]] = True
            elif parts[0] == 'full' and len(parts) == 2 and parts[1].startswith('rover') and parts[1].endswith('store'):
                 store_is_empty[parts[1]] = False # Explicitly mark as not empty
            elif parts[0] == 'calibrated' and len(parts) == 3:
                camera_is_calibrated[(parts[1], parts[2])] = True
            # elif parts[0] == 'at_soil_sample' and len(parts) == 2:
            #     soil_samples_left.add(parts[1]) # Not needed for heuristic calculation based on goals

        # Assume stores not explicitly marked full are empty (initial state semantics)
        all_stores = set(self.store_rover.keys())
        for store in all_stores:
             if store not in store_is_empty:
                  store_is_empty[store] = True


        # --- Compute Heuristic ---
        h = 0

        for goal in self.goals:
            # Check if goal is already satisfied
            if goal in state:
                continue

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                w = parts[1]
                h += 1 # Cost for communicate action

                # Check if sample is already collected by any rover
                has_sample = any(rover_has_soil.get((r, w), False) for r in self.rover_equipment if 'soil' in self.rover_equipment[r])

                if not has_sample:
                    # Need to sample and then communicate
                    h += 1 # Cost for sample action

                    # Cost for ensuring empty store for sampling
                    needs_empty_store_action = False
                    # Check if any soil-equipped rover has a full store
                    for r in self.rover_equipment:
                        if 'soil' in self.rover_equipment[r]:
                            store = next((s for s, rv in self.store_rover.items() if rv == r), None)
                            if store and not store_is_empty.get(store, True): # Check if NOT empty
                                needs_empty_store_action = True
                                break
                    if needs_empty_store_action:
                        h += 1 # Cost for drop action

                    # Find min cost path: curr -> sample_wp -> comm_wp
                    min_total_path_cost = float('inf')
                    sample_wp = w
                    possible_comm_wps = self.comm_wps
                    if not possible_comm_wps: return float('inf') # Unsolvable

                    suitable_rovers = [r for r in self.rover_equipment if 'soil' in self.rover_equipment[r]]
                    if not suitable_rovers: return float('inf') # Unsolvable

                    for r in suitable_rovers:
                        if r not in rover_curr_wp: continue # Rover location unknown

                        curr_wp = rover_curr_wp[r]

                        dist_curr_to_sample = self.dist.get(curr_wp, {}).get(sample_wp, float('inf'))
                        if dist_curr_to_sample == float('inf'): continue

                        min_dist_sample_to_comm = float('inf')
                        for comm_wp in possible_comm_wps:
                            dist_sample_to_comm = self.dist.get(sample_wp, {}).get(comm_wp, float('inf'))
                            min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist_sample_to_comm)

                        if min_dist_sample_to_comm == float('inf'): continue

                        min_path_cost_for_r = dist_curr_to_sample + min_dist_sample_to_comm
                        min_total_path_cost = min(min_total_path_cost, min_path_cost_for_r)

                    if min_total_path_cost == float('inf'): return float('inf') # Unsolvable
                    h += min_total_path_cost

                else: # Sample is already collected
                    # Need to communicate
                    # Find the rover(s) with the sample
                    rovers_with_sample = [r for r in self.rover_equipment if 'soil' in self.rover_equipment[r] and rover_has_soil.get((r, w), False)]
                    if not rovers_with_sample:
                         # This state shouldn't happen if has_sample is true, but safety check
                         return float('inf')

                    # Find min cost path: curr -> comm_wp
                    min_dist_to_comm = float('inf')
                    possible_comm_wps = self.comm_wps
                    if not possible_comm_wps: return float('inf') # Unsolvable

                    for r in rovers_with_sample:
                        if r not in rover_curr_wp: continue # Rover location unknown

                        curr_wp = rover_curr_wp[r]
                        for comm_wp in possible_comm_wps:
                            dist_curr_to_comm = self.dist.get(curr_wp, {}).get(comm_wp, float('inf'))
                            min_dist_to_comm = min(min_dist_to_comm, dist_curr_to_comm)

                    if min_dist_to_comm == float('inf'): return float('inf') # Unsolvable
                    h += min_dist_to_comm


            elif predicate == 'communicated_rock_data':
                w = parts[1]
                h += 1 # Cost for communicate action

                # Check if sample is already collected by any rover
                has_sample = any(rover_has_rock.get((r, w), False) for r in self.rover_equipment if 'rock' in self.rover_equipment[r])

                if not has_sample:
                    # Need to sample and then communicate
                    h += 1 # Cost for sample action

                    # Cost for ensuring empty store for sampling
                    needs_empty_store_action = False
                    # Check if any rock-equipped rover has a full store
                    for r in self.rover_equipment:
                        if 'rock' in self.rover_equipment[r]:
                            store = next((s for s, rv in self.store_rover.items() if rv == r), None)
                            if store and not store_is_empty.get(store, True): # Check if NOT empty
                                needs_empty_store_action = True
                                break
                    if needs_empty_store_action:
                        h += 1 # Cost for drop action

                    # Find min cost path: curr -> sample_wp -> comm_wp
                    min_total_path_cost = float('inf')
                    sample_wp = w
                    possible_comm_wps = self.comm_wps
                    if not possible_comm_wps: return float('inf') # Unsolvable

                    suitable_rovers = [r for r in self.rover_equipment if 'rock' in self.rover_equipment[r]]
                    if not suitable_rovers: return float('inf') # Unsolvable

                    for r in suitable_rovers:
                        if r not in rover_curr_wp: continue # Rover location unknown

                        curr_wp = rover_curr_wp[r]

                        dist_curr_to_sample = self.dist.get(curr_wp, {}).get(sample_wp, float('inf'))
                        if dist_curr_to_sample == float('inf'): continue

                        min_dist_sample_to_comm = float('inf')
                        for comm_wp in possible_comm_wps:
                            dist_sample_to_comm = self.dist.get(sample_wp, {}).get(comm_wp, float('inf'))
                            min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist_sample_to_comm)

                        if min_dist_sample_to_comm == float('inf'): continue

                        min_path_cost_for_r = dist_curr_to_sample + min_dist_sample_to_comm
                        min_total_path_cost = min(min_total_path_cost, min_path_cost_for_r)

                    if min_total_path_cost == float('inf'): return float('inf') # Unsolvable
                    h += min_total_path_cost

                else: # Sample is already collected
                    # Need to communicate
                    # Find the rover(s) with the sample
                    rovers_with_sample = [r for r in self.rover_equipment if 'rock' in self.rover_equipment[r] and rover_has_rock.get((r, w), False)]
                    if not rovers_with_sample:
                         # This state shouldn't happen if has_sample is true, but safety check
                         return float('inf')

                    # Find min cost path: curr -> comm_wp
                    min_dist_to_comm = float('inf')
                    possible_comm_wps = self.comm_wps
                    if not possible_comm_wps: return float('inf') # Unsolvable

                    for r in rovers_with_sample:
                        if r not in rover_curr_wp: continue # Rover location unknown

                        curr_wp = rover_curr_wp[r]
                        for comm_wp in possible_comm_wps:
                            dist_curr_to_comm = self.dist.get(curr_wp, {}).get(comm_wp, float('inf'))
                            min_dist_to_comm = min(min_dist_to_comm, dist_curr_to_comm)

                    if min_dist_to_comm == float('inf'): return float('inf') # Unsolvable
                    h += min_dist_to_comm


            elif predicate == 'communicated_image_data':
                o, m = parts[1], parts[2]
                h += 1 # Cost for communicate action

                # Check if image is already collected by any rover
                has_image = any(rover_has_image.get((r, o, m), False) for r in self.rover_equipment if 'imaging' in self.rover_equipment[r])

                if not has_image:
                    # Need to take image and then communicate
                    h += 1 # Cost for take_image action

                    # Find suitable (rover, camera) pairs that can potentially take this image
                    suitable_rc = []
                    for r in self.rover_equipment:
                        if 'imaging' in self.rover_equipment[r]:
                            for cam in self.rover_cameras.get(r, set()):
                                if m in self.camera_modes.get(cam, set()):
                                    suitable_rc.append((r, cam))

                    if not suitable_rc: return float('inf') # Unsolvable

                    # Find min cost path: curr -> cal_wp (if needed) -> img_wp -> comm_wp
                    min_total_path_cost = float('inf')
                    calibration_needed_for_best_path = False

                    possible_img_wps = self.obj_img_wps.get((o, m), set())
                    if not possible_img_wps: return float('inf') # Unsolvable

                    possible_comm_wps = self.comm_wps
                    if not possible_comm_wps: return float('inf') # Unsolvable

                    for r, cam in suitable_rc:
                        if r not in rover_curr_wp: continue # Rover location unknown

                        curr_wp = rover_curr_wp[r]
                        is_calibrated_now = camera_is_calibrated.get((cam, r), False)

                        if not is_calibrated_now:
                            # Path includes calibration
                            possible_cal_wps = self.cam_cal_wps.get((cam, r), set())
                            if not possible_cal_wps: continue # Cannot calibrate this camera/rover combo

                            # Find min cost path: curr -> cal_wp -> img_wp -> comm_wp
                            min_path_cost_for_rc = float('inf')
                            for cal_wp in possible_cal_wps:
                                dist_curr_to_cal = self.dist.get(curr_wp, {}).get(cal_wp, float('inf'))
                                if dist_curr_to_cal == float('inf'): continue
                                for img_wp in possible_img_wps:
                                    dist_cal_to_img = self.dist.get(cal_wp, {}).get(img_wp, float('inf'))
                                    if dist_cal_to_img == float('inf'): continue
                                    for comm_wp in possible_comm_wps:
                                        dist_img_to_comm = self.dist.get(img_wp, {}).get(comm_wp, float('inf'))
                                        if dist_img_to_comm == float('inf'): continue
                                        min_path_cost_for_rc = min(min_path_cost_for_rc, dist_curr_to_cal + dist_cal_to_img + dist_img_to_comm)

                            if min_path_cost_for_rc < min_total_path_cost:
                                min_total_path_cost = min_path_cost_for_rc
                                calibration_needed_for_best_path = True
                        else: # Camera is already calibrated
                            # Path is simpler: curr -> img_wp -> comm_wp
                            min_path_cost_for_rc = float('inf')
                            for img_wp in possible_img_wps:
                                dist_curr_to_img = self.dist.get(curr_wp, {}).get(img_wp, float('inf'))
                                if dist_curr_to_img == float('inf'): continue
                                for comm_wp in possible_comm_wps:
                                    dist_img_to_comm = self.dist.get(img_wp, {}).get(comm_wp, float('inf'))
                                    if dist_img_to_comm == float('inf'): continue
                                    min_path_cost_for_rc = min(min_path_cost_for_rc, dist_curr_to_img + dist_img_to_comm)

                            if min_path_cost_for_rc < min_total_path_cost:
                                min_total_path_cost = min_path_cost_for_rc
                                calibration_needed_for_best_path = False

                    if min_total_path_cost == float('inf'): return float('inf') # Unsolvable
                    h += min_total_path_cost
                    if calibration_needed_for_best_path:
                        h += 1 # Cost for calibrate action

                else: # Image is already collected
                    # Need to communicate
                    # Find the rover(s) with the image
                    rovers_with_image = [r for r in self.rover_equipment if 'imaging' in self.rover_equipment[r] and rover_has_image.get((r, o, m), False)]
                    if not rovers_with_image:
                         # This state shouldn't happen if has_image is true, but safety check
                         return float('inf')

                    # Find min cost path: curr -> comm_wp
                    min_dist_to_comm = float('inf')
                    possible_comm_wps = self.comm_wps
                    if not possible_comm_wps: return float('inf') # Unsolvable

                    for r in rovers_with_image:
                        if r not in rover_curr_wp: continue # Rover location unknown

                        curr_wp = rover_curr_wp[r]
                        for comm_wp in possible_comm_wps:
                            dist_curr_to_comm = self.dist.get(curr_wp, {}).get(comm_wp, float('inf'))
                            min_dist_to_comm = min(min_dist_to_comm, dist_curr_to_comm)

                    if min_dist_to_comm == float('inf'): return float('inf') # Unsolvable
                    h += min_dist_to_comm

        return h
