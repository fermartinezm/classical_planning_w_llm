# Assume Heuristic base class is available
from heuristics.heuristic_base import Heuristic
# If not, define a dummy base class for standalone testing
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
from collections import defaultdict, deque
import sys # Used for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # The fact must have the same number of parts as the pattern arguments
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Compute shortest distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        distance = distances[current_node]
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances

def min_dist_set(rover, W1, W2, rover_distances, all_rovers):
    """
    Computes min distance between any waypoint in W1 and any in W2,
    for a specific rover or any rover (if rover is None).
    Returns float('inf') if W1 or W2 is empty, or no path exists.
    """
    min_d = float('inf')
    if not W1 or not W2:
        return min_d

    rovers_to_check = all_rovers if rover is None else [rover]

    for r in rovers_to_check:
        if r in rover_distances:
            for w1 in W1:
                if w1 in rover_distances[r]: # Check if w1 is a valid start node for this rover
                    for w2 in W2:
                        if w2 in rover_distances[r][w1]: # Check if w2 is reachable from w1 for this rover
                            min_d = min(min_d, rover_distances[r][w1][w2])
    return min_d

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It sums up the estimated costs for each goal
    independently, considering necessary actions (sample, calibrate, take_image,
    communicate) and estimated travel costs.

    # Assumptions:
    - The heuristic is non-admissible; it may overestimate or underestimate
      the true cost.
    - Costs for different goal facts are summed independently, ignoring
      potential synergies (e.g., one trip for multiple samples/images).
    - Travel costs are estimated using precomputed shortest path distances
      on the `can_traverse` graph for each rover.
    - The cost of the `drop` action (to empty a store before sampling) is
      ignored for simplicity.
    - The cost of `calibrate` is added once per uncollected image goal,
      regardless of how many times calibration is actually needed due to
      the camera becoming uncalibrated after `take_image`.
    - It assumes solvable instances, meaning required samples exist at waypoints
      and necessary capabilities/locations exist. Unreachable goals result
      in a large finite heuristic value.

    # Heuristic Initialization
    The constructor precomputes and stores static information from the task:
    - The lander's location.
    - The set of waypoints visible from the lander (communication points).
    - Which rovers are equipped for soil, rock, and imaging tasks.
    - Camera information: which rover a camera is on, which modes it supports,
      and its calibration target objective.
    - Objective information: which waypoints are visible from each objective
      (potential imaging points).
    - Camera-specific calibration points: which waypoints are visible from
      a camera's calibration target objective.
    - Shortest path distances between all pairs of waypoints for each rover,
      based on the `can_traverse` graph.
    - A list of all rovers and waypoints in the problem for use in distance lookups.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Identify all goal facts that are not yet true in the current state.
    2.  Initialize the total heuristic cost to 0.
    3.  For each unmet goal fact:
        -   If the goal is `(communicated_soil_data ?w)`:
            -   Check if any equipped soil rover already has `(have_soil_analysis ?r ?w)`.
            -   If not collected: Add 1 (sample_soil) + minimum travel cost for any equipped soil rover from its current location to `?w` + minimum travel cost for any equipped soil rover from `?w` to any communication waypoint.
            -   If collected: Add minimum travel cost for any rover that has the sample from its current location to any communication waypoint.
            -   Add 1 (communicate_soil_data).
        -   If the goal is `(communicated_rock_data ?w)`:
            -   Similar logic as for soil data, using rock-equipped rovers and rock samples.
        -   If the goal is `(communicated_image_data ?o ?m)`:
            -   Check if any equipped imaging rover already has `(have_image ?r ?o ?m)`.
            -   If not collected:
                -   Add 1 (calibrate) + 1 (take_image).
                -   Identify suitable rovers (equipped for imaging, with camera supporting mode `?m`) and suitable cameras.
                -   Identify imaging waypoints for objective `?o`.
                -   Identify calibration waypoints for suitable cameras (visible from their calibration targets).
                -   Add minimum travel cost for any suitable rover from its current location to any imaging waypoint.
                -   Add minimum travel cost for any suitable rover from any imaging waypoint to any calibration waypoint.
                -   Add minimum travel cost for any suitable rover from any calibration waypoint back to any imaging waypoint (to take the picture).
                -   Add minimum travel cost for any suitable rover from any imaging waypoint to any communication waypoint.
            -   If collected: Add minimum travel cost for any rover that has the image from its current location to any communication waypoint.
            -   Add 1 (communicate_image_data).
    4.  Sum up the costs calculated for each unmet goal fact.
    5.  If any required waypoint or capability was unreachable/missing during calculation (resulting in infinity), return a large finite number instead.
    6.  Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.task = task # Store task for access to goals and static facts
        self.goal_facts = set(task.goals)

        # --- Precomputation ---

        # Collect all objects of relevant types from task.facts (or task.initial_state + task.goals + task.static)
        # task.facts contains all possible ground facts, which implies all objects are mentioned.
        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_landers = set()

        # A robust way is to parse objects from the initial state and goals,
        # or rely on task.facts if it lists all possible groundings.
        # Let's parse initial state and goals for objects.
        # This is not strictly necessary if task.facts is comprehensive,
        # but makes the precomputation less brittle.
        all_facts_in_task = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact in all_facts_in_task:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred in ('at', 'at_lander', 'can_traverse', 'visible', 'have_rock_analysis', 'have_soil_analysis', 'at_soil_sample', 'at_rock_sample', 'visible_from'):
                 for part in parts[1:]:
                     if part.startswith('waypoint'): self.all_waypoints.add(part)
                     elif part.startswith('rover'): self.all_rovers.add(part)
                     elif part.startswith('lander'): self.all_landers.add(part)
             elif pred in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'store_of', 'calibrated', 'on_board'):
                 for part in parts[1:]:
                     if part.startswith('rover'): self.all_rovers.add(part)
                     elif part.startswith('store'): pass # store type
                     elif part.startswith('camera'): self.all_cameras.add(part)
             elif pred in ('supports', 'have_image', 'communicated_image_data'):
                 for part in parts[1:]:
                     if part.startswith('camera'): self.all_cameras.add(part)
                     elif part.startswith('mode'): self.all_modes.add(part)
                     elif part.startswith('rover'): self.all_rovers.add(part)
                     elif part.startswith('objective'): self.all_objectives.add(part)
             elif pred in ('calibration_target', 'visible_from'):
                 for part in parts[1:]:
                     if part.startswith('camera'): self.all_cameras.add(part)
                     elif part.startswith('objective'): self.all_objectives.add(part)
                     elif part.startswith('waypoint'): self.all_waypoints.add(part)
             elif pred in ('empty', 'full'): pass # store state
             elif pred in ('communicated_soil_data', 'communicated_rock_data'):
                 for part in parts[1:]:
                     if part.startswith('waypoint'): self.all_waypoints.add(part)


        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible *from* the lander location
        self.equipped_rovers = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_on_board_camera = {} # camera -> rover
        self.camera_supports_mode = defaultdict(set) # camera -> {mode, ...}
        self.camera_calibration_targets = {} # camera -> objective
        self.objective_imaging_wps = defaultdict(set) # objective -> {waypoint, ...}
        self.objective_calibration_wps = defaultdict(set) # camera -> {waypoint, ...} # Store by camera, as calibration is camera-specific

        rover_can_traverse_graph = defaultdict(lambda: defaultdict(set)) # rover -> wp -> {neighbor_wp, ...}

        # First pass to get basic info like lander location and on_board cameras
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]
            elif match(fact, "on_board", "*", "*"):
                self.rover_on_board_camera[parts[1]] = parts[2]

        # Second pass for dependent info and graph building
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_rovers[parts[1]].add("soil")
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_rovers[parts[1]].add("rock")
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_rovers[parts[1]].add("imaging")
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover_can_traverse_graph[parts[1]][parts[2]].add(parts[3])
            elif match(fact, "visible", "*", "*"):
                # Check if the second waypoint is the lander location
                if self.lander_location and parts[2] == self.lander_location:
                    self.comm_waypoints.add(parts[1])
            elif match(fact, "supports", "*", "*"):
                 # Ensure camera is on board a known rover, though not strictly needed for supports
                 if parts[1] in self.rover_on_board_camera:
                    self.camera_supports_mode[parts[1]].add(parts[2])
            elif match(fact, "calibration_target", "*", "*"):
                self.camera_calibration_targets[parts[1]] = parts[2]
            elif match(fact, "visible_from", "*", "*"):
                obj = parts[1]
                wp = parts[2]
                self.objective_imaging_wps[obj].add(wp)
                # Also add to calibration wps if this objective is a calibration target for some camera
                for cam, cal_obj in self.camera_calibration_targets.items():
                    if cal_obj == obj:
                        self.objective_calibration_wps[cam].add(wp)


        # Compute shortest paths for each rover
        self.rover_distances = {}
        for rover in self.all_rovers:
            self.rover_distances[rover] = {}
            graph = rover_can_traverse_graph[rover]
            # Compute distances from every waypoint to every other waypoint for this rover
            for start_wp in self.all_waypoints:
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # Convert sets to lists for consistent iteration order if needed (not strictly necessary for min_dist_set)
        self.all_rovers = list(self.all_rovers)
        self.all_waypoints = list(self.all_waypoints)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state ---
        current_rover_locations = {}
        rovers_with_soil = set()
        rovers_with_rock = set()
        rovers_with_image = set()
        full_stores = set()
        soil_samples_at_wps = set()
        rock_samples_at_wps = set()
        calibrated_cameras = set()
        rover_stores = {} # rover -> store

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                current_rover_locations[parts[1]] = parts[2]
            elif pred == "have_soil_analysis" and len(parts) == 3:
                rovers_with_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis" and len(parts) == 3:
                rovers_with_rock.add((parts[1], parts[2]))
            elif pred == "have_image" and len(parts) == 4:
                rovers_with_image.add((parts[1], parts[2], parts[3]))
            elif pred == "full" and len(parts) == 2:
                full_stores.add(parts[1])
            elif pred == "at_soil_sample" and len(parts) == 2:
                soil_samples_at_wps.add(parts[1])
            elif pred == "at_rock_sample" and len(parts) == 2:
                rock_samples_at_wps.add(parts[1])
            elif pred == "calibrated" and len(parts) == 3:
                calibrated_cameras.add((parts[1], parts[2]))
            elif pred == "store_of" and len(parts) == 3:
                 rover_stores[parts[2]] = parts[1]


        total_cost = 0

        # --- Calculate cost for each unmet goal ---
        for goal in self.goal_facts:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                wp = parts[1]
                # Check if sample is collected by any equipped rover
                has_sample = any((r, wp) in rovers_with_soil for r in self.equipped_rovers if "soil" in self.equipped_rovers[r])

                if not has_sample:
                    # Need to sample
                    # Find equipped soil rovers
                    soil_rovers = [r for r in self.equipped_rovers if "soil" in self.equipped_rovers[r]]
                    if not soil_rovers: return 1000000 # Impossible

                    # Min travel cost for any soil rover from its current location to sample waypoint
                    min_t_to_sample = float('inf')
                    for r in soil_rovers:
                        current_wp = current_rover_locations.get(r)
                        if current_wp:
                            min_t_to_sample = min(min_t_to_sample, self.rover_distances[r][current_wp].get(wp, float('inf')))
                    if min_t_to_sample == float('inf'): return 1000000 # Cannot reach sample waypoint

                    total_cost += min_t_to_sample + 1 # travel + sample

                    # Min travel cost from sample waypoint to comm waypoint (any rover)
                    min_t_sample_to_comm = min_dist_set(None, {wp}, self.comm_waypoints, self.rover_distances, self.all_rovers)
                    if min_t_sample_to_comm == float('inf'): return 1000000 # Cannot reach comm point from sample point

                    total_cost += min_t_sample_to_comm + 1 # travel + communicate

                else: # Sample is already collected
                    # Find rovers with this sample
                    rovers_with_this_sample = [r for (r, collected_wp) in rovers_with_soil if collected_wp == wp]
                    if not rovers_with_this_sample: return 1000000 # Should not happen if has_sample is true

                    # Min travel cost from current location of any rover with sample to comm waypoint
                    min_t_to_comm = float('inf')
                    for r in rovers_with_this_sample:
                        current_wp = current_rover_locations.get(r)
                        if current_wp:
                            min_t_to_comm = min(min_t_to_comm, min_dist_set(r, {current_wp}, self.comm_waypoints, self.rover_distances, self.all_rovers))
                    if min_t_to_comm == float('inf'): return 1000000 # Cannot reach comm point

                    total_cost += min_t_to_comm + 1 # travel + communicate

            elif predicate == "communicated_rock_data":
                 wp = parts[1]
                 has_sample = any((r, wp) in rovers_with_rock for r in self.equipped_rovers if "rock" in self.equipped_rovers[r])

                 if not has_sample:
                     rock_rovers = [r for r in self.equipped_rovers if "rock" in self.equipped_rovers[r]]
                     if not rock_rovers: return 1000000

                     min_t_to_sample = float('inf')
                     for r in rock_rovers:
                         current_wp = current_rover_locations.get(r)
                         if current_wp:
                             min_t_to_sample = min(min_t_to_sample, self.rover_distances[r][current_wp].get(wp, float('inf')))
                     if min_t_to_sample == float('inf'): return 1000000

                     total_cost += min_t_to_sample + 1 # travel + sample

                     min_t_sample_to_comm = min_dist_set(None, {wp}, self.comm_waypoints, self.rover_distances, self.all_rovers) # Use None to indicate any rover
                     if min_t_sample_to_comm == float('inf'): return 1000000

                     total_cost += min_t_sample_to_comm + 1 # travel + communicate

                 else: # Sample is already collected
                     rovers_with_this_sample = [r for (r, collected_wp) in rovers_with_rock if collected_wp == wp]
                     if not rovers_with_this_sample: return 1000000

                     min_t_to_comm = float('inf')
                     for r in rovers_with_this_sample:
                         current_wp = current_rover_locations.get(r)
                         if current_wp:
                             min_t_to_comm = min(min_t_to_comm, min_dist_set(r, {current_wp}, self.comm_waypoints, self.rover_distances, self.all_rovers))
                     if min_t_to_comm == float('inf'): return 1000000

                     total_cost += min_t_to_comm + 1 # travel + communicate

            elif predicate == "communicated_image_data":
                obj = parts[1]
                mode = parts[2]

                has_image = any((r, obj, mode) in rovers_with_image for r in self.equipped_rovers if "imaging" in self.equipped_rovers[r])

                if not has_image:
                    # Need to take image
                    # Find suitable rover+camera combinations
                    suitable_rovers_cameras = []
                    for r in self.equipped_rovers:
                        if "imaging" in self.equipped_rovers[r]:
                            for cam, rover_on_cam in self.rover_on_board_camera.items():
                                if rover_on_cam == r and mode in self.camera_supports_mode[cam]:
                                    suitable_rovers_cameras.append((r, cam))

                    if not suitable_rovers_cameras: return 1000000 # No rover/camera can take this image

                    # Imaging WPs for this objective
                    imaging_wps = self.objective_imaging_wps.get(obj, set())
                    if not imaging_wps: return 1000000 # Cannot image this objective

                    # Calibration WPs for suitable cameras
                    cal_wps_for_suitable_cams = set()
                    can_calibrate_any_suitable_cam = False
                    for r, cam in suitable_rovers_cameras:
                        cal_target_obj = self.camera_calibration_targets.get(cam)
                        if cal_target_obj:
                             # Calibration waypoints are visible from the calibration target objective
                             cam_cal_wps = self.objective_imaging_wps.get(cal_target_obj, set()) # Use objective_imaging_wps for cal target
                             if cam_cal_wps:
                                 cal_wps_for_suitable_cams.update(cam_cal_wps)
                                 can_calibrate_any_suitable_cam = True

                    if not can_calibrate_any_suitable_cam or not cal_wps_for_suitable_cams: return 1000000 # Cannot calibrate a suitable camera

                    if not self.comm_waypoints: return 1000000 # No communication points

                    # Find the best rover among suitable ones for travel
                    min_total_travel_for_image = float('inf')

                    for r, cam in suitable_rovers_cameras:
                        current_wp = current_rover_locations.get(r)
                        if not current_wp: continue # Rover location unknown? Should not happen.

                        # Travel current -> imaging wp
                        t_curr_to_img = min_dist_set(r, {current_wp}, imaging_wps, self.rover_distances, self.all_rovers)
                        if t_curr_to_img == float('inf'): continue

                        # Travel imaging wp -> calibration wp
                        t_img_to_cal = min_dist_set(r, imaging_wps, cal_wps_for_suitable_cams, self.rover_distances, self.all_rovers)
                        if t_img_to_cal == float('inf'): continue

                        # Travel calibration wp -> imaging wp (back)
                        t_cal_to_img = min_dist_set(r, cal_wps_for_suitable_cams, imaging_wps, self.rover_distances, self.all_rovers)
                        if t_cal_to_img == float('inf'): continue

                        # Travel imaging wp -> comm wp
                        t_img_to_comm = min_dist_set(r, imaging_wps, self.comm_waypoints, self.rover_distances, self.all_rovers)
                        if t_img_to_comm == float('inf'): continue

                        # Total travel for this rover/camera combo
                        total_travel = t_curr_to_img + t_img_to_cal + t_cal_to_img + t_img_to_comm
                        min_total_travel_for_image = min(min_total_travel_for_image, total_travel)

                    if min_total_travel_for_image == float('inf'): return 1000000 # Cannot complete image task travel

                    total_cost += min_total_travel_for_image + 1 + 1 + 1 # travel + calibrate + take_image + communicate

                else: # Image is already collected
                    # Find rovers with this image
                    rovers_with_this_image = [r for (r, collected_obj, collected_mode) in rovers_with_image if collected_obj == obj and collected_mode == mode]
                    if not rovers_with_this_image: return 1000000 # Should not happen

                    # Min travel cost from current location of any rover with image to comm waypoint
                    min_t_to_comm = float('inf')
                    for r in rovers_with_this_image:
                        current_wp = current_rover_locations.get(r)
                        if current_wp:
                            min_t_to_comm = min(min_t_to_comm, min_dist_set(r, {current_wp}, self.comm_waypoints, self.rover_distances, self.all_rovers))
                    if min_t_to_comm == float('inf'): return 1000000 # Cannot reach comm point

                    total_cost += min_t_to_comm + 1 # travel + communicate

        # Return a large value for unreachable goals
        if total_cost == float('inf'):
             return 1000000 # Use a large finite number

        return total_cost
