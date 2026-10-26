# Import necessary classes
from heuristics.heuristic_base import Heuristic
from task import Operator, Task
from collections import defaultdict
import heapq # Although Floyd-Warshall is used, heapq might be useful for Dijkstra if needed later.

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Removes surrounding brackets and splits by space."""
    # Handle potential empty string or malformed fact
    if not fact_string or fact_string[0] != '(' or fact_string[-1] != ')':
        # This might indicate a malformed fact string, return None or handle appropriately
        # For robustness, let's return None, though valid PDDL facts should match
        return None
    parts = fact_string[1:-1].split()
    return tuple(parts)


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing the
    estimated costs for each unachieved goal predicate. For each unachieved
    goal (communicating soil data, rock data, or image data), it calculates
    the minimum cost across all possible ways to achieve that goal. This
    minimum cost considers whether the required data (sample or image) is
    already collected by a rover or needs to be collected first. The cost
    includes movement (estimated by shortest path on the waypoint graph),
    sampling/imaging actions, calibration actions (for images), dropping
    samples (if store is full), and the final communication action.

    Assumptions:
    - The PDDL domain and problem files are valid and follow the structure
      provided in the examples.
    - The planner provides the task definition including initial state, goals,
      operators, and static facts.
    - The waypoint graph defined by 'can_traverse' predicates is static.
    - 'visible' and 'visible_from' predicates are static.
    - Rover capabilities ('equipped_for_...') are static.
    - Store ownership ('store_of') is static.
    - Camera properties ('on_board', 'supports', 'calibration_target') are static.
    - Initial locations of soil/rock samples ('at_soil_sample', 'at_rock_sample')
      are static (they are consumed, but their initial presence is a static
      condition for possibility).
    - The lander location ('at_lander') is static.
    - Action costs are uniform (cost 1).

    Heuristic Initialization:
    The constructor precomputes static information and data structures needed
    for efficient heuristic calculation:
    - Extracts all objects (rovers, waypoints, stores, cameras, modes, objectives, landers).
    - Parses static facts to build mappings for rover capabilities, store ownership,
      camera properties, objective visibility, calibration targets, lander location,
      communication waypoints (visible from lander), and initial sample locations.
    - For each rover, it builds its specific traversal graph based on 'can_traverse'
      facts and computes the All-Pairs Shortest Path (APSP) distances between all
      waypoints using the Floyd-Warshall algorithm. This allows fast lookup of
      movement costs between any two waypoints for a given rover.

    Step-By-Step Thinking for Computing Heuristic:
    1.  **Parse Current State:** Extract dynamic information from the current
        state (node.state): rover locations, store statuses (empty/full),
        camera calibration statuses, which rovers have which samples/images,
        and which data has already been communicated.
    2.  **Identify Unachieved Goals:** Determine the set of goal predicates
        from task.goals that are not present in the communicated data in the
        current state.
    3.  **Calculate Goal Costs:** For each unachieved goal predicate:
        a.  Initialize the cost for this goal to infinity.
        b.  **Soil/Rock Goal `(communicated_soil_data ?w)` or `(communicated_rock_data ?w)`:**
            i.  Check if the sample was initially present at waypoint `?w`. If not, the goal is impossible (cost infinity).
            ii. Calculate the minimum cost if the sample is *already held* by some suitable rover: Find all rovers that currently have the sample. For each such rover, find the minimum movement cost from its current location to any communication waypoint, plus the cost of the communicate action (1). The minimum over all such rovers is the cost for this case.
            iii. Calculate the minimum cost if the sample *needs to be collected*: Find all rovers equipped for the analysis type (soil/rock) with a store. For each such rover, calculate the cost to move to `?w`, plus the cost of dropping a sample if the store is full (1), plus the cost of the sample action (1), plus the minimum cost to move from `?w` to any communication waypoint, plus the cost of the communicate action (1). The minimum over all such rovers is the cost for this case.
            iv. The cost for this goal is the minimum of the costs from steps ii and iii. If both are infinity, the goal is impossible.
        c.  **Image Goal `(communicated_image_data ?o ?m)`:**
            i.  Calculate the minimum cost if the image is *already held* by some suitable rover: Find all rovers that currently have the image for objective `?o` and mode `?m`. For each such rover, find the minimum movement cost from its current location to any communication waypoint, plus the cost of the communicate action (1). The minimum over all such rovers is the cost for this case.
            ii. Calculate the minimum cost if the image *needs to be collected*: Find all rovers equipped for imaging with a camera that is on board and supports mode `?m`. For each such rover/camera pair, find all possible calibration waypoints for that camera and all possible image waypoints for objective `?o`. Calculate the cost to move from the rover's current location to a calibration waypoint, plus the calibrate action (1), plus the cost to move from the calibration waypoint to an image waypoint, plus the take_image action (1), plus the minimum cost to move from the image waypoint to any communication waypoint, plus the communicate action (1). The minimum over all combinations of suitable rovers/cameras/calibration_wps/image_wps is the cost for this case.
            iii. The cost for this goal is the minimum of the costs from steps i and ii. If both are infinity, the goal is impossible.
    4.  **Sum Goal Costs:** Sum the calculated costs for all unachieved goals.
    5.  **Return Heuristic Value:** If the sum is infinity (meaning at least one unachieved goal is impossible), return infinity. Otherwise, return the sum. If there are no unachieved goals, the sum is 0, which is correct for a goal state.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # --- Precomputation ---

        # 1. Extract all objects
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        # Helper to add objects from a fact tuple
        def add_objects_from_fact(fact_tuple):
            if not fact_tuple: return
            pred = fact_tuple[0]
            # List of predicates and the indices of objects of specific types
            pred_obj_indices = {
                'at': {1: 'rover', 2: 'waypoint'},
                'at_lander': {1: 'lander', 2: 'waypoint'},
                'can_traverse': {1: 'rover', 2: 'waypoint', 3: 'waypoint'},
                'equipped_for_soil_analysis': {1: 'rover'},
                'equipped_for_rock_analysis': {1: 'rover'},
                'equipped_for_imaging': {1: 'rover'},
                'empty': {1: 'store'},
                'full': {1: 'store'},
                'have_rock_analysis': {1: 'rover', 2: 'waypoint'},
                'have_soil_analysis': {1: 'rover', 2: 'waypoint'},
                'calibrated': {1: 'camera', 2: 'rover'},
                'supports': {1: 'camera', 2: 'mode'},
                'visible': {1: 'waypoint', 2: 'waypoint'},
                'have_image': {1: 'rover', 2: 'objective', 3: 'mode'},
                'communicated_soil_data': {1: 'waypoint'},
                'communicated_rock_data': {1: 'waypoint'},
                'communicated_image_data': {1: 'objective', 2: 'mode'},
                'at_soil_sample': {1: 'waypoint'},
                'at_rock_sample': {1: 'waypoint'},
                'visible_from': {1: 'objective', 2: 'waypoint'},
                'store_of': {1: 'store', 2: 'rover'},
                'calibration_target': {1: 'camera', 2: 'objective'},
                'on_board': {1: 'camera', 2: 'rover'},
            }
            if pred in pred_obj_indices:
                for idx, obj_type in pred_obj_indices[pred].items():
                    if len(fact_tuple) > idx:
                        obj_name = fact_tuple[idx]
                        if obj_type == 'rover': self.rovers.add(obj_name)
                        elif obj_type == 'waypoint': self.waypoints.add(obj_name)
                        elif obj_type == 'store': self.stores.add(obj_name)
                        elif obj_type == 'camera': self.cameras.add(obj_name)
                        elif obj_type == 'mode': self.modes.add(obj_name)
                        elif obj_type == 'objective': self.objectives.add(obj_name)
                        elif obj_type == 'lander': self.landers.add(obj_name)


        # Collect objects from initial state, goals, and static facts
        # Also iterate through operators to find objects mentioned there (e.g., navigate parameters)
        all_relevant_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for op in task.operators:
             # Operator name might contain object names? Unlikely in PDDL, but safe.
             # Example: (action navigate ?x - rover ?y - waypoint ?z - waypoint)
             # The parameters themselves are not part of the operator.name string
             # We already extract objects from preconditions/effects below.
             pass # all_relevant_facts.add(op.name)

             all_relevant_facts.update(op.preconditions)
             all_relevant_facts.update(op.add_effects)
             all_relevant_facts.update(op.del_effects) # Del effects might mention objects

        for fact_str in all_relevant_facts:
             fact = parse_fact(fact_str)
             if fact:
                 add_objects_from_fact(fact)


        self.waypoints = sorted(list(self.waypoints)) # Ensure consistent order for matrix
        self.wp_to_idx = {wp: i for i, wp in enumerate(self.waypoints)}
        self.idx_to_wp = {i: wp for i, wp in enumerate(self.waypoints)}
        num_waypoints = len(self.waypoints)

        # 2. Parse static facts into useful structures
        self.lander_location = None
        self.communication_wps = set()
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_of_rover = {} # store -> rover
        self.rover_has_store = {} # rover -> store
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint
        self.objective_image_wps = defaultdict(set) # objective -> {waypoint}
        self.camera_calibration_targets = {} # camera -> objective (target)
        self.calibration_target_wps = defaultdict(set) # camera -> {waypoint} (waypoint visible from target)
        self.rover_cameras = defaultdict(set) # rover -> {camera}
        self.camera_supports_mode = defaultdict(set) # camera -> {mode}

        rover_can_traverse = defaultdict(set) # rover -> {(wp_from, wp_to)}

        # Temporary storage for visible_from facts to process calibration targets later
        visible_from_facts = defaultdict(set) # objective -> {waypoint}

        for fact_str in self.static_facts:
            fact = parse_fact(fact_str)
            if not fact: continue
            pred = fact[0]
            if pred == 'at_lander':
                if len(fact) > 2: self.lander_location = fact[2]
            elif pred == 'equipped_for_soil_analysis':
                if len(fact) > 1: self.rover_capabilities[fact[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                if len(fact) > 1: self.rover_capabilities[fact[1]].add('rock')
            elif pred == 'equipped_for_imaging':
                if len(fact) > 1: self.rover_capabilities[fact[1]].add('imaging')
            elif pred == 'store_of':
                if len(fact) > 2:
                    self.store_of_rover[fact[1]] = fact[2]
                    self.rover_has_store[fact[2]] = fact[1]
            elif pred == 'visible':
                # Used for communication waypoints
                if self.lander_location and len(fact) > 2:
                    if fact[2] == self.lander_location:
                        self.communication_wps.add(fact[1])
                    if fact[1] == self.lander_location: # visible is symmetric
                        self.communication_wps.add(fact[2])
            elif pred == 'can_traverse':
                if len(fact) > 3: rover_can_traverse[fact[1]].add((fact[2], fact[3]))
            elif pred == 'visible_from':
                # Used for image waypoints (objective visible from waypoint)
                if len(fact) > 2:
                    self.objective_image_wps[fact[1]].add(fact[2])
                    # Store for calibration targets
                    visible_from_facts[fact[1]].add(fact[2])
            elif pred == 'calibration_target':
                if len(fact) > 2: self.camera_calibration_targets[fact[1]] = fact[2] # camera -> objective
            elif pred == 'on_board':
                if len(fact) > 2: self.rover_cameras[fact[2]].add(fact[1]) # rover -> camera
            elif pred == 'supports':
                if len(fact) > 2: self.camera_supports_mode[fact[1]].add(fact[2]) # camera -> mode

        # Link calibration targets to waypoints they are visible from
        # Iterate through cameras and their calibration targets
        for cam, target_obj in self.camera_calibration_targets.items():
             # The waypoints visible from the target_obj are calibration wps for this camera
             if target_obj in visible_from_facts:
                 self.calibration_target_wps[cam].update(visible_from_facts[target_obj])


        # Initial sample locations are also static for the purpose of possibility
        for fact_str in self.initial_state:
            fact = parse_fact(fact_str)
            if not fact: continue
            if fact[0] == 'at_soil_sample':
                if len(fact) > 1: self.initial_soil_samples.add(fact[1])
            elif fact[0] == 'at_rock_sample':
                if len(fact) > 1: self.initial_rock_samples.add(fact[1])

        # 3. Compute APSP for each rover
        self.rover_apsp = {} # rover -> {wp_from -> {wp_to -> distance}}

        for rover in self.rovers:
            dist = [[float('inf')] * num_waypoints for _ in range(num_waypoints)]
            for i in range(num_waypoints):
                dist[i][i] = 0

            if rover in rover_can_traverse:
                for wp_from, wp_to in rover_can_traverse[rover]:
                    if wp_from in self.wp_to_idx and wp_to in self.wp_to_idx:
                         u = self.wp_to_idx[wp_from]
                         v = self.wp_to_idx[wp_to]
                         dist[u][v] = 1 # Assuming unit cost for navigation

            # Floyd-Warshall
            for k in range(num_waypoints):
                for i in range(num_waypoints):
                    for j in range(num_waypoints):
                        if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

            # Store result in a dictionary format
            self.rover_apsp[rover] = {}
            for i in range(num_waypoints):
                self.rover_apsp[rover][self.idx_to_wp[i]] = {}
                for j in range(num_waypoints):
                    self.rover_apsp[rover][self.idx_to_wp[i]][self.idx_to_wp[j]] = dist[i][j]

    def __call__(self, node):
        state = node.state

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        camera_calibration_status = {} # camera -> bool
        rover_have_soil = set() # (rover, waypoint)
        rover_have_rock = set() # (rover, waypoint)
        rover_have_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        # Initialize calibration status to False for all cameras
        for cam in self.cameras:
             camera_calibration_status[cam] = False

        # Initialize store status to Empty for all stores (will be overwritten if Full)
        for store in self.stores:
             store_status[store] = 'empty'

        for fact_str in state:
            fact = parse_fact(fact_str)
            if not fact: continue
            pred = fact[0]
            if pred == 'at':
                if len(fact) > 2: rover_locations[fact[1]] = fact[2]
            elif pred == 'full':
                if len(fact) > 1: store_status[fact[1]] = 'full'
            elif pred == 'calibrated':
                if len(fact) > 1: camera_calibration_status[fact[1]] = True
            elif pred == 'have_soil_analysis':
                if len(fact) > 2: rover_have_soil.add((fact[1], fact[2]))
            elif pred == 'have_rock_analysis':
                if len(fact) > 2: rover_have_rock.add((fact[1], fact[2]))
            elif pred == 'have_image':
                if len(fact) > 3: rover_have_image.add((fact[1], fact[2], fact[3]))
            elif pred == 'communicated_soil_data':
                if len(fact) > 1: communicated_soil.add(fact[1])
            elif pred == 'communicated_rock_data':
                if len(fact) > 1: communicated_rock.add(fact[1])
            elif pred == 'communicated_image_data':
                if len(fact) > 2: communicated_image.add((fact[1], fact[2]))

        # --- Calculate Heuristic ---
        h_value = 0
        # Goals are only communication facts in this domain
        unachieved_comm_goals = set()
        for goal_str in self.goals:
             goal = parse_fact(goal_str)
             if not goal: continue
             if goal[0] == 'communicated_soil_data':
                 if len(goal) > 1 and goal[1] not in communicated_soil:
                     unachieved_comm_goals.add(('soil', goal[1]))
             elif goal[0] == 'communicated_rock_data':
                 if len(goal) > 1 and goal[1] not in communicated_rock:
                     unachieved_comm_goals.add(('rock', goal[1]))
             elif goal[0] == 'communicated_image_data':
                 if len(goal) > 2 and (goal[1], goal[2]) not in communicated_image:
                     unachieved_comm_goals.add(('image', goal[1], goal[2]))


        if not unachieved_comm_goals:
            return 0 # Goal state reached

        for goal in unachieved_comm_goals:
            goal_type = goal[0]
            cost_g = float('inf')

            if goal_type == 'soil':
                w = goal[1]
                # Check if sample was initially present
                if w not in self.initial_soil_samples:
                    return float('inf') # Impossible goal

                min_total_cost = float('inf')

                # Case 1: Sample is already held by some rover
                rovers_with_sample = {r for r, wp in rover_have_soil if wp == w}
                min_cost_comm_if_have = float('inf')
                if rovers_with_sample:
                    for r in rovers_with_sample:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown? Should not happen.
                        min_cost_comm_from_current = float('inf')
                        for comm_wp in self.communication_wps:
                             dist = self.rover_apsp.get(r, {}).get(current_wp, {}).get(comm_wp, float('inf'))
                             if dist < float('inf'):
                                 min_cost_comm_from_current = min(min_cost_comm_from_current, dist + 1) # +1 for communicate action
                        min_cost_comm_if_have = min(min_cost_comm_if_have, min_cost_comm_from_current)

                # Case 2: Sample needs to be collected
                min_cost_get_and_comm = float('inf')
                for r in self.rovers:
                    if 'soil' in self.rover_capabilities.get(r, set()):
                        store = self.rover_has_store.get(r)
                        if store is None: continue # Rover has no store? Should not happen based on domain.
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        cost_move_to_sample = self.rover_apsp.get(r, {}).get(current_wp, {}).get(w, float('inf'))
                        if cost_move_to_sample < float('inf'):
                            cost_sample_action = 1 # sample_soil action
                            cost_drop_if_full = 1 if store_status.get(store) == 'full' else 0 # drop action if needed

                            cost_get_sample = cost_move_to_sample + cost_drop_if_full + cost_sample_action

                            min_cost_comm_from_sample_wp = float('inf')
                            # After sampling, rover is at waypoint w
                            for comm_wp in self.communication_wps:
                                dist = self.rover_apsp.get(r, {}).get(w, {}).get(comm_wp, float('inf'))
                                if dist < float('inf'):
                                    min_cost_comm_from_sample_wp = min(min_cost_comm_from_sample_wp, dist + 1) # +1 for communicate action

                            if min_cost_comm_from_sample_wp < float('inf'):
                                total_cost_via_get_sample = cost_get_sample + min_cost_comm_from_sample_wp
                                min_cost_get_and_comm = min(min_cost_get_and_comm, total_cost_via_get_sample)

                cost_g = min(min_cost_comm_if_have, min_cost_get_and_comm)

            elif goal_type == 'rock':
                w = goal[1]
                # Check if sample was initially present
                if w not in self.initial_rock_samples:
                    return float('inf') # Impossible goal

                min_total_cost = float('inf')

                # Case 1: Sample is already held by some rover
                rovers_with_sample = {r for r, wp in rover_have_rock if wp == w}
                min_cost_comm_if_have = float('inf')
                if rovers_with_sample:
                    for r in rovers_with_sample:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue
                        min_cost_comm_from_current = float('inf')
                        for comm_wp in self.communication_wps:
                             dist = self.rover_apsp.get(r, {}).get(current_wp, {}).get(comm_wp, float('inf'))
                             if dist < float('inf'):
                                 min_cost_comm_from_current = min(min_cost_comm_from_current, dist + 1) # +1 for communicate action
                        min_cost_comm_if_have = min(min_cost_comm_if_have, min_cost_comm_from_current)

                # Case 2: Sample needs to be collected
                min_cost_get_and_comm = float('inf')
                for r in self.rovers:
                    if 'rock' in self.rover_capabilities.get(r, set()):
                        store = self.rover_has_store.get(r)
                        if store is None: continue
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        cost_move_to_sample = self.rover_apsp.get(r, {}).get(current_wp, {}).get(w, float('inf'))
                        if cost_move_to_sample < float('inf'):
                            cost_sample_action = 1 # sample_rock action
                            cost_drop_if_full = 1 if store_status.get(store) == 'full' else 0 # drop action if needed

                            cost_get_sample = cost_move_to_sample + cost_drop_if_full + cost_sample_action

                            min_cost_comm_from_sample_wp = float('inf')
                            # After sampling, rover is at waypoint w
                            for comm_wp in self.communication_wps:
                                dist = self.rover_apsp.get(r, {}).get(w, {}).get(comm_wp, float('inf'))
                                if dist < float('inf'):
                                    min_cost_comm_from_sample_wp = min(min_cost_comm_from_sample_wp, dist + 1) # +1 for communicate action

                            if min_cost_comm_from_sample_wp < float('inf'):
                                total_cost_via_get_sample = cost_get_sample + min_cost_comm_from_sample_wp
                                min_cost_get_and_comm = min(min_cost_get_and_comm, total_cost_via_get_sample)

                cost_g = min(min_cost_comm_if_have, min_cost_get_and_comm)

            elif goal_type == 'image':
                o, m = goal[1], goal[2]

                min_total_cost = float('inf')

                # Case 1: Image is already held by some rover
                rovers_with_image = {r for r, obj, md in rover_have_image if obj == o and md == m}
                min_cost_comm_if_have = float('inf')
                if rovers_with_image:
                    for r in rovers_with_image:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue
                        min_cost_comm_from_current = float('inf')
                        for comm_wp in self.communication_wps:
                             dist = self.rover_apsp.get(r, {}).get(current_wp, {}).get(comm_wp, float('inf'))
                             if dist < float('inf'):
                                 min_cost_comm_from_current = min(min_cost_comm_from_current, dist + 1) # +1 for communicate action
                        min_cost_comm_if_have = min(min_cost_comm_if_have, min_cost_comm_from_current)

                # Case 2: Image needs to be collected
                min_cost_get_and_comm = float('inf')
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for cam in self.rover_cameras.get(r, set()):
                            if m in self.camera_supports_mode.get(cam, set()):
                                cal_wps = self.calibration_target_wps.get(cam, set())
                                if not cal_wps: continue # Camera has no calibration target visible from anywhere
                                img_wps = self.objective_image_wps.get(o, set())
                                if not img_wps: continue # Objective has no visible_from wps

                                current_wp = rover_locations.get(r)
                                if current_wp is None: continue

                                min_cost_get_image_for_rover_cam = float('inf')
                                best_img_wp_for_get_image = None # Track the image wp that achieved the min cost

                                for cal_wp in cal_wps:
                                    cost_move_to_cal = self.rover_apsp.get(r, {}).get(current_wp, {}).get(cal_wp, float('inf'))
                                    if cost_move_to_cal < float('inf'):
                                        cost_calibrate_action = 1 # calibrate action
                                        for img_wp in img_wps:
                                            cost_move_cal_to_img = self.rover_apsp.get(r, {}).get(cal_wp, {}).get(img_wp, float('inf'))
                                            if cost_move_cal_to_img < float('inf'):
                                                cost_take_image_action = 1 # take_image action
                                                total_cost_path_image = cost_move_to_cal + cost_calibrate_action + cost_move_cal_to_img + cost_take_image_action
                                                if total_cost_path_image < min_cost_get_image_for_rover_cam:
                                                     min_cost_get_image_for_rover_cam = total_cost_path_image
                                                     best_img_wp_for_get_image = img_wp # Store the wp where image was taken

                                if min_cost_get_image_for_rover_cam < float('inf'):
                                    # Now calculate communication cost from the image waypoint
                                    min_cost_comm_from_img_wp = float('inf')
                                    # The rover is at best_img_wp_for_get_image after taking the image
                                    # Need to find the best communication wp from there
                                    if best_img_wp_for_get_image is not None:
                                         for comm_wp in self.communication_wps:
                                             dist = self.rover_apsp.get(r, {}).get(best_img_wp_for_get_image, {}).get(comm_wp, float('inf'))
                                             if dist < float('inf'):
                                                 min_cost_comm_from_img_wp = min(min_cost_comm_from_img_wp, dist + 1) # +1 for communicate action

                                    if min_cost_comm_from_img_wp < float('inf'):
                                         total_cost_via_get_image = min_cost_get_image_for_rover_cam + min_cost_comm_from_img_wp
                                         min_cost_get_and_comm = min(min_cost_get_and_comm, total_cost_via_get_image)


                cost_g = min(min_cost_comm_if_have, min_cost_get_and_comm)


            # Add cost of this goal to total heuristic
            if cost_g == float('inf'):
                # If any unachieved goal is impossible, the state is a dead end
                return float('inf')
            h_value += cost_g

        return h_value
