# Import necessary modules
from fnmatch import fnmatch
import math # For float('inf')
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         return []
    # Get content inside parentheses
    content = fact[1:-1].strip() # strip leading/trailing whitespace
    if not content: # Handle empty fact like "()"
        return []
    return content.split() # Split by whitespace

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    all goal conditions. It decomposes the problem into individual goals
    (communicating soil, rock, or image data) and sums the estimated costs
    for each unachieved goal. The cost for each goal is estimated based on
    the actions needed to acquire the data (sample/image) and communicate it,
    including necessary navigation and resource preparation (store empty, camera calibrated).
    Navigation costs are estimated using precomputed shortest paths for each rover.

    # Assumptions
    - Each unachieved goal is considered independently (additive heuristic).
    - For sampling goals, a rover needs to navigate to the sample location, sample,
      navigate to a communication location, and communicate. A drop action is added
      if the rover's store is full *before* sampling.
    - For imaging goals, a rover needs to navigate to a calibration location, calibrate,
      navigate to an image location, take the image, navigate to a communication location,
      and communicate. Calibration is assumed needed before each image.
    - The heuristic finds the minimum cost among suitable rovers and relevant locations
      (calibration spots, image spots, communication spots) for each goal.
    - Unreachable goals or locations result in an infinite heuristic value.
    - Samples (soil/rock) are assumed to be available at their initial locations
      unless they have been successfully sampled and are currently held by a rover.
      If a sample was initially present but is neither at its location nor held,
      it is considered lost/impossible to collect again.

    # Heuristic Initialization
    - Parses static facts to identify rover capabilities, store ownership, camera properties,
      calibration targets, lander location, and waypoint visibility.
    - Builds navigation graphs for each rover based on `can_traverse` facts.
    - Precomputes shortest path distances between all pairs of waypoints for each rover using BFS.
    - Identifies sets of waypoints relevant for communication (visible from lander),
      imaging (visible from objectives), and calibration (visible from calibration targets).
    - Stores initial locations of soil and rock samples to check if they were originally present.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. For each goal fact in the task's goals:
        a. If the goal fact is already present in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Check if `(have_soil_analysis ?r ?w)` exists for any rover `?r` in the state.
            ii. If yes: Add 1 (communicate action). Find the rover `?r` holding the sample. Add the minimum navigation cost for `?r` from its current location to any waypoint visible from the lander.
            iii. If no: Check if `(at_soil_sample ?w)` is in the current state. If not, and `?w` was in the initial state's soil samples, the sample is lost and the goal is impossible (return infinity). If `?w` was never in the initial state's soil samples, the goal is also impossible (return infinity). Otherwise (sample is on the ground), find all rovers equipped for soil analysis. For each such rover `?r`:
                - Calculate cost to sample: Add 1 (sample_soil action). Add navigation cost for `?r` from its current location to `?w`. If `?r`'s store is full, add 1 (drop action).
                - Calculate cost to communicate: Add 1 (communicate action). Add navigation cost for `?r` from `?w` (where sampling occurs) to any waypoint visible from the lander.
                - The cost for this goal via this rover is the sum of sample cost and communicate cost.
                - Take the minimum cost over all suitable rovers. Add this minimum cost to the total.
        c. If the goal is `(communicated_rock_data ?w)`: Follow a similar process as for soil data, using rock-equipped rovers and rock samples. Apply the same logic for sample availability.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Check if `(have_image ?r ?o ?m)` exists for any rover `?r` in the state.
            ii. If yes: Add 1 (communicate action). Find the rover `?r` holding the image. Add the minimum navigation cost for `?r` from its current location to any waypoint visible from the lander.
            iii. If no: Find all rovers equipped for imaging that have a camera supporting mode `?m` and a calibration target. For each such rover `?r` and camera `?i`:
                - Find the calibration target `?t` for `?i`.
                - Find the set of waypoints `CalSpots` visible from `?t`.
                - Find the set of waypoints `ImageSpots` visible from `?o`.
                - Find the set of waypoints `CommSpots` visible from the lander.
                - If any of these spot sets are empty, this rover/camera cannot achieve the goal via this path; continue to the next suitable pair.
                - Calculate the minimum navigation cost for `?r` from its current location to any `w` in `CalSpots`, then from `w` to any `p` in `ImageSpots`, then from `p` to any `c` in `CommSpots`. This is `min_{w \in CalSpots, p \in ImageSpots, c \in CommSpots} (dist(current, w) + dist(w, p) + dist(p, c))`.
                - If this minimum navigation cost is infinite, this path is impossible; continue.
                - Add this minimum navigation cost + 1 (calibrate) + 1 (take_image) + 1 (communicate) to get the cost for this goal via this rover/camera.
                - Take the minimum cost over all suitable rover/camera pairs. Add this minimum cost to the total.
    3. Return the total heuristic cost. If any necessary location is unreachable for all suitable rovers, or if a sample is lost, the cost for that goal will be infinite, and the total heuristic will be infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Precompute Navigation Distances ---
        self.rover_distances = {} # rover -> {wp_from -> {wp_to -> dist}}
        waypoints = set()
        can_traverse_map = {} # rover -> {wp_from -> {wp_to}}

        # Collect all waypoints mentioned in static, initial, and goal facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            # Waypoints can appear in many predicates
            if parts[0] in ['at_lander', 'visible', 'can_traverse', 'visible_from', 'at']:
                 for part in parts[1:]:
                     # Simple check: if it starts with 'waypoint' or 'general' (lander)
                     if part.startswith('waypoint') or part == 'general':
                         waypoints.add(part)

        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in ['at', 'at_lander', 'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 for part in parts[1:]:
                      if part.startswith('waypoint') or part == 'general':
                         waypoints.add(part)

        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             if parts[0] in ['communicated_soil_data', 'communicated_rock_data']:
                 if len(parts) > 1 and parts[1].startswith('waypoint'):
                     waypoints.add(parts[1])


        all_waypoints = list(waypoints)

        # Build can_traverse map
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'can_traverse':
                if len(parts) == 4:
                    rover, wp_from, wp_to = parts[1:]
                    if rover not in can_traverse_map:
                        can_traverse_map[rover] = {}
                    if wp_from not in can_traverse_map[rover]:
                        can_traverse_map[rover][wp_from] = set()
                    can_traverse_map[rover][wp_from].add(wp_to)

        # Compute shortest paths for each rover using BFS
        for rover, graph in can_traverse_map.items():
            self.rover_distances[rover] = {}
            for start_wp in all_waypoints:
                self.rover_distances[rover][start_wp] = {}
                q = [(start_wp, 0)]
                visited = {start_wp}
                self.rover_distances[rover][start_wp][start_wp] = 0

                head = 0
                while head < len(q):
                    (current_wp, dist) = q[head]
                    head += 1

                    if current_wp in graph:
                        for neighbor in graph[current_wp]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                self.rover_distances[rover][start_wp][neighbor] = dist + 1
                                q.append((neighbor, dist + 1))

            # Fill unreachable waypoints with infinity
            for start_wp in all_waypoints:
                 for end_wp in all_waypoints:
                     if end_wp not in self.rover_distances[rover][start_wp]:
                         self.rover_distances[rover][start_wp][end_wp] = math.inf


        # --- Extract Static Information ---
        self.lander_location = None
        self.comm_spots = set() # Waypoints visible from lander

        self.objective_image_spots = {} # objective -> {waypoint}
        self.camera_cal_targets = {} # camera -> objective
        self.cal_target_cal_spots = {} # objective (calibration target) -> {waypoint}

        self.rover_equipment = {} # rover -> {equipment_type} ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store (rover -> store_name)
        self.rover_cameras = {} # rover -> {camera}
        self.camera_modes = {} # camera -> {mode}

        # Need initial soil/rock sample locations (they are consumed)
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        # First pass to get calibration targets and lander location
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'at_lander':
                if len(parts) > 2: self.lander_location = parts[2]
            elif parts[0] == 'calibration_target':
                if len(parts) > 2: self.camera_cal_targets[parts[1]] = parts[2]

        # Second pass to get visible_from, equipment, etc.
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'visible':
                 if len(parts) > 2 and parts[2] == self.lander_location:
                     self.comm_spots.add(parts[1])
            elif parts[0] == 'visible_from':
                if len(parts) > 2:
                    obj, wp = parts[1:]
                    # Image spots
                    if obj not in self.objective_image_spots:
                        self.objective_image_spots[obj] = set()
                    self.objective_image_spots[obj].add(wp)
                    # Calibration spots (if obj is a calibration target)
                    if obj in self.camera_cal_targets.values():
                        if obj not in self.cal_target_cal_spots:
                            self.cal_target_cal_spots[obj] = set()
                        self.cal_target_cal_spots[obj].add(wp)
            elif parts[0].startswith('equipped_for_'):
                if len(parts) > 1:
                    equip_type = parts[0].split('_')[-2] # soil or rock or imaging
                    rover = parts[1]
                    if rover not in self.rover_equipment:
                        self.rover_equipment[rover] = set()
                    self.rover_equipment[rover].add(equip_type)
            elif parts[0] == 'store_of':
                if len(parts) > 2: self.rover_stores[parts[2]] = parts[1] # rover -> store_name
            elif parts[0] == 'on_board':
                if len(parts) > 2:
                    camera, rover = parts[1:]
                    if rover not in self.rover_cameras:
                        self.rover_cameras[rover] = set()
                    self.rover_cameras[rover].add(camera)
            elif parts[0] == 'supports':
                if len(parts) > 2:
                    camera, mode = parts[1:]
                    if camera not in self.camera_modes:
                        self.camera_modes[camera] = set()
                    self.camera_modes[camera].add(mode)

        # Initial sample locations from initial state
        for fact in initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'at_soil_sample':
                if len(parts) > 1: self.initial_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                if len(parts) > 1: self.initial_rock_samples.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0.0 # Use float for infinity comparison

        # Helper to get rover location
        def get_rover_location(state, rover):
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == 'at' and len(parts) > 2 and parts[1] == rover:
                    return parts[2]
            return None # Should not happen for active rovers

        # Helper to check if fact is in state
        def is_in_state(state, predicate, *args):
            fact_str = "(" + predicate + (" " + " ".join(args) if args else "") + ")"
            return fact_str in state

        # Helper to find minimum distance for a rover from start_wp to any in target_wps
        def min_dist(rover, start_wp, target_wps):
            if not target_wps: return math.inf # Cannot reach an empty set of targets
            if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
                 # Rover cannot navigate from start_wp (e.g., start_wp not in graph)
                 return math.inf
            min_d = math.inf
            for target_wp in target_wps:
                if target_wp in self.rover_distances[rover][start_wp]:
                    min_d = min(min_d, self.rover_distances[rover][start_wp][target_wp])
            return min_d


        # Process each goal
        for goal in self.goals:
            if is_in_state(state, *get_parts(goal)):
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals
            predicate, *args = parts

            if predicate == 'communicated_soil_data':
                if len(args) < 1: continue # Malformed goal
                waypoint = args[0]

                # Check if sample is already collected
                have_sample = False
                sample_rover = None
                for fact in state:
                    fact_parts = get_parts(fact)
                    if fact_parts and fact_parts[0] == 'have_soil_analysis' and len(fact_parts) > 2 and fact_parts[2] == waypoint:
                        have_sample = True
                        sample_rover = fact_parts[1]
                        break

                if have_sample:
                    # Need to communicate the sample
                    total_cost += 1 # communicate action
                    # Need rover at communication spot
                    current_loc = get_rover_location(state, sample_rover)
                    if current_loc is None: return math.inf # Rover location unknown
                    nav_cost = min_dist(sample_rover, current_loc, self.comm_spots)
                    if nav_cost == math.inf: return math.inf # Cannot reach communication spot
                    total_cost += nav_cost
                else:
                    # Need to sample and then communicate
                    # Check if sample is available on the ground
                    sample_on_ground = is_in_state(state, 'at_soil_sample', waypoint)

                    # If sample is not on the ground and not held, check if it was initially present.
                    # If it was initially present but is now neither on ground nor held, it's lost.
                    # If sample was never initially present, it's impossible.
                    if waypoint not in self.initial_soil_samples or (not sample_on_ground and waypoint in self.initial_soil_samples):
                         return math.inf # Sample never existed or is lost

                    # Find a suitable rover (equipped for soil)
                    suitable_rovers = [r for r, equip in self.rover_equipment.items() if 'soil' in equip]
                    if not suitable_rovers: return math.inf # Cannot achieve goal

                    min_goal_cost = math.inf

                    for rover in suitable_rovers:
                        current_loc = get_rover_location(state, rover)
                        if current_loc is None: continue # Should not happen

                        store = self.rover_stores.get(rover) # Get store for this rover
                        if store is None: continue # Rover has no store? (Problematic instance?)

                        # Cost to sample
                        sample_cost = 1 # sample_soil action
                        nav_to_sample = min_dist(rover, current_loc, {waypoint})
                        if nav_to_sample == math.inf: continue # Cannot reach sample location
                        sample_cost += nav_to_sample

                        # Check if store is full, need drop
                        if is_in_state(state, 'full', store):
                             sample_cost += 1 # drop action

                        # Cost to communicate (assuming sampling happens first at 'waypoint')
                        comm_cost = 1 # communicate action
                        nav_to_comm = min_dist(rover, waypoint, self.comm_spots)
                        if nav_to_comm == math.inf: continue # Cannot reach communication spot

                        comm_cost += nav_to_comm

                        # Total cost for this rover for this goal
                        rover_goal_cost = sample_cost + comm_cost
                        min_goal_cost = min(min_goal_cost, rover_goal_cost)

                    if min_goal_cost == math.inf: return math.inf # No suitable rover can achieve this

                    total_cost += min_goal_cost

            elif predicate == 'communicated_rock_data':
                if len(args) < 1: continue # Malformed goal
                waypoint = args[0]

                # Check if sample is already collected
                have_sample = False
                sample_rover = None
                for fact in state:
                    fact_parts = get_parts(fact)
                    if fact_parts and fact_parts[0] == 'have_rock_analysis' and len(fact_parts) > 2 and fact_parts[2] == waypoint:
                        have_sample = True
                        sample_rover = fact_parts[1]
                        break

                if have_sample:
                    # Need to communicate the sample
                    total_cost += 1 # communicate action
                    # Need rover at communication spot
                    current_loc = get_rover_location(state, sample_rover)
                    if current_loc is None: return math.inf # Rover location unknown
                    nav_cost = min_dist(sample_rover, current_loc, self.comm_spots)
                    if nav_cost == math.inf: return math.inf # Cannot reach communication spot
                    total_cost += nav_cost
                else:
                    # Need to sample and then communicate
                    # Check if sample is available on the ground
                    sample_on_ground = is_in_state(state, 'at_rock_sample', waypoint)

                    # If sample is not on the ground and not held, check if it was initially present.
                    # If it was initially present but is now neither on ground nor held, it's lost.
                    # If sample was never initially present, it's impossible.
                    if waypoint not in self.initial_rock_samples or (not sample_on_ground and waypoint in self.initial_rock_samples):
                         return math.inf # Sample never existed or is lost

                    # Find a suitable rover (equipped for rock)
                    suitable_rovers = [r for r, equip in self.rover_equipment.items() if 'rock' in equip]
                    if not suitable_rovers: return math.inf # Cannot achieve goal

                    min_goal_cost = math.inf

                    for rover in suitable_rovers:
                        current_loc = get_rover_location(state, rover)
                        if current_loc is None: continue # Should not happen

                        store = self.rover_stores.get(rover) # Get store for this rover
                        if store is None: continue # Rover has no store?

                        # Cost to sample
                        sample_cost = 1 # sample_rock action
                        nav_to_sample = min_dist(rover, current_loc, {waypoint})
                        if nav_to_sample == math.inf: continue # Cannot reach sample location
                        sample_cost += nav_to_sample

                        # Check if store is full, need drop
                        if is_in_state(state, 'full', store):
                             sample_cost += 1 # drop action

                        # Cost to communicate (assuming sampling happens first at 'waypoint')
                        comm_cost = 1 # communicate action
                        nav_to_comm = min_dist(rover, waypoint, self.comm_spots)
                        if nav_to_comm == math.inf: continue # Cannot reach communication spot

                        comm_cost += nav_to_comm

                        # Total cost for this rover for this goal
                        rover_goal_cost = sample_cost + comm_cost
                        min_goal_cost = min(min_goal_cost, rover_goal_cost)

                    if min_goal_cost == math.inf: return math.inf # No suitable rover can achieve this
                    total_cost += min_goal_cost

            elif predicate == 'communicated_image_data':
                if len(args) < 2: continue # Malformed goal
                objective, mode = args[0], args[1]

                # Check if image is already taken
                have_image = False
                image_rover = None
                for fact in state:
                    fact_parts = get_parts(fact)
                    if fact_parts and fact_parts[0] == 'have_image' and len(fact_parts) > 3 and fact_parts[2] == objective and fact_parts[3] == mode:
                        have_image = True
                        image_rover = fact_parts[1]
                        break

                if have_image:
                    # Need to communicate the image
                    total_cost += 1 # communicate action
                    # Need rover at communication spot
                    current_loc = get_rover_location(state, image_rover)
                    if current_loc is None: return math.inf # Rover location unknown
                    nav_cost = min_dist(image_rover, current_loc, self.comm_spots)
                    if nav_cost == math.inf: return math.inf # Cannot reach communication spot
                    total_cost += nav_cost
                else:
                    # Need to take image and then communicate
                    # Find a suitable rover (equipped for imaging) with a suitable camera/mode
                    suitable_rover_camera_pairs = []
                    for rover, equip in self.rover_equipment.items():
                        if 'imaging' in equip:
                            # Check if rover has a camera supporting the mode
                            for camera in self.rover_cameras.get(rover, []):
                                if mode in self.camera_modes.get(camera, []):
                                    # Check if camera has a calibration target (needed for calibrate action)
                                    if camera in self.camera_cal_targets:
                                         suitable_rover_camera_pairs.append((rover, camera))
                                    # If camera has no calibration target, it cannot be calibrated, thus cannot take image.
                                    # We only consider cameras with calibration targets here.

                    if not suitable_rover_camera_pairs: return math.inf # Cannot achieve goal

                    min_goal_cost = math.inf

                    image_spots = self.objective_image_spots.get(objective, set())
                    if not image_spots: return math.inf # Cannot take image of this objective

                    if not self.comm_spots: return math.inf # Cannot communicate

                    for rover, camera in suitable_rover_camera_pairs:
                        current_loc = get_rover_location(state, rover)
                        if current_loc is None: continue # Should not happen

                        cal_target = self.camera_cal_targets.get(camera)
                        if cal_target is None: continue # Camera has no calibration target (should be caught earlier?)

                        cal_spots = self.cal_target_cal_spots.get(cal_target, set())
                        if not cal_spots: continue # Cannot calibrate at any known spot

                        # Cost breakdown:
                        # Navigate to calibration spot + Calibrate + Navigate to image spot + Take Image + Navigate to communication spot + Communicate
                        # = min_{w in CalSpots, p in ImageSpots, c in CommSpots} (dist(current, w) + 1 + dist(w, p) + 1 + dist(p, c) + 1)
                        # = min_{w, p, c} (dist(current, w) + dist(w, p) + dist(p, c)) + 3

                        min_nav_cost_sequence = math.inf

                        # Iterate through all combinations of spots to find the minimum navigation cost sequence
                        for w in cal_spots:
                            dist_curr_w = self.rover_distances[rover][current_loc].get(w, math.inf)
                            if dist_curr_w == math.inf: continue
                            for p in image_spots:
                                dist_w_p = self.rover_distances[rover][w].get(p, math.inf)
                                if dist_w_p == math.inf: continue
                                # Find min dist from p to any comm spot
                                dist_p_comm = min_dist(rover, p, self.comm_spots)
                                if dist_p_comm == math.inf: continue
                                min_nav_cost_sequence = min(min_nav_cost_sequence, dist_curr_w + dist_w_p + dist_p_comm)

                        if min_nav_cost_sequence == math.inf: continue # No path found for this rover/camera

                        # Total cost for this rover/camera for this goal
                        # Navigation (sequence) + Calibrate + Take Image + Communicate
                        rover_goal_cost = min_nav_cost_sequence + 3 # 1 for calibrate, 1 for take_image, 1 for communicate
                        min_goal_cost = min(min_goal_cost, rover_goal_cost)

                    if min_goal_cost == math.inf: return math.inf # No suitable rover/camera can achieve this

                    total_cost += min_goal_cost

            # Add handling for other potential goal types if they existed

        return total_cost
