from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Helper to split PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal condition independently.
    For each unachieved goal (communicating soil data, rock data, or image data),
    it calculates the minimum estimated cost across all capable rovers (and cameras for image goals).
    The estimated cost for a goal involves the actions needed (sample/take_image/calibrate, communicate)
    and the navigation costs between relevant waypoints (sample/image/calibration location, communication location).
    Navigation costs are precomputed shortest path distances for each rover.
    The heuristic is non-admissible as it ignores negative interactions (e.g., resource contention like stores or camera calibration state across different image goals)
    and simplifies sequential dependencies (e.g., assuming direct travel between required waypoints for image goals).

    Assumptions:
    - All goal conditions are reachable from the initial state.
    - Soil/rock samples, once taken from the ground, are held by a rover until communicated. Dropped samples are not re-samplable from the ground.
    - Camera calibration is required before each image taken with that camera.
    - The heuristic calculates costs for each goal independently and sums them up.

    Heuristic Initialization:
    - Parses static facts to identify:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Rover-camera mapping and camera modes.
        - Camera calibration targets.
        - Waypoints visible from objectives and calibration targets.
        - Waypoints visible from the lander location (communication waypoints).
        - Rover navigation graphs (based on can_traverse).
        - All relevant waypoints and rovers.
    - Stores initial locations of soil and rock samples from the initial state.
    - Precomputes all-pairs shortest path distances for each rover based on its navigation graph using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total_cost to 0.
    2. Parse dynamic facts from the current state (rover locations, store status, collected data, calibrated cameras, remaining samples on ground).
    3. For each goal condition in the task's goals:
        a. If the goal is already satisfied in the current state, continue to the next goal.
        b. If the goal is (communicated_soil_data ?w) or (communicated_rock_data ?w):
            i. Identify the waypoint ?w, predicate types (soil/rock), equipped rovers, and initial/current sample locations.
            ii. Initialize min_goal_cost for this goal to infinity.
            iii. Iterate through all relevant equipped rovers:
                - Get the rover's current location.
                - Calculate the estimated cost for this rover to get the sample:
                    - If the rover currently has the required sample data, sample_cost = 0.
                    - Else if the sample is still on the ground at ?w:
                        - Cost = (distance from rover's current location to ?w) + 1 (sample action) + (1 if rover's store is full).
                    - Else (sample gone from ground and this rover doesn't have it), sample_cost remains infinity (impossible for this rover).
                - If sample_cost is not infinity:
                    - Calculate communication cost:
                        - Determine the rover's location after sampling (if applicable) or its current location.
                        - Cost = (minimum distance from that location to any communication waypoint) + 1 (communicate action).
                    - Total cost for this rover = sample_cost + communication_cost.
                    - Update min_goal_cost = min(min_goal_cost, Total cost for this rover).
            iv. If min_goal_cost is not infinity, add it to total_cost.
        c. If the goal is (communicated_image_data ?o ?m):
            i. Identify the objective ?o and mode ?m.
            ii. Initialize min_goal_cost for this goal to infinity.
            iii. Iterate through all rovers equipped for imaging:
                - Iterate through all cameras on board the rover that support mode ?m:
                    - Get the rover's current location.
                    - Calculate the estimated cost for this rover/camera to get the image:
                        - If the rover currently has the required image data, image_cost = 0, and the location after image is the current location.
                        - Else:
                            - Find calibration target and relevant calibration waypoints for the camera.
                            - Find relevant image waypoints for the objective.
                            - If no suitable cal or image waypoints, image_cost remains infinity.
                            - Otherwise, calculate the minimum cost for the sequence: move_to_cal -> calibrate -> move_to_img -> take_image.
                            - Cost = min over cal_wp, img_wp: (dist(curr, cal_wp) + 1 + dist(cal_wp, img_wp) + 1).
                            - The location after image is the chosen img_wp.
                    - If image_cost is not infinity:
                        - Calculate communication cost:
                            - Cost = (minimum distance from the location after image to any communication waypoint) + 1 (communicate action).
                        - Total cost for this rover/camera = image_cost + communication_cost.
                        - Update min_goal_cost = min(min_goal_cost, Total cost for this rover/camera).
            iv. If min_goal_cost is not infinity, add it to total_cost.
    4. After processing all goals, if total_cost is 0 but the state is not a goal state, return 1 (to ensure non-goal states have a positive heuristic). Otherwise, return total_cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial samples

        # Data structures for static information
        self.lander_location = None
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.rover_store = {} # rover -> store
        self.rover_cameras = {} # rover -> {camera}
        self.camera_modes = {} # camera -> {mode}
        self.camera_cal_target = {} # camera -> objective (calibration target)
        self.obj_visible_from_wp = {} # objective -> {waypoint}
        self.target_visible_from_wp = {} # target -> {waypoint}
        self.comm_wps = set() # waypoints visible from lander
        self.rover_graphs = {} # rover -> {wp -> {neighbor_wp}}
        self.rover_dist = {} # (rover, wp_from, wp_to) -> distance
        self.initial_soil_samples = set() # {waypoint}
        self.initial_rock_samples = set() # {waypoint}
        self.all_waypoints = set() # {waypoint}
        self.all_rovers = set() # {rover}

        # First pass: Collect basic info and all waypoints/rovers from static facts
        lander_wp = None
        visible_from_facts = []
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
                lander_wp = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.soil_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rock_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.imaging_rovers.add(parts[1])
            elif parts[0] == 'store_of':
                self.rover_store[parts[2]] = parts[1]
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == 'calibration_target':
                self.camera_cal_target[parts[1]] = parts[2]
            elif parts[0] == 'visible_from':
                visible_from_facts.append(fact) # Store for later processing
            elif parts[0] == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)
            elif parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.all_rovers.add(rover)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
                self.rover_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)

        # Second pass: Process visible_from now that calibration targets are known
        for fact in visible_from_facts:
             parts = get_parts(fact)
             obj_or_target, wp = parts[1], parts[2]
             is_cal_target = any(target == obj_or_target for target in self.camera_cal_target.values())
             if is_cal_target:
                  self.target_visible_from_wp.setdefault(obj_or_target, set()).add(wp)
             else:
                  self.obj_visible_from_wp.setdefault(obj_or_target, set()).add(wp)
             self.all_waypoints.add(wp) # Ensure these waypoints are included

        # Add waypoints from initial samples and initial rover locations
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at_soil_sample':
                 self.initial_soil_samples.add(parts[1])
                 self.all_waypoints.add(parts[1])
             elif parts[0] == 'at_rock_sample':
                 self.initial_rock_samples.add(parts[1])
                 self.all_waypoints.add(parts[1])
             elif parts[0] == 'at' and parts[1] in self.all_rovers: # Add initial rover locations
                 self.all_waypoints.add(parts[2])

        # Identify communication waypoints based on lander location and visible facts
        if lander_wp:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if parts[0] == 'visible':
                     wp1, wp2 = parts[1], parts[2]
                     if wp2 == lander_wp:
                         self.comm_wps.add(wp1)
                         self.all_waypoints.add(wp1) # Ensure comm wps are in all_waypoints

        # Compute shortest paths for each rover
        for rover in self.all_rovers:
            graph = self.rover_graphs.get(rover, {})
            # Ensure all waypoints are in the graph keys, even if they have no outgoing edges
            for wp in self.all_waypoints:
                 graph.setdefault(wp, set())

            for start_wp in list(self.all_waypoints): # Iterate over a copy
                distances = self._bfs(graph, start_wp)
                for end_wp in self.all_waypoints:
                    if end_wp in distances:
                        self.rover_dist[(rover, start_wp, end_wp)] = distances[end_wp]
                    else:
                        self.rover_dist[(rover, start_wp, end_wp)] = float('inf')


    def _bfs(self, graph, start_node):
        """Performs BFS to find shortest distances from start_node."""
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            curr_node = queue.popleft()

            if curr_node in graph:
                for neighbor in graph[curr_node]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[curr_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, rover, wp_from, wp_to):
        """Gets precomputed shortest distance for a rover between waypoints."""
        return self.rover_dist.get((rover, wp_from, wp_to), float('inf'))

    def __call__(self, node):
        state = node.state

        # Extract dynamic facts from the current state
        rover_locations = {} # rover -> waypoint
        full_stores = set() # {store}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}
        current_soil_samples = set() # {waypoint}
        current_rock_samples = set() # {waypoint}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.all_rovers: # Check if it's a rover
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'full':
                full_stores.add(parts[1])
            elif parts[0] == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            elif parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))
            elif parts[0] == 'at_soil_sample':
                 current_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 current_rock_samples.add(parts[1])

        total_cost = 0

        # Handle Soil and Rock Goals
        for goal in self.goals:
            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                wp = parts[1]
                if wp in communicated_soil:
                    continue

                # Check if this goal was even possible initially
                if wp not in self.initial_soil_samples:
                     continue # Impossible goal, ignore

                min_goal_cost = float('inf')

                for rover in self.soil_rovers:
                    if rover not in rover_locations: continue # Should not happen

                    curr_wp = rover_locations[rover]

                    # Cost to get sample
                    sample_cost = float('inf')
                    wp_after_sample = None

                    if (rover, wp) in have_soil:
                        sample_cost = 0
                        wp_after_sample = curr_wp
                    elif wp in current_soil_samples:
                        move_to_sample_cost = self.get_distance(rover, curr_wp, wp)
                        if move_to_sample_cost != float('inf'):
                            sample_cost = move_to_sample_cost + 1 # sample action
                            store = self.rover_store.get(rover)
                            if store and store in full_stores:
                                sample_cost += 1 # drop action needed first
                            wp_after_sample = wp # Location after sampling
                    # If (rover, wp) not in have_soil and wp not in current_soil_samples, sample is gone and this rover doesn't have it. Impossible for this rover.

                    if sample_cost == float('inf'): continue # Cannot get sample with this rover

                    # Communication cost
                    min_move_to_comm = float('inf')
                    wp_before_comm = wp_after_sample # Location after sample/having sample

                    for comm_wp in self.comm_wps:
                         min_move_to_comm = min(min_move_to_comm, self.get_distance(rover, wp_before_comm, comm_wp))

                    if min_move_to_comm == float('inf'): continue # Cannot reach any communication location

                    comm_cost = min_move_to_comm + 1 # communicate action

                    # Total cost for this rover
                    current_rover_goal_cost = sample_cost + comm_cost
                    min_goal_cost = min(min_goal_cost, current_rover_goal_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost

            elif goal_type == 'communicated_rock_data':
                wp = parts[1]
                if wp in communicated_rock:
                    continue

                # Check if this goal was even possible initially
                if wp not in self.initial_rock_samples:
                     continue # Impossible goal, ignore

                min_goal_cost = float('inf')

                for rover in self.rock_rovers:
                    if rover not in rover_locations: continue
                    curr_wp = rover_locations[rover]

                    # Cost to get sample
                    sample_cost = float('inf')
                    wp_after_sample = None

                    if (rover, wp) in have_rock:
                        sample_cost = 0
                        wp_after_sample = curr_wp
                    elif wp in current_rock_samples:
                        move_to_sample_cost = self.get_distance(rover, curr_wp, wp)
                        if move_to_sample_cost != float('inf'):
                            sample_cost = move_to_sample_cost + 1 # sample action
                            store = self.rover_store.get(rover)
                            if store and store in full_stores:
                                sample_cost += 1 # drop action needed first
                            wp_after_sample = wp # Location after sampling
                    # If (rover, wp) not in have_rock and wp not in current_rock_samples, sample is gone and this rover doesn't have it. Impossible for this rover.

                    if sample_cost == float('inf'): continue # Cannot get sample with this rover

                    # Communication cost
                    min_move_to_comm = float('inf')
                    wp_before_comm = wp_after_sample # Location after sample/having sample

                    for comm_wp in self.comm_wps:
                         min_move_to_comm = min(min_move_to_comm, self.get_distance(rover, wp_before_comm, comm_wp))

                    if min_move_to_comm == float('inf'): continue # Cannot reach any communication location

                    comm_cost = min_move_to_comm + 1 # communicate action

                    # Total cost for this rover
                    current_rover_goal_cost = sample_cost + comm_cost
                    min_goal_cost = min(min_goal_cost, current_rover_goal_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost

            elif goal_type == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                if (obj, mode) in communicated_image:
                    continue

                min_goal_cost = float('inf')

                for rover in self.imaging_rovers:
                    if rover not in rover_locations: continue
                    curr_wp = rover_locations[rover]

                    for camera in self.rover_cameras.get(rover, set()):
                        if mode not in self.camera_modes.get(camera, set()):
                            continue # Camera doesn't support mode

                        # Cost to get image
                        image_cost = float('inf')
                        wp_after_image = None # To track location for comm cost

                        if (rover, obj, mode) in have_image:
                            image_cost = 0
                            wp_after_image = curr_wp # Location doesn't change if already held
                        else:
                            # Need to calibrate and take image
                            cal_target = self.camera_cal_target.get(camera)
                            if not cal_target: continue # Camera has no calibration target
                            cal_wps = self.target_visible_from_wp.get(cal_target, set())
                            img_wps = self.obj_visible_from_wp.get(obj, set())
                            if not cal_wps or not img_wps: continue # No suitable waypoints

                            # Cost = min_dist(curr, cal) + 1 (cal) + min_dist(cal, img) + 1 (img)
                            min_cal_img_cost = float('inf')
                            best_img_wp = None # Keep track of the image wp for comm cost

                            for cal_wp in cal_wps:
                                dist_curr_to_cal = self.get_distance(rover, curr_wp, cal_wp)
                                if dist_curr_to_cal != float('inf'):
                                    for img_wp in img_wps:
                                        dist_cal_to_img = self.get_distance(rover, cal_wp, img_wp)
                                        if dist_cal_to_img != float('inf'):
                                            current_cal_img_cost = dist_curr_to_cal + 1 + dist_cal_to_img + 1
                                            if current_cal_img_cost < min_cal_img_cost:
                                                min_cal_img_cost = current_cal_img_cost
                                                best_img_wp = img_wp

                            if min_cal_img_cost != float('inf'):
                                image_cost = min_cal_img_cost
                                wp_after_image = best_img_wp # Location after taking image

                        if image_cost == float('inf'): continue # Cannot get image with this rover/camera

                        # Communication cost
                        min_move_to_comm = float('inf')
                        wp_before_comm = wp_after_image # Use location after image/having image

                        for comm_wp in self.comm_wps:
                             min_move_to_comm = min(min_move_to_comm, self.get_distance(rover, wp_before_comm, comm_wp))

                        if min_move_to_comm == float('inf'): continue # Cannot reach any communication location

                        comm_cost = min_move_to_comm + 1 # communicate action

                        # Total cost for this rover/camera
                        current_rc_goal_cost = image_cost + comm_cost
                        min_goal_cost = min(min_goal_cost, current_rc_goal_cost)

                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost


        # Ensure heuristic is 0 only at goal state
        if total_cost == 0 and not self.goals <= state:
            return 1

        return total_cost
