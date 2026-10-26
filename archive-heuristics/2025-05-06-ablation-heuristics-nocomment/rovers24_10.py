from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import sys # For infinity

def get_parts(fact):
    """Helper to parse PDDL fact string into parts."""
    # Assumes fact is like '(predicate arg1 arg2)'
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to match PDDL fact parts against patterns."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might exist but have no outgoing edges, or not be in the graph keys
         # If it's a known waypoint, initialize its distance
         if start_node in distances:
              distances[start_node] = 0
         else:
              # If start_node is not even a known node in the graph structure,
              # it's unreachable from anywhere within this graph context.
              # However, the graph should ideally contain all waypoints.
              # Let's assume all waypoints are keys in the graph dict, even if empty set of neighbors.
              pass # distances is already initialized with inf

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances.get(neighbor, float('inf')) == float('inf'): # Use .get for safety
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal predicate. For each unachieved goal (communicating
    soil, rock, or image data), it calculates the minimum number of actions required
    to achieve that specific goal, assuming other goals do not interfere (delete
    relaxation). The cost includes navigation steps (estimated by shortest path
    distance) and specific actions like sampling, dropping, calibrating, taking
    images, and communicating.

    Assumptions:
    - The heuristic is non-admissible and focuses on minimizing expanded nodes
      in a greedy best-first search.
    - Resource constraints (like store capacity beyond full/empty) are simplified.
    - Camera calibration state is only considered in the current state; subsequent
      uncalibration by take_image is ignored for future steps within the same
      heuristic calculation.
    - Navigation costs are precomputed shortest paths assuming traversability.
    - Any waypoint visible from the lander can be used for communication.
    - Any waypoint visible from an objective/calibration target can be used for
      imaging/calibration.
    - Costs for different goals are summed independently.
    - If a goal requires a sample/image that is no longer available (e.g., sample
      already taken by another rover and not a goal for this state), the goal is
      considered impossible.

    Heuristic Initialization:
    The constructor precomputes static information from the task definition:
    - Lander location.
    - Rover capabilities (soil, rock, imaging).
    - Rover store mapping.
    - Cameras on board each rover and modes they support.
    - Calibration target for each camera.
    - Waypoints visible from each objective and calibration target.
    - Initial locations of soil and rock samples.
    - Rover navigation graphs based on 'can_traverse' predicates.
    - All-pairs shortest paths for navigation for each rover using BFS.
    - Waypoints visible from the lander (communication points).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h` to 0.
    2. Parse the current state to extract dynamic information: rover locations,
       which rovers have which samples/images, store status, camera calibration
       status, and which data has already been communicated. Also, update the
       locations of available soil/rock samples.
    3. Iterate through each goal predicate in `task.goals`.
    4. For an unachieved goal `g`:
        a. If `g` is `(communicated_soil_data ?w)`:
           - Find the minimum cost to achieve this goal among all soil-equipped rovers.
           - For a given soil-equipped rover `r`:
             - If rover `r` does not have `(have_soil_analysis ?r ?w)`:
               - Check if `(at_soil_sample ?w)` is still in the state. If not, this goal is impossible via sampling for this rover.
               - Add 1 for `drop` if rover `r`'s store is full.
               - Add navigation cost from `r`'s current location to `w` (using precomputed shortest paths).
               - Add 1 for `sample_soil`.
               - The rover is now conceptually at `w`.
             - Add navigation cost from the rover's current location (or `w` if sample was just taken) to the nearest waypoint visible from the lander (using precomputed shortest paths).
             - Add 1 for `communicate_soil_data`.
           - The minimum cost for this goal is the minimum over all suitable rovers. If no suitable rover can achieve it, the goal is impossible (return infinity). Add this minimum cost to `h`.
        b. If `g` is `(communicated_rock_data ?w)`:
           - Similar logic as soil data, using rock-equipped rovers and `sample_rock`.
        c. If `g` is `(communicated_image_data ?o ?m)`:
           - First, check if any rover already possesses the required image data `(have_image ?r ?o ?m)`.
           - If yes, the cost is simply the minimum navigation cost for such a rover from its current location to a communication waypoint, plus 1 for `communicate_image_data`.
           - If no rover has the image, find the minimum cost to acquire the image and then communicate it, among all imaging-equipped rovers `r` with a camera `i` supporting mode `m`.
           - For a given rover `r` and camera `i`/mode `m`:
             - Cost to get the image:
               - If camera `i` on rover `r` is not `calibrated`:
                 - Add navigation cost from `r`'s current location to the nearest waypoint visible from `i`'s calibration target.
                 - Add 1 for `calibrate`.
                 - The rover is now conceptually at the chosen calibration waypoint.
               - Add navigation cost from the rover's current location (or calibration waypoint if just calibrated) to the nearest waypoint visible from objective `o`.
               - Add 1 for `take_image`.
               - The rover is now conceptually at the chosen image waypoint.
             - The minimum cost to get the image is minimized over choices of calibration and image waypoints. Add this minimum cost to the goal cost.
             - Add navigation cost from the rover's current location (or image waypoint if image was just taken) to the nearest waypoint visible from the lander.
             - Add 1 for `communicate_image_data`.
           - The minimum cost for this goal is the minimum over all suitable rover/camera pairs (if image needs acquiring) or over rovers already having the image (if applicable). If no path exists, the goal is impossible (return infinity). Add this minimum cost to `h`.
    5. Return the total heuristic value `h`. If any goal was determined to be impossible, the total heuristic will be infinity.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_location = None
        self.rover_capabilities = defaultdict(set) # rover -> {cap1, cap2, ...}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = defaultdict(list) # rover -> [camera1, camera2, ...]
        self.camera_modes = defaultdict(set) # camera -> {mode1, mode2, ...}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visibility_locations = defaultdict(set) # objective -> {wp1, wp2, ...}
        self.calibration_target_visibility_locations = defaultdict(set) # objective -> {wp1, wp2, ...}
        self.initial_soil_sample_locations = set() # {wp1, wp2, ...}
        self.initial_rock_sample_locations = set() # {wp1, wp2, ...}
        self.rover_navigation_graphs = defaultdict(lambda: defaultdict(set)) # rover -> wp -> {neighbor_wp1, ...}
        self.waypoints = set() # Set of all waypoints
        self.visible_graph = defaultdict(set) # wp -> {visible_wp1, ...}

        # Extract static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                self.rover_capabilities[parts[1]].add(parts[0])
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # store -> rover (corrected mapping)
            elif parts[0] == 'on_board':
                self.rover_cameras[parts[2]].append(parts[1]) # rover -> camera
            elif parts[0] == 'supports':
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif parts[0] == 'calibration_target':
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
            elif parts[0] == 'visible_from':
                self.objective_visibility_locations[parts[1]].add(parts[2])
                self.waypoints.add(parts[2]) # Add waypoint to known waypoints
            elif parts[0] == 'can_traverse':
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.rover_navigation_graphs[rover][wp_from].add(wp_to)
                self.waypoints.add(wp_from)
                self.waypoints.add(wp_to)
            elif parts[0] == 'at_soil_sample':
                self.initial_soil_sample_locations.add(parts[1])
                self.waypoints.add(parts[1]) # Add waypoint to known waypoints
            elif parts[0] == 'at_rock_sample':
                self.initial_rock_sample_locations.add(parts[1])
                self.waypoints.add(parts[1]) # Add waypoint to known waypoints
            elif parts[0] == 'at': # Initial rover locations are static
                 self.waypoints.add(parts[2]) # Add waypoint to known waypoints
            elif parts[0] == 'visible':
                 self.visible_graph[parts[1]].add(parts[2])
                 self.waypoints.add(parts[1]) # Add waypoint to known waypoints
                 self.waypoints.add(parts[2]) # Add waypoint to known waypoints
            elif parts[0] in ['rover', 'waypoint', 'store', 'camera', 'mode', 'lander', 'objective']:
                 # These are type declarations, ignore
                 pass
            # Note: 'empty' and 'full' are dynamic, handled in __call__

        # Populate calibration_target_visibility_locations based on camera_calibration_target
        for camera, cal_target_obj in self.camera_calibration_target.items():
             if cal_target_obj in self.objective_visibility_locations:
                 self.calibration_target_visibility_locations[cal_target_obj].update(self.objective_visibility_locations[cal_target_obj])


        # Precompute all-pairs shortest paths for each rover
        self.rover_distances = {} # rover -> start_wp -> {end_wp -> distance}
        all_waypoints_list = list(self.waypoints) # Ensure BFS covers all known waypoints
        for rover, graph in self.rover_navigation_graphs.items():
            self.rover_distances[rover] = {}
            # Ensure all waypoints are potential start nodes for BFS
            graph_with_all_wps = {wp: graph.get(wp, set()) for wp in all_waypoints_list}
            for start_wp in all_waypoints_list:
                 self.rover_distances[rover][start_wp] = bfs(graph_with_all_wps, start_wp)

        # Identify communication waypoints (visible from lander location)
        self.communication_waypoints = set()
        if self.lander_location:
            # Find all waypoints from which the lander location is visible
            for wp in self.waypoints:
                 if self.lander_location in self.visible_graph.get(wp, set()):
                      self.communication_waypoints.add(wp)


    def __call__(self, node):
        state = node.state
        goals = self.goals

        # Check if goal is reached
        if goals <= state:
            return 0

        # Parse dynamic state information
        rover_locations = {} # rover -> wp
        rover_soil_data = defaultdict(set) # rover -> {wp1, wp2, ...}
        rover_rock_data = defaultdict(set) # rover -> {wp1, wp2, ...}
        rover_image_data = defaultdict(set) # rover -> {(obj1, mode1), ...}
        rover_store_status = {} # rover -> 'empty' or 'full'
        rover_camera_calibration = {} # (camera, rover) -> True/False
        # communicated_soil_data, communicated_rock_data, communicated_image_data are implicitly handled by checking goal in state
        current_soil_sample_locations = set() # {wp1, wp2, ...}
        current_rock_sample_locations = set() # {wp1, wp2, ...}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                rover_soil_data[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                rover_rock_data[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                rover_image_data[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'empty':
                # Find the rover for this store
                store = parts[1]
                rover = None
                # Corrected lookup: store_of maps store -> rover
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover = r
                        break
                if rover:
                    rover_store_status[rover] = 'empty'
            elif parts[0] == 'full':
                 # Find the rover for this store
                store = parts[1]
                rover = None
                # Corrected lookup: store_of maps store -> rover
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover = r
                        break
                if rover:
                    rover_store_status[rover] = 'full'
            elif parts[0] == 'calibrated':
                rover_camera_calibration[(parts[1], parts[2])] = True
            elif parts[0] == 'at_soil_sample':
                 current_soil_sample_locations.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                 current_rock_sample_locations.add(parts[1])


        total_heuristic = 0

        # Process each goal
        for goal in goals:
            if goal in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal)
            goal_predicate = goal_parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint = goal_parts[1]
                min_cost = float('inf')

                # Find a suitable rover
                for rover, capabilities in self.rover_capabilities.items():
                    if 'equipped_for_soil_analysis' in capabilities:
                        cost = 0
                        curr_wp = rover_locations.get(rover)

                        if curr_wp is None: continue # Rover location unknown

                        # Cost to get the sample
                        if waypoint not in rover_soil_data[rover]:
                            # Need to sample
                            if waypoint not in current_soil_sample_locations:
                                # Sample is gone from the ground and no rover has it
                                # This goal is impossible for this rover path
                                continue # Try next rover

                            # Add drop cost if store is full
                            if rover_store_status.get(rover) == 'full':
                                cost += 1 # drop

                            # Add navigation cost to sample location
                            if rover not in self.rover_distances or curr_wp not in self.rover_distances[rover]:
                                 cost = float('inf') # Cannot navigate from current location
                            else:
                                 dist_to_sample = self.rover_distances[rover][curr_wp].get(waypoint, float('inf'))
                                 if dist_to_sample == float('inf'):
                                     cost = float('inf') # Cannot reach sample location
                                 else:
                                     cost += dist_to_sample
                                     cost += 1 # sample_soil
                                     curr_wp_after_sample = waypoint # Rover is now at sample location
                                     # Update temp location for communication step
                                     curr_wp = curr_wp_after_sample

                        # If cost is already infinity, skip communication part
                        if cost == float('inf'):
                             continue

                        # Cost to communicate
                        min_dist_to_comm = float('inf')
                        if rover in self.rover_distances and curr_wp in self.rover_distances[rover]:
                            for comm_wp in self.communication_waypoints:
                                if comm_wp in self.rover_distances[rover][curr_wp]:
                                    min_dist_to_comm = min(min_dist_to_comm, self.rover_distances[rover][curr_wp][comm_wp])

                        if min_dist_to_comm == float('inf'):
                            cost = float('inf') # Cannot reach communication waypoint
                        else:
                            cost += min_dist_to_comm
                            cost += 1 # communicate_soil_data

                        min_cost = min(min_cost, cost)

                if min_cost == float('inf'):
                     # If after checking all rovers, it's still infinity, the goal is impossible
                     return float('inf') # Return infinity for the whole heuristic
                total_heuristic += min_cost

            elif goal_predicate == 'communicated_rock_data':
                waypoint = goal_parts[1]
                min_cost = float('inf')

                # Find a suitable rover
                for rover, capabilities in self.rover_capabilities.items():
                    if 'equipped_for_rock_analysis' in capabilities:
                        cost = 0
                        curr_wp = rover_locations.get(rover)

                        if curr_wp is None: continue # Rover location unknown

                        # Cost to get the sample
                        if waypoint not in rover_rock_data[rover]:
                            # Need to sample
                            if waypoint not in current_rock_sample_locations:
                                # Sample is gone
                                continue # Try next rover

                            # Add drop cost if store is full
                            if rover_store_status.get(rover) == 'full':
                                cost += 1 # drop

                            # Add navigation cost to sample location
                            if rover not in self.rover_distances or curr_wp not in self.rover_distances[rover]:
                                 cost = float('inf') # Cannot navigate from current location
                            else:
                                dist_to_sample = self.rover_distances[rover][curr_wp].get(waypoint, float('inf'))
                                if dist_to_sample == float('inf'):
                                    cost = float('inf') # Cannot reach sample location
                                else:
                                    cost += dist_to_sample
                                    cost += 1 # sample_rock
                                    curr_wp_after_sample = waypoint # Rover is now at sample location
                                    # Update temp location for communication step
                                    curr_wp = curr_wp_after_sample

                        # If cost is already infinity, skip communication part
                        if cost == float('inf'):
                             continue

                        # Cost to communicate
                        min_dist_to_comm = float('inf')
                        if rover in self.rover_distances and curr_wp in self.rover_distances[rover]:
                            for comm_wp in self.communication_waypoints:
                                if comm_wp in self.rover_distances[rover][curr_wp]:
                                    min_dist_to_comm = min(min_dist_to_comm, self.rover_distances[rover][curr_wp][comm_wp])

                        if min_dist_to_comm == float('inf'):
                            cost = float('inf') # Cannot reach communication waypoint
                        else:
                            cost += min_dist_to_comm
                            cost += 1 # communicate_rock_data

                        min_cost = min(min_cost, cost)

                if min_cost == float('inf'):
                     return float('inf')
                total_heuristic += min_cost

            elif goal_predicate == 'communicated_image_data':
                objective, mode = goal_parts[1], goal_parts[2]
                min_cost = float('inf')

                # Check if any rover already has the image
                rover_with_image = None
                for rover, images in rover_image_data.items():
                    if (objective, mode) in images:
                        rover_with_image = rover
                        break

                if rover_with_image:
                    # Image is already acquired by rover_with_image, just need to communicate
                    rover = rover_with_image
                    curr_wp = rover_locations.get(rover)
                    if curr_wp is None: # Should not happen
                         min_cost = float('inf')
                    else:
                        # Cost to communicate
                        min_dist_to_comm = float('inf')
                        if rover in self.rover_distances and curr_wp in self.rover_distances[rover]:
                            for comm_wp in self.communication_waypoints:
                                if comm_wp in self.rover_distances[rover][curr_wp]:
                                    min_dist_to_comm = min(min_dist_to_comm, self.rover_distances[rover][curr_wp][comm_wp])

                        if min_dist_to_comm == float('inf'):
                            min_cost = float('inf') # Cannot reach communication waypoint
                        else:
                            min_cost = min_dist_to_comm + 1 # navigate + communicate

                else:
                    # No rover has the image, need to acquire it and then communicate
                    # Find a suitable rover and camera
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'equipped_for_imaging' in capabilities:
                            curr_wp = rover_locations.get(rover)
                            if curr_wp is None: continue # Rover location unknown

                            for camera in self.rover_cameras.get(rover, []):
                                if mode in self.camera_modes.get(camera, set()):
                                    # Found a suitable rover/camera pair
                                    cost = 0
                                    temp_curr_wp = curr_wp # Use a temporary variable for location tracking

                                    cal_target_obj = self.camera_calibration_target.get(camera)
                                    if cal_target_obj is None:
                                         # Camera has no calibration target, cannot be calibrated
                                         continue # Skip this camera/rover pair

                                    cal_wps = self.calibration_target_visibility_locations.get(cal_target_obj, set())
                                    img_wps = self.objective_visibility_locations.get(objective, set())

                                    if not cal_wps or not img_wps:
                                         # No calibration or image waypoints available
                                         continue # Skip this camera/rover pair

                                    # Cost to get the image: Navigate to calibration (if needed), calibrate (if needed), navigate to image location, take image.
                                    cost_get_image = float('inf')

                                    # Option 1: Calibrate then Image
                                    if not rover_camera_calibration.get((camera, rover), False):
                                        # Need to calibrate
                                        min_dist_to_cal = float('inf')
                                        best_cal_wp = None
                                        if rover in self.rover_distances and temp_curr_wp in self.rover_distances[rover]:
                                            for cal_wp in cal_wps:
                                                if cal_wp in self.rover_distances[rover][temp_curr_wp]:
                                                    dist = self.rover_distances[rover][temp_curr_wp][cal_wp]
                                                    if dist < min_dist_to_cal:
                                                        min_dist_to_cal = dist
                                                        best_cal_wp = cal_wp

                                        if min_dist_to_cal != float('inf'):
                                            temp_cost_part1 = min_dist_to_cal + 1 # navigate + calibrate
                                            temp_wp_after_cal = best_cal_wp

                                            # Navigate to image waypoint after calibration
                                            min_dist_cal_to_img = float('inf')
                                            best_img_wp_after_cal = None
                                            if rover in self.rover_distances and temp_wp_after_cal in self.rover_distances[rover]:
                                                for img_wp in img_wps:
                                                    if img_wp in self.rover_distances[rover][temp_wp_after_cal]:
                                                        dist = self.rover_distances[rover][temp_wp_after_cal][img_wp]
                                                        if dist < min_dist_cal_to_img:
                                                            min_dist_cal_to_img = dist
                                                            best_img_wp_after_cal = img_wp

                                            if min_dist_cal_to_img != float('inf'):
                                                temp_cost_part1 += min_dist_cal_to_img + 1 # navigate + take_image
                                                temp_wp_after_image = best_img_wp_after_cal
                                                cost_get_image = min(cost_get_image, temp_cost_part1)
                                                # Store the waypoint where the image was taken for communication step
                                                wp_after_image = temp_wp_after_image

                                    # Option 2: Already calibrated, just need to Image
                                    else: # Camera is calibrated in the current state
                                        min_dist_curr_to_img = float('inf')
                                        best_img_wp_from_curr = None
                                        if rover in self.rover_distances and temp_curr_wp in self.rover_distances[rover]:
                                            for img_wp in img_wps:
                                                if img_wp in self.rover_distances[rover][temp_curr_wp]:
                                                    dist = self.rover_distances[rover][temp_curr_wp][img_wp]
                                                    if dist < min_dist_curr_to_img:
                                                        min_dist_curr_to_img = dist
                                                        best_img_wp_from_curr = img_wp

                                        if min_dist_curr_to_img != float('inf'):
                                            temp_cost_part2 = min_dist_curr_to_img + 1 # navigate + take_image
                                            temp_wp_after_image = best_img_wp_from_curr
                                            cost_get_image = min(cost_get_image, temp_cost_part2)
                                            # Store the waypoint where the image was taken for communication step
                                            wp_after_image = temp_wp_after_image


                                    if cost_get_image == float('inf'):
                                         # Cannot get the image with this rover/camera
                                         continue # Skip this rover/camera pair
                                    else:
                                         cost += cost_get_image
                                         temp_curr_wp = wp_after_image # Rover is now at the image waypoint

                                    # Cost to communicate
                                    min_dist_to_comm = float('inf')
                                    if rover in self.rover_distances and temp_curr_wp in self.rover_distances[rover]:
                                        for comm_wp in self.communication_waypoints:
                                            if comm_wp in self.rover_distances[rover][temp_curr_wp]:
                                                min_dist_to_comm = min(min_dist_to_comm, self.rover_distances[rover][temp_curr_wp][comm_wp])

                                    if min_dist_to_comm == float('inf'):
                                        cost = float('inf') # Cannot reach communication waypoint
                                    else:
                                        cost += min_dist_to_comm
                                        cost += 1 # communicate_image_data

                                    min_cost = min(min_cost, cost)

                if min_cost == float('inf'):
                     return float('inf')
                total_heuristic += min_cost

        return total_heuristic

