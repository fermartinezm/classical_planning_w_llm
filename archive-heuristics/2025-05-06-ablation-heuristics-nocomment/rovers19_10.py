from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

# Helper function to match PDDL fact parts with patterns
def match(fact, *args):
    """Checks if a fact string matches a sequence of patterns."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest paths
def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all reachable nodes.
    Graph is represented as an adjacency list: {node: [neighbor1, neighbor2, ...]}
    Returns a dictionary {node: distance}.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Ensure current_node is in graph and has neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing up the estimated costs
    for each unsatisfied goal predicate. The cost for each predicate is
    estimated based on the required actions (sample/image, calibrate,
    communicate, drop) and the navigation costs (shortest path distance)
    to reach the necessary locations (sample/image site, calibration site,
    communication site). It simplifies by selecting the first suitable
    rover/camera/waypoint found and summing costs without detailed action
    sequencing or resource contention.

    Assumptions:
    - Action costs are uniform (cost 1).
    - The problem instance is solvable (required equipment, locations, etc., exist).
    - Navigation is possible between required waypoints for the chosen rover.
    - Shortest path distance is a reasonable estimate for navigation cost.
    - Resource contention (multiple rovers needing the same store/camera,
      multiple goals needing the same rover) is ignored.
    - Camera recalibration after taking an image is assumed to be a separate
      cost added only if the camera is not calibrated when needed for an image.
    - Store capacity is only checked once per sample goal (if a drop is needed).
    - If multiple rovers can satisfy a 'have_X' prerequisite, the heuristic
      arbitrarily picks one (the first one found in the state iteration).
    - If multiple waypoints are suitable for imaging or calibration, the
      heuristic arbitrarily picks one (the first one found in the static data).

    Heuristic Initialization:
    - Parses static facts to build data structures:
        - Lander location.
        - Navigation graph for each rover based on 'can_traverse'.
        - Shortest path distances between all reachable waypoint pairs for each rover using BFS.
        - Sets of rovers equipped for soil, rock, and imaging.
        - Mapping from store to rover.
        - Mapping from rover to its cameras and their supported modes.
        - Mapping from camera to its calibration target objective.
        - Mapping from objective to waypoints from which it is visible.
        - Set of communication waypoints (visible from lander location).
    - Parses goal facts to identify required communications (soil, rock, image).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h = 0`.
    2. Parse the current state to extract dynamic information: rover locations,
       which samples/images are held by which rovers, camera calibration status,
       store status (empty/full), and which communication goals are already satisfied.
    3. If all goal communication predicates are already in the current state,
       return `h = 0`.
    4. For each waypoint `w` that needs `(communicated_soil_data w)` and is not yet communicated:
        a. Add 1 to `h` for the `communicate_soil_data` action.
        b. Check if any rover currently has `(have_soil_analysis r w)`.
        c. If no rover has the sample:
            i. Add 1 to `h` for the `sample_soil` action.
            ii. Find a soil-equipped rover (pick the first one found in static data that is also in the current state). Let it be `r_soil`. If none found, return infinity.
            iii. Get `r_soil`'s current location `loc_r_soil`.
            iv. Calculate shortest distance `dist_to_sample = dist(r_soil, loc_r_soil, w)`. If unreachable, return infinity. Add `dist_to_sample` to `h`.
            v. Find the store `s` for `r_soil`. If `s` is not in the set of `empty_stores` in the current state, add 1 to `h` for a `drop` action.
            vi. Assume `r_soil` will perform the communication. Set `rover_for_comm = r_soil`.
        d. If a rover `r_have` already has the sample:
            i. Set `rover_for_comm = r_have`.
        e. Get `rover_for_comm`'s current location `loc_rover_for_comm`. If rover not in state, return infinity.
        f. Calculate the minimum shortest distance from `loc_rover_for_comm` to any communication waypoint visible from the lander for `rover_for_comm`. Let this be `min_dist_to_comm`. If no communication waypoint is reachable, return infinity. Add `min_dist_to_comm` to `h`.
    5. For each waypoint `w` that needs `(communicated_rock_data w)` and is not yet communicated:
        a. Follow similar steps as for soil data, using rock-specific predicates and equipment.
    6. For each objective `o` and mode `m` that needs `(communicated_image_data o m)` and is not yet communicated:
        a. Add 1 to `h` for the `communicate_image_data` action.
        b. Check if any rover currently has `(have_image r o m)`.
        c. If no rover has the image:
            i. Add 1 to `h` for the `take_image` action.
            ii. Find an imaging-equipped rover `r_img` with a camera `i` supporting mode `m` (pick the first one found in static data that is also in the current state). If none found, return infinity.
            iii. Get `r_img`'s current location `loc_r_img`.
            iv. Find a waypoint `p` from which objective `o` is visible (pick the first one found in static data). If none found, return infinity.
            v. Calculate shortest distance `dist_to_image_loc = dist(r_img, loc_r_img, p)`. If unreachable, return infinity. Add `dist_to_image_loc` to `h`.
            vi. Check if camera `i` on rover `r_img` is calibrated in the current state. If not:
                - Add 1 to `h` for the `calibrate` action.
                - Find the calibration target `t` for camera `i`. If none found, return infinity.
                - Find a waypoint `w` from which `t` is visible (pick the first one found in static data). If none found, return infinity.
                - Calculate shortest distance `dist_to_cal_loc = dist(r_img, loc_r_img, w)`. If unreachable, return infinity. Add `dist_to_cal_loc` to `h`. (Note: This adds navigation cost to calibration site independently of navigation to image site).
            vii. Assume `r_img` will perform the communication. Set `rover_for_comm = r_img`.
        d. If a rover `r_have` already has the image:
            i. Set `rover_for_comm = r_have`.
        e. Get `rover_for_comm`'s current location `loc_rover_for_comm`. If rover not in state, return infinity.
        f. Calculate the minimum shortest distance from `loc_rover_for_comm` to any communication waypoint visible from the lander for `rover_for_comm`. Let this be `min_dist_to_comm`. If no communication waypoint is reachable, return infinity. Add `min_dist_to_comm` to `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.lander_location = None
        self.rover_graphs = {} # {rover: {wp: [neighbor_wp, ...]}}
        self.rover_distances = {} # {rover: {start_wp: {end_wp: distance}}}
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.store_to_rover = {} # {store: rover}
        self.rover_cameras = {} # {rover: {camera: [mode, ...]}}
        self.calibration_targets = {} # {camera: objective}
        self.objective_locations = {} # {objective: [waypoint, ...]}
        self.visible_waypoints = {} # {wp: [visible_wp, ...]}
        self.comm_waypoints = set() # waypoints visible from lander

        # --- Parse static facts (Two passes for camera info) ---

        # Pass 1: Basic info and camera-to-rover mapping
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.rover_graphs:
                    self.rover_graphs[rover] = {}
                if wp1 not in self.rover_graphs[rover]:
                    self.rover_graphs[rover][wp1] = []
                self.rover_graphs[rover][wp1].append(wp2)
            elif parts[0] == 'equipped_for_soil_analysis':
                self.soil_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rock_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.imaging_rovers.add(parts[1])
            elif parts[0] == 'store_of':
                self.store_to_rover[parts[1]] = parts[2]
            elif parts[0] == 'visible':
                wp1, wp2 = parts[1], parts[2]
                if wp1 not in self.visible_waypoints:
                    self.visible_waypoints[wp1] = []
                self.visible_waypoints[wp1].append(wp2)
            elif parts[0] == 'visible_from':
                objective, wp = parts[1], parts[2]
                if objective not in self.objective_locations:
                    self.objective_locations[objective] = []
                self.objective_locations[objective].append(wp)
            elif parts[0] == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                if rover not in self.rover_cameras:
                    self.rover_cameras[rover] = {}
                self.rover_cameras[rover][camera] = [] # Initialize list for modes

        # Pass 2: Add supported modes to cameras
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'supports':
                 camera, mode = parts[1], parts[2]
                 # Find which rover this camera is on and add the mode
                 for r, cams_on_r in self.rover_cameras.items():
                     if camera in cams_on_r:
                         self.rover_cameras[r][camera].append(mode)
                         break # Found the rover

        # --- Precompute distances ---
        for rover, graph in self.rover_graphs.items():
            self.rover_distances[rover] = {}
            # Collect all waypoints relevant to this rover's graph
            all_wps_for_rover = set(graph.keys()).union(set(wp for neighbors in graph.values() for wp in neighbors))
            for start_wp in all_wps_for_rover:
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # --- Identify communication waypoints ---
        if self.lander_location and self.lander_location in self.visible_waypoints:
             self.comm_waypoints = set(self.visible_waypoints[self.lander_location])

        # --- Parse goal facts ---
        self.goal_communicated_soil = set()
        self.goal_communicated_rock = set()
        self.goal_communicated_image = set() # {(objective, mode)}

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                self.goal_communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                self.goal_communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                self.goal_communicated_image.add((parts[1], parts[2]))

    def get_distance(self, rover, start_wp, end_wp):
        """Gets the precomputed shortest distance or infinity if unreachable."""
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover] or end_wp not in self.rover_distances[rover][start_wp]:
            return float('inf') # Unreachable
        return self.rover_distances[rover][start_wp][end_wp]

    def get_min_distance_to_comm(self, rover, start_wp):
        """Gets the minimum distance from start_wp to any communication waypoint for the rover."""
        min_dist = float('inf')
        if not self.comm_waypoints:
             return float('inf') # No communication waypoints

        found_reachable_comm_wp = False
        for comm_wp in self.comm_waypoints:
            dist = self.get_distance(rover, start_wp, comm_wp)
            if dist != float('inf'):
                 min_dist = min(min_dist, dist)
                 found_reachable_comm_wp = True

        return min_dist if found_reachable_comm_wp else float('inf')


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Parse current state
        rover_locations = {} # {rover: wp}
        have_soil = set() # {(rover, wp)}
        have_rock = set() # {(rover, wp)}
        have_image = set() # {(rover, objective, mode)}
        calibrated = set() # {(camera, rover)}
        empty_stores = set() # {store}
        # at_soil_samples = set() # Not needed for this heuristic logic
        # at_rock_samples = set() # Not needed for this heuristic logic
        communicated_soil = set() # {wp}
        communicated_rock = set() # {wp}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rover_graphs: # Check if it's a rover location we care about
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'empty':
                empty_stores.add(parts[1])
            # elif parts[0] == 'at_soil_sample': # Not needed
            #      at_soil_samples.add(parts[1])
            # elif parts[0] == 'at_rock_sample': # Not needed
            #      at_rock_samples.add(parts[1])
            elif parts[0] == 'communicated_soil_data':
                 communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                 communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                 communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # Process unsatisfied soil goals
        for wp in self.goal_communicated_soil:
            if wp not in communicated_soil:
                cost_for_goal = 0
                cost_for_goal += 1 # communicate action

                # Check if sample is already collected by any rover
                sample_held = False
                rover_with_sample = None
                for r, w in have_soil:
                    if w == wp:
                        sample_held = True
                        rover_with_sample = r
                        break

                if not sample_held:
                    cost_for_goal += 1 # sample action
                    # Find a suitable rover for sampling (equipped and in state)
                    r_soil = None
                    for r in self.soil_rovers:
                         if r in rover_locations:
                              r_soil = r
                              break # Pick the first available soil rover

                    if r_soil is None:
                         return float('inf') # No soil rover available

                    loc_r_soil = rover_locations[r_soil]
                    dist_to_sample = self.get_distance(r_soil, loc_r_soil, wp)
                    if dist_to_sample == float('inf'):
                         return float('inf') # Cannot reach sample location
                    cost_for_goal += dist_to_sample # navigate to sample

                    # Check if store is full for this rover
                    store_is_full = True # Assume full unless found empty
                    for s, r in self.store_to_rover.items():
                         if r == r_soil:
                              if s in empty_stores:
                                   store_is_full = False
                              break
                    if store_is_full:
                         cost_for_goal += 1 # drop action

                    # Assume this rover will communicate
                    rover_for_comm = r_soil
                else:
                    # Sample is held, find the rover holding it
                    rover_for_comm = rover_with_sample

                # Add cost to navigate to communication waypoint
                loc_rover_for_comm = rover_locations.get(rover_for_comm)
                if loc_rover_for_comm is None:
                     return float('inf') # Rover not found in state

                min_dist_to_comm = self.get_min_distance_to_comm(rover_for_comm, loc_rover_for_comm)
                if min_dist_to_comm == float('inf'):
                     return float('inf') # Cannot reach any communication waypoint
                cost_for_goal += min_dist_to_comm # navigate to communicate

                total_cost += cost_for_goal

        # Process unsatisfied rock goals
        for wp in self.goal_communicated_rock:
            if wp not in communicated_rock:
                cost_for_goal = 0
                cost_for_goal += 1 # communicate action

                # Check if sample is already collected by any rover
                sample_held = False
                rover_with_sample = None
                for r, w in have_rock:
                    if w == wp:
                        sample_held = True
                        rover_with_sample = r
                        break

                if not sample_held:
                    cost_for_goal += 1 # sample action
                    # Find a suitable rover for sampling (equipped and in state)
                    r_rock = None
                    for r in self.rock_rovers:
                         if r in rover_locations:
                              r_rock = r
                              break

                    if r_rock is None:
                         return float('inf') # No rock rover available

                    loc_r_rock = rover_locations[r_rock]
                    dist_to_sample = self.get_distance(r_rock, loc_r_rock, wp)
                    if dist_to_sample == float('inf'):
                         return float('inf') # Cannot reach sample location
                    cost_for_goal += dist_to_sample # navigate to sample

                    # Check if store is full for this rover
                    store_is_full = True
                    for s, r in self.store_to_rover.items():
                         if r == r_rock:
                              if s in empty_stores:
                                   store_is_full = False
                              break
                    if store_is_full:
                         cost_for_goal += 1 # drop action

                    # Assume this rover will communicate
                    rover_for_comm = r_rock
                else:
                    # Sample is held, find the rover holding it
                    rover_for_comm = rover_with_sample

                # Add cost to navigate to communication waypoint
                loc_rover_for_comm = rover_locations.get(rover_for_comm)
                if loc_rover_for_comm is None:
                     return float('inf') # Rover not found in state

                min_dist_to_comm = self.get_min_distance_to_comm(rover_for_comm, loc_rover_for_comm)
                if min_dist_to_comm == float('inf'):
                     return float('inf') # Cannot reach any communication waypoint
                cost_for_goal += min_dist_to_comm # navigate to communicate

                total_cost += cost_for_goal

        # Process unsatisfied image goals
        for o, m in self.goal_communicated_image:
            if (o, m) not in communicated_image:
                cost_for_goal = 0
                cost_for_goal += 1 # communicate action

                # Check if image is already taken by any rover
                image_taken = False
                rover_with_image = None
                for r, obj, mode in have_image:
                    if obj == o and mode == m:
                        image_taken = True
                        rover_with_image = r
                        break

                if not image_taken:
                    cost_for_goal += 1 # take_image action
                    # Find a suitable rover/camera for imaging (equipped, camera on board, supports mode, and rover in state)
                    r_img = None
                    cam_img = None
                    for r in self.imaging_rovers:
                         if r in rover_locations and r in self.rover_cameras:
                              for c, modes in self.rover_cameras[r].items():
                                   if m in modes:
                                        r_img = r
                                        cam_img = c
                                        break
                              if r_img: break # Found a suitable rover and camera

                    if r_img is None or cam_img is None:
                         return float('inf') # No suitable imaging setup available

                    loc_r_img = rover_locations[r_img]

                    # Find a waypoint to take the image from
                    img_wp = None
                    if o in self.objective_locations and self.objective_locations[o]:
                         # Pick the first visible waypoint
                         img_wp = self.objective_locations[o][0]

                    if img_wp is None:
                         return float('inf') # No waypoint to view objective from

                    dist_to_image_loc = self.get_distance(r_img, loc_r_img, img_wp)
                    if dist_to_image_loc == float('inf'):
                         return float('inf') # Cannot reach image location
                    cost_for_goal += dist_to_image_loc # navigate to image location

                    # Check if camera is calibrated
                    if (cam_img, r_img) not in calibrated:
                         cost_for_goal += 1 # calibrate action
                         # Find calibration target and location
                         cal_target = self.calibration_targets.get(cam_img)
                         if cal_target is None:
                              return float('inf') # Camera has no calibration target

                         cal_wp = None
                         if cal_target in self.objective_locations and self.objective_locations[cal_target]:
                              # Pick the first visible waypoint for the calibration target
                              cal_wp = self.objective_locations[cal_target][0]

                         if cal_wp is None:
                              return float('inf') # No waypoint to view calibration target from

                         dist_to_cal_loc = self.get_distance(r_img, loc_r_img, cal_wp)
                         if dist_to_cal_loc == float('inf'):
                              return float('inf') # Cannot reach calibration location
                         cost_for_goal += dist_to_cal_loc # navigate to calibration location

                    # Assume this rover will communicate
                    rover_for_comm = r_img
                else:
                    # Image is held, find the rover holding it
                    rover_for_comm = rover_with_image

                # Add cost to navigate to communication waypoint
                loc_rover_for_comm = rover_locations.get(rover_for_comm)
                if loc_rover_for_comm is None:
                     return float('inf') # Rover not found in state

                min_dist_to_comm = self.get_min_distance_to_comm(rover_for_comm, loc_rover_for_comm)
                if min_dist_to_comm == float('inf'):
                     return float('inf') # Cannot reach any communication waypoint
                cost_for_goal += min_dist_to_comm # navigate to communicate

                total_cost += cost_for_goal

        return total_cost
