# Assuming the Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing up the estimated costs
    for each uncommunicated goal fact. The cost for each uncommunicated
    goal is estimated based on the minimum actions needed to collect the
    required data (sample or image) and communicate it, including movement.

    Movement costs are estimated using precomputed shortest paths on the
    waypoint graph.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing waypoint distances.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

        # --- Extract Static Information ---
        self.lander_location = None
        self.waypoint_graph = {} # Adjacency list {w1: {w2, w3}, ...}
        self.objects_by_type = {} # {type: {obj1, obj2}, ...}
        self.rover_equipment = {} # {rover: {soil, rock, imaging}, ...}
        self.rover_store = {} # {rover: store, ...}
        self.camera_info = {} # {camera: {'rover': rover, 'modes': {mode1, mode2}, 'cal_target': cal_target}, ...}
        self.objective_visibility = {} # {objective: {waypoint1, waypoint2}, ...}
        self.calibration_targets = {} # {camera: objective, ...}

        # First pass to identify all objects and their types
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] in ['rover', 'waypoint', 'store', 'camera', 'mode', 'lander', 'objective']:
                 obj_type, obj_name = parts[0], parts[1]
                 if obj_type not in self.objects_by_type:
                     self.objects_by_type[obj_type] = set()
                 self.objects_by_type[obj_type].add(obj_name)

        # Initialize waypoint graph nodes
        for wp in self.objects_by_type.get('waypoint', set()):
             self.waypoint_graph[wp] = set()

        # Second pass for relationships and equipment
        for fact in self.static:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                # Ensure lander object exists before storing location
                if len(parts) == 3 and parts[1] in self.objects_by_type.get('lander', set()) and parts[2] in self.objects_by_type.get('waypoint', set()):
                    self.lander_location = parts[2]
            elif predicate == "visible":
                w1, w2 = parts[1], parts[2]
                # Add edge if both are valid waypoints
                if w1 in self.waypoint_graph and w2 in self.waypoint_graph:
                    self.waypoint_graph[w1].add(w2)
                    self.waypoint_graph[w2].add(w1) # Assume symmetric
            # Note: Ignoring can_traverse for graph building as per simplified assumption based on examples
            # If can_traverse was rover-specific and restrictive, we'd need per-rover graphs or a more complex model.

            elif predicate == "equipped_for_soil_analysis":
                rover = parts[1]
                if rover in self.objects_by_type.get('rover', set()):
                    if rover not in self.rover_equipment: self.rover_equipment[rover] = set()
                    self.rover_equipment[rover].add('soil')
            elif predicate == "equipped_for_rock_analysis":
                rover = parts[1]
                if rover in self.objects_by_type.get('rover', set()):
                    if rover not in self.rover_equipment: self.rover_equipment[rover] = set()
                    self.rover_equipment[rover].add('rock')
            elif predicate == "equipped_for_imaging":
                rover = parts[1]
                if rover in self.objects_by_type.get('rover', set()):
                    if rover not in self.rover_equipment: self.rover_equipment[rover] = set()
                    self.rover_equipment[rover].add('imaging')
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                if store in self.objects_by_type.get('store', set()) and rover in self.objects_by_type.get('rover', set()):
                    self.rover_store[rover] = store
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                if camera in self.objects_by_type.get('camera', set()) and rover in self.objects_by_type.get('rover', set()):
                    if camera not in self.camera_info: self.camera_info[camera] = {'modes': set()}
                    self.camera_info[camera]['rover'] = rover
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                if camera in self.objects_by_type.get('camera', set()) and mode in self.objects_by_type.get('mode', set()):
                    if camera not in self.camera_info: self.camera_info[camera] = {'modes': set()}
                    self.camera_info[camera]['modes'].add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                if camera in self.objects_by_type.get('camera', set()) and objective in self.objects_by_type.get('objective', set()):
                    self.calibration_targets[camera] = objective
                    if camera not in self.camera_info: self.camera_info[camera] = {'modes': set()} # Ensure camera entry exists
                    self.camera_info[camera]['cal_target'] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                if objective in self.objects_by_type.get('objective', set()) and waypoint in self.objects_by_type.get('waypoint', set()):
                    if objective not in self.objective_visibility: self.objective_visibility[objective] = set()
                    self.objective_visibility[objective].add(waypoint)

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        waypoints = list(self.waypoint_graph.keys()) # Use list for consistent order if needed, set is fine too
        for start_node in waypoints:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[(start_node, start_node)] = 0

            while q:
                current_node, current_dist = q.popleft()

                for neighbor in self.waypoint_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[(start_node, neighbor)] = current_dist + 1
                        q.append((neighbor, current_dist + 1))

        # Identify communication points (waypoints visible from lander)
        self.communication_points = set()
        if self.lander_location and self.lander_location in self.waypoint_graph:
             # Communication requires (at ?r ?x), (at_lander ?l ?y), (visible ?x ?y)
             # So, communication points are waypoints ?x visible from lander_location ?y.
             # Assuming 'visible' is symmetric, these are neighbors of lander_location in the graph.
             self.communication_points = self.waypoint_graph.get(self.lander_location, set())


        # Store initial sample locations
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at_soil_sample":
                 if parts[1] in self.objects_by_type.get('waypoint', set()):
                    self.initial_soil_samples.add(parts[1])
             elif parts[0] == "at_rock_sample":
                 if parts[1] in self.objects_by_type.get('waypoint', set()):
                    self.initial_rock_samples.add(parts[1])


    def get_distance(self, w1, w2):
        """Helper to get shortest distance, returns infinity if no path."""
        # Handle cases where w1 or w2 are not valid waypoints or not in computed distances
        if w1 not in self.waypoint_graph or w2 not in self.waypoint_graph:
             return float('inf')
        return self.dist.get((w1, w2), float('inf'))

    def get_min_dist_to_comm(self, waypoint):
        """Helper to get minimum distance from a waypoint to any communication point."""
        if not self.communication_points:
            return float('inf') # Cannot communicate if no comm points
        min_d = float('inf')
        for comm_point in self.communication_points:
             min_d = min(min_d, self.get_distance(waypoint, comm_point))
        return min_d


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # --- Extract State Information ---
        rover_locations = {} # {rover: waypoint, ...}
        store_status = {} # {store: 'empty'/'full', ...}
        have_soil = set() # {(rover, waypoint), ...}
        have_rock = set() # {(rover, waypoint), ...}
        have_image = set() # {(rover, objective, mode), ...}
        calibrated_cameras = set() # {(camera, rover), ...}
        communicated_soil = set() # {waypoint, ...}
        communicated_rock = set() # {waypoint, ...}
        communicated_image = set() # {(objective, mode), ...}
        current_soil_samples = set() # {waypoint, ...}
        current_rock_samples = set() # {waypoint, ...}


        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.objects_by_type.get('rover', set()) and loc in self.objects_by_type.get('waypoint', set()):
                    rover_locations[obj] = loc
            elif predicate == "empty":
                if parts[1] in self.objects_by_type.get('store', set()):
                    store_status[parts[1]] = 'empty'
            elif predicate == "full":
                 if parts[1] in self.objects_by_type.get('store', set()):
                    store_status[parts[1]] = 'full'
            elif predicate == "have_soil_analysis":
                if parts[1] in self.objects_by_type.get('rover', set()) and parts[2] in self.objects_by_type.get('waypoint', set()):
                    have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                 if parts[1] in self.objects_by_type.get('rover', set()) and parts[2] in self.objects_by_type.get('waypoint', set()):
                    have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                 if parts[1] in self.objects_by_type.get('rover', set()) and parts[2] in self.objects_by_type.get('objective', set()) and parts[3] in self.objects_by_type.get('mode', set()):
                    have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "calibrated":
                 if parts[1] in self.objects_by_type.get('camera', set()) and parts[2] in self.objects_by_type.get('rover', set()):
                    calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == "communicated_soil_data":
                 if parts[1] in self.objects_by_type.get('waypoint', set()):
                    communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                 if parts[1] in self.objects_by_type.get('waypoint', set()):
                    communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                 if parts[1] in self.objects_by_type.get('objective', set()) and parts[2] in self.objects_by_type.get('mode', set()):
                    communicated_image.add((parts[1], parts[2]))
            elif predicate == "at_soil_sample":
                 if parts[1] in self.objects_by_type.get('waypoint', set()):
                    current_soil_samples.add(parts[1])
            elif predicate == "at_rock_sample":
                 if parts[1] in self.objects_by_type.get('waypoint', set()):
                    current_rock_samples.add(parts[1])


        total_heuristic_cost = 0

        # --- Estimate cost for each uncommunicated goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                if waypoint not in communicated_soil:
                    min_goal_cost = float('inf')

                    # Option 1: Sample is already collected by a rover
                    rovers_with_sample = [r for r in self.objects_by_type.get('rover', set()) if (r, waypoint) in have_soil]
                    if rovers_with_sample:
                        for rover_w_sample in rovers_with_sample:
                            rover_loc = rover_locations.get(rover_w_sample)
                            if rover_loc:
                                cost = 1 # communicate action
                                cost += self.get_min_dist_to_comm(rover_loc) # Move to comm point
                                min_goal_cost = min(min_goal_cost, cost)

                    # Option 2: Sample needs to be collected
                    if waypoint in current_soil_samples:
                        soil_rovers = [r for r in self.objects_by_type.get('rover', set()) if 'soil' in self.rover_equipment.get(r, set())]
                        if soil_rovers:
                            for rover_to_use in soil_rovers:
                                rover_loc = rover_locations.get(rover_to_use)
                                if rover_loc:
                                    cost = 1 # sample_soil action
                                    cost += self.get_distance(rover_loc, waypoint) # Move to sample location
                                    # Check if rover's store is full
                                    store = self.rover_store.get(rover_to_use)
                                    if store and store_status.get(store) == 'full':
                                        cost += 1 # drop action
                                    cost += 1 # communicate action
                                    cost += self.get_min_dist_to_comm(waypoint) # Move from sample location to comm point
                                    min_goal_cost = min(min_goal_cost, cost)

                    total_heuristic_cost += min_goal_cost

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                if waypoint not in communicated_rock:
                    min_goal_cost = float('inf')

                    # Option 1: Sample is already collected by a rover
                    rovers_with_sample = [r for r in self.objects_by_type.get('rover', set()) if (r, waypoint) in have_rock]
                    if rovers_with_sample:
                        for rover_w_sample in rovers_with_sample:
                            rover_loc = rover_locations.get(rover_w_sample)
                            if rover_loc:
                                cost = 1 # communicate action
                                cost += self.get_min_dist_to_comm(rover_loc) # Move to comm point
                                min_goal_cost = min(min_goal_cost, cost)

                    # Option 2: Sample needs to be collected
                    if waypoint in current_rock_samples:
                        rock_rovers = [r for r in self.objects_by_type.get('rover', set()) if 'rock' in self.rover_equipment.get(r, set())]
                        if rock_rovers:
                            for rover_to_use in rock_rovers:
                                rover_loc = rover_locations.get(rover_to_use)
                                if rover_loc:
                                    cost = 1 # sample_rock action
                                    cost += self.get_distance(rover_loc, waypoint) # Move to sample location
                                    # Check if rover's store is full
                                    store = self.rover_store.get(rover_to_use)
                                    if store and store_status.get(store) == 'full':
                                        cost += 1 # drop action
                                    cost += 1 # communicate action
                                    cost += self.get_min_dist_to_comm(waypoint) # Move from sample location to comm point
                                    min_goal_cost = min(min_goal_cost, cost)

                    total_heuristic_cost += min_goal_cost


            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                if (objective, mode) not in communicated_image:
                    min_goal_cost = float('inf')

                    # Option 1: Image is already taken by a rover/camera
                    rovers_with_image = [r for r in self.objects_by_type.get('rover', set()) if (r, objective, mode) in have_image]
                    if rovers_with_image:
                         for rover_w_image in rovers_with_image:
                            rover_loc = rover_locations.get(rover_w_image)
                            if rover_loc:
                                cost = 1 # communicate action
                                cost += self.get_min_dist_to_comm(rover_loc) # Move to comm point
                                min_goal_cost = min(min_goal_cost, cost)

                    # Option 2: Image needs to be taken
                    suitable_rover_camera_pairs = []
                    for r in self.objects_by_type.get('rover', set()):
                        if 'imaging' in self.rover_equipment.get(r, set()):
                            for cam, info in self.camera_info.items():
                                if info.get('rover') == r and mode in info.get('modes', set()):
                                    suitable_rover_camera_pairs.append((r, cam))

                    if suitable_rover_camera_pairs:
                        image_waypoints = self.objective_visibility.get(objective, set())
                        if image_waypoints:
                            for rover_to_use, camera_to_use in suitable_rover_camera_pairs:
                                rover_loc = rover_locations.get(rover_to_use)
                                if rover_loc:
                                    # Find best image waypoint for this rover
                                    min_img_dist_from_loc = float('inf')
                                    best_img_waypoint = None
                                    for img_wp in image_waypoints:
                                        dist = self.get_distance(rover_loc, img_wp)
                                        if dist < min_img_dist_from_loc:
                                            min_img_dist_from_loc = dist
                                            best_img_waypoint = img_wp

                                    if best_img_waypoint:
                                        cost = 1 # take_image action
                                        cost += 1 # communicate action
                                        cost += self.get_min_dist_to_comm(best_img_waypoint) # Move from image location to comm point

                                        # Calibration cost and movement
                                        if (camera_to_use, rover_to_use) not in calibrated_cameras:
                                            cal_target = self.calibration_targets.get(camera_to_use)
                                            if cal_target:
                                                cal_waypoints = self.objective_visibility.get(cal_target, set())
                                                if cal_waypoints:
                                                    # Find best calibration waypoint for this rover
                                                    min_cal_dist_from_loc = float('inf')
                                                    best_cal_waypoint = None
                                                    for cal_wp in cal_waypoints:
                                                        dist = self.get_distance(rover_loc, cal_wp)
                                                        if dist < min_cal_dist_from_loc:
                                                            min_cal_dist_from_loc = dist
                                                            best_cal_waypoint = cal_wp

                                                    if best_cal_waypoint:
                                                        cost += 1 # calibrate action
                                                        # Movement: L -> W -> P
                                                        cost += self.get_distance(rover_loc, best_cal_waypoint) # Move to cal point
                                                        cost += self.get_distance(best_cal_waypoint, best_img_waypoint) # Move from cal point to image point
                                                    else:
                                                        cost += float('inf') # No reachable cal waypoint
                                                else:
                                                    cost += float('inf') # No waypoint to view cal target from
                                            else:
                                                cost += float('inf') # No cal target for camera
                                        else:
                                            # Camera is calibrated, just need to move to image point
                                            cost += self.get_distance(rover_loc, best_img_waypoint)

                                        min_goal_cost = min(min_goal_cost, cost)

                    total_heuristic_cost += min_goal_cost

        return total_heuristic_cost
