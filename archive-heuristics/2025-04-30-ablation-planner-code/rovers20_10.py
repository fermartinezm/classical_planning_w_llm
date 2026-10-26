from fnmatch import fnmatch
import collections
# Assuming heuristic_base.py exists and contains:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        # These attributes are typically provided by the planning task object
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state
        self.objects = task.objects # Need objects to identify types

    def __call__(self, node):
        raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(rover_graph, start_node, target_nodes):
    """
    Performs BFS to find the shortest path distance from start_node to any node in target_nodes
    using the given rover_graph (adjacency list: waypoint -> set of reachable waypoints).
    Returns float('inf') if no path exists or target_nodes is empty.
    """
    if not target_nodes:
        return float('inf')
    if start_node in target_nodes:
        return 0

    queue = collections.deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        # Check if current_node is in the graph keys before accessing neighbors
        # A waypoint might exist but have no outgoing can_traverse links for this rover
        if current_node in rover_graph:
            for neighbor in rover_graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in target_nodes:
                        return dist + 1
                    queue.append((neighbor, dist + 1))
    return float('inf') # No path found

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal condition. It calculates the cost to acquire necessary
    data (sample soil/rock, take image) and the cost to communicate that data.
    For sampling/imaging, it greedily selects the capable rover/camera that appears
    cheapest based on current location and state (store full, camera calibrated).
    Communication cost is added once per rover that needs to communicate any data.
    Navigation costs are estimated using BFS on the rover-specific traverse graph.

    # Assumptions:
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Soil and rock samples, calibration targets, and objectives are static entities
      in the environment, although samples can be removed by sampling.
    - The lander location is static.
    - Waypoint visibility and rover traverse capabilities are static.
    - The heuristic does not account for resource contention (e.g., multiple rovers
      needing the same store or camera simultaneously).
    - The heuristic does not account for complex interactions like dropping samples
      to free up a store for a *different* sample type needed for a goal. It only
      considers dropping the current sample if the store is full and a *new* sample
      needs to be collected.
    - The heuristic does not try to find optimal paths that combine multiple goal
      achievements (e.g., visiting a waypoint to sample soil and rock in one trip).
      It calculates costs for each unachieved goal independently (except for grouping
      communication costs per rover).

    # Heuristic Initialization
    The heuristic pre-processes static information from the task definition:
    - Identifies rover capabilities (soil, rock, imaging).
    - Maps rovers to their stores and cameras.
    - Maps cameras to supported modes and calibration targets.
    - Builds the waypoint traversal graph for each rover.
    - Stores the lander location and waypoint visibility information.
    - Identifies initial locations of soil and rock samples (from initial state).
    - Pre-calculates sets of waypoints suitable for communication (visible from lander),
      calibration (visible from calibration target), and imaging (visible from objective).
    - Extracts the set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the location of each rover, the status
       of stores, which samples/images rovers have, which cameras are calibrated,
       and which data has been communicated. Also, identify remaining soil/rock samples
       on the ground.
    2. Identify the set of goal conditions that are not yet satisfied in the current state.
    3. If all goal conditions are satisfied, the heuristic value is 0.
    4. Initialize the total estimated cost to 0.
    5. Initialize a set `rovers_needing_communication` to track which rovers will
       eventually need to communicate data to satisfy the remaining goals.
    6. For each unachieved `communicated_soil_data ?w` goal:
       - Check if any rover currently has `(have_soil_analysis ?r ?w)`. If yes,
         identify one such rover and mark it as needing communication. No sampling cost is added.
       - If no rover has the sample and `(at_soil_sample ?w)` is true:
         - Find the soil-equipped rover `?r` that minimizes the cost to get the sample:
           navigation from `?r`'s current location to `?w` + (1 if `?r`'s store is full, for a drop action) + 1 (sample action).
         - Add this minimum cost to the total cost.
         - Add the chosen rover `?r` to `rovers_needing_communication`.
       - If no rover has the sample and `(at_soil_sample ?w)` is false: The goal is impossible; return infinity.
    7. For each unachieved `communicated_rock_data ?w` goal: Follow the same logic as for soil data, using rock-equipped rovers and rock samples.
    8. For each unachieved `communicated_image_data ?o ?m` goal:
       - Check if any rover currently has `(have_image ?r ?o ?m)`. If yes,
         identify one such rover and mark it as needing communication. No imaging cost is added.
       - If no rover has the image:
         - Find the imaging-equipped rover `?r` with a camera `?i` supporting mode `?m`
           that minimizes the cost to take the image:
           navigation from `?r`'s current location to a suitable calibration waypoint for `?i`
           + (1 if `?i` is not calibrated, for a calibrate action) + 1 (calibrate action)
           + navigation from the calibration waypoint to a suitable imaging waypoint for `?o`
           + 1 (take_image action).
         - Add this minimum cost to the total cost.
         - Add the chosen rover `?r` to `rovers_needing_communication`.
       - If no such rover/camera exists or required waypoints are unreachable: The goal is impossible; return infinity.
    9. For each rover in the `rovers_needing_communication` set:
       - Calculate the minimum navigation cost from the rover's current location to any waypoint
         from which it can communicate (visible from the lander).
       - Add this navigation cost + 1 (communicate action) to the total cost.
       - If no communication point is reachable, the state is unsolvable; return infinity.
    10. Return the total estimated cost.
    """
    def __init__(self, task):
        # Use the dummy Heuristic base class attributes if not inheriting from a real one
        # super().__init__(task) # Uncomment if inheriting

        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state # Need initial state for sample locations
        task_objects = task.objects # Need objects to identify types

        # --- Static Information ---
        self.rover_capabilities = collections.defaultdict(set) # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.store_to_rover = {} # store -> rover
        self.rover_cameras = collections.defaultdict(list) # rover -> list of cameras
        self.camera_modes = collections.defaultdict(set) # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.rover_waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> set of reachable waypoints
        self.lander_location = None
        self.visible_waypoints = collections.defaultdict(set) # waypoint -> set of visible waypoints
        self.objective_visible_from = collections.defaultdict(set) # objective -> set of waypoints

        # Get all object names by type
        self.all_rovers = [obj_name for obj_name, obj_type in task_objects if obj_type == 'rover']
        self.all_cameras = [obj_name for obj_name, obj_type in task_objects if obj_type == 'camera']
        self.all_stores = [obj_name for obj_name, obj_type in task_objects if obj_type == 'store']
        self.all_objectives = [obj_name for obj_name, obj_type in task_objects if obj_type == 'objective']
        self.all_modes = [obj_name for obj_name, obj_type in task_objects if obj_type == 'mode']
        self.all_waypoints = [obj_name for obj_name, obj_type in task_objects if obj_type == 'waypoint']

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif predicate == 'store_of':
                self.store_to_rover[parts[1]] = parts[2] # store -> rover
            elif predicate == 'on_board':
                self.rover_cameras[parts[2]].append(parts[1]) # rover -> camera
            elif predicate == 'supports':
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == 'calibration_target':
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective
            elif predicate == 'can_traverse':
                # Ensure rover exists in graph even if no links yet
                if parts[1] not in self.rover_waypoint_graph:
                     self.rover_waypoint_graph[parts[1]] = collections.defaultdict(set)
                self.rover_waypoint_graph[parts[1]][parts[2]].add(parts[3]) # rover -> from -> to
            elif predicate == 'at_lander':
                self.lander_location = parts[2]
            elif predicate == 'visible':
                self.visible_waypoints[parts[1]].add(parts[2]) # from -> to
            elif predicate == 'visible_from':
                self.objective_visible_from[parts[1]].add(parts[2]) # objective -> waypoint

        # Initial sample locations are in the initial state
        self.soil_sample_locations_init = set()
        self.rock_sample_locations_init = set()
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'at_soil_sample':
                 self.soil_sample_locations_init.add(parts[1])
             elif predicate == 'at_rock_sample':
                 self.rock_sample_locations_init.add(parts[1])

        # Pre-calculate communication points for each rover
        self.rover_comm_points = collections.defaultdict(set)
        lander_comm_points = self.visible_waypoints.get(self.lander_location, set())
        lander_comm_points.add(self.lander_location) # Lander location itself is a comm point

        for rover in self.all_rovers:
             # A rover can communicate from any waypoint visible from the lander
             # that the rover can actually reach.
             # For simplicity, let's assume any waypoint visible from the lander is a potential comm point.
             # BFS will handle reachability.
             self.rover_comm_points[rover] = lander_comm_points


        # Pre-calculate calibration points for each camera
        self.camera_cal_points = collections.defaultdict(set) # camera -> set of waypoints
        for camera, target_obj in self.camera_calibration_target.items():
            self.camera_cal_points[camera] = self.objective_visible_from.get(target_obj, set())

        # Pre-calculate imaging points for each objective
        self.objective_img_points = self.objective_visible_from # objective -> set of waypoints


    def __call__(self, node):
        state = node.state

        # --- State Information ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        rover_soil_samples = collections.defaultdict(set) # rover -> set of waypoints
        rover_rock_samples = collections.defaultdict(set) # rover -> set of waypoints
        rover_images = collections.defaultdict(set) # rover -> set of (objective, mode)
        camera_calibrated = set() # set of (camera, rover)
        communicated_soil = set() # set of waypoints
        communicated_rock = set() # set of waypoints
        communicated_images = set() # set of (objective, mode)
        soil_samples_remaining = set() # set of waypoints on the ground
        rock_samples_remaining = set() # set of waypoints on the ground

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                # Could be rover or lander, only care about rovers here
                if parts[1] in self.all_rovers:
                     rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty':
                store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis':
                rover_soil_samples[parts[1]].add(parts[2])
            elif predicate == 'have_rock_analysis':
                rover_rock_samples[parts[1]].add(parts[2])
            elif predicate == 'have_image':
                rover_images[parts[1]].add((parts[2], parts[3]))
            elif predicate == 'calibrated':
                camera_calibrated.add((parts[1], parts[2]))
            elif predicate == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                communicated_images.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample':
                soil_samples_remaining.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_remaining.add(parts[1])

        # --- Identify Unachieved Goals ---
        is_goal = True
        soil_goals_to_achieve = set()
        rock_goals_to_achieve = set()
image_goals_to_achieve = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint not in communicated_soil:
                    is_goal = False
                    soil_goals_to_achieve.add(waypoint)
            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint not in communicated_rock:
                    is_goal = False
                    rock_goals_to_achieve.add(waypoint)
            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) not in communicated_images:
                    is_goal = False
                    image_goals_to_achieve.add((objective, mode))
            # Assuming only these goal types exist based on domain

        if is_goal:
            return 0

        # --- Calculate Heuristic Cost ---
        total_cost = 0
        rovers_needing_communication = set() # Track which rovers will end up with data to communicate

        # Process unachieved soil goals
        for w in soil_goals_to_achieve:
            min_cost_sample = float('inf')
            best_rover_for_sample = None

            # 1. Check if any rover already has the sample
            rover_already_has = None
            for r in self.all_rovers:
                if w in rover_soil_samples[r]:
                    rover_already_has = r
                    break

            if rover_already_has:
                best_rover_for_sample = rover_already_has
                min_cost_sample = 0 # No sampling cost needed
            # 2. If not, check if sample is on the ground and can be collected
            elif w in soil_samples_remaining:
                for r in self.all_rovers:
                    if 'soil' in self.rover_capabilities.get(r, set()):
                        # Find the store for this rover
                        store = None
                        for s, r_owner in self.store_to_rover.items():
                            if r_owner == r:
                                store = s
                                break
                        if store is None: continue # Rover must have a store

                        current_rover_wp = rover_locations.get(r)
                        if current_rover_wp is None: continue # Rover location unknown

                        rover_graph = self.rover_waypoint_graph.get(r, {})
                        nav_cost = bfs(rover_graph, current_rover_wp, {w})
                        if nav_cost == float('inf'): continue # Cannot reach sample location

                        store_full = (store_status.get(store) == 'full')
                        current_cost = nav_cost + (1 if store_full else 0) + 1 # nav + drop (if needed) + sample

                        if current_cost < min_cost_sample:
                            min_cost_sample = current_cost
                            best_rover_for_sample = r
            else: # Sample is gone from the ground and no rover has it
                 # This goal is impossible from this state
                 return float('inf')

            if min_cost_sample == float('inf'):
                 # Cannot sample or get sample data
                 return float('inf')

            total_cost += min_cost_sample
            if best_rover_for_sample:
                rovers_needing_communication.add(best_rover_for_sample)


        # Process unachieved rock goals
        for w in rock_goals_to_achieve:
            min_cost_sample = float('inf')
            best_rover_for_sample = None

            # 1. Check if any rover already has the sample
            rover_already_has = None
            for r in self.all_rovers:
                if w in rover_rock_samples[r]:
                    rover_already_has = r
                    break

            if rover_already_has:
                best_rover_for_sample = rover_already_has
                min_cost_sample = 0 # No sampling cost needed
            # 2. If not, check if sample is on the ground and can be collected
            elif w in rock_samples_remaining:
                for r in self.all_rovers:
                    if 'rock' in self.rover_capabilities.get(r, set()):
                        # Find the store for this rover
                        store = None
                        for s, r_owner in self.store_to_rover.items():
                            if r_owner == r:
                                store = s
                                break
                        if store is None: continue # Rover must have a store

                        current_rover_wp = rover_locations.get(r)
                        if current_rover_wp is None: continue # Rover location unknown

                        rover_graph = self.rover_waypoint_graph.get(r, {})
                        nav_cost = bfs(rover_graph, current_rover_wp, {w})
                        if nav_cost == float('inf'): continue # Cannot reach sample location

                        store_full = (store_status.get(store) == 'full')
                        current_cost = nav_cost + (1 if store_full else 0) + 1 # nav + drop (if needed) + sample

                        if current_cost < min_cost_sample:
                            min_cost_sample = current_cost
                            best_rover_for_sample = r
            else: # Sample is gone from the ground and no rover has it
                 # This goal is impossible from this state
                 return float('inf')

            if min_cost_sample == float('inf'):
                 # Cannot sample or get sample data
                 return float('inf')

            total_cost += min_cost_sample
            if best_rover_for_sample:
                rovers_needing_communication.add(best_rover_for_sample)

        # Process unachieved image goals
        for o, m in image_goals_to_achieve:
            min_cost_image = float('inf')
            best_rover_for_image = None

            # 1. Check if any rover already has the image
            rover_already_has = None
            for r in self.all_rovers:
                if (o, m) in rover_images[r]:
                    rover_already_has = r
                    break

            if rover_already_has:
                best_rover_for_image = rover_already_has
                min_cost_image = 0 # No imaging cost needed
            # 2. If not, calculate cost to take the image
            else:
                for r in self.all_rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        for i in self.rover_cameras.get(r, []):
                            if m in self.camera_modes.get(i, set()):
                                # Found a capable rover and camera

                                cal_points = self.camera_cal_points.get(i, set())
                                img_points = self.objective_img_points.get(o, set())

                                if not cal_points or not img_points: continue # Cannot calibrate or image this objective/mode

                                rover_graph = self.rover_waypoint_graph.get(r, {})
                                current_rover_wp = rover_locations.get(r)
                                if current_rover_wp is None: continue # Rover location unknown?

                                # Find the cal point closest to the rover's current location that also allows reaching an img point
                                min_total_nav_cal_img = float('inf')
                                best_cal_wp_for_nav = None # The cal point that minimizes rover->cal + cal->img

                                for cal_wp in cal_points:
                                    nav_rover_to_this_cal = bfs(rover_graph, current_rover_wp, {cal_wp})
                                    if nav_rover_to_this_cal == float('inf'): continue

                                    min_nav_this_cal_to_img = bfs(rover_graph, cal_wp, img_points)

                                    if min_nav_this_cal_to_img == float('inf'): continue # Cannot reach any img point from this cal point

                                    total_nav_cost_via_this_cal = nav_rover_to_this_cal + min_nav_this_cal_to_img

                                    if total_nav_cost_via_this_cal < min_total_nav_cal_img:
                                        min_total_nav_cal_img = total_nav_cost_via_this_cal
                                        best_cal_wp_for_nav = cal_wp # Store the cal point that achieved this minimum

                                if min_total_nav_cal_img == float('inf'): continue # Cannot find a path via any cal point

                                # The cost is nav_rover_to_best_cal + cal_cost + calibrate + nav_best_cal_to_best_img + take_image
                                # min_total_nav_cal_img already includes nav_rover_to_best_cal + nav_best_cal_to_best_img
                                calibrated_status = (i, r) in camera_calibrated
                                cal_cost = 0 if calibrated_status else 1 # Cost to calibrate if not already

                                current_cost = min_total_nav_cal_img + cal_cost + 1 + 1 # nav_total + cal_cost + calibrate + take_image

                                if current_cost < min_cost_image:
                                    min_cost_image = current_cost
                                    best_rover_for_image = r

        if min_cost_image == float('inf'):
             # Cannot image or get image data
             return float('inf')

        total_cost += min_cost_image
        if best_rover_for_image:
            rovers_needing_communication.add(best_rover_for_image)

    # Process communication costs for rovers that need to communicate
    for r in rovers_needing_communication:
        comm_points = self.rover_comm_points.get(r, set())
        if not comm_points:
            # Rover cannot reach any communication point
            return float('inf')

        current_rover_wp = rover_locations.get(r)
        if current_rover_wp is None:
             # Rover location unknown? Should not happen if rover exists.
             return float('inf')

        rover_graph = self.rover_waypoint_graph.get(r, {})
        min_nav_to_comm = bfs(rover_graph, current_rover_wp, comm_points)

        if min_nav_to_comm == float('inf'):
            # Cannot reach any communication point
            return float('inf')

        total_cost += min_nav_to_comm + 1 # nav + communicate

    return total_cost
