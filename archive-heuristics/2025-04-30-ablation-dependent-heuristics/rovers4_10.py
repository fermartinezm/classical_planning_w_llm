import collections
import math

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """
    Parses a PDDL fact string into a predicate and its arguments.
    e.g., '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    """
    # Removes parentheses and splits by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper functions to parse specific facts (for clarity, can be done inline)
def parse_at(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1], parts[2] # rover, waypoint

def parse_have_soil(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1], parts[2] # rover, waypoint

def parse_have_rock(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1], parts[2] # rover, waypoint

def parse_have_image(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1], parts[2], parts[3] # rover, objective, mode

def parse_full(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1] # store

def parse_calibrated(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1], parts[2] # camera, rover

def parse_communicated_soil(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1] # waypoint

def parse_communicated_rock(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1] # waypoint

def parse_communicated_image(fact_str):
    parts = fact_str[1:-1].split()
    return parts[1], parts[2] # objective, mode


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up:
    1. The number of 'sample_soil', 'sample_rock', and 'take_image' actions
       required to obtain the items needed for unachieved goal facts.
    2. The number of 'communicate' actions required for items that are
       already obtained but not yet communicated.
    3. The number of 'calibrate' actions needed for imaging tasks, counting
       one calibration per image needed per camera-rover pair if not calibrated.
    4. The minimum number of 'drop' actions needed for sampling tasks, based
       on the number of samples required and the number of capable rovers
       with empty stores.
    5. An estimate of the navigation cost, calculated as the sum, over all rovers,
       of the minimum distance from the rover's current location to any waypoint
       relevant to its potential tasks (sampling locations, imaging locations,
       calibration locations, or communication points).

    Assumptions:
    - The navigation graph defined by 'can_traverse' is symmetric and connected
      for relevant waypoints for each rover.
    - 'can_traverse ?r ?y ?z' implies 'visible ?y ?z'.
    - 'visible ?y ?z' implies 'visible ?z ?y'.
    - 'at_soil_sample' and 'at_rock_sample' facts are removed permanently after sampling.
    - A rover can only hold one sample/image at a time (due to single store).
    - Calibration state is per camera-rover pair and is lost after 'take_image'.
    - Any rover capable of sampling can sample any needed soil/rock sample.
    - Any imaging rover with a suitable camera can take any needed image.
    - Any rover with an item can communicate it from a communication point.
    - The set of waypoints and objects is static.

    Heuristic Initialization:
    The constructor precomputes static information from the task:
    - Lander location and communication points (waypoints visible from lander).
    - Rover capabilities (equipped for soil/rock/imaging).
    - Store mapping (store to rover).
    - Camera information (on board, supports mode, calibration target).
    - Waypoint visibility and traversability per rover.
    - Shortest path distances between all pairs of waypoints for each rover using BFS.
    - Waypoints visible from objectives and calibration targets.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to get dynamic facts: rover locations, samples/images held,
       store status (full/empty), camera calibration status.
    2. Identify unachieved goal facts.
    3. Categorize unachieved goals:
       - `needed_soil_samples`: Waypoints where soil data is needed and no rover has the sample yet.
       - `needed_rock_samples`: Waypoints where rock data is needed and no rover has the sample yet.
       - `needed_images`: (objective, mode) pairs where image data is needed and no rover has the image yet.
       - `items_to_communicate`: (type, item) pairs (soil waypoint, rock waypoint, image (o,m))
         where data is needed, some rover *does* have the item, but it hasn't been communicated.
    4. Initialize heuristic `h = 0`.
    5. Add action costs for obtaining items:
       - `h += len(needed_soil_samples)` (for sample_soil actions)
       - `h += len(needed_rock_samples)` (for sample_rock actions)
       - `h += len(needed_images)` (for take_image actions)
    6. Add action costs for communication:
       - `h += len(items_to_communicate)` (for communicate actions)
    7. Add calibration costs:
       - Identify `uncalibrated_suitable_cameras`: Set of (rover, camera) pairs where the camera is needed
         for a `needed_image` and is not currently calibrated on that rover.
       - Count how many `needed_images` require each such uncalibrated `(r, i)` pair. Sum these counts.
       - `h += calibration_actions_count` (for calibrate actions)
    8. Add drop costs:
       - Calculate `num_needed_samples = len(needed_soil_samples) + len(needed_rock_samples)`.
       - Identify capable sampling rovers (`soil_equipped` or `rock_equipped`).
       - Count capable sampling rovers with empty stores.
       - `drops_needed_count = max(0, num_needed_samples - num_sampling_rovers_empty_store)`.
       - `h += drops_needed_count` (for drop actions)
    9. Estimate navigation cost:
       - Initialize `nav_cost = 0`.
       - For each rover `?r`:
         - Get its current location `?r_loc`.
         - Identify `potential_targets` for this rover based on its capabilities and the needed tasks.
         - If `potential_targets` is not empty and `?r_loc` is reachable by `?r`, find the minimum distance from `?r_loc` to any waypoint in `potential_targets` using the precomputed distances for `?r`. Add this minimum distance to `nav_cost`.
       - `h += nav_cost`.
    10. Return `h`. If `h` is 0, it must be a goal state (checked first).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.
        """
        self.task = task
        self.static = task.static

        # Precompute static information
        self.lander_location = None
        self.comm_points = set()
        self.rover_capabilities = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_store_map = {} # store -> rover
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = collections.defaultdict(set) # camera -> {modes}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_locations = collections.defaultdict(set) # objective -> {waypoints visible from objective}
        self.calibration_locations = collections.defaultdict(set) # calibration_target -> {waypoints visible from target}
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # Navigation graph: rover -> waypoint -> {neighbor_waypoint}
        self.nav_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        # Shortest distances: rover -> waypoint -> waypoint -> distance
        self.dist = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: math.inf)))

        # Parse static facts
        for fact_str in self.static:
            predicate, args = parse_fact(fact_str)
            if predicate == 'at_lander':
                self.landers.add(args[0])
                self.lander_location = args[1]
                self.waypoints.add(args[1])
            elif predicate == 'equipped_for_soil_analysis':
                self.rovers.add(args[0])
                self.rover_capabilities[args[0]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                self.rovers.add(args[0])
                self.rover_capabilities[args[0]].add('rock')
            elif predicate == 'equipped_for_imaging':
                self.rovers.add(args[0])
                self.rover_capabilities[args[0]].add('imaging')
            elif predicate == 'store_of':
                self.stores.add(args[0])
                self.rovers.add(args[1])
                self.rover_store_map[args[0]] = args[1]
            elif predicate == 'on_board':
                self.cameras.add(args[0])
                self.rovers.add(args[1])
                self.camera_on_board[args[0]] = args[1]
            elif predicate == 'supports':
                self.cameras.add(args[0])
                self.modes.add(args[1])
                self.camera_supports[args[0]].add(args[1])
            elif predicate == 'calibration_target':
                self.cameras.add(args[0])
                self.objectives.add(args[1])
                self.camera_calibration_target[args[0]] = args[1]
            elif predicate == 'visible':
                self.waypoints.add(args[0])
                self.waypoints.add(args[1])
                # Visibility is symmetric
            elif predicate == 'can_traverse':
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                self.waypoints.add(args[2])
                # Assuming can_traverse is symmetric based on example
                self.nav_graph[args[0]][args[1]].add(args[2])
                self.nav_graph[args[0]][args[2]].add(args[1])
            elif predicate == 'visible_from':
                self.objectives.add(args[0])
                self.waypoints.add(args[1])
                self.objective_locations[args[0]].add(args[1])
                # Check if this objective is a calibration target
                for cam, obj in self.camera_calibration_target.items():
                    if obj == args[0]:
                        self.calibration_locations[obj].add(args[1]) # Store by objective for easier lookup

        # Precompute communication points (waypoints visible from lander)
        if self.lander_location:
             # Build a temporary graph just for visibility to find comm points
            visible_graph = collections.defaultdict(set)
            for fact_str in self.static:
                 predicate, args = parse_fact(fact_str)
                 if predicate == 'visible':
                     visible_graph[args[0]].add(args[1])
                     visible_graph[args[1]].add(args[0]) # Assuming symmetric
            if self.lander_location in visible_graph:
                 self.comm_points = visible_graph[self.lander_location]


        # Precompute shortest paths for each rover
        for rover in self.rovers:
            if rover not in self.nav_graph:
                 # Rover cannot traverse anywhere, skip pathfinding for this rover
                 continue
            for start_node in self.waypoints:
                # Only compute if the start_node is part of the rover's graph
                # Check if start_node is a key in the nav_graph for this rover
                if start_node not in self.nav_graph[rover]:
                     continue # Rover cannot start from this waypoint based on can_traverse

                q = collections.deque([(start_node, 0)])
                visited = {start_node}
                self.dist[rover][start_node][start_node] = 0

                while q:
                    curr_node, curr_dist = q.popleft()

                    for neighbor in self.nav_graph[rover].get(curr_node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dist[rover][start_node][neighbor] = curr_dist + 1
                            q.append((neighbor, curr_dist + 1))

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # --- Parse State ---
        rovers_current_loc = {} # rover -> waypoint
        rover_has_soil = set() # (rover, waypoint)
        rover_has_rock = set() # (rover, waypoint)
        rover_has_image = set() # (rover, objective, mode)
        rover_store_full = set() # {rover}
        camera_calibrated = set() # (camera, rover)
        # at_soil_sample_state = set() # {waypoint} # Not strictly needed for heuristic logic
        # at_rock_sample_state = set() # {waypoint} # Not strictly needed for heuristic logic

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'at':
                rovers_current_loc[args[0]] = args[1]
            elif predicate == 'have_soil_analysis':
                rover_has_soil.add((args[0], args[1]))
            elif predicate == 'have_rock_analysis':
                rover_has_rock.add((args[0], args[1]))
            elif predicate == 'have_image':
                rover_has_image.add(tuple(args)) # (rover, objective, mode)
            elif predicate == 'full':
                 # Find the rover associated with this store
                 for store, rover in self.rover_store_map.items():
                     if store == args[0]:
                         rover_store_full.add(rover)
                         break
            elif predicate == 'calibrated':
                camera_calibrated.add((args[0], args[1])) # (camera, rover)
            # elif predicate == 'at_soil_sample':
            #      at_soil_sample_state.add(args[0])
            # elif predicate == 'at_rock_sample':
            #      at_rock_sample_state.add(args[0])


        # --- Identify Needed Tasks ---
        needed_soil_samples = set() # {waypoint}
        needed_rock_samples = set() # {waypoint}
        needed_images = set() # {(objective, mode)}
        items_to_communicate = set() # {('soil', waypoint), ('rock', waypoint), ('image', (objective, mode))}

        for goal_fact_str in self.task.goals:
            if goal_fact_str in state:
                continue # Goal already achieved

            predicate, args = parse_fact(goal_fact_str)

            if predicate == 'communicated_soil_data':
                w = args[0]
                # Check if any rover has the sample
                if any((r, w) in rover_has_soil for r in rovers_current_loc):
                    items_to_communicate.add(('soil', w))
                else:
                    # Need to sample. Assuming sample was initially present if needed.
                    needed_soil_samples.add(w)

            elif predicate == 'communicated_rock_data':
                w = args[0]
                # Check if any rover has the sample
                if any((r, w) in rover_has_rock for r in rovers_current_loc):
                    items_to_communicate.add(('rock', w))
                else:
                    # Need to sample
                    needed_rock_samples.add(w)

            elif predicate == 'communicated_image_data':
                o, m = args
                # Check if any rover has the image
                if any((r, o, m) in rover_has_image for r in rovers_current_loc):
                    items_to_communicate.add(('image', (o, m)))
                else:
                    # Need to take image
                    needed_images.add((o, m))

        # --- Calculate Heuristic Components ---
        h = 0

        # 1. Action costs for obtaining items (sample, take_image)
        h += len(needed_soil_samples)
        h += len(needed_rock_samples)
        h += len(needed_images)

        # 2. Action costs for communication
        h += len(items_to_communicate)

        # 3. Calibration costs (refined)
        calibration_actions_count = 0
        # Identify (rover, camera) pairs that *could* be used for needed images
        # and are not calibrated.
        uncalibrated_suitable_cameras = set() # {(rover, camera)}
        suitable_camera_for_image = collections.defaultdict(set) # (o, m) -> {(r, i)}

        for o, m in needed_images:
             for cam in self.cameras:
                 if m in self.camera_supports.get(cam, set()):
                     rover = self.camera_on_board.get(cam)
                     if rover and rover in self.rover_capabilities and 'imaging' in self.rover_capabilities[rover]:
                          suitable_camera_for_image[(o, m)].add((rover, cam))
                          if (cam, rover) not in camera_calibrated:
                              uncalibrated_suitable_cameras.add((rover, cam))

        # For each uncalibrated (rover, camera) pair, count how many needed images require it
        for r, i in uncalibrated_suitable_cameras:
             num_images_this_pair_needed_for = 0
             for o, m in needed_images:
                  if (r, i) in suitable_camera_for_image[(o, m)]:
                       num_images_this_pair_needed_for += 1
             calibration_actions_count += num_images_this_pair_needed_for # Assumes re-calibration needed per image

        h += calibration_actions_count

        # 4. Drop costs (simplified)
        num_needed_samples = len(needed_soil_samples) + len(needed_rock_samples)
        sampling_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps or 'rock' in caps}
        num_sampling_rovers_empty_store = sum(1 for r in sampling_rovers if r not in rover_store_full)
        drops_needed_count = max(0, num_needed_samples - num_sampling_rovers_empty_store)
        h += drops_needed_count

        # 5. Navigation costs
        nav_cost = 0
        for rover, r_loc in rovers_current_loc.items():
            potential_targets = set()

            # Targets for sampling
            if 'soil' in self.rover_capabilities.get(rover, set()):
                potential_targets.update(needed_soil_samples)
            if 'rock' in self.rover_capabilities.get(rover, set()):
                potential_targets.update(needed_rock_samples)

            # Targets for imaging/calibration
            if 'imaging' in self.rover_capabilities.get(rover, set()):
                rover_cameras = {cam for cam, r in self.camera_on_board.items() if r == rover}
                for o, m in needed_images:
                    # Check if rover has a camera supporting this mode
                    suitable_cams_on_rover = {cam for cam in rover_cameras if m in self.camera_supports.get(cam, set())}
                    if suitable_cams_on_rover:
                        # Add image locations
                        potential_targets.update(self.objective_locations.get(o, set()))
                        # Add calibration locations if needed for this rover/camera
                        for cam in suitable_cams_on_rover:
                            if (rover, cam) in uncalibrated_suitable_cameras: # Check if this specific (rover, cam) needs calibration for *any* needed image
                                 cal_target = self.camera_calibration_target.get(cam)
                                 if cal_target:
                                     potential_targets.update(self.calibration_locations.get(cal_target, set()))


            # Targets for communication
            # A rover needs to go to a comm point if it has an item that needs communicating
            # or if it is capable of obtaining items and there are items needing communication.
            rover_has_item_to_comm = False
            for item_type, item_val in items_to_communicate:
                 if item_type == 'soil':
                      if (rover, item_val) in rover_has_soil:
                           rover_has_item_to_comm = True
                           break
                 elif item_type == 'rock':
                      if (rover, item_val) in rover_has_rock:
                           rover_has_item_to_comm = True
                           break
                 elif item_type == 'image':
                      if (rover, item_val[0], item_val[1]) in rover_has_image:
                           rover_has_item_to_comm = True
                           break
            if rover_has_item_to_comm:
                 potential_targets.update(self.comm_points)
            # Add comm points as potential targets if there are items to communicate
            # and this rover is capable of obtaining items (might be assigned a task).
            # This helps guide rovers towards comm points even if they don't hold the item yet.
            elif items_to_communicate:
                 is_capable_of_obtaining = ('soil' in self.rover_capabilities.get(rover, set())) or \
                                          ('rock' in self.rover_capabilities.get(rover, set())) or \
                                          ('imaging' in self.rover_capabilities.get(rover, set()))
                 if is_capable_of_obtaining:
                      potential_targets.update(self.comm_points)


            # Calculate min distance to any potential target
            min_dist = math.inf
            if potential_targets:
                # Ensure rover can actually navigate from its current location
                if rover in self.dist and r_loc in self.dist[rover]:
                    for target_loc in potential_targets:
                        if target_loc in self.dist[rover][r_loc]:
                             min_dist = min(min_dist, self.dist[rover][r_loc][target_loc])

            if min_dist != math.inf:
                nav_cost += min_dist

        h += nav_cost

        # The goal_reached check at the beginning ensures h=0 only at the goal.
        # If the calculation somehow resulted in 0 for a non-goal state,
        # the initial check would have already returned 0.

        return h
