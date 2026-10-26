from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections
import math # For float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove surrounding parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid facts


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs_min_distance(graph, start_node, target_waypoint_set):
    """
    Computes the minimum navigation cost (number of moves) for a rover
    from a start waypoint to any waypoint in the target set using BFS.
    Returns float('inf') if no path exists to any target or if inputs are invalid.
    """
    if not start_node or not target_waypoint_set:
         return math.inf # Invalid input or no targets

    if start_node in target_waypoint_set:
        return 0

    if not graph or start_node not in graph: # Check if graph is valid and start_node is in graph keys
        return math.inf # Rover cannot traverse from start_node

    queue = collections.deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        if current_node in graph: # Ensure current_node is a key in the graph
            for neighbor in graph[current_node]:
                if neighbor in target_waypoint_set:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return math.inf # No path found


def find_closest_waypoint(graph, start_waypoint, target_waypoint_set):
    """
    Finds the waypoint in target_waypoint_set closest to start_waypoint
    for the given rover's traversal graph, and returns the navigation cost
    and the waypoint.
    Returns (math.inf, None) if no path exists or inputs are invalid.
    """
    if not start_waypoint or not target_waypoint_set:
         return (math.inf, None)

    if start_waypoint in target_waypoint_set:
        return (0, start_waypoint)

    if not graph or start_waypoint not in graph: # Check if graph is valid and start_node is in graph keys
        return (math.inf, None) # Cannot traverse from start_node

    queue = collections.deque([(start_waypoint, 0)])
    visited = {start_waypoint}

    while queue:
        current_node, dist = queue.popleft()

        if current_node in graph: # Ensure current_node is a key in the graph
            for neighbor in graph[current_node]:
                if neighbor in target_waypoint_set:
                    # Found a target, this is the shortest path to *a* target
                    return (dist + 1, neighbor) # Return immediately

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return (math.inf, None) # No path found


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the total number of actions and significant navigation steps
    required to achieve all unachieved goal conditions. It sums the estimated costs
    for each goal independently. For each goal, it estimates the cost to obtain the
    required data (soil sample, rock sample, or image) and the cost to communicate
    that data. Navigation costs are estimated using BFS on the rover's traversal graph.

    # Assumptions
    - The cost of each action (sample, drop, calibrate, take_image, communicate) is 1.
    - The cost of navigation is the number of 'navigate' actions (shortest path).
    - The heuristic sums costs for goals independently, ignoring potential synergies
      (e.g., one trip for multiple samples/images/communications).
    - When estimating navigation costs for data acquisition (sampling/imaging/calibrating),
      the starting point is the rover's current location.
    - When estimating navigation costs for communication, the starting point is the
      estimated location where the data was acquired (sample waypoint, imaging waypoint),
      or the rover's current location if the data was already held.
    - If a required resource (equipped rover, visible waypoint, calibration target)
      does not exist or is unreachable, the cost for that goal component is infinite.

    # Heuristic Initialization
    - Extracts static information from the task, including:
        - Rover capabilities (soil, rock, imaging).
        - Store ownership for rovers.
        - Cameras on board rovers and modes they support.
        - Calibration targets for cameras.
        - Waypoints visible from objectives (for imaging) and calibration targets.
        - Lander location and communication waypoints (visible from lander).
        - Rover traversal graphs (assuming symmetric traversal).
    - Identifies all rovers mentioned in static facts and the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get the current status of all relevant predicates
       (rover locations, samples, analysis, images, calibration, communication, store status).
    3. Iterate through each goal condition specified in the task.
    4. For each goal:
       - If the goal is already satisfied in the current state, add 0 to the total cost.
       - If the goal is `(communicated_soil_data ?w)` and not satisfied:
         - Find the minimum cost over all rovers that can achieve this goal (either already have the data or are equipped for soil analysis).
         - For each candidate rover `?r`:
           - Estimate the cost to get the soil analysis data for `?w`:
             - If `?r` already has the data, cost is 0, and the data location is `?r`'s current location.
             - Otherwise, cost is 1 (`sample_soil`) + navigation cost from `?r`'s current location to `?w`. If `?r`'s store is full, add 1 (`drop`). The data location is `?w`.
           - Estimate the cost to communicate the data:
             - Cost is 1 (`communicate_soil_data`) + navigation cost from the data location (where the sample was obtained or where the rover currently is if it already had the data) to any communication waypoint.
           - The total cost for this rover for this goal is the sum of the data acquisition cost and the communication cost.
         - Add the minimum total cost over all candidate rovers to the overall total cost. If no rover can achieve the goal, the total cost is infinity.
       - If the goal is `(communicated_rock_data ?w)` and not satisfied:
         - Follow similar logic as soil data, using rock-specific predicates and capabilities.
       - If the goal is `(communicated_image_data ?o ?m)` and not satisfied:
         - Find the minimum cost over all rover/camera combinations that can achieve this goal (either already have the image or are suitable for imaging).
         - For each candidate `?r`/`?i` combination:
           - Estimate the cost to get the image data for `?o` and `?m`:
             - If `?r` already has the image, cost is 0, and the data location is `?r`'s current location.
             - Otherwise, cost is 1 (`take_image`).
               - If `?i` is not calibrated on `?r`, add 1 (`calibrate`) + navigation cost from `?r`'s current location to any waypoint visible from `?i`'s calibration target. The rover's estimated location after calibration is the chosen calibration waypoint.
               - Add navigation cost from the rover's estimated location (current or calibration waypoint) to any waypoint visible from `?o` (imaging spot). The data location is the chosen imaging waypoint.
           - Estimate the cost to communicate the data:
             - Cost is 1 (`communicate_image_data`) + navigation cost from the data location (where the image was taken or where the rover currently is if it already had the image) to any communication waypoint.
           - The total cost for this combo for this goal is the sum of the data acquisition cost and the communication cost.
         - Add the minimum total cost over all candidate combos to the overall total cost. If no combo can achieve the goal, the total cost is infinity.
5. Return the total estimated cost. If any goal component was impossible, the total cost will be infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Static mappings and graphs
        self.rover_capabilities = collections.defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.rover_cameras = collections.defaultdict(set) # rover -> {cameras}
        self.camera_modes = collections.defaultdict(set) # camera -> {modes}
        self.camera_calibration_targets = {} # camera -> objective (target)
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoints}
        self.calibration_target_visible_from = collections.defaultdict(set) # objective (target) -> {waypoints}
        self.lander_location = None
        self.communication_waypoint_candidates = set() # waypoints visible from lander
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> {neighbors}

        # Set to identify calibration target objectives
        calibration_objectives_set = set()

        # Process static facts to build core static structures
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'on_board':
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif parts[0] == 'supports':
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif parts[0] == 'calibration_target':
                self.camera_calibration_targets[parts[1]] = parts[2] # camera -> objective (target)
                calibration_objectives_set.add(parts[2])
            elif parts[0] == 'visible_from':
                 self.objective_visible_from[parts[1]].add(parts[2])
            elif parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'can_traverse':
                # Assuming can_traverse is symmetric based on instance examples
                r, w_from, w_to = parts[1], parts[2], parts[3]
                self.rover_traversal_graph[r][w_from].add(w_to)
                self.rover_traversal_graph[r][w_to].add(w_from) # Add reverse edge

        # Now that lander_location is likely populated, process visible facts for communication points
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'visible':
                 w1, w2 = parts[1], parts[2]
                 if self.lander_location:
                     if w1 == self.lander_location:
                         self.communication_waypoint_candidates.add(w2)
                     if w2 == self.lander_location:
                          self.communication_waypoint_candidates.add(w1)

        # Populate calibration_target_visible_from based on camera_calibration_targets and objective_visible_from
        for cal_target_obj in calibration_objectives_set:
             if cal_target_obj in self.objective_visible_from:
                 self.calibration_target_visible_from[cal_target_obj].update(self.objective_visible_from[cal_target_obj])

        # Get the set of all rovers mentioned in static facts and initial state
        self.rovers = set(self.rover_capabilities.keys()) | set(self.rover_stores.keys()) | set(self.rover_cameras.keys()) | set(self.rover_traversal_graph.keys())
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at':
                  self.rovers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse current state into convenient data structures
        current_rover_locations = {} # rover -> waypoint
        current_store_status = {} # store -> 'empty' or 'full'
        current_rover_soil_analysis = collections.defaultdict(set) # rover -> {waypoints}
        current_rover_rock_analysis = collections.defaultdict(set) # rover -> {waypoints}
        current_rover_image_data = collections.defaultdict(set) # rover -> {(objective, mode)}
        current_camera_calibrated = set() # {(camera, rover)}
        current_communicated_soil = set() # {waypoint}
        current_communicated_rock = set() # {waypoint}
        current_communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at':
                current_rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'empty':
                current_store_status[parts[1]] = 'empty'
            elif parts[0] == 'full':
                current_store_status[parts[1]] = 'full'
            elif parts[0] == 'have_soil_analysis':
                current_rover_soil_analysis[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                current_rover_rock_analysis[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                current_rover_image_data[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                current_camera_calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # Process unachieved goals
        for goal in self.goals:
            g_parts = get_parts(goal)
            if not g_parts: continue

            predicate = g_parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = g_parts[1]
                if waypoint not in current_communicated_soil:
                    # Cost for this goal = min_rover_cost(get_data + communicate)
                    best_goal_cost = math.inf

                    # Iterate through all rovers that could potentially achieve this goal
                    candidate_rovers = {r for r in self.rovers if waypoint in current_rover_soil_analysis[r] or 'soil' in self.rover_capabilities[r]}

                    if not candidate_rovers or not self.communication_waypoint_candidates:
                         return math.inf # Goal is unreachable

                    for rover in candidate_rovers:
                         rover_curr_w = current_rover_locations.get(rover)
                         if not rover_curr_w: continue # Rover location unknown

                         # --- Cost to get soil analysis data for this rover ---
                         get_data_cost = 0
                         data_location = rover_curr_w # Where the data is obtained (or already exists)

                         if waypoint not in current_rover_soil_analysis[rover]:
                             # Need to sample
                             get_data_cost += 1 # sample_soil action
                             nav_to_sample_cost, sample_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), rover_curr_w, {waypoint})
                             if nav_to_sample_cost == math.inf: continue # Cannot reach sample spot

                             get_data_cost += nav_to_sample_cost
                             data_location = sample_wp # Data is obtained at the sample waypoint

                             store = self.rover_stores.get(rover)
                             if store and current_store_status.get(store) == 'full':
                                 get_data_cost += 1 # drop action

                         # --- Cost to communicate soil data for this rover ---
                         # Cost = communicate + nav_from_data_location_to_comm
                         comm_cost = 1 # communicate_soil_data action
                         nav_to_comm_cost, comm_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), data_location, self.communication_waypoint_candidates)
                         if nav_to_comm_cost == math.inf: continue # Cannot reach communication spot

                         comm_cost += nav_to_comm_cost

                         # Total cost for this rover for this goal
                         rover_goal_cost = get_data_cost + comm_cost
                         best_goal_cost = min(best_goal_cost, rover_goal_cost)

                    if best_goal_cost == math.inf:
                         return math.inf # Goal is unreachable by any candidate rover

                    total_cost += best_goal_cost


            elif predicate == 'communicated_rock_data':
                waypoint = g_parts[1]
                if waypoint not in current_communicated_rock:
                    # Cost for this goal = min_rover_cost(get_data + communicate)
                    best_goal_cost = math.inf

                    # Iterate through all rovers that could potentially achieve this goal
                    candidate_rovers = {r for r in self.rovers if waypoint in current_rover_rock_analysis[r] or 'rock' in self.rover_capabilities[r]}

                    if not candidate_rovers or not self.communication_waypoint_candidates:
                         return math.inf # Goal is unreachable

                    for rover in candidate_rovers:
                         rover_curr_w = current_rover_locations.get(rover)
                         if not rover_curr_w: continue # Rover location unknown

                         # --- Cost to get rock analysis data for this rover ---
                         get_data_cost = 0
                         data_location = rover_curr_w # Where the data is obtained (or already exists)

                         if waypoint not in current_rover_rock_analysis[rover]:
                             # Need to sample
                             get_data_cost += 1 # sample_rock action
                             nav_to_sample_cost, sample_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), rover_curr_w, {waypoint})
                             if nav_to_sample_cost == math.inf: continue # Cannot reach sample spot

                             get_data_cost += nav_to_sample_cost
                             data_location = sample_wp # Data is obtained at the sample waypoint

                             store = self.rover_stores.get(rover)
                             if store and current_store_status.get(store) == 'full':
                                 get_data_cost += 1 # drop action

                         # --- Cost to communicate rock data for this rover ---
                         # Cost = communicate + nav_from_data_location_to_comm
                         comm_cost = 1 # communicate_rock_data action
                         nav_to_comm_cost, comm_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), data_location, self.communication_waypoint_candidates)
                         if nav_to_comm_cost == math.inf: continue # Cannot reach communication spot

                         comm_cost += nav_to_comm_cost

                         # Total cost for this rover for this goal
                         rover_goal_cost = get_data_cost + comm_cost
                         best_goal_cost = min(best_goal_cost, rover_goal_cost)

                    if best_goal_cost == math.inf:
                         return math.inf # Goal is unreachable by any candidate rover

                    total_cost += best_goal_cost


            elif predicate == 'communicated_image_data':
                objective, mode = g_parts[1], g_parts[2]
                if (objective, mode) not in current_communicated_image:
                    # Cost for this goal = min_rover_camera_combo_cost(get_data + communicate)
                    best_goal_cost = math.inf

                    # Iterate through all rover/camera combos that could potentially achieve this goal
                    candidate_combos = [] # [(rover, camera)]
                    for rover in self.rovers:
                         if 'imaging' in self.rover_capabilities[rover]:
                             for camera in self.rover_cameras[rover]:
                                 if mode in self.camera_modes[camera]:
                                     # Check if imaging is actually possible (visible_from exists)
                                     if self.objective_visible_from.get(objective):
                                         # Check if calibration is possible if needed
                                         if (camera, rover) in current_camera_calibrated:
                                             candidate_combos.append((rover, camera))
                                         else:
                                             cal_target_obj = self.camera_calibration_targets.get(camera)
                                             if cal_target_obj and self.calibration_target_visible_from.get(cal_target_obj):
                                                  candidate_combos.append((rover, camera))

                    if not candidate_combos or not self.communication_waypoint_candidates:
                         return math.inf # Goal is unreachable

                    for rover, camera in candidate_combos:
                         rover_curr_w = current_rover_locations.get(rover)
                         if not rover_curr_w: continue # Rover location unknown

                         # --- Cost to get image data for this rover/camera ---
                         get_data_cost = 0
                         current_pos_for_nav = rover_curr_w # Starting point for navigation legs

                         if (objective, mode) not in current_rover_image_data[rover]:
                             # Need to take the image
                             get_data_cost += 1 # take_image action

                             # Calibration cost if needed (assumes calibration happens before imaging)
                             if (camera, rover) not in current_camera_calibrated:
                                 cal_cost = 1 # calibrate action
                                 cal_target_obj = self.camera_calibration_targets.get(camera)
                                 if not cal_target_obj: continue # Camera has no calibration target (shouldn't be in candidate_combos)

                                 calibration_waypoint_candidates = self.calibration_target_visible_from.get(cal_target_obj, set())
                                 if not calibration_waypoint_candidates: continue # No calibration spots (shouldn't be in candidate_combos)

                                 nav_to_calibrate_cost, cal_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), current_pos_for_nav, calibration_waypoint_candidates)
                                 if nav_to_calibrate_cost == math.inf: continue # Cannot reach calibration spot

                                 cal_cost += nav_to_calibrate_cost
                                 get_data_cost += cal_cost
                                 # Rover is now at cal_wp for the next step (nav to image spot)
                                 current_pos_for_nav = cal_wp

                             # Nav to imaging spot (from current pos or calibration spot)
                             imaging_waypoint_candidates = self.objective_visible_from.get(objective, set())
                             if not imaging_waypoint_candidates: continue # Cannot image this objective (shouldn't be in candidate_combos)

                             nav_to_image_cost, image_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), current_pos_for_nav, imaging_waypoint_candidates)
                             if nav_to_image_cost == math.inf: continue # Cannot reach imaging spot

                             get_data_cost += nav_to_image_cost
                             data_location = image_wp # Data is obtained at the imaging waypoint
                             current_pos_for_nav = image_wp # Rover is now at image_wp


                         # --- Cost to communicate image data for this rover ---
                         # Cost = communicate + nav_from_data_location_to_comm
                         comm_cost = 1 # communicate_image_data action
                         # If image was already held, data_location is the initial rover_curr_w
                         # If image was just taken, data_location is image_wp
                         data_location_for_comm = current_pos_for_nav # Use the final position after getting data

                         nav_to_comm_cost, comm_wp = self.find_closest_waypoint(self.rover_traversal_graph.get(rover), data_location_for_comm, self.communication_waypoint_candidates)
                         if nav_to_comm_cost == math.inf: continue # Cannot reach communication spot

                         comm_cost += nav_to_comm_cost

                         # Total cost for this rover/camera combo for this goal
                         combo_goal_cost = get_data_cost + comm_cost
                         best_goal_cost = min(best_goal_cost, combo_goal_cost)


                    if best_goal_cost == math.inf:
                         return math.inf # Goal is unreachable by any candidate combo

                    total_cost += best_goal_cost

        return total_cost
