from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# --- Helper functions ---

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start_node.
        Nodes not reachable will not be in the dictionary.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not a valid waypoint in the domain
         return {}

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Handle cases where current_node might not be in the graph dict
        # (e.g., if all_nodes contains nodes not in any 'can_traverse' fact,
        # or if the graph was filtered and removed nodes)
        if current_node not in graph:
             continue

        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    # Return distances only for reachable nodes
    return {node: dist for node, dist in distances.items() if dist != float('inf')}


# --- Heuristic Class ---

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated costs for each unachieved goal fact.
    For soil/rock goals, the cost is estimated as the minimum actions to sample
    the data and then communicate it. For image goals, the cost is estimated
    as the minimum actions to calibrate a camera, take the image, and then
    communicate it. Movement costs are estimated using precomputed shortest
    paths on the navigation graph for each rover.

    # Assumptions
    - The heuristic assumes that any equipped rover can perform the necessary
      sampling/imaging tasks and communication tasks.
    - For sampling, it adds a cost of 1 if the rover's store is currently full
      and needs to be emptied before sampling. It does not model dropping
      samples from other rovers or complex store management.
    - For imaging, it assumes calibration is needed before taking an image
      if the camera is not currently calibrated. It does not explicitly model
      recalibration needs if multiple images are taken with the same camera
      without intermediate calibration, but the cost structure implicitly
      includes a calibration step for each image goal if the image fact is
      not already achieved.
    - The heuristic uses shortest path distances on the navigation graph
      (considering both `can_traverse` and `visible` predicates) as movement costs.
    - If a required location (sample, image, calibrate, lander communication point)
      is unreachable by any suitable rover, the cost for that goal component
      is considered infinite (represented by a large number).

    # Heuristic Initialization
    The heuristic precomputes and stores the following information from the task definition:
    - A set of all waypoints, rovers, cameras, objectives, modes, and calibration targets.
    - The navigation graph for each rover, considering both `can_traverse` and `visible` facts.
    - Shortest path distances between all pairs of waypoints for each rover using BFS.
    - The lander's location.
    - The set of waypoints visible from the lander's location (communication points).
    - The locations of soil and rock samples.
    - For each objective, the set of waypoints from which it is visible (imaging points).
    - For each camera, its calibration target.
    - For each calibration target, the set of waypoints from which it is visible (calibration points).
    - For each rover, its equipment capabilities (soil, rock, imaging).
    - For each rover, the cameras on board.
    - For each camera, the modes it supports.
    - For each rover, its associated store.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Parse the current state to determine:
        -   The current location of each rover.
        -   Whether each rover's store is full.
        -   Which soil/rock analyses each rover possesses.
        -   Which images each rover possesses.
        -   Which cameras on which rovers are calibrated.
    2.  Initialize the total heuristic cost to 0.
    3.  Iterate through each goal fact specified in the task:
        -   If the goal fact is already true in the current state, add 0 to the total cost and continue to the next goal.
        -   If the goal fact is `(communicated_soil_data ?w)`:
            -   Check if any rover currently has `(have_soil_analysis ?r ?w)`.
            -   If yes: The cost is the minimum actions for *that* rover to reach *any* communication waypoint and communicate (shortest path + 1). Minimize over all rovers having the data and all communication waypoints.
            -   If no: The cost is the minimum actions for an equipped soil rover `?r_s` to go to `?w`, sample (adding 1 if its store is full), then go to *any* communication waypoint `?x_c` and communicate (shortest path from current to `?w` + drop_cost + 1 + shortest path from `?w` to `?x_c` + 1). Minimize over all equipped soil rovers and all communication waypoints.
            -   Add the minimum calculated cost for this goal to the total cost. If no valid sequence is possible (e.g., no equipped rover, unreachable waypoint), add a large constant representing infinity.
        -   If the goal fact is `(communicated_rock_data ?w)`:
            -   Follow the symmetric logic as for `communicated_soil_data`.
        -   If the goal fact is `(communicated_image_data ?o ?m)`:
            -   Check if any rover currently has `(have_image ?r ?o ?m)`.
            -   If yes: The cost is the minimum actions for *that* rover to reach *any* communication waypoint and communicate (shortest path + 1). Minimize over all rovers having the image and all communication waypoints.
            -   If no: The cost is the minimum actions for an equipped imaging rover `?r_i` with a camera `?i` supporting mode `?m` to go to a calibration waypoint `?w_c` for `?i`, calibrate (1 action), go to an imaging waypoint `?p_i` for `?o`, take the image (1 action), then go to *any* communication waypoint `?x_c` and communicate (1 action). The cost is (shortest path from current to `?w_c` + 1 + shortest path from `?w_c` to `?p_i` + 1 + shortest path from `?p_i` to `?x_c` + 1). Minimize over all suitable rovers, cameras, calibration targets, calibration waypoints, imaging waypoints, and communication waypoints.
            -   Add the minimum calculated cost for this goal to the total cost. If no valid sequence is possible, add a large constant representing infinity.
    4.  Return the total calculated cost.
    """

    UNREACHABLE = 10000 # Use a large number to represent infinity

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Precomputation from static facts and objects ---

        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_targets = set() # Calibration targets are objectives, but we track them separately here

        # Extract objects by type
        for obj_name, obj_type in task.objects:
            if obj_type == 'waypoint':
                self.all_waypoints.add(obj_name)
            elif obj_type == 'rover':
                self.all_rovers.add(obj_name)
            elif obj_type == 'camera':
                self.all_cameras.add(obj_name)
            elif obj_type == 'objective':
                self.all_objectives.add(obj_name)
            elif obj_type == 'mode':
                self.all_modes.add(obj_name)
            elif obj_type == 'lander':
                # Assume one lander, store its name
                self.lander_name = obj_name
            # store type is not needed explicitly here

        self.rover_graph = {rover: {} for rover in self.all_rovers} # {rover: {wp1: [wp2, ...], ...}}
        self.rover_dist = {rover: {} for rover in self.all_rovers} # {rover: {wp1: {wp2: dist, ...}, ...}}
        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible from lander location
        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.objective_imaging_waypoints = {} # {objective: {wp1, wp2, ...}}
        self.camera_calibration_target = {} # {camera: target}
        self.target_calibration_waypoints = {} # {target: {wp1, wp2, ...}}
        self.rover_equipment = {rover: set() for rover in self.all_rovers} # {rover: {soil, rock, imaging}}
        self.rover_cameras = {rover: set() for rover in self.all_rovers} # {rover: {cam1, cam2, ...}}
        self.camera_modes = {camera: set() for camera in self.all_cameras} # {camera: {mode1, mode2, ...}}
        self.rover_store = {} # {rover: store}

        visible_facts = set()
        can_traverse_facts = set()

        # First pass to collect basic facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at_lander':
                # (at_lander ?x - lander ?y - waypoint)
                self.lander_location = parts[2]
            elif predicate == 'can_traverse':
                # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                can_traverse_facts.add(tuple(parts[1:])) # (rover, from_wp, to_wp)
            elif predicate == 'visible':
                # (visible ?w - waypoint ?p - waypoint)
                visible_facts.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample':
                # (at_soil_sample ?w - waypoint)
                self.soil_sample_locations.add(parts[1])
            elif predicate == 'at_rock_sample':
                # (at_rock_sample ?w - waypoint)
                self.rock_sample_locations.add(parts[1])
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis ?r - rover)
                self.rover_equipment[parts[1]].add('soil_analysis')
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis ?r - rover)
                self.rover_equipment[parts[1]].add('rock_analysis')
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging ?r - rover)
                self.rover_equipment[parts[1]].add('imaging')
            elif predicate == 'visible_from':
                # (visible_from ?o - objective ?w - waypoint)
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_imaging_waypoints:
                    self.objective_imaging_waypoints[objective] = set()
                self.objective_imaging_waypoints[objective].add(waypoint)
            elif predicate == 'store_of':
                # (store_of ?s - store ?r - rover)
                self.rover_store[parts[2]] = parts[1] # rover -> store
            elif predicate == 'calibration_target':
                # (calibration_target ?i - camera ?o - objective)
                self.camera_calibration_target[parts[1]] = parts[2] # camera -> target (which is an objective)
                self.all_targets.add(parts[2]) # Add the objective name to targets set
            elif predicate == 'on_board':
                # (on_board ?i - camera ?r - rover)
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif predicate == 'supports':
                # (supports ?c - camera ?m - mode)
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode

        # Build rover graph considering both can_traverse and visible
        for rover, wp_from, wp_to in can_traverse_facts:
             if (wp_from, wp_to) in visible_facts:
                 if wp_from not in self.rover_graph[rover]:
                     self.rover_graph[rover][wp_from] = []
                 self.rover_graph[rover][wp_from].append(wp_to)


        # Compute BFS distances for each rover
        for rover in self.all_rovers:
            self.rover_dist[rover] = {}
            for start_wp in self.all_waypoints:
                 # BFS returns distances only for reachable nodes
                 self.rover_dist[rover][start_wp] = bfs(self.rover_graph.get(rover, {}), start_wp, self.all_waypoints)

        # Determine communication waypoints (visible from lander location)
        if self.lander_location:
             for wp1, wp2 in visible_facts:
                 if wp1 == self.lander_location:
                     self.comm_waypoints.add(wp2)
                 if wp2 == self.lander_location:
                     self.comm_waypoints.add(wp1) # visible is symmetric

        # Populate target_calibration_waypoints based on calibration_target and visible_from
        # Iterate through all visible_from facts again, but this time check if the objective is a calibration target
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'visible_from':
                 obj, waypoint = parts[1], parts[2]
                 # Check if this objective is a calibration target
                 if obj in self.all_targets:
                     if obj not in self.target_calibration_waypoints:
                         self.target_calibration_waypoints[obj] = set()
                     self.target_calibration_waypoints[obj].add(waypoint)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state ---
        current_rover_location = {} # {rover: waypoint}
        rover_store_full = {} # {rover: bool}
        rover_has_soil = {} # {rover: {waypoint, ...}}
        rover_has_rock = {} # {rover: {waypoint, ...}}
        rover_has_image = {} # {rover: {objective: {mode, ...}, ...}}
        rover_camera_calibrated = {} # {rover: {camera, ...}}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                # (at ?x - rover ?y - waypoint)
                if parts[1] in self.all_rovers: # Ensure it's a rover
                    current_rover_location[parts[1]] = parts[2]
            elif predicate == 'full':
                # (full ?s - store)
                store_name = parts[1]
                # Find which rover this store belongs to
                for rover, store in self.rover_store.items():
                    if store == store_name:
                        rover_store_full[rover] = True
                        break
            elif predicate == 'have_soil_analysis':
                # (have_soil_analysis ?r - rover ?w - waypoint)
                rover, waypoint = parts[1], parts[2]
                if rover not in rover_has_soil:
                    rover_has_soil[rover] = set()
                rover_has_soil[rover].add(waypoint)
            elif predicate == 'have_rock_analysis':
                # (have_rock_analysis ?r - rover ?w - waypoint)
                rover, waypoint = parts[1], parts[2]
                if rover not in rover_has_rock:
                    rover_has_rock[rover] = set()
                rover_has_rock[rover].add(waypoint)
            elif predicate == 'have_image':
                # (have_image ?r - rover ?o - objective ?m - mode)
                rover, objective, mode = parts[1], parts[2], parts[3]
                if rover not in rover_has_image:
                    rover_has_image[rover] = {}
                if objective not in rover_has_image[rover]:
                    rover_has_image[rover][objective] = set()
                rover_has_image[rover][objective].add(mode)
            elif predicate == 'calibrated':
                # (calibrated ?c - camera ?r - rover)
                camera, rover = parts[1], parts[2]
                if rover not in rover_camera_calibrated:
                    rover_camera_calibrated[rover] = set()
                rover_camera_calibrated[rover].add(camera)

        total_cost = 0

        # --- Calculate cost for each unachieved goal ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                goal_cost = self.UNREACHABLE

                # Check if data is already collected by any rover
                rover_with_data = None
                for rover in self.all_rovers:
                    if rover in rover_has_soil and waypoint in rover_has_soil[rover]:
                        rover_with_data = rover
                        break

                if rover_with_data:
                    # Data exists, need to communicate it
                    current_loc = current_rover_location.get(rover_with_data)
                    if current_loc:
                        min_comm_dist = self.UNREACHABLE
                        for comm_wp in self.comm_waypoints:
                            dist = self.rover_dist.get(rover_with_data, {}).get(current_loc, {}).get(comm_wp, self.UNREACHABLE)
                            min_comm_dist = min(min_comm_dist, dist)

                        if min_comm_dist != self.UNREACHABLE:
                            goal_cost = min_comm_dist + 1 # +1 for communicate action
                else:
                    # Data does not exist, need to sample and communicate
                    min_sample_comm_cost = self.UNREACHABLE
                    if waypoint in self.soil_sample_locations: # Check if there is a sample to collect
                        for rover_s in self.all_rovers:
                            if 'soil_analysis' in self.rover_equipment.get(rover_s, set()):
                                current_loc_s = current_rover_location.get(rover_s)
                                if current_loc_s:
                                    dist_to_sample = self.rover_dist.get(rover_s, {}).get(current_loc_s, {}).get(waypoint, self.UNREACHABLE)

                                    if dist_to_sample != self.UNREACHABLE:
                                        sample_action_cost = 1 # sample_soil
                                        # Add cost if store is full and needs drop before sampling
                                        drop_cost = 1 if rover_store_full.get(rover_s, False) else 0

                                        min_comm_dist_after_sample = self.UNREACHABLE
                                        for comm_wp in self.comm_waypoints:
                                            dist = self.rover_dist.get(rover_s, {}).get(waypoint, {}).get(comm_wp, self.UNREACHABLE)
                                            min_comm_dist_after_sample = min(min_comm_dist_after_sample, dist)

                                        if min_comm_dist_after_sample != self.UNREACHABLE:
                                            comm_action_cost = 1 # communicate_soil_data
                                            path_cost = dist_to_sample + drop_cost + sample_action_cost + min_comm_dist_after_sample + comm_action_cost
                                            min_sample_comm_cost = min(min_sample_comm_cost, path_cost)

                    goal_cost = min_sample_comm_cost # Use the calculated min cost (could still be UNREACHABLE)

                total_cost += goal_cost

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                goal_cost = self.UNREACHABLE

                # Check if data is already collected by any rover
                rover_with_data = None
                for rover in self.all_rovers:
                    if rover in rover_has_rock and waypoint in rover_has_rock[rover]:
                        rover_with_data = rover
                        break

                if rover_with_data:
                    # Data exists, need to communicate it
                    current_loc = current_rover_location.get(rover_with_data)
                    if current_loc:
                        min_comm_dist = self.UNREACHABLE
                        for comm_wp in self.comm_waypoints:
                            dist = self.rover_dist.get(rover_with_data, {}).get(current_loc, {}).get(comm_wp, self.UNREACHABLE)
                            min_comm_dist = min(min_comm_dist, dist)

                        if min_comm_dist != self.UNREACHABLE:
                            goal_cost = min_comm_dist + 1 # +1 for communicate action
                else:
                    # Data does not exist, need to sample and communicate
                    min_sample_comm_cost = self.UNREACHABLE
                    if waypoint in self.rock_sample_locations: # Check if there is a sample to collect
                        for rover_s in self.all_rovers:
                            if 'rock_analysis' in self.rover_equipment.get(rover_s, set()):
                                current_loc_s = current_rover_location.get(rover_s)
                                if current_loc_s:
                                    dist_to_sample = self.rover_dist.get(rover_s, {}).get(current_loc_s, {}).get(waypoint, self.UNREACHABLE)

                                    if dist_to_sample != self.UNREACHABLE:
                                        sample_action_cost = 1 # sample_rock
                                        # Add cost if store is full and needs drop before sampling
                                        drop_cost = 1 if rover_store_full.get(rover_s, False) else 0

                                        min_comm_dist_after_sample = self.UNREACHABLE
                                        for comm_wp in self.comm_waypoints:
                                            dist = self.rover_dist.get(rover_s, {}).get(waypoint, {}).get(comm_wp, self.UNREACHABLE)
                                            min_comm_dist_after_sample = min(min_comm_dist_after_sample, dist)

                                        if min_comm_dist_after_sample != self.UNREACHABLE:
                                            comm_action_cost = 1 # communicate_rock_data
                                            path_cost = dist_to_sample + drop_cost + sample_action_cost + min_comm_dist_after_sample + comm_action_cost
                                            min_sample_comm_cost = min(min_sample_comm_cost, path_cost)

                    goal_cost = min_sample_comm_cost # Use the calculated min cost

                total_cost += goal_cost

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                goal_cost = self.UNREACHABLE

                # Check if image is already collected by any rover
                rover_with_image = None
                for rover in self.all_rovers:
                    if rover in rover_has_image and objective in rover_has_image[rover] and mode in rover_has_image[rover][objective]:
                        rover_with_image = rover
                        break

                if rover_with_image:
                    # Image exists, need to communicate it
                    current_loc = current_rover_location.get(rover_with_image)
                    if current_loc:
                        min_comm_dist = self.UNREACHABLE
                        for comm_wp in self.comm_waypoints:
                            dist = self.rover_dist.get(rover_with_image, {}).get(current_loc, {}).get(comm_wp, self.UNREACHABLE)
                            min_comm_dist = min(min_comm_dist, dist)

                        if min_comm_dist != self.UNREACHABLE:
                            goal_cost = min_comm_dist + 1 # +1 for communicate action
                else:
                    # Image does not exist, need to calibrate, take image, and communicate
                    min_img_comm_cost = self.UNREACHABLE

                    # Iterate through all possible ways to achieve the image goal
                    for rover_i in self.all_rovers:
                        if 'imaging' in self.rover_equipment.get(rover_i, set()):
                            current_loc_i = current_rover_location.get(rover_i)
                            if current_loc_i:
                                for camera in self.rover_cameras.get(rover_i, set()):
                                    if mode in self.camera_modes.get(camera, set()):
                                        cal_target = self.camera_calibration_target.get(camera)
                                        if cal_target:
                                            cal_wps = self.target_calibration_waypoints.get(cal_target, set())
                                            img_wps = self.objective_imaging_waypoints.get(objective, set())
                                            comm_wps = self.comm_waypoints

                                            if cal_wps and img_wps and comm_wps:
                                                # Find min cost path: current -> cal_wp -> img_wp -> comm_wp
                                                min_path_cost_for_rover_cam = self.UNREACHABLE
                                                for w_c in cal_wps:
                                                    dist1 = self.rover_dist.get(rover_i, {}).get(current_loc_i, {}).get(w_c, self.UNREACHABLE)
                                                    if dist1 == self.UNREACHABLE: continue

                                                    for p_i in img_wps:
                                                        dist2 = self.rover_dist.get(rover_i, {}).get(w_c, {}).get(p_i, self.UNREACHABLE)
                                                        if dist2 == self.UNREACHABLE: continue

                                                        for x_c in comm_wps:
                                                            dist3 = self.rover_dist.get(rover_i, {}).get(p_i, {}).get(x_c, self.UNREACHABLE)
                                                            if dist3 == self.UNREACHABLE: continue

                                                            # +1 calibrate, +1 take_image, +1 communicate
                                                            path_cost = dist1 + 1 + dist2 + 1 + dist3 + 1
                                                            min_path_cost_for_rover_cam = min(min_path_cost_for_rover_cam, path_cost)

                                                min_img_comm_cost = min(min_img_comm_cost, min_path_cost_for_rover_cam)

                    goal_cost = min_img_comm_cost # Use the calculated min cost

                total_cost += goal_cost

            # Add other goal types if necessary (although the examples only show communicated data)
            # Based on the domain file, the only goal predicates are the communicated ones.

        return total_cost

