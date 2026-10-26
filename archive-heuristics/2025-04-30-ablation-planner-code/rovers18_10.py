from fnmatch import fnmatch
import collections

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest path distances from start_node in an unweighted graph."""
    distances = {node: float('inf') for node in graph}
    if start_node in distances: # Ensure start_node is in the graph keys
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            current_node = queue.popleft()
            # Use graph.get to handle nodes with no outgoing edges gracefully
            for neighbor in graph.get(current_node, set()):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph, all_nodes):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_dist = {}
    # Ensure all_nodes are keys in the graph dict for BFS to initialize distances correctly
    graph_with_all_nodes = {node: graph.get(node, set()) for node in all_nodes}

    for start_node in all_nodes:
        all_dist[start_node] = bfs(graph_with_all_nodes, start_node)
    return all_dist

# Assume Heuristic base class is not provided, define the class directly.
# If a base class is required, add 'from heuristics.heuristic_base import Heuristic'
# and change class definition to 'class roversHeuristic(Heuristic):'

class roversHeuristic:
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    uncommunicated data and image goals. It calculates the minimum cost for
    each unachieved goal independently by considering the necessary steps:
    acquiring the data/image (moving to location, sampling/imaging, handling
    store/calibration) and communicating the data/image (moving to a lander-visible
    waypoint, communicating). The total heuristic is the sum of these minimum costs
    for all unachieved goals.

    # Assumptions
    - Each navigation step between adjacent waypoints costs 1 action.
    - Sampling (soil/rock), dropping, calibrating, taking image, and communicating
      each cost 1 action.
    - The heuristic assumes that if a goal requires a sample at a waypoint, that
      sample is either currently present at the waypoint or has already been
      acquired by a rover. It does not consider samples that were initially present
      but are no longer at the waypoint and not held by any rover.
    - Calibration cost is added only if the camera is not currently calibrated
      when the image is needed. It does not account for subsequent recalibrations
      needed if the same camera is used for multiple images after the first one.
    - The heuristic finds the minimum cost over available rovers and relevant
      waypoints (sample locations, image locations, calibration locations,
      communication locations).
    - The heuristic assumes solvable instances, meaning necessary resources
      (equipped rovers, cameras, visible waypoints) exist to achieve the goals.
      If a goal appears impossible based on the current state and available static
      information, it is assigned an infinite cost.

    # Heuristic Initialization
    The heuristic is initialized by processing the task's goal conditions and
    static facts. This involves:
    - Storing the set of goal conditions.
    - Identifying the lander's location.
    - Building rover-specific traversal graphs based on `can_traverse` facts.
    - Building the general waypoint visibility graph based on `visible` facts.
    - Precomputing all-pairs shortest path distances for each rover's traversal graph using BFS.
    - Identifying the set of waypoints visible from the lander's location (communication waypoints).
    - Extracting rover capabilities (`equipped_for_...`).
    - Extracting camera information (`on_board`, `supports`, `calibration_target`).
    - Extracting objective and calibration target visibility information (`visible_from`).
    - Extracting store ownership (`store_of`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify all goal conditions that are *not* currently satisfied in the state.
    2.  Initialize the total heuristic cost to 0.
    3.  For each unachieved goal condition:
        a.  Determine the type of goal: `communicated_soil_data`, `communicated_rock_data`, or `communicated_image_data`.
        b.  Find all rovers (and cameras, for image goals) that are capable of contributing to this goal based on their equipment and onboard instruments.
        c.  Calculate the minimum cost to achieve this specific goal fact, considering all capable agents:
            i.  **If the required data/image is already held by an agent:** The cost to acquire is 0. The cost is just the communication cost (move to a communication waypoint + communicate action). Find the minimum communication cost over all agents holding the data/image and all communication waypoints.
            ii. **If the required data/image is NOT held by any agent:**
                -   **For Soil/Rock:** Check if the sample is still at the waypoint (`at_soil_sample`/`at_rock_sample`). If yes, the cost to acquire involves moving to the waypoint, potentially dropping a full sample, and sampling. Then add the communication cost (move from sample waypoint to communication waypoint + communicate). Find the minimum total cost over all capable rovers and communication waypoints. If the sample is not at the waypoint, this path is not possible for this goal segment via sampling.
                -   **For Image:** Find waypoints from which the objective is visible (`visible_from`). Find the nearest such waypoint. The cost to acquire involves moving to this waypoint, potentially calibrating the camera (move to calibration target waypoint + calibrate + move back, if not already calibrated), and taking the image. Then add the communication cost (move from image waypoint to communication waypoint + communicate). Find the minimum total cost over all suitable rover/camera pairs, image waypoints, calibration waypoints, and communication waypoints.
            iii. If no path is found (e.g., cannot reach necessary waypoints, no capable agent), the cost for this goal segment is infinite.
        d.  Add the minimum calculated cost for this unachieved goal to the total heuristic cost. If the minimum cost for a goal is infinite, the total cost becomes infinite.
    4.  Return the total heuristic cost. A cost of `float('inf')` indicates an apparently impossible goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = set(task.goals) # Use a set for faster lookup
        static_facts = set(task.static) # Use a set

        # Data structures to store parsed information
        self.lander_location = None
        self.rover_capabilities = {'soil': set(), 'rock': set(), 'imaging': set()}
        self.rover_stores = {} # {rover: store}
        # Use a directed graph representation for can_traverse
        self.rover_graph_adj = collections.defaultdict(lambda: collections.defaultdict(set)) # {rover: {from_wp: {to_wp}}}
        self.camera_info = {} # {camera: {'rover': r, 'modes': {m}, 'cal_target': t}}
        self.objective_visibility = collections.defaultdict(set) # {objective: {wp}}
        self.calibration_targets = {} # {camera: target}
        self.target_visibility = collections.defaultdict(set) # {target: {wp}}
        self.waypoint_visibility = collections.defaultdict(set) # {wp: {visible_wp}}
        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_landers = set()
        self.all_stores = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_location = parts[2]
                self.all_landers.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities['soil'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities['rock'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities['imaging'].add(parts[1])
                self.all_rovers.add(parts[1])
            elif pred == 'store_of':
                self.rover_stores[parts[2]] = parts[1]
                self.all_stores.add(parts[1])
                self.all_rovers.add(parts[2])
            elif pred == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                # Only add the directed edge specified
                self.rover_graph_adj[rover][from_wp].add(to_wp)
                self.all_rovers.add(rover)
                self.all_waypoints.add(from_wp)
                self.all_waypoints.add(to_wp)
            elif pred == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 self.waypoint_visibility[wp1].add(wp2)
                 self.waypoint_visibility[wp2].add(wp1) # Assuming symmetric visibility for visible predicate
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['rover'] = rover
                self.all_cameras.add(camera)
                self.all_rovers.add(rover)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'modes' not in self.camera_info[camera]: self.camera_info[camera]['modes'] = set()
                self.camera_info[camera]['modes'].add(mode)
                self.all_cameras.add(camera)
                self.all_modes.add(mode)
            elif pred == 'calibration_target':
                camera, target = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['cal_target'] = target
                self.calibration_targets[camera] = target
                self.all_cameras.add(camera)
                self.all_objectives.add(target) # Calibration targets are objectives
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility[objective].add(waypoint)
                self.target_visibility[objective].add(waypoint) # Calibration targets are objectives
                self.all_objectives.add(objective)
                self.all_waypoints.add(waypoint)

        # 2. Precompute shortest paths for each rover
        self.dist = {}
        for rover in self.all_rovers:
             # Use the rover's specific directed graph
             self.dist[rover] = compute_all_pairs_shortest_paths(self.rover_graph_adj.get(rover, {}), self.all_waypoints)

        # 3. Identify communication waypoints (visible from lander)
        self.comm_waypoint_set = self.waypoint_visibility.get(self.lander_location, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # frozenset of strings

        # Use sets for faster lookups
        state_set = set(state)

        # Extract current state information
        current_rover_locations = {} # {rover: waypoint}
        current_soil_analysis = collections.defaultdict(set) # {rover: {waypoint}}
        current_rock_analysis = collections.defaultdict(set) # {rover: {waypoint}}
        current_images = collections.defaultdict(lambda: collections.defaultdict(set)) # {rover: {objective: {mode}}}
        current_calibrated_cameras = collections.defaultdict(set) # {rover: {camera}}
        current_store_status = {} # {rover: 'empty' or 'full'}
        current_soil_samples = set() # {waypoint}
        current_rock_samples = set() # {waypoint}

        for fact in state_set:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                current_rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                current_soil_analysis[parts[1]].add(parts[2])
            elif pred == 'have_rock_analysis':
                current_rock_analysis[parts[1]].add(parts[2])
            elif pred == 'have_image':
                # (have_image ?r - rover ?o - objective ?m - mode)
                rover, objective, mode = parts[1], parts[2], parts[3]
                current_images[rover][objective].add(mode)
            elif pred == 'calibrated':
                current_calibrated_cameras[parts[2]].add(parts[1])
            elif pred == 'empty':
                 store = parts[1]
                 # Find which rover owns this store
                 for rover, s in self.rover_stores.items():
                     if s == store:
                         current_store_status[rover] = 'empty'
                         break
            elif pred == 'full':
                 store = parts[1]
                 # Find which rover owns this store
                 for rover, s in self.rover_stores.items():
                     if s == store:
                         current_store_status[rover] = 'full'
                         break
            elif pred == 'at_soil_sample':
                 current_soil_samples.add(parts[1])
            elif pred == 'at_rock_sample':
                 current_rock_samples.add(parts[1])


        total_cost = 0

        # Iterate through goals not yet achieved
        for goal in self.goals:
            if goal in state_set:
                continue # Goal already achieved

            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Find rovers capable of soil analysis
                capable_rovers = self.rover_capabilities['soil']

                for rover in capable_rovers:
                    if rover not in current_rover_locations: continue # Rover not in state (shouldn't happen?)

                    r_loc = current_rover_locations[rover]
                    cost_to_get_sample = float('inf')
                    source_loc_for_comm = None

                    # Case 1: Rover already has the soil analysis
                    if waypoint in current_soil_analysis.get(rover, set()):
                        cost_to_get_sample = 0
                        source_loc_for_comm = r_loc
                    # Case 2: Sample is available at the waypoint
                    elif waypoint in current_soil_samples:
                        # Need to move to waypoint, sample, potentially drop first
                        move_c = self.dist[rover][r_loc].get(waypoint, float('inf'))
                        if move_c == float('inf'): continue # Cannot reach sample location

                        sample_c = 1 # sample_soil action
                        # Find the store for this rover
                        rover_store = self.rover_stores.get(rover)
                        drop_c = 1 if current_store_status.get(rover) == 'full' else 0 # Cost to drop if store is full

                        cost_to_get_sample = move_c + drop_c + sample_c
                        source_loc_for_comm = waypoint

                    # If sample is obtained or already held, calculate communication cost
                    if cost_to_get_sample != float('inf'):
                        min_comm_c = float('inf')
                        # Find nearest communication waypoint from source_loc_for_comm
                        for comm_wp in self.comm_waypoint_set:
                             comm_move_c = self.dist[rover][source_loc_for_comm].get(comm_wp, float('inf'))
                             if comm_move_c != float('inf'):
                                 min_comm_c = min(min_comm_c, comm_move_c + 1) # +1 for communicate action

                        if min_comm_c != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_to_get_sample + min_comm_c)

                # Add the minimum cost found for this specific goal fact
                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # If min_goal_cost is still infinity, this goal is currently impossible
                    # based on the heuristic's logic (e.g., sample gone, no capable rover, unreachable).
                    # Add infinity to indicate impossibility.
                    total_cost = float('inf')
                    break # No need to check other goals if one is impossible


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Find rovers capable of rock analysis
                capable_rovers = self.rover_capabilities['rock']

                for rover in capable_rovers:
                    if rover not in current_rover_locations: continue

                    r_loc = current_rover_locations[rover]
                    cost_to_get_sample = float('inf')
                    source_loc_for_comm = None

                    # Case 1: Rover already has the rock analysis
                    if waypoint in current_rock_analysis.get(rover, set()):
                        cost_to_get_sample = 0
                        source_loc_for_comm = r_loc
                    # Case 2: Sample is available at the waypoint
                    elif waypoint in current_rock_samples:
                        # Need to move to waypoint, sample, potentially drop first
                        move_c = self.dist[rover][r_loc].get(waypoint, float('inf'))
                        if move_c == float('inf'): continue # Cannot reach sample location

                        sample_c = 1 # sample_rock action
                        # Find the store for this rover
                        rover_store = self.rover_stores.get(rover)
                        drop_c = 1 if current_store_status.get(rover) == 'full' else 0 # Cost to drop if store is full

                        cost_to_get_sample = move_c + drop_c + sample_c
                        source_loc_for_comm = waypoint

                    # If sample is obtained or already held, calculate communication cost
                    if cost_to_get_sample != float('inf'):
                        min_comm_c = float('inf')
                        # Find nearest communication waypoint from source_loc_for_comm
                        for comm_wp in self.comm_waypoint_set:
                             comm_move_c = self.dist[rover][source_loc_for_comm].get(comm_wp, float('inf'))
                             if comm_move_c != float('inf'):
                                 min_comm_c = min(min_comm_c, comm_move_c + 1) # +1 for communicate action

                        if min_comm_c != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_to_get_sample + min_comm_c)

                # Add the minimum cost found for this specific goal fact
                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost = float('inf')
                    break # No need to check other goals if one is impossible


            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Find rover/camera pairs capable of taking this image
                # Need rover equipped for imaging, camera on board, camera supports mode
                suitable_rover_camera_pairs = []
                for camera, info in self.camera_info.items():
                    rover = info.get('rover')
                    supported_modes = info.get('modes', set())
                    # Check if rover exists and is equipped for imaging
                    if rover in self.all_rovers and rover in self.rover_capabilities['imaging'] and mode in supported_modes:
                         suitable_rover_camera_pairs.append((rover, camera))

                for rover, camera in suitable_rover_camera_pairs:
                    if rover not in current_rover_locations: continue

                    r_loc = current_rover_locations[rover]
                    cost_to_get_image = float('inf')
                    source_loc_for_comm = None

                    # Case 1: Rover already has the image
                    if objective in current_images.get(rover, {}).get(mode, set()):
                        cost_to_get_image = 0
                        source_loc_for_comm = r_loc
                    # Case 2: Image needs to be taken
                    else:
                        # Need to move to image waypoint, calibrate (if needed), take image
                        image_wps = self.objective_visibility.get(objective, set())
                        if not image_wps: continue # Cannot see objective from anywhere

                        # Find nearest image waypoint
                        nearest_img_wp = None
                        min_move_to_img_c = float('inf')
                        for img_wp in image_wps:
                            move_c = self.dist[rover][r_loc].get(img_wp, float('inf'))
                            if move_c != float('inf') and move_c < min_move_to_img_c:
                                min_move_to_img_c = move_c
                                nearest_img_wp = img_wp

                        if nearest_img_wp is None: continue # Cannot reach any image waypoint

                        # Calibration cost
                        cal_c = 0
                        # Check if camera is calibrated for this rover
                        if camera not in current_calibrated_cameras.get(rover, set()):
                            cal_target = self.calibration_targets.get(camera)
                            if cal_target:
                                cal_wps = self.target_visibility.get(cal_target, set())
                                if cal_wps:
                                    # Find nearest calibration waypoint from the image waypoint
                                    nearest_cal_wp = None
                                    min_move_to_cal_c = float('inf')
                                    for cal_wp in cal_wps:
                                        move_c = self.dist[rover][nearest_img_wp].get(cal_wp, float('inf'))
                                        if move_c != float('inf') and move_c < min_move_to_cal_c:
                                            min_move_to_cal_c = move_c
                                            nearest_cal_wp = cal_wp

                                    if nearest_cal_wp is not None:
                                        # Move to cal_wp + calibrate + move back to img_wp
                                        move_back_c = self.dist[rover][nearest_cal_wp].get(nearest_img_wp, float('inf'))
                                        if move_back_c != float('inf'):
                                            cal_c = min_move_to_cal_c + 1 + move_back_c # +1 for calibrate action
                                        else:
                                            # Cannot move back from cal_wp to img_wp
                                            cal_c = float('inf')
                                    else:
                                        # Cannot reach any calibration waypoint
                                        cal_c = float('inf')
                                else:
                                    # No visible_from waypoint for calibration target
                                    cal_c = float('inf')
                            else:
                                # Camera has no calibration target defined
                                cal_c = float('inf') # Cannot calibrate

                        if cal_c == float('inf'): continue # Cannot calibrate if needed

                        # Take image cost
                        take_image_c = 1 # take_image action

                        cost_to_get_image = min_move_to_img_c + cal_c + take_image_c
                        source_loc_for_comm = nearest_img_wp


                    # If image is obtained or already held, calculate communication cost
                    if cost_to_get_image != float('inf'):
                        min_comm_c = float('inf')
                        # Find nearest communication waypoint from source_loc_for_comm
                        for comm_wp in self.comm_waypoint_set:
                             comm_move_c = self.dist[rover][source_loc_for_comm].get(comm_wp, float('inf'))
                             if comm_move_c != float('inf'):
                                 min_comm_c = min(min_comm_c, comm_move_c + 1) # +1 for communicate action

                        if min_comm_c != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_to_get_image + min_comm_c)

                # Add the minimum cost found for this specific goal fact
                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost = float('inf')
                    break # No need to check other goals if one is impossible

        # Return infinity if any goal segment was impossible
        return total_cost if total_cost != float('inf') else float('inf')
