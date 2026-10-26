from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict
import sys

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    # Handle potential empty facts or malformed strings gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts against a pattern
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

# Helper function for Breadth-First Search to find shortest paths
def bfs(graph, start_node):
    """
    Computes shortest distances from start_node to all other nodes in the graph
    using BFS. Assumes graph is represented as {node: {neighbor1, neighbor2, ...}}.
    Returns a dictionary {node: distance}. Unreachable nodes have distance float('inf').
    """
    # Ensure all potential nodes in the graph are initialized with infinity distance
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    distances = {node: float('inf') for node in all_nodes}

    if start_node not in all_nodes:
         # Start node is not in the graph of waypoints, cannot navigate from here.
         # All distances remain infinity.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure the current_node has neighbors defined in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions and navigation steps required
    to achieve all unachieved goal conditions. It breaks down each goal
    (communicating soil, rock, or image data) into necessary sub-tasks
    (sampling, imaging, calibrating, navigating) and sums up the estimated
    costs for each unachieved goal independently. Navigation costs are estimated
    using precomputed shortest path distances between waypoints for each rover.

    # Heuristic Initialization
    - Extracts static information like lander locations, rover capabilities,
      camera details, objective visibility, and waypoint connectivity.
    - Builds a navigation graph for each rover based on `can_traverse` facts.
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover using BFS.
    - Stores initial locations of soil and rock samples.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Initialize heuristic value `h` to 0.
    2. Extract current state information (rover locations, store status,
       calibration status, collected data, available samples).
    3. Iterate through each goal fact defined in the task.
    4. If a goal fact is already true in the current state, skip it.
    5. For each unachieved goal fact, estimate the minimum cost to achieve it:
       - **For `(communicated_soil_data ?w)`:**
         - Find equipped rovers.
         - Check if any equipped rover already has `(have_soil_analysis ?r ?w)`.
           - If yes: Cost is 1 (communicate) + navigation cost from rover's
             current location to the nearest lander-visible waypoint.
           - If no: Check if `(at_soil_sample ?w)` is true in the current state.
             - If no sample: Goal is likely unreachable (sample was taken and lost), cost is infinity.
             - If sample exists: Cost is 1 (sample) + 1 (communicate) + navigation
               cost from rover's current location to `w` + navigation cost from `w`
               to the nearest lander-visible waypoint. Add 1 if the rover's store is full (for `drop`).
         - Take the minimum cost over all equipped rovers.
       - **For `(communicated_rock_data ?w)`:** Similar logic to soil data.
       - **For `(communicated_image_data ?o ?m)`:**
         - Find rovers with cameras supporting mode `m` and equipped for imaging.
         - Check if any suitable rover already has `(have_image ?r ?o ?m)`.
           - If yes: Cost is 1 (communicate) + navigation cost from rover's
             current location to the nearest lander-visible waypoint.
           - If no: Check if objective `o` is visible from any waypoint.
             - If not visible: Goal is unreachable, cost is infinity.
             - If visible: Cost is 1 (take_image) + 1 (communicate) + navigation
               cost from rover's current location to the nearest waypoint visible
               from `o` + navigation cost from that waypoint to the nearest
               lander-visible waypoint.
             - If calibration is needed for the chosen camera (`(calibrated ?i ?r)` is false):
               Add 1 (calibrate) + navigation cost from rover's current location
               to the nearest waypoint visible from the camera's calibration target.
               The navigation path then becomes current_loc -> cal_wp -> img_wp -> lander_wp.
         - Take the minimum cost over all suitable rovers/cameras.
       - Handle unreachable cases by assigning `float('inf')` cost.
    6. Sum the minimum costs for all unachieved goals to get the total heuristic value `h`.
    7. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing
        navigation distances.
        """
        super().__init__(task)

        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Precompute Static Information ---

        self.lander_at = {} # {lander: waypoint}
        self.lander_visible_wps = defaultdict(set) # {lander: {waypoint}}
        self.rover_capabilities = defaultdict(set) # {rover: {soil, rock, imaging}}
        self.rover_store = {} # {rover: store}
        self.rover_cameras = defaultdict(set) # {rover: {camera}}
        self.camera_modes = defaultdict(set) # {camera: {mode}}
        self.camera_cal_target = {} # {camera: objective}
        self.objective_visible_wps = defaultdict(set) # {objective: {waypoint}}

        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_objectives = set()
        self.all_cameras = set()
        self.all_modes = set()
        self.all_landers = set()
        self.all_stores = set()

        # Extract all objects by type from initial state, static facts, and goals
        all_facts = static_facts | initial_state | self.goals
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue

             pred = parts[0]
             if pred == 'at':
                 if len(parts) == 3: # (at ?x - rover ?y - waypoint)
                     self.all_rovers.add(parts[1])
                     self.all_waypoints.add(parts[2])
             elif pred == 'at_lander': # (at_lander ?x - lander ?y - waypoint)
                 self.all_landers.add(parts[1])
                 self.all_waypoints.add(parts[2])
             elif pred == 'can_traverse': # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 self.all_rovers.add(parts[1])
                 self.all_waypoints.add(parts[2])
                 self.all_waypoints.add(parts[3])
             elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']: # (equipped_for_... ?r - rover)
                 if len(parts) == 2: self.all_rovers.add(parts[1])
             elif pred in ['empty', 'full']: # (empty ?s - store), (full ?s - store)
                 if len(parts) == 2: self.all_stores.add(parts[1])
             elif pred in ['have_rock_analysis', 'have_soil_analysis']: # (have_... ?r - rover ?w - waypoint)
                 if len(parts) == 3:
                     self.all_rovers.add(parts[1])
                     self.all_waypoints.add(parts[2])
             elif pred == 'calibrated': # (calibrated ?c - camera ?r - rover)
                 if len(parts) == 3:
                     self.all_cameras.add(parts[1])
                     self.all_rovers.add(parts[2])
             elif pred == 'supports': # (supports ?c - camera ?m - mode)
                 if len(parts) == 3:
                     self.all_cameras.add(parts[1])
                     self.all_modes.add(parts[2])
             elif pred == 'visible': # (visible ?w - waypoint ?p - waypoint)
                 if len(parts) == 3:
                     self.all_waypoints.add(parts[1])
                     self.all_waypoints.add(parts[2])
             elif pred == 'have_image': # (have_image ?r - rover ?o - objective ?m - mode)
                 if len(parts) == 4:
                     self.all_rovers.add(parts[1])
                     self.all_objectives.add(parts[2])
                     self.all_modes.add(parts[3])
             elif pred in ['communicated_soil_data', 'communicated_rock_data']: # (communicated_... ?w - waypoint)
                 if len(parts) == 2: self.all_waypoints.add(parts[1])
             elif pred == 'communicated_image_data': # (communicated_image_data ?o - objective ?m - mode)
                 if len(parts) == 3:
                     self.all_objectives.add(parts[1])
                     self.all_modes.add(parts[2])
             elif pred in ['at_soil_sample', 'at_rock_sample']: # (at_..._sample ?w - waypoint)
                 if len(parts) == 2: self.all_waypoints.add(parts[1])
             elif pred == 'visible_from': # (visible_from ?o - objective ?w - waypoint)
                 if len(parts) == 3:
                     self.all_objectives.add(parts[1])
                     self.all_waypoints.add(parts[2])
             elif pred == 'store_of': # (store_of ?s - store ?r - rover)
                 if len(parts) == 3:
                     self.all_stores.add(parts[1])
                     self.all_rovers.add(parts[2])
             elif pred == 'calibration_target': # (calibration_target ?i - camera ?o - objective)
                 if len(parts) == 3:
                     self.all_cameras.add(parts[1])
                     self.all_objectives.add(parts[2])
             elif pred == 'on_board': # (on_board ?i - camera ?r - rover)
                 if len(parts) == 3:
                     self.all_cameras.add(parts[1])
                     self.all_rovers.add(parts[2])


        # Build rover-specific navigation graphs
        self.rover_graph = defaultdict(lambda: defaultdict(set)) # {rover: {from_wp: {to_wp}}}
        # Ensure all waypoints are nodes in the graph even if they have no edges initially
        for rover in self.all_rovers:
             for wp in self.all_waypoints:
                 self.rover_graph[rover][wp] = set() # Initialize with empty set of neighbors

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "at_lander", "*", "*"):
                self.lander_at[parts[1]] = parts[2]
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities[parts[1]].add('soil')
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities[parts[1]].add('rock')
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities[parts[1]].add('imaging')
            elif match(fact, "store_of", "*", "*"):
                self.rover_store[parts[2]] = parts[1] # rover -> store
            elif match(fact, "on_board", "*", "*"):
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif match(fact, "supports", "*", "*"):
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif match(fact, "calibration_target", "*", "*"):
                self.camera_cal_target[parts[1]] = parts[2] # camera -> objective
            elif match(fact, "visible_from", "*", "*"):
                self.objective_visible_wps[parts[1]].add(parts[2]) # objective -> waypoint
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                # Ensure waypoints exist in the graph structure before adding edges
                if from_wp in self.all_waypoints and to_wp in self.all_waypoints:
                     self.rover_graph[rover][from_wp].add(to_wp)

        # Precompute lander visible waypoints
        all_visible_facts = {fact for fact in static_facts if match(fact, "visible", "*", "*")}
        for lander, lander_wp in self.lander_at.items():
             for fact in all_visible_facts:
                 wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                 if wp1 == lander_wp:
                     self.lander_visible_wps[lander].add(wp2)
                 if wp2 == lander_wp:
                     self.lander_visible_wps[lander].add(wp1) # Assuming visible is symmetric for communication

        # Store initial sample locations (they are consumed by sample actions)
        self.initial_soil_samples_at = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples_at = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_rock_sample", "*")}


        # Precompute shortest paths for each rover from all waypoints
        self.rover_dist = defaultdict(dict) # {rover: {from_wp: {to_wp: dist}}}
        for rover in self.all_rovers:
            for start_wp in self.all_waypoints:
                 self.rover_dist[rover][start_wp] = bfs(self.rover_graph[rover], start_wp)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Extract Current State Information ---
        current_rover_at = {} # {rover: waypoint}
        current_store_status = {} # {store: 'empty' or 'full'}
        current_calibrated = {} # {camera: True/False}
        current_have_soil = set() # {(rover, waypoint)}
        current_have_rock = set() # {(rover, waypoint)}
        current_have_image = set() # {(rover, objective, mode)}
        current_soil_samples_at = set() # {waypoint}
        current_rock_samples_at = set() # {waypoint}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at":
                current_rover_at[parts[1]] = parts[2]
            elif pred == "empty":
                current_store_status[parts[1]] = 'empty'
            elif pred == "full":
                current_store_status[parts[1]] = 'full'
            elif pred == "calibrated":
                current_calibrated[parts[1]] = True
            elif pred == "have_soil_analysis":
                current_have_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis":
                current_have_rock.add((parts[1], parts[2]))
            elif pred == "have_image":
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif pred == "at_soil_sample":
                 current_soil_samples_at.add(parts[1])
            elif pred == "at_rock_sample":
                 current_rock_samples_at.add(parts[1])

        # Assume cameras not explicitly listed as calibrated are not calibrated
        for cam in self.all_cameras:
             if cam not in current_calibrated:
                 current_calibrated[cam] = False

        # --- Calculate Heuristic ---

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals
            goal_type = parts[0]

            min_goal_cost = float('inf')

            if goal_type == "communicated_soil_data":
                if len(parts) != 2: continue # Malformed goal
                w = parts[1]

                # Find equipped rovers
                equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

                # Find if ANY equipped rover already has the required analysis
                any_equipped_rover_has_soil = any((r, w) in current_have_soil for r in equipped_rovers)

                if any_equipped_rover_has_soil:
                    # Cost = 1 (communicate) + navigation cost from rover's current location to lander
                    # Find the equipped rover that has the sample and is closest to a lander waypoint
                    current_min_comm_nav = float('inf')
                    for r in equipped_rovers:
                         if (r, w) in current_have_soil:
                             r_loc = current_rover_at.get(r)
                             if r_loc is None or r_loc not in self.rover_dist.get(r, {}): continue # Rover location unknown or not in graph

                             # Find min dist from r_loc to any lander waypoint
                             min_dist_to_lander = float('inf')
                             for lander in self.lander_at:
                                 for lander_wp in self.lander_visible_wps[lander]:
                                     dist = self.rover_dist[r][r_loc].get(lander_wp, float('inf'))
                                     min_dist_to_lander = min(min_dist_to_lander, dist)

                             current_min_comm_nav = min(current_min_comm_nav, min_dist_to_lander)

                    if current_min_comm_nav != float('inf'):
                        min_goal_cost = 1 + current_min_comm_nav # 1 for communicate action
                    # else: min_goal_cost remains inf

                else: # Sample needs to be taken
                    # Check if sample is still at the waypoint
                    if w not in current_soil_samples_at:
                         # Sample is gone and no equipped rover has it -> Unreachable
                         min_goal_cost = float('inf')
                    else:
                        # Find the best equipped rover to take the sample and communicate
                        min_total_cost = float('inf')

                        for r in equipped_rovers:
                            r_loc = current_rover_at.get(r)
                            if r_loc is None or r_loc not in self.rover_dist.get(r, {}): continue # Rover location unknown or not in graph

                            store = self.rover_store.get(r)
                            if store is None: continue # Rover has no store? Should not happen.

                            # Cost to sample: 1 (sample) + nav from r_loc to w + (1 if drop needed)
                            nav_to_sample_cost = self.rover_dist[r][r_loc].get(w, float('inf'))
                            if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                            drop_cost = 0
                            if current_store_status.get(store) == 'full':
                                drop_cost = 1 # Need to drop before sampling

                            sample_action_cost = 1 # sample_soil

                            # Cost to communicate: 1 (communicate) + nav from w to lander waypoint
                            min_nav_from_sample_to_lander = float('inf')
                            for lander in self.lander_at:
                                for lander_wp in self.lander_visible_wps[lander]:
                                    dist = self.rover_dist[r][w].get(lander_wp, float('inf'))
                                    min_nav_from_sample_to_lander = min(min_nav_from_sample_to_lander, dist)

                            if min_nav_from_sample_to_lander == float('inf'): continue # Cannot reach lander from sample location

                            comm_action_cost = 1 # communicate_soil_data

                            total_cost = nav_to_sample_cost + drop_cost + sample_action_cost + min_nav_from_sample_to_lander + comm_action_cost

                            min_total_cost = min(min_total_cost, total_cost)

                        min_goal_cost = min_total_cost


            elif goal_type == "communicated_rock_data":
                if len(parts) != 2: continue # Malformed goal
                w = parts[1]

                # Find equipped rovers
                equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

                # Find if ANY equipped rover already has the required analysis
                any_equipped_rover_has_rock = any((r, w) in current_have_rock for r in equipped_rovers)

                if any_equipped_rover_has_rock:
                    # Cost = 1 (communicate) + navigation cost from rover's current location to lander
                    # Find the equipped rover that has the sample and is closest to a lander waypoint
                    current_min_comm_nav = float('inf')
                    for r in equipped_rovers:
                         if (r, w) in current_have_rock:
                             r_loc = current_rover_at.get(r)
                             if r_loc is None or r_loc not in self.rover_dist.get(r, {}): continue # Rover location unknown or not in graph

                             # Find min dist from r_loc to any lander waypoint
                             min_dist_to_lander = float('inf')
                             for lander in self.lander_at:
                                 for lander_wp in self.lander_visible_wps[lander]:
                                     dist = self.rover_dist[r][r_loc].get(lander_wp, float('inf'))
                                     min_dist_to_lander = min(min_dist_to_lander, dist)

                             current_min_comm_nav = min(current_min_comm_nav, min_dist_to_lander)

                    if current_min_comm_nav != float('inf'):
                        min_goal_cost = 1 + current_min_comm_nav # 1 for communicate action
                    # else: min_goal_cost remains inf

                else: # Sample needs to be taken
                    # Check if sample is still at the waypoint
                    if w not in current_rock_samples_at:
                         # Sample is gone and no equipped rover has it -> Unreachable
                         min_goal_cost = float('inf')
                    else:
                        # Find the best equipped rover to take the sample and communicate
                        min_total_cost = float('inf')

                        for r in equipped_rovers:
                            r_loc = current_rover_at.get(r)
                            if r_loc is None or r_loc not in self.rover_dist.get(r, {}): continue # Rover location unknown or not in graph

                            store = self.rover_store.get(r)
                            if store is None: continue # Rover has no store?

                            # Cost to sample: 1 (sample) + nav from r_loc to w + (1 if drop needed)
                            nav_to_sample_cost = self.rover_dist[r][r_loc].get(w, float('inf'))
                            if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                            drop_cost = 0
                            if current_store_status.get(store) == 'full':
                                drop_cost = 1 # Need to drop before sampling

                            sample_action_cost = 1 # sample_rock

                            # Cost to communicate: 1 (communicate) + nav from w to lander waypoint
                            min_nav_from_sample_to_lander = float('inf')
                            for lander in self.lander_at:
                                for lander_wp in self.lander_visible_wps[lander]:
                                    dist = self.rover_dist[r][w].get(lander_wp, float('inf'))
                                    min_nav_from_sample_to_lander = min(min_nav_from_sample_to_lander, dist)

                            if min_nav_from_sample_to_lander == float('inf'): continue # Cannot reach lander from sample location

                            comm_action_cost = 1 # communicate_rock_data

                            total_cost = nav_to_sample_cost + drop_cost + sample_action_cost + min_nav_from_sample_to_lander + comm_action_cost

                            min_total_cost = min(min_total_cost, total_cost)

                        min_goal_cost = min_total_cost


            elif goal_type == "communicated_image_data":
                if len(parts) != 3: continue # Malformed goal
                o, m = parts[1], parts[2]

                # Find suitable rovers/cameras
                suitable_rovers_cameras = []
                for r, caps in self.rover_capabilities.items():
                    if 'imaging' in caps:
                        for i in self.rover_cameras.get(r, []):
                            if m in self.camera_modes.get(i, []):
                                suitable_rovers_cameras.append((r, i))

                # Find if ANY suitable rover/camera already has the image
                any_suitable_has_image = any((r, o, m) in current_have_image for r, i in suitable_rovers_cameras)

                if any_suitable_has_image:
                    # Cost = 1 (communicate) + navigation cost from rover's current location to lander
                    # Find the suitable rover/camera that has the image and is closest to a lander waypoint
                    current_min_comm_nav = float('inf')
                    for r, i in suitable_rovers_cameras:
                         if (r, o, m) in current_have_image:
                             r_loc = current_rover_at.get(r)
                             if r_loc is None or r_loc not in self.rover_dist.get(r, {}): continue # Rover location unknown or not in graph

                             # Find min dist from r_loc to any lander waypoint
                             min_dist_to_lander = float('inf')
                             for lander in self.lander_at:
                                 for lander_wp in self.lander_visible_wps[lander]:
                                     dist = self.rover_dist[r][r_loc].get(lander_wp, float('inf'))
                                     min_dist_to_lander = min(min_dist_to_lander, dist)

                             current_min_comm_nav = min(current_min_comm_nav, min_dist_to_lander)

                    if current_min_comm_nav != float('inf'):
                        min_goal_cost = 1 + current_min_comm_nav # 1 for communicate action
                    # else: min_goal_cost remains inf

                else: # Image needs to be taken
                    # Find waypoints visible from the objective
                    img_wps = self.objective_visible_wps.get(o, set())
                    if not img_wps:
                        min_goal_cost = float('inf') # Cannot see objective -> Unreachable
                    else:
                        # Find the best suitable rover/camera to take the image and communicate
                        min_total_cost = float('inf')

                        for r, i in suitable_rovers_cameras:
                            r_loc = current_rover_at.get(r)
                            if r_loc is None or r_loc not in self.rover_dist.get(r, {}): continue # Rover location unknown or not in graph

                            # Cost to take image: 1 (take_image) + nav to img_wp + cal_cost
                            image_action_cost = 1 # take_image

                            # Cost for calibration step
                            is_calibrated = current_calibrated.get(i, False)
                            cal_cost = 0
                            img_step_start_loc = r_loc # Default start for image step nav

                            if not is_calibrated:
                                # Calibration needed: 1 (calibrate) + nav from r_loc to cal_wp
                                cal_action_cost = 1 # calibrate
                                cal_target = self.camera_cal_target.get(i)
                                if cal_target is None:
                                     cal_cost = float('inf') # Camera has no calibration target? Unreachable.
                                else:
                                    cal_wps = self.objective_visible_wps.get(cal_target, set())
                                    if not cal_wps:
                                        cal_cost = float('inf') # Cannot see calibration target -> Unreachable
                                    else:
                                        # Find closest calibration waypoint reachable from r_loc
                                        min_dist_to_cal = float('inf')
                                        chosen_cal_wp = None
                                        for cal_wp in cal_wps:
                                            dist = self.rover_dist[r][r_loc].get(cal_wp, float('inf'))
                                            if dist < min_dist_to_cal:
                                                min_dist_to_cal = dist
                                                chosen_cal_wp = cal_wp

                                        if chosen_cal_wp is None or min_dist_to_cal == float('inf'):
                                            cal_cost = float('inf') # Cannot reach any calibration waypoint
                                        else:
                                            cal_cost = cal_action_cost + min_dist_to_cal
                                            img_step_start_loc = chosen_cal_wp # Next step starts from calibration waypoint

                            if cal_cost == float('inf'): continue # Cannot calibrate, this rover/camera is unsuitable

                            # Nav from cal_loc (or r_loc) to image_wp
                            min_dist_to_img = float('inf')
                            chosen_img_wp = None
                            for img_wp in img_wps:
                                # Ensure img_step_start_loc is a valid key in rover_dist
                                if img_step_start_loc not in self.rover_dist.get(r, {}): continue
                                dist = self.rover_dist[r][img_step_start_loc].get(img_wp, float('inf'))
                                if dist < min_dist_to_img:
                                    min_dist_to_img = dist
                                    chosen_img_wp = img_wp

                            if chosen_img_wp is None or min_dist_to_img == float('inf'):
                                total_image_cost = float('inf') # Cannot reach any image waypoint
                            else:
                                total_image_cost = cal_cost + min_dist_to_img + image_action_cost
                                comm_step_start_loc = chosen_img_wp # Next step starts from image waypoint

                            if total_image_cost == float('inf'): continue # Cannot take image, this rover/camera is unsuitable

                            # Cost for communication step: 1 (communicate) + nav from img_wp to lander waypoint
                            comm_action_cost = 1 # communicate_image_data
                            min_nav_from_image_to_lander = float('inf')
                            for lander in self.lander_at:
                                for lander_wp in self.lander_visible_wps[lander]:
                                     # Ensure comm_step_start_loc is a valid key in rover_dist
                                     if comm_step_start_loc not in self.rover_dist.get(r, {}): continue
                                     dist = self.rover_dist[r][comm_step_start_loc].get(lander_wp, float('inf'))
                                     min_nav_from_image_to_lander = min(min_nav_from_image_to_lander, dist)

                            if min_nav_from_image_to_lander == float('inf'): continue # Cannot reach lander from image location

                            total_cost = total_image_cost + min_nav_from_image_to_lander + comm_action_cost

                            min_total_cost = min(min_total_cost, total_cost)

                        min_goal_cost = min_total_cost


            # Add the minimum cost for this goal to the total heuristic value
            if min_goal_cost == float('inf'):
                 # If a goal is unreachable, the whole state is likely a dead end.
                 # Returning infinity is appropriate for greedy best-first search
                 # to prune this branch.
                 return float('inf')
            h += min_goal_cost

        return h
