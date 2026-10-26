import heapq
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic
# Assuming task.py is available in the environment
# from task import Operator, Task # Assuming Operator and Task are defined elsewhere and imported by the planner

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal predicate. For each unachieved
    goal, it calculates the minimum cost across all capable rovers to either
    communicate existing data/images or to first collect/take the data/image
    and then communicate it. The cost estimation considers navigation steps
    (using precomputed shortest paths) and action costs (sample, drop,
    calibrate, take_image, communicate).

    Assumptions:
    - Soil/rock samples only exist at initial locations specified by
      (at_soil_sample) / (at_rock_sample) facts in the initial state.
    - Data/image transfer between rovers is not possible (based on domain actions).
    - All objects (rovers, waypoints, cameras, objectives, etc.) mentioned
      in the state and goals are defined in the problem instance.
    - The initial state provided to the heuristic constructor is the task's
      initial state, used to check for initial sample locations.

    Heuristic Initialization:
    The constructor precomputes static information from the task's initial
    state and static facts:
    - Waypoint graph for each rover based on 'can_traverse'.
    - All-pairs shortest paths (navigation costs) for each rover using BFS.
    - Lander locations.
    - Rover capabilities (equipped_for_soil/rock/imaging).
    - Store ownership.
    - Camera properties (on_board, supports, calibration_target).
    - Waypoints visible from landers (communication points).
    - Waypoints visible from calibration targets (calibration points per camera).
    - Waypoints from which objectives are visible (image points per objective).
    - Initial soil/rock sample locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h = 0`.
    2. Extract dynamic information from the current state: rover positions,
       store contents, held data/images, camera calibration status.
    3. Iterate through each goal predicate in `self.goals`.
    4. If a goal predicate is already true in the current state, its cost is 0. Continue to the next goal.
    5. If a goal predicate `g` is not true:
        a. Initialize `min_goal_cost = infinity`.
        b. If `g` is `(communicated_soil_data ?w)`:
            i. Iterate through all rovers `r` that are equipped for soil analysis and have a store `s`.
            ii. Get the current position of rover `r`.
            iii. Calculate the cost for this rover `r` to achieve and communicate the goal:
                - If `(have_soil_analysis r w)` is true in the current state:
                    - Cost to get data = 0.
                    - Cost to communicate = `min_nav_cost(current_pos(r), any_comms_wp, r) + 1` (communicate).
                    - Total cost = Cost to communicate.
                - Else (`(have_soil_analysis r w)` is false):
                    - Check if `(at_soil_sample ?w)` was true in the initial state. If not, this rover cannot sample here. Continue to next rover.
                    - Cost to get data = `min_nav_cost(current_pos(r), ?w, r)` (nav to sample)
                                        `+ (1 if store_of(s, r) is full in state else 0)` (drop)
                                        `+ 1` (sample).
                    - Cost to communicate = `min_nav_cost(?w, any_comms_wp, r)` (nav from sample)
                                          `+ 1` (communicate).
                    - Total cost = Cost to get data + Cost to communicate.
            iv. Update `min_goal_cost = min(min_goal_cost, Total cost for r)`.
        c. If `g` is `(communicated_rock_data ?w)`: Similar logic as soil data, checking rock capabilities, rock samples, and rock data.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            i. Iterate through all rover/camera pairs `(r, c)` where `r` is equipped for imaging, `c` is on board `r`, `c` supports mode `m`, and `c` has a calibration target `t`.
            ii. Get the current position of rover `r`.
            iii. Calculate the cost for this pair `(r, c)` to achieve and communicate the goal:
                - If `(have_image r o m)` is true in the current state:
                    - Cost to get image = 0.
                    - Cost to communicate = `min_nav_cost(current_pos(r), any_comms_wp, r) + 1` (communicate).
                    - Total cost = Cost to communicate.
                - Else (`(have_image r o m)` is false):
                    - Get the set of calibration waypoints for `c` and image waypoints for `o`. If either set is empty, this pair cannot achieve the goal. Continue to next pair.
                    - Cost to get image = `min_nav(current_pos(r), cal_wps_for_c, r)` (nav to cal)
                                        `+ 1` (calibrate)
                                        `+ min_nav(cal_wps_for_c, image_wps_for_o, r)` (nav cal to image)
                                        `+ 1` (take_image).
                    - Cost to communicate = `min_nav(image_wps_for_o, comms_wps, r)` (nav image to comms)
                                          `+ 1` (communicate).
                    - Total cost = Cost to get image + Cost to communicate.
            iv. Update `min_goal_cost = min(min_goal_cost, Total cost for (r, c))`.
        e. Add `min_goal_cost` to `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state # Need initial state for sample locations
        self.static_facts = task.static

        # --- Precompute static information ---

        # Waypoint graph per rover
        self.rover_graphs = defaultdict(lambda: defaultdict(list))
        # Rover capabilities
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        # Store info
        self.store_owner = {} # store -> rover
        self.rover_store = {} # rover -> store
        # Camera info
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = defaultdict(set) # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        # Lander info
        self.lander_pos = {} # lander -> waypoint
        # Waypoint visibility
        self.visible_wps = defaultdict(set) # wp -> set of visible wps
        # Objective visibility
        self.visible_from_obj = defaultdict(set) # objective -> set of wps it's visible from
        # Calibration target visibility (subset of visible_from_obj)
        self.visible_from_cal_target = defaultdict(set) # objective (cal target) -> set of wps it's visible from
        # Initial sample locations
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        # Parse static facts
        for fact_str in self.static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'can_traverse':
                rover, wp1, wp2 = args
                self.rover_graphs[rover][wp1].append(wp2)
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(args[0])
            elif pred == 'store_of':
                store, rover = args
                self.store_owner[store] = rover
                self.rover_store[rover] = store
            elif pred == 'on_board':
                camera, rover = args
                self.camera_on_board[camera] = rover
            elif pred == 'supports':
                camera, mode = args
                self.camera_supports[camera].add(mode)
            elif pred == 'calibration_target':
                camera, objective = args
                self.camera_cal_target[camera] = objective
            elif pred == 'at_lander':
                lander, waypoint = args
                self.lander_pos[lander] = waypoint
            elif pred == 'visible':
                wp1, wp2 = args
                self.visible_wps[wp1].add(wp2)
            elif pred == 'visible_from':
                objective, waypoint = args
                self.visible_from_obj[objective].add(waypoint)


        # Link calibration targets to visible_from
        for camera, cal_obj in self.camera_cal_target.items():
             if cal_obj in self.visible_from_obj:
                  self.visible_from_cal_target[cal_obj].update(self.visible_from_obj[cal_obj])


        # Parse initial state for initial sample locations
        for fact_str in self.initial_state:
             pred, args = self._parse_fact(fact_str)
             if pred == 'at_soil_sample':
                 self.initial_soil_samples.add(args[0])
             elif pred == 'at_rock_sample':
                 self.initial_rock_samples.add(args[0])

        # Precompute navigation costs (all-pairs shortest paths per rover)
        self.min_nav_cost = defaultdict(dict) # rover -> start_wp -> end_wp -> cost

        # Collect all waypoints relevant for navigation starts/ends for each rover
        rover_relevant_waypoints = defaultdict(set)
        for rover, graph in self.rover_graphs.items():
             # Waypoints in the rover's graph
             rover_relevant_waypoints[rover].update(graph.keys())
             for neighbors in graph.values():
                 rover_relevant_waypoints[rover].update(neighbors)

        # Add waypoints from static facts that might be start/end points for any rover
        all_static_wps = set(self.lander_pos.values())
        all_static_wps.update(self.initial_soil_samples)
        all_static_wps.update(self.initial_rock_samples)
        for wps_set in self.visible_from_obj.values():
             all_static_wps.update(wps_set)
        for wps_set in self.visible_wps.values():
             all_static_wps.update(wps_set)


        for rover, graph in self.rover_graphs.items():
            # Run BFS from all relevant waypoints for this rover
            # A waypoint is relevant if it's in the rover's graph or mentioned in static facts
            relevant_start_wps = rover_relevant_waypoints[rover] | all_static_wps
            for start_wp in relevant_start_wps:
                 # Only run BFS if the start_wp is connected to the rover's graph
                 # (i.e., it's in the set of nodes within the rover's graph)
                 if start_wp in rover_relevant_waypoints[rover]:
                     self.min_nav_cost[rover][start_wp] = self._bfs(graph, start_wp)
                 # If start_wp is not in rover_relevant_waypoints, BFS from it won't reach
                 # any node in the graph, distances remain infinity, handled by .get()


        # Precompute communication waypoints
        self.comms_wps = set()
        lander_wps_set = set(self.lander_pos.values())
        # Consider all waypoints that could potentially be comms points (those mentioned in visible)
        potential_comms_wps = set(self.visible_wps.keys()) | set.union(*self.visible_wps.values()) if self.visible_wps else set()
        for wp in potential_comms_wps:
            for lander_wp in lander_wps_set:
                if lander_wp in self.visible_wps.get(wp, set()):
                    self.comms_wps.add(wp)
                    break # No need to check other lander wps for this waypoint


        # Precompute calibration waypoints per camera
        self.cal_wps = defaultdict(set) # camera -> set of wps
        for camera, cal_target in self.camera_cal_target.items():
            self.cal_wps[camera] = self.visible_from_cal_target.get(cal_target, set())

        # Precompute image waypoints per objective
        self.image_wps = self.visible_from_obj # objective -> set of wps


    def _parse_fact(self, fact_string):
        """Helper function to parse a fact string."""
        # Remove surrounding brackets and split by space
        parts = fact_string.strip('()').split()
        return parts[0], parts[1:]

    def _bfs(self, graph, start_node):
        """Helper function for BFS."""
        # Collect all nodes present in the graph (keys and values)
        all_nodes_in_graph = set(graph.keys())
        for neighbors in graph.values():
             all_nodes_in_graph.update(neighbors)

        distances = {node: float('inf') for node in all_nodes_in_graph}

        if start_node not in distances:
             # Start node is not part of the graph at all.
             # Return distances initialized to inf for all nodes in the graph.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Consider neighbors reachable from current_node
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances


    def min_nav(self, start_loc, target_wps_set, rover):
        """
        Calculates minimum navigation cost for a rover from a start location
        (single waypoint or set of waypoints) to any waypoint in a target set.
        Returns float('inf') if no path exists.
        """
        if not target_wps_set:
            return float('inf')

        min_cost = float('inf')
        start_wps = {start_loc} if isinstance(start_loc, str) else start_loc

        for s_wp in start_wps:
            # Check if the rover exists and can navigate *from* this start waypoint
            if rover in self.min_nav_cost and s_wp in self.min_nav_cost[rover]:
                for t_wp in target_wps_set:
                    min_cost = min(min_cost, self.min_nav_cost[rover][s_wp].get(t_wp, float('inf')))
            # else: rover cannot navigate from s_wp, min_cost remains inf for this s_wp

        return min_cost

    def __call__(self, node):
        state = node.state
        h = 0

        # Extract dynamic state info
        rover_pos = {} # rover -> waypoint
        store_status = {} # store -> 'full' or 'empty'
        have_soil = defaultdict(dict) # rover -> wp -> True
        have_rock = defaultdict(dict) # rover -> wp -> True
        have_image = defaultdict(lambda: defaultdict(dict)) # rover -> obj -> mode -> True
        calibrated = defaultdict(dict) # camera -> rover -> True

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                rover, wp = args
                rover_pos[rover] = wp
            elif pred == 'full' and len(args) == 1:
                store_status[args[0]] = 'full'
            elif pred == 'empty' and len(args) == 1:
                 store_status[args[0]] = 'empty'
            elif pred == 'have_soil_analysis' and len(args) == 2:
                rover, wp = args
                have_soil[rover][wp] = True
            elif pred == 'have_rock_analysis' and len(args) == 2:
                rover, wp = args
                have_rock[rover][wp] = True
            elif pred == 'have_image' and len(args) == 3:
                rover, obj, mode = args
                have_image[rover][obj][mode] = True
            elif pred == 'calibrated' and len(args) == 2:
                camera, rover = args
                calibrated[camera][rover] = True
            # Communicated goals are checked directly below

        # Evaluate each goal
        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already achieved

            pred, args = self._parse_fact(goal_str)

            if pred == 'communicated_soil_data' and len(args) == 1:
                w = args[0]
                min_goal_cost = float('inf')

                # Find capable rovers
                capable_rovers = [r for r in self.equipped_soil if r in self.rover_store]

                for r in capable_rovers:
                    s = self.rover_store[r]
                    current_pos = rover_pos.get(r)

                    if current_pos is None:
                         # Rover is not at any known waypoint? Should not happen in valid states.
                         continue

                    cost_to_get_data = float('inf')
                    if have_soil.get(r, {}).get(w, False):
                        # Rover already has the data
                        cost_to_get_data = 0
                    else:
                        # Need to sample
                        if w in self.initial_soil_samples: # Check if sample exists initially
                            nav_to_sample = self.min_nav(current_pos, {w}, r)
                            if nav_to_sample != float('inf'):
                                drop_cost = 1 if store_status.get(s) == 'full' else 0
                                sample_cost = 1
                                cost_to_get_data = nav_to_sample + drop_cost + sample_cost

                    if cost_to_get_data != float('inf'):
                        # Now communicate
                        # Nav from current pos if data existed, else from sample pos (w)
                        nav_start_pos = current_pos if cost_to_get_data == 0 else w
                        nav_to_comms = self.min_nav(nav_start_pos, self.comms_wps, r)
                        if nav_to_comms != float('inf'):
                            communicate_cost = 1
                            total_cost = cost_to_get_data + nav_to_comms + communicate_cost
                            min_goal_cost = min(min_goal_cost, total_cost)

                h += min_goal_cost

            elif pred == 'communicated_rock_data' and len(args) == 1:
                w = args[0]
                min_goal_cost = float('inf')

                # Find capable rovers
                capable_rovers = [r for r in self.equipped_rock if r in self.rover_store]

                for r in capable_rovers:
                    s = self.rover_store[r]
                    current_pos = rover_pos.get(r)

                    if current_pos is None: continue

                    cost_to_get_data = float('inf')
                    if have_rock.get(r, {}).get(w, False):
                        cost_to_get_data = 0
                    else:
                        if w in self.initial_rock_samples:
                            nav_to_sample = self.min_nav(current_pos, {w}, r)
                            if nav_to_sample != float('inf'):
                                drop_cost = 1 if store_status.get(s) == 'full' else 0
                                sample_cost = 1
                                cost_to_get_data = nav_to_sample + drop_cost + sample_cost

                    if cost_to_get_data != float('inf'):
                        nav_start_pos = current_pos if cost_to_get_data == 0 else w
                        nav_to_comms = self.min_nav(nav_start_pos, self.comms_wps, r)
                        if nav_to_comms != float('inf'):
                            communicate_cost = 1
                            total_cost = cost_to_get_data + nav_to_comms + communicate_cost
                            min_goal_cost = min(min_goal_cost, total_cost)

                h += min_goal_cost

            elif pred == 'communicated_image_data' and len(args) == 2:
                o, m = args
                min_goal_cost = float('inf')

                # Find capable rover/camera pairs
                capable_pairs = [] # list of (rover, camera)
                for r in self.equipped_imaging:
                    for c, r_on_board in self.camera_on_board.items():
                        if r_on_board == r and m in self.camera_supports.get(c, set()) and c in self.camera_cal_target:
                             capable_pairs.append((r, c))

                for r, c in capable_pairs:
                    current_pos = rover_pos.get(r)

                    if current_pos is None: continue

                    cost_to_get_image = float('inf')
                    if have_image.get(r, {}).get(o, {}).get(m, False):
                        # Rover already has the image
                        cost_to_get_image = 0
                    else:
                        # Need to take image
                        cal_wps_for_c = self.cal_wps.get(c, set())
                        image_wps_for_o = self.image_wps.get(o, set())

                        if cal_wps_for_c and image_wps_for_o:
                            # Cost = Nav to cal + Calibrate + Nav cal to image + Take image
                            nav_to_cal = self.min_nav(current_pos, cal_wps_for_c, r)
                            if nav_to_cal != float('inf'):
                                calibrate_cost = 1
                                # Need min nav from *any* cal wp to *any* image wp
                                nav_cal_to_image = self.min_nav(cal_wps_for_c, image_wps_for_o, r)
                                if nav_cal_to_image != float('inf'):
                                    take_image_cost = 1
                                    cost_to_get_image = nav_to_cal + calibrate_cost + nav_cal_to_image + take_image_cost


                    if cost_to_get_image != float('inf'):
                        # Now communicate
                        # Nav from current pos if image existed, else from image location(s)
                        # If image was just taken, the rover is at *an* image_wp_for_o.
                        # We need the min nav from *any* image_wp_for_o to *any* comms_wp.
                        nav_start_loc_set = {current_pos} if cost_to_get_image == 0 else image_wps_for_o
                        nav_to_comms = self.min_nav(nav_start_loc_set, self.comms_wps, r)

                        if nav_to_comms != float('inf'):
                            communicate_cost = 1
                            total_cost = cost_to_get_image + nav_to_comms + communicate_cost
                            min_goal_cost = min(min_goal_cost, total_cost)

                h += min_goal_cost

            # Add other goal types if any (none in this domain)

        return h
