import heapq
from collections import deque, defaultdict
import re

# Assume Heuristic and Task classes are available from the planner environment
# from heuristics.heuristic_base import Heuristic
# from task import Operator, Task

# If running standalone for testing, uncomment these mock classes:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# class Operator:
#     def __init__(self, name, preconditions, add_effects, del_effects):
#         self.name = name
#         self.preconditions = frozenset(preconditions)
#         self.add_effects = frozenset(add_effects)
#         self.del_effects = frozenset(del_effects)

# class Task:
#     def __init__(self, name, facts, initial_state, goals, operators, static):
#         self.name = name
#         self.facts = facts
#         self.initial_state = initial_state
#         self.goals = goals
#         self.operators = operators
#         self.static = static

# class Node:
#     def __init__(self, state, parent=None, operator=None, cost=0):
#         self.state = state
#         self.parent = parent
#         self.operator = operator
#         self.cost = cost
#         self.depth = parent.depth + 1 if parent else 0

#     def __lt__(self, other):
#          # Dummy comparison for potential use in priority queue
#          return self.cost < other.cost


# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """
    Parses a PDDL fact string into a predicate name and a list of arguments.
    e.g., '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    """
    # Removes surrounding brackets and splits by spaces
    parts = fact_str[1:-1].split()
    if not parts:
        return None, [] # Should not happen with valid PDDL facts
    return parts[0], parts[1:]


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs for each unachieved goal fact. For each unachieved goal (communicating
    soil, rock, or image data), it calculates the minimum cost to first acquire
    the necessary data (if not already held by a rover) and then navigate to
    a communication waypoint and perform the communication action. Navigation
    costs are estimated using precomputed shortest paths (BFS) on the rover's
    traversal graph.

    Assumptions:
    - The waypoint graph defined by `visible` and `can_traverse` is static.
    - `visible w1 w2` implies `visible w2 w1`.
    - `can_traverse r w1 w2` implies `can_traverse r w2 w1` for navigation cost calculation (BFS).
    - Action costs are 1.
    - Dropping a sample always frees the store.
    - Taking an image always uncalibrates the camera.
    - Calibration requires navigating to a waypoint visible from the calibration target.
    - Sampling requires navigating to the sample location.
    - Imaging requires navigating to a waypoint visible from the objective.
    - Communication requires navigating to a waypoint visible from the lander.
    - The heuristic does not need to be admissible.

    Heuristic Initialization:
    - Parses static facts from the task description.
    - Identifies lander location(s), equipped rovers, camera capabilities,
      calibration targets, visibility relations (`visible`, `visible_from`).
    - Stores initial locations of soil and rock samples.
    - For each rover, builds a traversal graph based on `can_traverse` facts
      and precomputes all-pairs shortest paths using BFS.
    - Identifies communication waypoints (visible from lander).
    - Performs initial checks for goal reachability based on static/initial facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state and identify unachieved goal facts.
    2. Initialize total heuristic value `h = 0`.
    3. Extract dynamic information from the current state (rover locations,
       store status, acquired data, camera calibration status, current sample locations).
    4. For each unachieved goal fact `g`:
        a. Parse the goal fact (e.g., `(communicated_soil_data waypointX)`).
        b. Initialize the cost for this goal `cost_g = infinity`.
        c. Check if the required data is already held by any rover in the current state.
            - If yes: Calculate the minimum navigation cost for any rover holding the data to reach any communication waypoint, plus 1 for the communicate action. `cost_g = min_rover_with_data(min_comm_wp(dist(current_wp, comm_wp))) + 1`.
            - If no: Calculate the minimum cost to acquire the data and then communicate it. This involves finding a suitable rover and minimizing over required waypoints (sample location, image waypoint, calibration waypoint) and communication waypoints.
                - For soil/rock goals: Find an equipped rover. If the sample is still on the ground, calculate `min_equipped_rover(dist(current_wp, sample_wp) + 1 (sample) + (1 if store full else 0) (drop) + dist(sample_wp, comm_wp) + 1 (communicate))`. If the sample is gone from the ground and no rover has it, the goal is unreachable (`cost_g = infinity`).
                - For image goals: Find an equipped rover with a suitable camera/mode. Find image waypoints for the objective and calibration waypoints for the camera. Calculate `min_rover_camera_mode(min_image_wp(min_cal_wp(dist(current_wp, cal_wp) + 1 (calibrate) + dist(cal_wp, image_wp) + 1 (take_image) + dist(image_wp, comm_wp) + 1 (communicate))))`. If any required waypoint set is empty or no suitable rover/camera exists, the goal is unreachable (`cost_g = infinity`).
        d. If `cost_g` is infinity, the problem is likely unsolvable from this state. Return infinity immediately.
        e. Add `cost_g` to the total heuristic `h`.
    5. Return `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task

        # --- Precompute Static Information ---
        self.lander_locations = set()
        self.soil_equipped_rovers = set()
        self.rock_equipped_rovers = set()
        self.imaging_equipped_rovers = set()
        self.store_of_rover = {} # {store: rover}
        self.rover_stores = {} # {rover: store}
        self.camera_on_board = {} # {camera: rover}
        self.rover_cameras = defaultdict(set) # {rover: {camera}}
        self.camera_supports_mode = defaultdict(set) # {camera: {mode}}
        self.camera_calibration_target = {} # {camera: objective}
        self.objective_visible_from = defaultdict(set) # {objective: {waypoint}}
        self.caltarget_visible_from = defaultdict(set) # {objective: {waypoint}} # Calibration targets are objectives
        self.initial_soil_samples = set() # {waypoint}
        self.initial_rock_samples = set() # {waypoint}

        # Waypoint graph and rover traversal graphs
        self.waypoints = set()
        self.visible_graph = defaultdict(set) # Undirected
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set)) # {rover: {wp1: {wp2, ...}}}
        self.rovers = set() # Collect all rover names

        # Parse all facts (initial state + static) to get all objects and graph info
        all_facts = set(task.initial_state) | set(task.static)
        for fact_str in all_facts:
            pred, args = parse_fact(fact_str)
            if pred == 'at_lander':
                self.lander_locations.add(args[1])
            elif pred == 'equipped_for_soil_analysis':
                self.soil_equipped_rovers.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.rock_equipped_rovers.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.imaging_equipped_rovers.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'store_of':
                self.store_of_rover[args[0]] = args[1]
                self.rover_stores[args[1]] = args[0]
            elif pred == 'on_board':
                self.camera_on_board[args[0]] = args[1]
                self.rover_cameras[args[1]].add(args[0])
            elif pred == 'supports':
                self.camera_supports_mode[args[0]].add(args[1])
            elif pred == 'calibration_target':
                self.camera_calibration_target[args[0]] = args[1]
            elif pred == 'visible_from':
                # visible_from can be for objectives or calibration targets (which are objectives)
                obj_or_target, wp = args
                self.objective_visible_from[obj_or_target].add(wp)
                # Assume calibration targets are a subset of objectives and use the same mapping
                self.caltarget_visible_from[obj_or_target].add(wp)
            elif pred == 'at_soil_sample' and fact_str in task.initial_state:
                 self.initial_soil_samples.add(args[0])
            elif pred == 'at_rock_sample' and fact_str in task.initial_state:
                 self.initial_rock_samples.add(args[0])
            elif pred == 'visible':
                wp1, wp2 = args
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.visible_graph[wp1].add(wp2)
                self.visible_graph[wp2].add(wp1) # Assume symmetric visibility
            elif pred == 'can_traverse':
                rover, wp1, wp2 = args
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.rover_traversal_graphs[rover][wp1].add(wp2)
                self.rover_traversal_graphs[rover][wp2].add(wp1) # Assume symmetric traversal for BFS
            elif pred == 'at': # Need to collect all waypoints mentioned in initial 'at' facts too
                 if args[0] in self.rovers: # Ensure it's a rover
                     self.waypoints.add(args[1])


        # Precompute shortest paths for each rover
        self.rover_distances = {} # {rover: {wp1: {wp2: dist}}}
        for rover in self.rovers: # Iterate over all known rovers
             # Build graph only with waypoints reachable by this rover
             rover_specific_graph = self.rover_traversal_graphs.get(rover, defaultdict(set))
             self.rover_distances[rover] = self._bfs_all_pairs(rover_specific_graph)


        # Identify communication waypoints (visible from any lander location)
        self.comm_wps = set()
        for lander_wp in self.lander_locations:
             if lander_wp in self.visible_graph: # Lander location might not be in visible graph if no waypoints are visible from it
                 self.comm_wps.update(self.visible_graph[lander_wp])

        # Check initial goal reachability based on static/initial facts
        self._check_initial_reachability()


    def _bfs_all_pairs(self, graph):
        """Computes all-pairs shortest paths on a graph using BFS."""
        distances = {}
        waypoints_in_graph = list(graph.keys())
        for start_node in waypoints_in_graph:
            distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_node, dist = q.popleft()
                distances[start_node][current_node] = dist
                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))
        return distances

    def _get_distance(self, rover, wp1, wp2):
        """Get shortest distance for a rover between two waypoints."""
        # Check if rover exists and waypoints are in its distance map
        if rover not in self.rover_distances or \
           wp1 not in self.rover_distances[rover] or \
           wp2 not in self.rover_distances[rover][wp1]:
            return float('inf') # Not reachable by this rover
        return self.rover_distances[rover][wp1][wp2]

    def _check_initial_reachability(self):
        """Checks if goals are initially impossible based on static/initial facts."""
        self.initially_unreachable_goals = set()

        # If no communication waypoints exist, all communication goals are impossible
        if not self.comm_wps:
             self.initially_unreachable_goals.update(self.task.goals)
             return # No need to check further if communication is impossible

        for goal_str in self.task.goals:
            pred, args = parse_fact(goal_str)
            if pred == 'communicated_soil_data':
                waypoint = args[0]
                # Need equipped rover AND initial sample AND reachable sample AND reachable comm_wp
                if not self.soil_equipped_rovers or waypoint not in self.initial_soil_samples:
                     self.initially_unreachable_goals.add(goal_str)
                     continue

                # Check if sample wp is reachable by *any* equipped rover AND if *any* comm_wp is reachable from that same rover
                possible_rover_found = False
                for r in self.soil_equipped_rovers:
                    # Check if sample waypoint is in the rover's graph
                    if waypoint in self.rover_distances.get(r, {}):
                         # Check if any comm_wp is reachable from the sample waypoint by this rover
                         if any(self._get_distance(r, waypoint, comm_wp) != float('inf') for comm_wp in self.comm_wps):
                             possible_rover_found = True
                             break # Found at least one path for this goal

                if not possible_rover_found:
                     self.initially_unreachable_goals.add(goal_str)


            elif pred == 'communicated_rock_data':
                waypoint = args[0]
                # Need equipped rover AND initial sample AND reachable sample AND reachable comm_wp
                if not self.rock_equipped_rovers or waypoint not in self.initial_rock_samples:
                     self.initially_unreachable_goals.add(goal_str)
                     continue

                # Check if sample wp is reachable by *any* equipped rover AND if *any* comm_wp is reachable from that same rover
                possible_rover_found = False
                for r in self.rock_equipped_rovers:
                    if waypoint in self.rover_distances.get(r, {}):
                         if any(self._get_distance(r, waypoint, comm_wp) != float('inf') for comm_wp in self.comm_wps):
                             possible_rover_found = True
                             break
                if not possible_rover_found:
                     self.initially_unreachable_goals.add(goal_str)

            elif pred == 'communicated_image_data':
                objective, mode = args
                # Need equipped rover AND camera supporting mode AND camera with cal target AND visible_from for objective AND visible_from for cal target AND reachable image_wp AND reachable cal_wp AND reachable comm_wp

                image_wps = self.objective_visible_from.get(objective, set())
                if not image_wps: # No waypoints to image from
                     self.initially_unreachable_goals.add(goal_str)
                     continue

                possible_rover_camera_mode_found = False
                for r in self.imaging_equipped_rovers:
                    for camera in self.rover_cameras.get(r, set()):
                        if mode in self.camera_supports_mode.get(camera, set()):
                            cal_target = self.camera_calibration_target.get(camera)
                            if not cal_target: continue # Camera has no calibration target

                            cal_wps = self.caltarget_visible_from.get(cal_target, set())
                            if not cal_wps: continue # No waypoints to calibrate from

                            # Check if there is a path: any cal_wp -> any image_wp -> any comm_wp reachable by this rover
                            path_exists = False
                            # Check if any cal_wp, any image_wp, any comm_wp are individually reachable by this rover first for a quick fail
                            reachable_cal_wp_by_r = any(wp in self.rover_distances.get(r, {}) for wp in cal_wps)
                            reachable_image_wp_by_r = any(wp in self.rover_distances.get(r, {}) for wp in image_wps)
                            reachable_comm_wp_by_r = any(wp in self.rover_distances.get(r, {}) for wp in self.comm_wps)

                            if reachable_cal_wp_by_r and reachable_image_wp_by_r and reachable_comm_wp_by_r:
                                for cal_wp in cal_wps:
                                    for image_wp in image_wps:
                                        for comm_wp in self.comm_wps:
                                            if self._get_distance(r, cal_wp, image_wp) != float('inf') and \
                                               self._get_distance(r, image_wp, comm_wp) != float('inf'):
                                                path_exists = True
                                                break
                                        if path_exists: break
                                    if path_exists: break

                            if path_exists:
                                 possible_rover_camera_mode_found = True
                                 break # Found at least one path for this goal
                    if possible_rover_camera_mode_found:
                         break

                if not possible_rover_camera_mode_found:
                     self.initially_unreachable_goals.add(goal_str)


    def __call__(self, node):
        state = node.state
        unachieved_goals = self.task.goals - set(state)

        # If any goal was determined unreachable from the start, return infinity
        if any(goal in self.initially_unreachable_goals for goal in unachieved_goals):
             return float('inf')

        h = 0

        # --- Extract Dynamic Information from State ---
        rover_locations = {} # {rover: waypoint}
        store_status = {} # {store: 'empty' or 'full'}
        current_soil_samples = set() # {waypoint}
        current_rock_samples = set() # {waypoint}
        have_soil = set() # {(pred, rover, waypoint)}
        have_rock = set() # {(pred, rover, waypoint)}
        have_image = set() # {(pred, rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at':
                rover_locations[args[0]] = args[1]
            elif pred == 'empty':
                store_status[args[0]] = 'empty'
            elif pred == 'full':
                store_status[args[0]] = 'full'
            elif pred == 'at_soil_sample':
                current_soil_samples.add(args[0])
            elif pred == 'at_rock_sample':
                current_rock_samples.add(args[0])
            elif pred == 'have_soil_analysis':
                have_soil.add((pred, args[0], args[1]))
            elif pred == 'have_rock_analysis':
                have_rock.add((pred, args[0], args[1]))
            elif pred == 'have_image':
                have_image.add((pred, args[0], args[1], args[2]))
            elif pred == 'calibrated':
                calibrated_cameras.add((args[0], args[1]))

        # --- Calculate Cost for Each Unachieved Goal ---
        for goal_str in unachieved_goals:
            pred, args = parse_fact(goal_str)
            cost_g = float('inf')

            if pred == 'communicated_soil_data':
                waypoint = args[0]
                goal_achieved_by_rover = False
                best_comm_cost = float('inf')

                # Check if data is already acquired by any rover
                rovers_with_data = {r for (_, r, w) in have_soil if w == waypoint}
                if rovers_with_data:
                    goal_achieved_by_rover = True
                    for r in rovers_with_data:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                             min_comm_dist = float('inf')
                             for comm_wp in self.comm_wps:
                                 min_comm_dist = min(min_comm_dist, self._get_distance(r, current_wp, comm_wp))
                             if min_comm_dist != float('inf'):
                                 best_comm_cost = min(best_comm_cost, min_comm_dist + 1) # +1 for communicate action

                if goal_achieved_by_rover and best_comm_cost != float('inf'):
                    cost_g = best_comm_cost
                else: # Data not yet acquired or held by a rover
                    # Check if sample is still available on the ground
                    if waypoint not in self.initial_soil_samples or waypoint not in current_soil_samples:
                         # Sample was initially there but is now gone and no one has it
                         return float('inf') # Goal unreachable

                    min_acquire_comm_cost = float('inf')
                    for r in self.soil_equipped_rovers:
                        current_wp = rover_locations.get(r)
                        if not current_wp: continue # Rover location unknown (shouldn't happen in valid state)

                        dist_to_sample = self._get_distance(r, current_wp, waypoint)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample

                        drop_cost = 0
                        rover_store = self.rover_stores.get(r)
                        if rover_store and store_status.get(rover_store) == 'full':
                            drop_cost = 1 # Cost to drop before sampling

                        acquire_cost = dist_to_sample + 1 + drop_cost # Nav + Sample + Drop(if needed)

                        min_dist_from_sample_to_comm = float('inf')
                        # Distance from the sample waypoint to the communication waypoint
                        if waypoint in self.rover_distances.get(r, {}):
                            for comm_wp in self.comm_wps:
                                min_dist_from_sample_to_comm = min(min_dist_from_sample_to_comm, self._get_distance(r, waypoint, comm_wp))

                        if min_dist_from_sample_to_comm != float('inf'):
                            total_cost_for_r = acquire_cost + min_dist_from_sample_to_comm + 1 # +1 for communicate
                            min_acquire_comm_cost = min(min_acquire_comm_cost, total_cost_for_r)

                    cost_g = min_acquire_comm_cost


            elif pred == 'communicated_rock_data':
                waypoint = args[0]
                goal_achieved_by_rover = False
                best_comm_cost = float('inf')

                # Check if data is already acquired by any rover
                rovers_with_data = {r for (_, r, w) in have_rock if w == waypoint}
                if rovers_with_data:
                    goal_achieved_by_rover = True
                    for r in rovers_with_data:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                             min_comm_dist = float('inf')
                             for comm_wp in self.comm_wps:
                                 min_comm_dist = min(min_comm_dist, self._get_distance(r, current_wp, comm_wp))
                             if min_comm_dist != float('inf'):
                                 best_comm_cost = min(best_comm_cost, min_comm_dist + 1) # +1 for communicate action

                if goal_achieved_by_rover and best_comm_cost != float('inf'):
                    cost_g = best_comm_cost
                else: # Data not yet acquired or held by a rover
                    # Check if sample is still available on the ground
                    if waypoint not in self.initial_rock_samples or waypoint not in current_rock_samples:
                         # Sample was initially there but is now gone and no one has it
                         return float('inf') # Goal unreachable

                    min_acquire_comm_cost = float('inf')
                    for r in self.rock_equipped_rovers:
                        current_wp = rover_locations.get(r)
                        if not current_wp: continue # Rover location unknown

                        dist_to_sample = self._get_distance(r, current_wp, waypoint)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample

                        drop_cost = 0
                        rover_store = self.rover_stores.get(r)
                        if rover_store and store_status.get(rover_store) == 'full':
                            drop_cost = 1 # Cost to drop before sampling

                        acquire_cost = dist_to_sample + 1 + drop_cost # Nav + Sample + Drop(if needed)

                        min_dist_from_sample_to_comm = float('inf')
                        # Distance from the sample waypoint to the communication waypoint
                        if waypoint in self.rover_distances.get(r, {}):
                            for comm_wp in self.comm_wps:
                                min_dist_from_sample_to_comm = min(min_dist_from_sample_to_comm, self._get_distance(r, waypoint, comm_wp))

                        if min_dist_from_sample_to_comm != float('inf'):
                            total_cost_for_r = acquire_cost + min_dist_from_sample_to_comm + 1 # +1 for communicate
                            min_acquire_comm_cost = min(min_acquire_comm_cost, total_cost_for_r)

                    cost_g = min_acquire_comm_cost


            elif pred == 'communicated_image_data':
                objective, mode = args
                goal_achieved_by_rover = False
                best_comm_cost = float('inf')

                # Check if data is already acquired by any rover
                rovers_with_data = {r for (_, r, o, m) in have_image if o == objective and m == mode}
                if rovers_with_data:
                    goal_achieved_by_rover = True
                    for r in rovers_with_data:
                        current_wp = rover_locations.get(r)
                        if current_wp:
                             min_comm_dist = float('inf')
                             for comm_wp in self.comm_wps:
                                 min_comm_dist = min(min_comm_dist, self._get_distance(r, current_wp, comm_wp))
                             if min_comm_dist != float('inf'):
                                 best_comm_cost = min(best_comm_cost, min_comm_dist + 1) # +1 for communicate action

                if goal_achieved_by_rover and best_comm_cost != float('inf'):
                    cost_g = best_comm_cost
                else: # Data not yet acquired or held by a rover
                    min_acquire_comm_cost = float('inf')

                    image_wps = self.objective_visible_from.get(objective, set())
                    # Initial reachability check already covers empty image_wps/cal_wps/comm_wps/no suitable rover
                    # if not image_wps: continue # Should be caught by initial check

                    for r in self.imaging_equipped_rovers:
                        current_wp = rover_locations.get(r)
                        if not current_wp: continue # Rover location unknown

                        for camera in self.rover_cameras.get(r, set()):
                            if mode in self.camera_supports_mode.get(camera, set()):
                                cal_target = self.camera_calibration_target.get(camera)
                                if not cal_target: continue # Camera has no calibration target

                                cal_wps = self.caltarget_visible_from.get(cal_target, set())
                                # if not cal_wps: continue # Should be caught by initial check

                                # Find min cost path: current -> cal_wp -> image_wp -> comm_wp
                                min_path_cost_for_r_i_m = float('inf')

                                for image_wp in image_wps:
                                    # Check if image_wp is reachable by this rover
                                    if image_wp not in self.rover_distances.get(r, {}).get(current_wp, {}): continue

                                    for cal_wp in cal_wps:
                                        # Check if cal_wp is reachable by this rover
                                        if cal_wp not in self.rover_distances.get(r, {}).get(current_wp, {}): continue
                                        # Check if image_wp is reachable from cal_wp by this rover
                                        if image_wp not in self.rover_distances.get(r, {}).get(cal_wp, {}): continue

                                        dist_curr_to_cal = self._get_distance(r, current_wp, cal_wp)
                                        dist_cal_to_image = self._get_distance(r, cal_wp, image_wp)

                                        # Cost to acquire = Nav to cal + Calibrate + Nav to image + TakeImage
                                        acquire_cost = dist_curr_to_cal + 1 + dist_cal_to_image + 1

                                        # Cost to communicate = Nav from image to comm + Communicate
                                        min_dist_image_to_comm = float('inf')
                                        # Distance from the image waypoint to the communication waypoint
                                        if image_wp in self.rover_distances.get(r, {}):
                                            for comm_wp in self.comm_wps:
                                                min_dist_image_to_comm = min(min_dist_image_to_comm, self._get_distance(r, image_wp, comm_wp))

                                        if min_dist_image_to_comm != float('inf'):
                                            total_cost_for_path = acquire_cost + min_dist_image_to_comm + 1 # +1 for communicate
                                            min_path_cost_for_r_i_m = min(min_path_cost_for_r_i_m, total_cost_for_path)

                                # Update min cost over all r, i, m combinations
                                min_acquire_comm_cost = min(min_acquire_comm_cost, min_path_cost_for_r_i_m)

                    cost_g = min_acquire_comm_cost

            # If cost_g is still infinity, this specific goal is unreachable from this state
            if cost_g == float('inf'):
                 return float('inf') # Return infinity for the whole heuristic

            h += cost_g

        # Heuristic is 0 only if all goals are achieved
        # If unachieved_goals is empty, h is 0.
        return h
