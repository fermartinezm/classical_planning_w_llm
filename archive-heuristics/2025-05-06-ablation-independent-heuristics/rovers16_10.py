import heapq
import collections
import math

# Assuming heuristic_base and task modules are available in the environment
# from heuristics.heuristic_base import Heuristic
# from task import Operator, Task

# Define dummy classes if running standalone for testing or testing environment doesn't provide them directly
try:
    from heuristics.heuristic_base import Heuristic
    from task import Operator, Task
except ImportError:
    # Define minimal dummy classes if imports fail
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

    class Operator:
        def __init__(self, name, preconditions, add_effects, del_effects):
            self.name = name
            self.preconditions = frozenset(preconditions)
            self.add_effects = frozenset(add_effects)
            self.del_effects = frozenset(del_effects)
        def applicable(self, state): return self.preconditions <= state
        def apply(self, state): return (state - self.del_effects) | self.add_effects
        def __eq__(self, other): return self.__dict__ == other.__dict__
        def __hash__(self): return hash((self.name, self.preconditions, self.add_effects, self.del_effects))
        def __repr__(self): return f"<Op {self.name}>"


    class Task:
        def __init__(self, name, facts, initial_state, goals, operators, static):
            self.name = name
            self.facts = facts
            self.initial_state = initial_state
            self.goals = goals
            self.operators = operators
            self.static = static
        def goal_reached(self, state): return self.goals <= state


# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding brackets and split by space
    parts = fact_string.strip('()').split()
    return tuple(parts)

# Helper function for BFS
def bfs(start_wp, graph):
    """
    Performs BFS from start_wp on the given graph.
    Returns a dictionary mapping reachable waypoints to their distance from start_wp.
    Returns float('inf') for unreachable waypoints within the graph.
    """
    # Initialize distances for all waypoints present in the graph
    distances = {wp: float('inf') for wp in graph}
    if start_wp not in graph:
        # If start_wp is not even a key in the graph, it's isolated or doesn't exist
        # We can still return the distances dict which has inf for all known wps
        return distances

    distances[start_wp] = 0
    queue = collections.deque([start_wp])

    while queue:
        current_wp = queue.popleft()
        current_dist = distances[current_wp]

        # Check if current_wp has neighbors in the graph
        if current_wp in graph:
            for neighbor_wp in graph[current_wp]:
                # Ensure neighbor_wp is also a known waypoint in the graph keys
                if neighbor_wp in graph and distances[neighbor_wp] == float('inf'):
                    distances[neighbor_wp] = current_dist + 1
                    queue.append(neighbor_wp)
    return distances

# Helper function to get minimum distance from a BFS result to a set of target waypoints
def get_min_dist_from_bfs(dist_dict, target_wps):
    """
    Given a distance dictionary from BFS and a set of target waypoints,
    returns the minimum distance to any waypoint in the target set.
    Returns float('inf') if no target waypoint is reachable or in the dist_dict.
    """
    min_dist = float('inf')
    for wp in target_wps:
        # Check if the target waypoint exists in the distance dictionary (i.e., is in the graph)
        if wp in dist_dict:
             min_dist = min(min_dist, dist_dict[wp])
    return min_dist

# Helper function to find the closest waypoint from a BFS result among a set of target waypoints
def find_closest_wp_from_bfs(dist_dict, target_wps):
    """
    Given a distance dictionary from BFS and a set of target waypoints,
    returns the waypoint in the target set that has the minimum distance.
    Returns None if target_wps is empty or no target waypoint is reachable or in the dist_dict.
    """
    min_dist = float('inf')
    closest_wp = None
    for wp in target_wps:
        # Check if the target waypoint exists in the distance dictionary (i.e., is in the graph)
        if wp in dist_dict:
            if dist_dict[wp] < min_dist:
                min_dist = dist_dict[wp]
                closest_wp = wp
    return closest_wp


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs for each unachieved goal fact. For each unachieved goal, it finds the
    minimum cost across all capable rovers (and cameras for image goals). The
    cost for a single goal and rover is estimated as the sum of necessary actions
    (sample, take_image, calibrate, communicate, drop) and the minimum number
    of navigation steps required to visit the necessary locations (sample waypoint,
    image waypoint, calibration waypoint, communication waypoint) in sequence.
    Shortest path distances between waypoints for each rover are computed on
    demand using BFS.

    Assumptions:
    - Samples (soil/rock) are static and do not disappear unless collected.
    - Calibration targets are static objectives.
    - Visibility and traversability are static.
    - Stores are tied to specific rovers and their status (empty/full) changes only
      via sample/drop actions.
    - The heuristic assumes a sequential path for achieving goals:
      - Soil/Rock: current_location -> sample_waypoint -> communication_waypoint
      - Image (calibrated): current_location -> image_waypoint -> communication_waypoint
      - Image (not calibrated): current_location -> calibration_waypoint -> image_waypoint -> communication_waypoint
    - The heuristic uses shortest path distances for navigation costs.
    - The heuristic is non-admissible (it can overestimate).

    Heuristic Initialization:
    The constructor pre-processes static information from the task definition:
    - Builds a waypoint graph for each rover based on `can_traverse` and `visible` facts.
      Assumes `visible ?y ?z` implies `visible ?z ?y` and `can_traverse ?r ?y ?z`
      implies `can_traverse ?r ?z ?y` if `visible ?y ?z` is true.
    - Identifies lander location and communication waypoints (visible from lander).
    - Stores rover capabilities (soil, rock, imaging).
    - Stores rover-store mapping.
    - Stores camera information (on-board rover, supported modes, calibration target).
    - Stores visibility information for objectives and calibration targets from waypoints.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Initialize total heuristic value `h = 0`.
    2.  Parse the current state to extract dynamic information: rover locations,
        sample locations, collected data, camera calibration status, store status.
    3.  For each goal fact specified in `self.goals`:
        a.  Check if the goal fact is already true in the current state. If yes,
            continue to the next goal.
        b.  If the goal is `(communicated_soil_data ?w)`:
            i.  Initialize `min_cost_for_goal = infinity`.
            ii. Find all rovers `?r` equipped for soil analysis.
            iii. If no such rover exists, the goal is unreachable, return infinity.
            iv. For each capable rover `?r`:
                - Get the rover's current location. If unknown, skip rover.
                - Compute shortest path distances from the current location using BFS for this rover's graph.
                - Calculate the cost to achieve this goal using this rover, considering if the soil data at `?w` is already collected and if the rover's store is full.
                  - If `(have_soil_analysis ?r ?w)` is true: Cost is `dist(current, best_comm_wp) + 1 (communicate)`. `best_comm_wp` is the communication waypoint closest to the current location. If `best_comm_wp` is unreachable, this path is impossible.
                  - If `(have_soil_analysis ?r ?w)` is false: Check if the rover has a store. If not, this rover cannot sample, skip it. If it has a store, compute `dist(current, ?w)`. If unreachable, skip rover. Compute `dist(?w, best_comm_wp)` where `best_comm_wp` is the communication waypoint closest to `?w`. If unreachable, skip rover. Cost is `dist(current, ?w) + 1 (sample) + dist(?w, best_comm_wp) + 1 (communicate) + (1 if store is full)`.
                - Update `min_cost_for_goal` with the minimum cost found for this rover.
            v. If `min_cost_for_goal` is still infinity after checking all rovers, the goal is unreachable, return infinity.
            vi. Add `min_cost_for_goal` to `h`.
        c.  If the goal is `(communicated_rock_data ?w)`: Follow the same logic as for soil data, replacing soil-specific predicates with rock-specific ones.
        d.  If the goal is `(communicated_image_data ?o ?m)`:
            i.  Initialize `min_cost_for_goal = infinity`.
            ii. Find all rovers `?r` equipped for imaging, with a camera `?i` on board supporting mode `?m`.
            iii. If no such rover/camera exists, the goal is unreachable, return infinity.
            iv. Identify potential image waypoints (`W_img`), calibration waypoints (`W_cal`), and communication waypoints (`W_comm`). If `W_img` or `W_comm` is empty, the goal is unreachable, return infinity.
            v. For each capable rover `?r` and camera `?i`:
                - Get the rover's current location. If unknown, skip rover.
                - Compute shortest path distances from the current location using BFS.
                - Calculate the cost to achieve this goal using this rover/camera combination, considering if the image data is already collected and if the camera is calibrated.
                  - If `(have_image ?r ?o ?m)` is true: Cost is `dist(current, best_comm_wp) + 1 (communicate)`. `best_comm_wp` is the communication waypoint closest to the current location. If unreachable, this path is impossible.
                  - If `(have_image ?r ?o ?m)` is false:
                    - Find `best_img_wp_from_current` (closest image wp from current). If unreachable, skip rover/camera.
                    - Compute shortest path distances from `best_img_wp_from_current` using BFS. Find `best_comm_wp_from_img` (closest comm wp from `best_img_wp_from_current`). If unreachable, skip rover/camera.
                    - If `(calibrated ?i ?r)` is true: Cost is `dist(current, best_img_wp_from_current) + 1 (take_image) + dist(best_img_wp_from_current, best_comm_wp_from_img) + 1 (communicate)`.
                    - If `(calibrated ?i ?r)` is false:
                      - Find the calibration target `?t` for camera `?i`. If none, skip rover/camera.
                      - Get calibration waypoints `W_cal` for `?t`. If empty, skip rover/camera.
                      - Find `best_cal_wp_from_current` (closest cal wp from current). If unreachable, skip rover/camera.
                      - Compute shortest path distances from `best_cal_wp_from_current` using BFS. Compute `dist(best_cal_wp_from_current, best_img_wp_from_current)`. If unreachable, skip rover/camera.
                      - Cost is `dist(current, best_cal_wp_from_current) + 1 (calibrate) + dist(best_cal_wp_from_current, best_img_wp_from_current) + 1 (take_image) + dist(best_img_wp_from_current, best_comm_wp_from_img) + 1 (communicate)`.
                - Update `min_cost_for_goal` with the minimum cost found for this rover/camera.
            v. If `min_cost_for_goal` is still infinity after checking all rovers/cameras, the goal is unreachable, return infinity.
            vi. Add `min_cost_for_goal` to `h`.
    4.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

        # --- Preprocess Static Information ---
        self.rover_waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        self.rover_capabilities = collections.defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        self.rover_stores = {} # {rover: store}
        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible from lander location

        self.camera_info = {} # {camera: {rover: r, modes: {m1, m2}, cal_target: obj}}
        self.objective_visibility = collections.defaultdict(set) # {objective: {wp1, wp2}}
        self.calibration_target_visibility = collections.defaultdict(set) # {cal_target_obj: {wp1, wp2}}

        # Collect all waypoints and build initial graph based on visible
        visible_map = collections.defaultdict(set)
        all_waypoints_set = set() # Use a set to collect all waypoints mentioned
        all_rovers_set = set() # Use a set to collect all rovers mentioned

        for fact_string in task.initial_state | task.static:
             parts = parse_fact(fact_string)
             if parts[0] == 'at': all_rovers_set.add(parts[1]); all_waypoints_set.add(parts[2])
             elif parts[0] == 'at_lander': all_waypoints_set.add(parts[2])
             elif parts[0] == 'can_traverse': all_rovers_set.add(parts[1]); all_waypoints_set.add(parts[2]); all_waypoints_set.add(parts[3])
             elif parts[0] == 'equipped_for_soil_analysis': all_rovers_set.add(parts[1])
             elif parts[0] == 'equipped_for_rock_analysis': all_rovers_set.add(parts[1])
             elif parts[0] == 'equipped_for_imaging': all_rovers_set.add(parts[1])
             elif parts[0] == 'empty': pass # Stores are objects, collected via store_of
             elif parts[0] == 'full': pass # Stores are objects, collected via store_of
             elif parts[0] == 'have_rock_analysis': all_rovers_set.add(parts[1]); all_waypoints_set.add(parts[2])
             elif parts[0] == 'have_soil_analysis': all_rovers_set.add(parts[1]); all_waypoints_set.add(parts[2])
             elif parts[0] == 'calibrated': all_rovers_set.add(parts[2]) # Camera is object
             elif parts[0] == 'supports': pass # Camera and mode are objects
             elif parts[0] == 'visible': all_waypoints_set.add(parts[1]); all_waypoints_set.add(parts[2]); visible_map[parts[1]].add(parts[2]); visible_map[parts[2]].add(parts[1]) # Assuming visible is symmetric
             elif parts[0] == 'have_image': all_rovers_set.add(parts[1]) # Objective and mode are objects
             elif parts[0] == 'communicated_soil_data': all_waypoints_set.add(parts[1])
             elif parts[0] == 'communicated_rock_data': all_waypoints_set.add(parts[1])
             elif parts[0] == 'communicated_image_data': pass # Objective and mode are objects
             elif parts[0] == 'at_soil_sample': all_waypoints_set.add(parts[1])
             elif parts[0] == 'at_rock_sample': all_waypoints_set.add(parts[1])
             elif parts[0] == 'visible_from': all_waypoints_set.add(parts[2]) # Objective is object
             elif parts[0] == 'store_of': all_rovers_set.add(parts[2]) # Store is object
             elif parts[0] == 'calibration_target': pass # Camera and objective are objects
             elif parts[0] == 'on_board': all_rovers_set.add(parts[2]) # Camera is object

        # Process static facts
        for fact_string in self.static_facts:
            parts = parse_fact(fact_string)
            predicate = parts[0]

            if predicate == 'at_lander':
                self.lander_location = parts[2]
            elif predicate == 'can_traverse':
                r, w1, w2 = parts[1:]
                # Add edge only if visible
                if w2 in visible_map.get(w1, set()):
                     self.rover_waypoint_graph[r][w1].add(w2)
                     self.rover_waypoint_graph[r][w2].add(w1) # Assuming traverse is symmetric if visible is
            elif predicate == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]]['soil'] = True
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]]['rock'] = True
            elif predicate == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]]['imaging'] = True
            elif predicate == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # {rover: store}
            elif predicate == 'on_board':
                cam, r = parts[1:]
                if cam not in self.camera_info: self.camera_info[cam] = {}
                self.camera_info[cam]['rover'] = r
            elif predicate == 'supports':
                cam, mode = parts[1:]
                if cam not in self.camera_info: self.camera_info[cam] = {}
                if 'modes' not in self.camera_info[cam]: self.camera_info[cam]['modes'] = set()
                self.camera_info[cam]['modes'].add(mode)
            elif predicate == 'calibration_target':
                 cam, obj = parts[1:]
                 if cam not in self.camera_info: self.camera_info[cam] = {}
                 self.camera_info[cam]['cal_target'] = obj

        # Process visible_from after identifying calibration targets
        calibration_targets = {info['cal_target'] for info in self.camera_info.values() if 'cal_target' in info}
        for fact_string in self.static_facts:
             parts = parse_fact(fact_string)
             if parts[0] == 'visible_from':
                 obj, wp = parts[1:]
                 if obj in calibration_targets:
                     self.calibration_target_visibility[obj].add(wp)
                 else:
                     self.objective_visibility[obj].add(wp)

        # Determine communication waypoints (visible from lander)
        if self.lander_location:
             self.comm_waypoints = visible_map.get(self.lander_location, set())

        # Ensure all waypoints mentioned in can_traverse or visible are in the graph keys for BFS
        for rover, graph in self.rover_waypoint_graph.items():
             for wp in all_waypoints_set:
                  if wp not in graph:
                       graph[wp] = set() # Add waypoint with no outgoing edges if not already present


    def __call__(self, node):
        state = node.state
        h = 0

        # --- Extract Dynamic Information from State ---
        current_rover_locations = {} # {rover: wp}
        soil_samples_at = set() # {wp}
        rock_samples_at = set() # {wp}
        rover_soil_data = collections.defaultdict(bool) # {(rover, wp): bool}
        rover_rock_data = collections.defaultdict(bool) # {(rover, wp): bool}
        rover_image_data = collections.defaultdict(bool) # {(rover, obj, mode): bool}
        rover_calibration_status = collections.defaultdict(bool) # {(camera, rover): bool}
        rover_store_status = collections.defaultdict(lambda: 'empty') # {store: 'empty' or 'full'}
        communicated_soil_goals = set() # {wp}
        communicated_rock_goals = set() # {wp}
        communicated_image_goals = set() # {(obj, mode)}

        for fact_string in state:
            parts = parse_fact(fact_string)
            predicate = parts[0]
            if predicate == 'at':
                current_rover_locations[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis':
                rover_soil_data[(parts[1], parts[2])] = True
            elif predicate == 'have_rock_analysis':
                rover_rock_data[(parts[1], parts[2])] = True
            elif predicate == 'have_image':
                rover_image_data[(parts[1], parts[2], parts[3])] = True
            elif predicate == 'calibrated':
                rover_calibration_status[(parts[1], parts[2])] = True
            elif predicate == 'empty':
                rover_store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                rover_store_status[parts[1]] = 'full'
            elif predicate == 'communicated_soil_data':
                communicated_soil_goals.add(parts[1])
            elif predicate == 'communicated_rock_data':
                communicated_rock_goals.add(parts[1])
            elif predicate == 'communicated_image_data':
                communicated_image_goals.add((parts[1], parts[2]))

        # Group goals by type
        soil_goals_wp = {parse_fact(g)[1] for g in self.goals if parse_fact(g)[0] == 'communicated_soil_data'}
        rock_goals_wp = {parse_fact(g)[1] for g in self.goals if parse_fact(g)[0] == 'communicated_rock_data'}
        image_goals_obj_mode = {(parse_fact(g)[1], parse_fact(g)[2]) for g in self.goals if parse_fact(g)[0] == 'communicated_image_data'}

        # Soil Goals
        for wp in soil_goals_wp:
            if wp in communicated_soil_goals:
                continue # Goal already achieved

            min_cost_for_goal = float('inf')

            # Find capable rovers
            capable_rovers = [r for r in self.rover_capabilities if self.rover_capabilities[r]['soil']]

            if not capable_rovers:
                 return float('inf') # Goal unreachable if no rover can sample soil

            for rover_name in capable_rovers:
                if rover_name not in current_rover_locations:
                    continue # Rover location unknown or rover doesn't exist in this instance

                rover_current_wp = current_rover_locations[rover_name]
                rover_graph = self.rover_waypoint_graph.get(rover_name, {}) # Get graph for this rover

                # Precompute BFS from current location
                dist_from_current = bfs(rover_current_wp, rover_graph)

                cost_for_rover = float('inf')
                rover_has_data = rover_soil_data[(rover_name, wp)]

                if rover_has_data:
                    # Already have data, just need to communicate
                    min_dist_current_to_comm = get_min_dist_from_bfs(dist_from_current, self.comm_waypoints)
                    if min_dist_current_to_comm != float('inf'):
                         cost_for_rover = min_dist_current_to_comm + 1 # communicate
                else:
                    # Need to sample
                    rover_store = self.rover_stores.get(rover_name)
                    if rover_store is None:
                        continue # Rover cannot sample if it has no store

                    dist_current_to_sample = dist_from_current.get(wp, float('inf'))
                    if dist_current_to_sample != float('inf'):
                        # Need BFS from sample waypoint to find path to comm waypoint
                        dist_from_sample = bfs(wp, rover_graph)
                        min_dist_sample_to_comm = get_min_dist_from_bfs(dist_from_sample, self.comm_waypoints)

                        if min_dist_sample_to_comm != float('inf'):
                            cost = dist_current_to_sample + 1 # sample
                            cost += min_dist_sample_to_comm + 1 # communicate
                            if rover_store_status[rover_store] == 'full':
                                cost += 1 # drop
                            cost_for_rover = cost

                min_cost_for_goal = min(min_cost_for_goal, cost_for_rover)

            if min_cost_for_goal == float('inf'):
                 return float('inf') # Goal unreachable
            h += min_cost_for_goal

        # Rock Goals
        for wp in rock_goals_wp:
            if wp in communicated_rock_goals:
                continue # Goal already achieved

            min_cost_for_goal = float('inf')

            # Find capable rovers
            capable_rovers = [r for r in self.rover_capabilities if self.rover_capabilities[r]['rock']]

            if not capable_rovers:
                 return float('inf') # Goal unreachable

            for rover_name in capable_rovers:
                if rover_name not in current_rover_locations:
                    continue

                rover_current_wp = current_rover_locations[rover_name]
                rover_graph = self.rover_waypoint_graph.get(rover_name, {})

                # Precompute BFS from current location
                dist_from_current = bfs(rover_current_wp, rover_graph)

                cost_for_rover = float('inf')
                rover_has_data = rover_rock_data[(rover_name, wp)]

                if rover_has_data:
                    # Already have data, just need to communicate
                    min_dist_current_to_comm = get_min_dist_from_bfs(dist_from_current, self.comm_waypoints)
                    if min_dist_current_to_comm != float('inf'):
                         cost_for_rover = min_dist_current_to_comm + 1 # communicate
                else:
                    # Need to sample
                    rover_store = self.rover_stores.get(rover_name)
                    if rover_store is None:
                        continue # Rover cannot sample if it has no store

                    dist_current_to_sample = dist_from_current.get(wp, float('inf'))
                    if dist_current_to_sample != float('inf'):
                        # Need BFS from sample waypoint to find path to comm waypoint
                        dist_from_sample = bfs(wp, rover_graph)
                        min_dist_sample_to_comm = get_min_dist_from_bfs(dist_from_sample, self.comm_waypoints)

                        if min_dist_sample_to_comm != float('inf'):
                            cost = dist_current_to_sample + 1 # sample
                            cost += min_dist_sample_to_comm + 1 # communicate
                            if rover_store_status[rover_store] == 'full':
                                cost += 1 # drop
                            cost_for_rover = cost

                min_cost_for_goal = min(min_cost_for_goal, cost_for_rover)

            if min_cost_for_goal == float('inf'):
                 return float('inf') # Goal unreachable
            h += min_cost_for_goal


        # Image Goals
        for obj, mode in image_goals_obj_mode:
            if (obj, mode) in communicated_image_goals:
                continue # Goal already achieved

            min_cost_for_goal = float('inf')

            # Find capable rovers/cameras
            capable_rover_cameras = [] # [(rover, camera)]
            for cam_name, cam_info in self.camera_info.items():
                 rover_name = cam_info.get('rover')
                 supported_modes = cam_info.get('modes', set())
                 if rover_name and self.rover_capabilities[rover_name]['imaging'] and mode in supported_modes:
                      capable_rover_cameras.append((rover_name, cam_name))

            if not capable_rover_cameras:
                 return float('inf') # Goal unreachable

            img_wps = self.objective_visibility.get(obj, set())
            comm_wps = self.comm_waypoints

            if not img_wps or not comm_wps:
                 return float('inf') # Cannot image or communicate

            for rover_name, camera_name in capable_rover_cameras:
                if rover_name not in current_rover_locations:
                    continue

                rover_current_wp = current_rover_locations[rover_name]
                rover_graph = self.rover_waypoint_graph.get(rover_name, {})

                # Precompute BFS from current location
                dist_from_current = bfs(rover_current_wp, rover_graph)

                cost_for_rover = float('inf')
                rover_has_image = rover_image_data[(rover_name, obj, mode)]

                if rover_has_image:
                    # Already have image, just need to communicate
                    min_dist_current_to_comm = get_min_dist_from_bfs(dist_from_current, comm_wps)
                    if min_dist_current_to_comm != float('inf'):
                         cost_for_rover = min_dist_current_to_comm + 1 # communicate
                else:
                    # Need to take image and communicate
                    min_dist_current_to_img = get_min_dist_from_bfs(dist_from_current, img_wps)
                    if min_dist_current_to_img == float('inf'):
                         continue # Cannot reach any image waypoint

                    best_img_wp_from_current = find_closest_wp_from_bfs(dist_from_current, img_wps)
                    if best_img_wp_from_current is None: # Should not happen if min_dist is not inf
                         continue

                    # Need BFS from the best image waypoint to find path to comm waypoint
                    dist_from_best_img = bfs(best_img_wp_from_current, rover_graph)
                    min_dist_img_to_comm = get_min_dist_from_bfs(dist_from_best_img, comm_wps)
                    if min_dist_img_to_comm == float('inf'):
                         continue # Cannot reach any comm waypoint from best image waypoint

                    cost = min_dist_current_to_img + 1 # take_image
                    cost += min_dist_img_to_comm + 1 # communicate

                    camera_calibrated = rover_calibration_status[(camera_name, rover_name)]
                    if not camera_calibrated:
                        cal_target = self.camera_info[camera_name].get('cal_target')
                        if cal_target is None:
                             # Camera requires calibration but has no target defined? Unreachable.
                             continue

                        cal_wps = self.calibration_target_visibility.get(cal_target, set())
                        if not cal_wps:
                             # Cannot calibrate
                             continue

                        min_dist_current_to_cal = get_min_dist_from_bfs(dist_from_current, cal_wps)
                        if min_dist_current_to_cal == float('inf'):
                             continue # Cannot reach any calibration waypoint

                        best_cal_wp_from_current = find_closest_wp_from_bfs(dist_from_current, cal_wps)
                        if best_cal_wp_from_current is None: # Should not happen if min_dist is not inf
                             continue

                        # Need BFS from the best calibration waypoint to find path to the best image waypoint
                        dist_from_best_cal = bfs(best_cal_wp_from_current, rover_graph)
                        dist_cal_to_img = dist_from_best_cal.get(best_img_wp_from_current, float('inf'))
                        if dist_cal_to_img == float('inf'):
                             continue # Cannot reach best image waypoint from best calibration waypoint

                        # Recalculate cost for the path: current -> cal -> img -> comm
                        cost = min_dist_current_to_cal + 1 # calibrate
                        cost += dist_cal_to_img + 1 # take_image
                        # Use the previously computed distance from best_img_wp_from_current to best_comm_wp_from_img
                        cost += min_dist_img_to_comm + 1 # communicate

                    cost_for_rover = cost

                min_cost_for_goal = min(min_cost_for_goal, cost_for_rover)

            if min_cost_for_goal == float('inf'):
                 return float('inf') # Goal unreachable
            h += min_cost_for_goal

        return h
