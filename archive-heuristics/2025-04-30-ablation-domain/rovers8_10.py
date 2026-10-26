from fnmatch import fnmatch
from collections import deque, defaultdict
import math

# Assume Heuristic base class is defined elsewhere and imported
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# Helper function for BFS
def bfs(start_node, graph):
    """
    Performs BFS to find shortest distances from start_node in a graph.
    Graph is represented as {node: set_of_neighbors}.
    Returns a dictionary {node: distance}.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure current_node exists in graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

# Assume Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    each uncommunicated goal fact (soil data, rock data, image data).
    The total heuristic value is the sum of the minimum estimated costs
    for each individual unachieved goal fact. The cost for each goal
    is estimated by finding the minimum number of moves, sample collections,
    calibrations, image captures, and communications required by any
    available rover, considering its current state and capabilities.
    Distances between waypoints are precomputed using BFS on the
    traversal graph.

    # Assumptions
    - The traversal graph is undirected and the same for all rovers
      (based on all `can_traverse` facts).
    - Dropping a sample costs 1 action and can be done anywhere.
    - A rover needs an empty store to pick up a sample. If the store
      is not empty, it needs to drop the current sample first (cost +1).
    - Communication requires the rover to be at a waypoint visible
      from the lander's waypoint.
    - Image capture requires the rover to be at a waypoint visible
      from the objective and the camera to be calibrated.
    - Camera calibration requires the rover to be at a waypoint visible
      from the camera's calibration target.
    - All action costs are assumed to be 1.
    - The heuristic does not consider resource contention (e.g., multiple
      rovers needing the same waypoint or store simultaneously).
    - Unreachable goals (e.g., due to disconnected graph or missing equipment)
      contribute infinity to the heuristic sum.

    # Heuristic Initialization
    The constructor precomputes static information from the task:
    - The lander's waypoint.
    - The set of waypoints visible from the lander's waypoint (communication points).
    - The undirected graph of traversable waypoints based on all `can_traverse` facts.
    - Shortest distances between all pairs of waypoints using BFS on the traversal graph.
    - Locations of soil and rock samples in the initial state.
    - Rover capabilities (soil, rock, imaging).
    - Camera information (which rover has it, supported modes, calibration target).
    - Objective visibility (from which waypoints an objective is visible).
    - Rover store mapping (which store belongs to which rover).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get dynamic facts: rover locations, sample locations,
       store contents, camera calibration status, image status, and communicated status.
    3. Iterate through each goal fact specified in `task.goals`.
    4. If a goal fact is already true in the current state, it contributes 0 to the heuristic.
    5. If a goal fact is not true, estimate the minimum cost to achieve it:
        - **For `(communicated_soil_data ?w)`:**
            - Find all soil-equipped rovers.
            - For each such rover, calculate the cost:
                - If the rover already has the sample `(have_soil_sample ?store ?w)`:
                    - Cost = `dist(rover_loc, nearest_comm_waypoint) + 1` (communicate).
                - If the sample is still at the waypoint `(at_soil_sample ?w)`:
                    - If the rover's store is empty:
                        - Cost = `dist(rover_loc, ?w) + 1` (move + take sample) + `dist(?w, nearest_comm_waypoint) + 1` (move + communicate).
                    - If the rover's store is not empty:
                        - Cost = `1` (drop sample) + `dist(rover_loc, ?w) + 1` (move + take sample) + `dist(?w, nearest_comm_waypoint) + 1` (move + communicate).
                - If the sample is held by another rover `(have_soil_sample ?store' ?w)`:
                    - Cost = `dist(other_rover_loc, nearest_comm_waypoint) + 1` (move + communicate).
            - The cost for this goal is the minimum cost over all considered rovers.
        - **For `(communicated_rock_data ?w)`:**
            - Similar logic as soil data, using rock-equipped rovers and rock samples.
        - **For `(communicated_image_data ?obj ?mode)`:**
            - Find all rovers with cameras supporting `?mode`.
            - For each such rover/camera pair:
                - If the rover already has the image `(have_image ?r ?obj ?mode)`:
                    - Cost = `dist(rover_loc, nearest_comm_waypoint) + 1` (move + communicate).
                - If the rover does not have the image:
                    - Find imaging waypoints visible from `?obj`. If none, this path is impossible.
                    - Find the nearest imaging waypoint `img_w` from the rover's current location.
                    - If the camera is calibrated `(calibrated ?c ?r)`:
                        - Cost = `dist(rover_loc, img_w) + 1` (move + take image) + `dist(img_w, nearest_comm_waypoint) + 1` (move + communicate).
                    - If the camera is not calibrated:
                        - Find calibration target `cal_obj` for the camera.
                        - Find calibration waypoints visible from `cal_obj`. If none, this path is impossible.
                        - Find the nearest calibration waypoint `cal_w` from the rover's current location.
                        - Find the nearest imaging waypoint `img_w_after_cal` from `cal_w`.
                        - Cost = `dist(rover_loc, cal_w) + 1` (move + calibrate) + `dist(cal_w, img_w_after_cal) + 1` (move + take image) + `dist(img_w_after_cal, nearest_comm_waypoint) + 1` (move + communicate).
            - The cost for this goal is the minimum cost over all considered rover/camera pairs.
    6. Sum the minimum costs for all unachieved goals to get the total heuristic value.
    7. If the total cost is infinity (meaning at least one unachieved goal is unreachable), return a large number. Otherwise, return the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---

        self.lander_waypoint = None
        self.visible_graph = defaultdict(set)
        self.traverse_graph = defaultdict(set)
        self.soil_sample_locations_init = set()
        self.rock_sample_locations_init = set()
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'cal_target': obj}
        self.objective_visibility = defaultdict(set) # objective -> {waypoint}
        self.rover_stores = {} # rover -> store

        all_waypoints = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == 'at_lander':
                # (at_lander ?lander ?waypoint)
                if len(parts) == 3:
                    self.lander_waypoint = parts[2]
                    all_waypoints.add(self.lander_waypoint)
            elif predicate == 'visible':
                # (visible ?waypoint1 ?waypoint2)
                if len(parts) == 3:
                    w1, w2 = parts[1], parts[2]
                    self.visible_graph[w1].add(w2)
                    self.visible_graph[w2].add(w1) # Assuming visibility is symmetric
                    all_waypoints.add(w1)
                    all_waypoints.add(w2)
            elif predicate == 'can_traverse':
                # (can_traverse ?rover ?waypoint1 ?waypoint2)
                if len(parts) == 4:
                    w1, w2 = parts[2], parts[3]
                    self.traverse_graph[w1].add(w2)
                    self.traverse_graph[w2].add(w1) # Assuming traversability is symmetric for the simplified graph
                    all_waypoints.add(w1)
                    all_waypoints.add(w2)
            elif predicate == 'at_soil_sample':
                # (at_soil_sample ?waypoint)
                if len(parts) == 2:
                    self.soil_sample_locations_init.add(parts[1])
                    all_waypoints.add(parts[1])
            elif predicate == 'at_rock_sample':
                # (at_rock_sample ?waypoint)
                if len(parts) == 2:
                    self.rock_sample_locations_init.add(parts[1])
                    all_waypoints.add(parts[1])
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis ?rover)
                if len(parts) == 2:
                    self.rover_capabilities[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis ?rover)
                if len(parts) == 2:
                    self.rover_capabilities[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging ?rover)
                if len(parts) == 2:
                    self.rover_capabilities[parts[1]].add('imaging')
            elif predicate == 'on_board':
                # (on_board ?camera ?rover)
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    if camera not in self.camera_info:
                        self.camera_info[camera] = {'rover': rover, 'modes': set(), 'cal_target': None}
                    self.camera_info[camera]['rover'] = rover
            elif predicate == 'supports':
                # (supports ?camera ?mode)
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    if camera not in self.camera_info:
                         self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                    self.camera_info[camera]['modes'].add(mode)
            elif predicate == 'calibration_target':
                # (calibration_target ?camera ?objective)
                if len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    if camera not in self.camera_info:
                         self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                    self.camera_info[camera]['cal_target'] = objective
            elif predicate == 'visible_from':
                # (visible_from ?objective ?waypoint)
                if len(parts) == 3:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visibility[objective].add(waypoint)
                    all_waypoints.add(waypoint)
            elif predicate == 'store_of':
                 # (store_of ?store ?rover)
                 if len(parts) == 3:
                     store, rover = parts[1], parts[2]
                     self.rover_stores[rover] = store

        # Ensure all waypoints mentioned in static facts are in the graph keys
        # This handles cases where a waypoint exists but has no traversal edges in static facts
        for w in all_waypoints:
             if w not in self.traverse_graph:
                 self.traverse_graph[w] = set() # Add isolated waypoints

        # Precompute distances between all waypoints
        self.waypoint_distances = {}
        # Only compute BFS from waypoints that actually exist in the graph keys
        for start_w in self.traverse_graph.keys():
            distances_from_start = bfs(start_w, self.traverse_graph)
            for end_w, dist in distances_from_start.items():
                self.waypoint_distances[(start_w, end_w)] = dist

        # Determine communication waypoints (visible from lander waypoint)
        self.comm_waypoints = set()
        if self.lander_waypoint:
             # Waypoints visible from the lander waypoint
             if self.lander_waypoint in self.visible_graph:
                  self.comm_waypoints.update(self.visible_graph[self.lander_waypoint])
             # Also, any waypoint W such that (visible W lander_waypoint) is a comm point.
             # Since visible is assumed symmetric, this is the same set.
             # Let's explicitly check all visible facts to be sure.
             for w1, w2_set in self.visible_graph.items():
                 if self.lander_waypoint in w2_set:
                     self.comm_waypoints.add(w1)


    def get_distance(self, w1, w2):
        """Looks up precomputed distance between two waypoints."""
        if w1 is None or w2 is None:
             return math.inf # Cannot calculate distance if waypoint is None
        return self.waypoint_distances.get((w1, w2), math.inf)

    def find_nearest_waypoint_distance(self, from_waypoint, target_waypoints):
        """Finds the minimum distance from from_waypoint to any waypoint in target_waypoints."""
        if from_waypoint is None or not target_waypoints:
            return math.inf
        min_dist = math.inf
        for target_w in target_waypoints:
            min_dist = min(min_dist, self.get_distance(from_waypoint, target_w))
        return min_dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Goal states have heuristic 0
        if self.goals <= state:
            return 0

        # --- Extract Dynamic Information ---
        rover_locations = {} # rover -> waypoint
        soil_samples_at_waypoint = set() # waypoint
        rock_samples_at_waypoint = set() # waypoint
        store_status = {} # store -> 'empty' or 'soil' or 'rock'
        rover_samples_held = {} # rover -> (sample_type, waypoint) or None
        camera_calibrated = set() # (camera, rover)
        rover_images_held = defaultdict(set) # rover -> {(objective, mode)}
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at':
                # (at ?obj ?loc) - used for rovers
                if len(parts) == 3 and parts[1].startswith('rover') and parts[2].startswith('waypoint'):
                     rover_locations[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample':
                # (at_soil_sample ?waypoint)
                if len(parts) == 2:
                    soil_samples_at_waypoint.add(parts[1])
            elif predicate == 'at_rock_sample':
                # (at_rock_sample ?waypoint)
                if len(parts) == 2:
                    rock_samples_at_waypoint.add(parts[1])
            elif predicate == 'empty':
                # (empty ?store)
                if len(parts) == 2:
                    store_status[parts[1]] = 'empty'
            elif predicate == 'have_soil_sample':
                # (have_soil_sample ?store ?waypoint)
                if len(parts) == 3:
                    store, waypoint = parts[1], parts[2]
                    store_status[store] = 'soil'
                    # Find the rover holding this store
                    rover = next((r for r, s in self.rover_stores.items() if s == store), None)
                    if rover:
                        rover_samples_held[rover] = ('soil', waypoint)
            elif predicate == 'have_rock_sample':
                # (have_rock_sample ?store ?waypoint)
                if len(parts) == 3:
                    store, waypoint = parts[1], parts[2]
                    store_status[store] = 'rock'
                     # Find the rover holding this store
                    rover = next((r for r, s in self.rover_stores.items() if s == store), None)
                    if rover:
                        rover_samples_held[rover] = ('rock', waypoint)
            elif predicate == 'calibrated':
                # (calibrated ?camera ?rover)
                if len(parts) == 3:
                    camera_calibrated.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                # (have_image ?rover ?objective ?mode)
                if len(parts) == 4:
                    rover_images_held[parts[1]].add((parts[2], parts[3]))
            elif predicate == 'communicated_soil_data':
                # (communicated_soil_data ?waypoint)
                if len(parts) == 2:
                    communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                # (communicated_rock_data ?waypoint)
                if len(parts) == 2:
                    communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                # (communicated_image_data ?objective ?mode)
                if len(parts) == 3:
                    communicated_image.add((parts[1], parts[2]))

        total_heuristic_cost = 0

        # --- Estimate Cost for Each Unachieved Goal ---

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                # Goal: (communicated_soil_data ?w)
                if len(parts) != 2: continue
                waypoint = parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Consider all rovers equipped for soil analysis
                for rover, capabilities in self.rover_capabilities.items():
                    if 'soil' not in capabilities:
                        continue # Rover cannot collect soil

                    loc_r = rover_locations.get(rover)
                    if loc_r is None:
                         continue # Rover location unknown (shouldn't happen in valid states?)

                    store_r = self.rover_stores.get(rover)
                    if store_r is None:
                         continue # Rover has no store (shouldn't happen)

                    current_rover_cost = math.inf

                    # Case 1: Rover already has the sample
                    if rover_samples_held.get(rover) == ('soil', waypoint):
                        dist_to_comm = self.find_nearest_waypoint_distance(loc_r, self.comm_waypoints)
                        if dist_to_comm != math.inf:
                            current_rover_cost = dist_to_comm + 1 # communicate

                    # Case 2: Sample is at the waypoint and needs collecting
                    elif waypoint in soil_samples_at_waypoint:
                        dist_to_sample = self.get_distance(loc_r, waypoint)
                        if dist_to_sample != math.inf:
                            dist_sample_to_comm = self.find_nearest_waypoint_distance(waypoint, self.comm_waypoints)
                            if dist_sample_to_comm != math.inf:
                                # Cost to collect + cost to communicate
                                cost_collect = dist_to_sample + 1 # move + take_soil_sample
                                if store_status.get(store_r) != 'empty':
                                     cost_collect += 1 # Add cost for drop_sample if store is not empty
                                cost_communicate = dist_sample_to_comm + 1 # move + communicate_soil_data
                                current_rover_cost = cost_collect + cost_communicate

                    # Case 3: Sample is held by another rover
                    else:
                         # Find which rover holds the sample
                         rover_with_sample = None
                         for r_other, (sample_type, sample_w) in rover_samples_held.items():
                              if sample_type == 'soil' and sample_w == waypoint:
                                   rover_with_sample = r_other
                                   break

                         if rover_with_sample:
                              loc_r_other = rover_locations.get(rover_with_sample)
                              if loc_r_other:
                                   dist_to_comm = self.find_nearest_waypoint_distance(loc_r_other, self.comm_waypoints)
                                   if dist_to_comm != math.inf:
                                        current_rover_cost = dist_to_comm + 1 # communicate

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                total_heuristic_cost += min_goal_cost # Add infinity if unreachable

            elif predicate == 'communicated_rock_data':
                # Goal: (communicated_rock_data ?w)
                if len(parts) != 2: continue
                waypoint = parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Consider all rovers equipped for rock analysis
                for rover, capabilities in self.rover_capabilities.items():
                    if 'rock' not in capabilities:
                        continue # Rover cannot collect rock

                    loc_r = rover_locations.get(rover)
                    if loc_r is None:
                         continue # Rover location unknown

                    store_r = self.rover_stores.get(rover)
                    if store_r is None:
                         continue # Rover has no store

                    current_rover_cost = math.inf

                    # Case 1: Rover already has the sample
                    if rover_samples_held.get(rover) == ('rock', waypoint):
                        dist_to_comm = self.find_nearest_waypoint_distance(loc_r, self.comm_waypoints)
                        if dist_to_comm != math.inf:
                            current_rover_cost = dist_to_comm + 1 # communicate

                    # Case 2: Sample is at the waypoint and needs collecting
                    elif waypoint in rock_samples_at_waypoint:
                        dist_to_sample = self.get_distance(loc_r, waypoint)
                        if dist_to_sample != math.inf:
                            dist_sample_to_comm = self.find_nearest_waypoint_distance(waypoint, self.comm_waypoints)
                            if dist_sample_to_comm != math.inf:
                                # Cost to collect + cost to communicate
                                cost_collect = dist_to_sample + 1 # move + take_rock_sample
                                if store_status.get(store_r) != 'empty':
                                     cost_collect += 1 # Add cost for drop_sample
                                cost_communicate = dist_sample_to_comm + 1 # move + communicate_rock_data
                                current_rover_cost = cost_collect + cost_communicate

                    # Case 3: Sample is held by another rover
                    else:
                         # Find which rover holds the sample
                         rover_with_sample = None
                         for r_other, (sample_type, sample_w) in rover_samples_held.items():
                              if sample_type == 'rock' and sample_w == waypoint:
                                   rover_with_sample = r_other
                                   break

                         if rover_with_sample:
                              loc_r_other = rover_locations.get(rover_with_sample)
                              if loc_r_other:
                                   dist_to_comm = self.find_nearest_waypoint_distance(loc_r_other, self.comm_waypoints)
                                   if dist_to_comm != math.inf:
                                        current_rover_cost = dist_to_comm + 1 # communicate

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                total_heuristic_cost += min_goal_cost # Add infinity if unreachable


            elif predicate == 'communicated_image_data':
                # Goal: (communicated_image_data ?obj ?mode)
                if len(parts) != 3: continue
                objective, mode = parts[1], parts[2]
                if (objective, mode) in communicated_image:
                    continue # Goal already achieved

                min_goal_cost = math.inf

                # Find cameras supporting this mode and their rovers
                suitable_camera_rover_pairs = []
                for camera, info in self.camera_info.items():
                    if mode in info['modes'] and info['rover']:
                        suitable_camera_rover_pairs.append((camera, info['rover']))

                for camera, rover in suitable_camera_rover_pairs:
                    loc_r = rover_locations.get(rover)
                    if loc_r is None:
                         continue # Rover location unknown

                    current_rover_cost = math.inf

                    # Case 1: Rover already has the image
                    if (objective, mode) in rover_images_held.get(rover, set()):
                        dist_to_comm = self.find_nearest_waypoint_distance(loc_r, self.comm_waypoints)
                        if dist_to_comm != math.inf:
                            current_rover_cost = dist_to_comm + 1 # communicate

                    # Case 2: Rover needs to take the image
                    else:
                        # Find imaging waypoints for the objective
                        img_waypoints = self.objective_visibility.get(objective, set())
                        if not img_waypoints:
                             continue # Cannot image this objective

                        # Find nearest imaging waypoint
                        dist_to_img_w = self.find_nearest_waypoint_distance(loc_r, img_waypoints)
                        if dist_to_img_w == math.inf:
                             continue # Cannot reach any imaging waypoint

                        # Find the specific nearest imaging waypoint to use for subsequent distance calcs
                        nearest_img_w = None
                        min_dist_img = math.inf
                        for img_w in img_waypoints:
                             dist = self.get_distance(loc_r, img_w)
                             if dist < min_dist_img:
                                  min_dist_img = dist
                                  nearest_img_w = img_w
                        if nearest_img_w is None: continue # Should not happen if dist_to_img_w is not inf, but safety check

                        if (camera, rover) in camera_calibrated:
                            # Camera is calibrated
                            dist_img_to_comm = self.find_nearest_waypoint_distance(nearest_img_w, self.comm_waypoints)
                            if dist_img_to_comm != math.inf:
                                # Cost to image + cost to communicate
                                cost_image = dist_to_img_w + 1 # move + take_image
                                cost_communicate = dist_img_to_comm + 1 # move + communicate_image_data
                                current_rover_cost = cost_image + cost_communicate
                        else:
                            # Camera needs calibration
                            cal_target = self.camera_info[camera].get('cal_target')
                            if not cal_target:
                                 continue # Camera has no calibration target

                            # Find calibration waypoints for the target
                            cal_waypoints = self.objective_visibility.get(cal_target, set())
                            if not cal_waypoints:
                                 continue # Cannot calibrate this camera

                            # Find nearest calibration waypoint
                            dist_to_cal_w = self.find_nearest_waypoint_distance(loc_r, cal_waypoints)
                            if dist_to_cal_w == math.inf:
                                 continue # Cannot reach any calibration waypoint

                            # Find the specific nearest calibration waypoint
                            nearest_cal_w = None
                            min_dist_cal = math.inf
                            for cal_w in cal_waypoints:
                                 dist = self.get_distance(loc_r, cal_w)
                                 if dist < min_dist_cal:
                                      min_dist_cal = dist
                                      nearest_cal_w = cal_w
                            if nearest_cal_w is None: continue # Safety check

                            # Find nearest imaging waypoint *from the calibration waypoint*
                            dist_cal_to_img_w = self.find_nearest_waypoint_distance(nearest_cal_w, img_waypoints)
                            if dist_cal_to_img_w == math.inf:
                                 continue # Cannot reach imaging waypoint after calibration

                            # Find the specific nearest imaging waypoint from cal_w
                            nearest_img_w_after_cal = None
                            min_dist_img_after_cal = math.inf
                            for img_w in img_waypoints:
                                 dist = self.get_distance(nearest_cal_w, img_w)
                                 if dist < min_dist_img_after_cal:
                                      min_dist_img_after_cal = dist
                                      nearest_img_w_after_cal = img_w
                            if nearest_img_w_after_cal is None: continue # Safety check


                            dist_img_to_comm = self.find_nearest_waypoint_distance(nearest_img_w_after_cal, self.comm_waypoints)
                            if dist_img_to_comm != math.inf:
                                # Cost to calibrate + cost to image + cost to communicate
                                cost_calibrate = dist_to_cal_w + 1 # move + calibrate_camera
                                cost_image = dist_cal_to_img_w + 1 # move + take_image
                                cost_communicate = dist_img_to_comm + 1 # move + communicate_image_data
                                current_rover_cost = cost_calibrate + cost_image + cost_communicate

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                total_heuristic_cost += min_goal_cost # Add infinity if unreachable

        # If any goal was unreachable, the total cost will be infinity.
        # Return a large number instead of infinity for search compatibility.
        return total_heuristic_cost if total_heuristic_cost != math.inf else 1000000
