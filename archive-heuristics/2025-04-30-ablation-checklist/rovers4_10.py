from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base.py is available in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone execution/testing
# In a real planning system, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS on a graph to find shortest distances from a start node.
    Graph is represented as a dictionary: {node: set(neighbor1, neighbor2, ...)}
    Returns a dictionary {node: distance}. Unreachable nodes have distance infinity.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         return distances # Start node not in graph, all unreachable

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key and has neighbors
        if current_node in graph:
            for neighbor in graph.get(current_node, set()):
                if distances.get(neighbor, float('inf')) == float('inf'): # Check if neighbor is in distances and unreachable
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact independently. The cost for each goal
    involves acquiring the necessary data (sampling or imaging) and
    communicating it. Navigation costs are estimated using precomputed
    shortest paths (BFS) on the traversable graph for each rover.

    This heuristic is non-admissible. It simplifies the problem by ignoring
    resource contention (e.g., multiple goals needing the same rover or store)
    and assuming optimal sequencing for each goal independently.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing navigation distances and
        extracting static information and goal requirements.
        """
        self.goals = task.goals
        self.static = task.static

        # --- Precompute Static Information ---
        # Collect all objects by type by iterating through all facts
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.landers = set()

        all_facts = set(task.initial_state) | set(self.static) | set(self.goals)
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip empty facts
            pred = parts[0]
            # Infer types based on predicate structure and common usage
            if pred in ['at', 'at_lander', 'can_traverse', 'have_rock_analysis', 'have_soil_analysis', 'at_soil_sample', 'at_rock_sample', 'visible', 'visible_from', 'communicated_soil_data', 'communicated_rock_data']:
                 for part in parts[1:]:
                     if part.startswith('waypoint'): self.waypoints.add(part)
            if pred in ['at', 'can_traverse', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'have_rock_analysis', 'have_soil_analysis', 'have_image', 'calibrated', 'store_of', 'on_board']:
                 for part in parts[1:]:
                     if part.startswith('rover'): self.rovers.add(part)
            if pred in ['empty', 'full', 'store_of']:
                 for part in parts[1:]:
                     # Store names typically start with rover name + 'store'
                     if 'store' in part: self.stores.add(part)
            if pred in ['calibrated', 'supports', 'calibration_target', 'on_board', 'have_image']:
                 for part in parts[1:]:
                     if part.startswith('camera'): self.cameras.add(part)
            if pred in ['supports', 'have_image', 'communicated_image_data']:
                 for part in parts[1:]:
                     # Assuming modes are simple names like colour, high_res, low_res
                     if not (part.startswith('rover') or part.startswith('camera') or part.startswith('objective') or part.startswith('waypoint') or part.startswith('store') or part.startswith('lander')):
                         self.modes.add(part)
            if pred in ['have_image', 'communicated_image_data', 'visible_from', 'calibration_target']:
                 for part in parts[1:]:
                     if part.startswith('objective'): self.objectives.add(part)
            if pred == 'at_lander':
                 for part in parts[1:]:
                     if part.startswith('lander'): self.landers.add(part)


        self.equipped_soil = {p[1] for p in map(get_parts, self.static) if p[0] == 'equipped_for_soil_analysis' and len(p) > 1 and p[1] in self.rovers}
        self.equipped_rock = {p[1] for p in map(get_parts, self.static) if p[0] == 'equipped_for_rock_analysis' and len(p) > 1 and p[1] in self.rovers}
        self.equipped_imaging = {p[1] for p in map(get_parts, self.static) if p[0] == 'equipped_for_imaging' and len(p) > 1 and p[1] in self.rovers}
        self.store_owner = {p[1]: p[2] for p in map(get_parts, self.static) if p[0] == 'store_of' and len(p) > 2 and p[1] in self.stores and p[2] in self.rovers}

        self.camera_modes = {} # camera -> set of modes
        for p in map(get_parts, self.static):
            if p[0] == 'supports' and len(p) > 2:
                camera, mode = p[1], p[2]
                if camera in self.cameras and mode in self.modes:
                    if camera not in self.camera_modes: self.camera_modes[camera] = set()
                    self.camera_modes[camera].add(mode)

        self.rover_cameras = {} # rover -> set of cameras
        for p in map(get_parts, self.static):
             if p[0] == 'on_board' and len(p) > 2:
                 camera, rover = p[1], p[2]
                 if camera in self.cameras and rover in self.rovers:
                     if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
                     self.rover_cameras[rover].add(camera)

        self.camera_cal_target = {p[1]: p[2] for p in map(get_parts, self.static) if p[0] == 'calibration_target' and len(p) > 2 and p[1] in self.cameras and p[2] in self.objectives}

        self.visible_map = {wp: set() for wp in self.waypoints} # Initialize with all waypoints
        for p in map(get_parts, self.static):
            if p[0] == 'visible' and len(p) > 2:
                w1, w2 = p[1], p[2]
                if w1 in self.waypoints and w2 in self.waypoints: # Only add if waypoints are known
                    self.visible_map[w1].add(w2)
                    self.visible_map[w2].add(w1) # Assuming visible is symmetric

        self.can_traverse_map = {rover: {wp: set() for wp in self.waypoints} for rover in self.rovers} # Initialize
        for p in map(get_parts, self.static):
            if p[0] == 'can_traverse' and len(p) > 3:
                rover, w1, w2 = p[1], p[2], p[3]
                if rover in self.rovers and w1 in self.waypoints and w2 in self.waypoints:
                    # Only add if visible
                    if w2 in self.visible_map.get(w1, set()):
                        self.can_traverse_map[rover][w1].add(w2)
                        self.can_traverse_map[rover][w2].add(w1) # Assuming can_traverse is symmetric if visible is symmetric


        self.lander_wp = next((p[2] for p in map(get_parts, self.static) if p[0] == 'at_lander' and len(p) > 2 and p[2] in self.waypoints), None) # Assuming only one lander
        self.comm_wps = {wp for wp in self.waypoints if self.lander_wp and wp in self.visible_map and self.lander_wp in self.visible_map[wp]}

        self.obj_wps = {obj: set() for obj in self.objectives} # objective -> set of visible wps
        # Calibration targets are objectives, so use objectives set to initialize
        self.cal_target_wps = {target: set() for target in self.objectives if target in self.camera_cal_target.values()} # calibration_target (objective) -> set of visible wps

        for p in map(get_parts, self.static):
            if p[0] == 'visible_from' and len(p) > 2:
                obj_or_target, wp = p[1], p[2]
                if wp in self.waypoints: # Only add if waypoint is known
                    if obj_or_target in self.objectives:
                        self.obj_wps[obj_or_target].add(wp)
                    # Check if it's a known calibration target objective
                    if obj_or_target in self.cal_target_wps:
                         self.cal_target_wps[obj_or_target].add(wp)


        # --- Precompute Navigation Distances (BFS) ---
        self.dist = {} # dist[rover][from_wp][to_wp]
        for rover in self.rovers:
            self.dist[rover] = {}
            rover_graph = self.can_traverse_map.get(rover, {}) # Use traversable map for the rover
            # Ensure all waypoints are in the graph for BFS, even if isolated
            full_rover_graph = {wp: rover_graph.get(wp, set()) for wp in self.waypoints}

            for start_wp in self.waypoints:
                 self.dist[rover][start_wp] = bfs(full_rover_graph, start_wp)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Extract Current State Information ---
        rover_locs = {}
        store_full = {}
        have_soil = set() # (rover, wp)
        have_rock = set() # (rover, wp)
        have_image = set() # (rover, obj, mode)
        calibrated_cams = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) > 2 and parts[1] in self.rovers:
                rover_locs[parts[1]] = parts[2]
            elif pred == 'full' and len(parts) > 1 and parts[1] in self.stores:
                store_full[parts[1]] = True
            elif pred == 'have_soil_analysis' and len(parts) > 2 and parts[1] in self.rovers and parts[2] in self.waypoints:
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) > 2 and parts[1] in self.rovers and parts[2] in self.waypoints:
                have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) > 3 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated' and len(parts) > 2 and parts[1] in self.cameras and parts[2] in self.rovers:
                 calibrated_cams.add((parts[1], parts[2]))

        # Default store state is empty if not full
        for store in self.stores:
             if store not in store_full:
                 store_full[store] = False

        # --- Calculate Heuristic for Unachieved Goals ---
        unachieved_goals = self.goals - state

        for goal in unachieved_goals:
            parts = get_parts(goal)
            if not parts: continue
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data' and len(parts) > 1:
                wp = parts[1]
                if wp not in self.waypoints: continue # Goal waypoint not in domain

                min_total_cost_for_goal = float('inf')

                # Check if sample is already held by any rover
                sample_held = any((r, wp) in have_soil for r in self.rovers)

                if sample_held:
                    # If sample is held, just need to communicate
                    for rover in self.rovers:
                         if (rover, wp) in have_soil:
                             wp_r = rover_locs.get(rover)
                             if wp_r is None or wp_r not in self.waypoints: continue

                             min_comm_nav = float('inf')
                             for wp_comm in self.comm_wps:
                                 if wp_r in self.dist[rover] and wp_comm in self.dist[rover][wp_r]:
                                     min_comm_nav = min(min_comm_nav, self.dist[rover][wp_r][wp_comm])

                             if min_comm_nav != float('inf'):
                                 cost = min_comm_nav + 1 # Nav + Communicate
                                 min_total_cost_for_goal = min(min_total_cost_for_goal, cost)
                else:
                    # Sample is not held, need to sample AND communicate
                    # Find best equipped rover to do the whole chain: Sample -> Communicate
                    for rover in self.equipped_soil:
                        wp_r = rover_locs.get(rover)
                        if wp_r is None or wp_r not in self.waypoints: continue

                        # Cost to sample
                        cost_to_sample_nav = self.dist[rover].get(wp_r, {}).get(wp, float('inf'))
                        if cost_to_sample_nav == float('inf'): continue # Cannot reach sample location

                        cost_sample = cost_to_sample_nav + 1 # Nav + Sample
                        store = next((s for s, r in self.store_owner.items() if r == rover), None)
                        if store and store_full.get(store, False):
                            cost_sample += 1 # Drop needed before sampling

                        # Cost to communicate from sample location (which is 'wp')
                        min_comm_nav_from_sample = float('inf')
                        for wp_comm in self.comm_wps:
                             if wp in self.dist[rover] and wp_comm in self.dist[rover][wp]:
                                 min_comm_nav_from_sample = min(min_comm_nav_from_sample, self.dist[rover][wp][wp_comm])

                        if min_comm_nav_from_sample != float('inf'):
                            cost_comm = min_comm_nav_from_sample + 1 # Nav + Communicate
                            total_cost = cost_sample + cost_comm
                            min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost)

                h += min_total_cost_for_goal if min_total_cost_for_goal != float('inf') else 1000


            elif goal_type == 'communicated_rock_data' and len(parts) > 1:
                wp = parts[1]
                if wp not in self.waypoints: continue # Goal waypoint not in domain

                min_total_cost_for_goal = float('inf')

                # Check if sample is already held by any rover
                sample_held = any((r, wp) in have_rock for r in self.rovers)

                if sample_held:
                    # If sample is held, just need to communicate
                    for rover in self.rovers:
                         if (rover, wp) in have_rock:
                             wp_r = rover_locs.get(rover)
                             if wp_r is None or wp_r not in self.waypoints: continue

                             min_comm_nav = float('inf')
                             for wp_comm in self.comm_wps:
                                 if wp_r in self.dist[rover] and wp_comm in self.dist[rover][wp_r]:
                                     min_comm_nav = min(min_comm_nav, self.dist[rover][wp_r][wp_comm])

                             if min_comm_nav != float('inf'):
                                 cost = min_comm_nav + 1 # Nav + Communicate
                                 min_total_cost_for_goal = min(min_total_cost_for_goal, cost)
                else:
                    # Sample is not held, need to sample AND communicate
                    # Find best equipped rover to do the whole chain: Sample -> Communicate
                    for rover in self.equipped_rock:
                        wp_r = rover_locs.get(rover)
                        if wp_r is None or wp_r not in self.waypoints: continue

                        # Cost to sample
                        cost_to_sample_nav = self.dist[rover].get(wp_r, {}).get(wp, float('inf'))
                        if cost_to_sample_nav == float('inf'): continue # Cannot reach sample location

                        cost_sample = cost_to_sample_nav + 1 # Nav + Sample
                        store = next((s for s, r in self.store_owner.items() if r == rover), None)
                        if store and store_full.get(store, False):
                            cost_sample += 1 # Drop needed before sampling

                        # Cost to communicate from sample location (which is 'wp')
                        min_comm_nav_from_sample = float('inf')
                        for wp_comm in self.comm_wps:
                             if wp in self.dist[rover] and wp_comm in self.dist[rover][wp]:
                                 min_comm_nav_from_sample = min(min_comm_nav_from_sample, self.dist[rover][wp][wp_comm])

                        if min_comm_nav_from_sample != float('inf'):
                            cost_comm = min_comm_nav_from_sample + 1 # Nav + Communicate
                            total_cost = cost_sample + cost_comm
                            min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost)

                h += min_total_cost_for_goal if min_total_cost_for_goal != float('inf') else 1000


            elif goal_type == 'communicated_image_data' and len(parts) > 2:
                obj, mode = parts[1], parts[2]
                if obj not in self.objectives or mode not in self.modes: continue # Goal object/mode not in domain

                min_total_cost_for_goal = float('inf')

                # Check if image is already held by any rover
                image_held = any((r, obj, mode) in have_image for r in self.rovers)

                if image_held:
                    # If image is held, just need to communicate
                    for rover in self.rovers:
                         if (rover, obj, mode) in have_image:
                             wp_r = rover_locs.get(rover)
                             if wp_r is None or wp_r not in self.waypoints: continue

                             min_comm_nav = float('inf')
                             for wp_comm in self.comm_wps:
                                 if wp_r in self.dist[rover] and wp_comm in self.dist[rover][wp_r]:
                                     min_comm_nav = min(min_comm_nav, self.dist[rover][wp_r][wp_comm])

                             if min_comm_nav != float('inf'):
                                 cost = min_comm_nav + 1 # Nav + Communicate
                                 min_total_cost_for_goal = min(min_total_cost_for_goal, cost)
                else:
                    # Image is not held, need to acquire AND communicate
                    # Find best equipped rover/camera to do the whole chain: Calibrate -> Image -> Communicate
                    for rover in self.equipped_imaging:
                        wp_r = rover_locs.get(rover)
                        if wp_r is None or wp_r not in self.waypoints: continue

                        if rover not in self.rover_cameras: continue # Rover has no camera

                        for camera in self.rover_cameras[rover]:
                            if mode not in self.camera_modes.get(camera, set()): continue # Camera doesn't support mode

                            cal_target = self.camera_cal_target.get(camera)
                            if not cal_target or cal_target not in self.objectives: continue # Camera has no calibration target or target not in objectives

                            cal_wps = self.cal_target_wps.get(cal_target, set())
                            if not cal_wps: continue # No waypoints to calibrate from

                            img_wps = self.obj_wps.get(obj, set())
                            if not img_wps: continue # No waypoints to image from

                            # Find best path: wp_r -> wp_cal -> wp_img -> wp_comm
                            # Minimize (dist(wp_r, w_cal) + 1 + dist(w_cal, w_img) + 1 + dist(w_img, w_comm) + 1)
                            # over w_cal in cal_wps, w_img in img_wps, w_comm in comm_wps

                            min_acquire_comm_path_cost = float('inf')

                            for wp_cal in cal_wps:
                                nav_to_cal = self.dist[rover].get(wp_r, {}).get(wp_cal, float('inf'))
                                if nav_to_cal == float('inf'): continue

                                for wp_img in img_wps:
                                    nav_cal_to_img = self.dist[rover].get(wp_cal, {}).get(wp_img, float('inf'))
                                    if nav_cal_to_img == float('inf'): continue

                                    min_comm_nav_from_img = float('inf')
                                    for wp_comm in self.comm_wps:
                                         if wp_img in self.dist[rover] and wp_comm in self.dist[rover][wp_img]:
                                             min_comm_nav_from_img = min(min_comm_nav_from_img, self.dist[rover][wp_img][wp_comm])

                                    if min_comm_nav_from_img != float('inf'):
                                        # Total cost for this specific path (wp_r -> wp_cal -> wp_img -> wp_comm)
                                        path_cost = nav_to_cal + 1 + nav_cal_to_img + 1 + min_comm_nav_from_img + 1
                                        min_acquire_comm_path_cost = min(min_acquire_comm_path_cost, path_cost)

                            if min_acquire_comm_path_cost != float('inf'):
                                min_total_cost_for_goal = min(min_total_cost_for_goal, min_acquire_comm_path_cost)


                h += min_total_cost_for_goal if min_total_cost_for_goal != float('inf') else 1000


        return h
