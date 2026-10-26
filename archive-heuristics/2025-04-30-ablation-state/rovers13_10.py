from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list for malformed facts
         return []
    return fact[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node):
    """
    Performs BFS from a start node on a graph.
    Returns a dictionary mapping each reachable node to its shortest distance from the start node.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        return distances # Start node not in graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Ensure node exists in graph keys
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing up the estimated costs for each unachieved goal fact independently.
    The cost for each goal is estimated based on the current state (whether data/image
    is collected/taken) and includes action costs (sample, take_image, calibrate, drop,
    communicate) and navigation costs (shortest path distances between required waypoints).

    # Assumptions
    - The waypoint graph formed by 'visible' and 'can_traverse' predicates (for any rover)
      is connected for all relevant waypoints (sample locations, image locations,
      calibration locations, lander-visible locations) for solvable problems.
    - Every required sample location exists as an 'at_soil_sample' or 'at_rock_sample'
      fact in the initial state.
    - Suitable rovers (equipped for soil, rock, or imaging), cameras (on board, supporting
      mode, with calibration target), stores, and lander-visible waypoints exist to
      achieve the goals in solvable problems.
    - Action costs are uniform (cost 1). Navigation cost is the shortest path distance.
    - When multiple rovers/cameras/waypoints can fulfill a requirement, the heuristic
      considers the minimum cost among them for that specific goal.
    - The heuristic does not account for resource contention (e.g., multiple goals
      requiring the same store simultaneously) or complex multi-goal path planning
      for a single rover.

    # Heuristic Initialization
    - Parses goal facts to identify required data/images.
    - Parses static facts to build data structures:
        - Lander location.
        - Locations of soil and rock samples.
        - Mappings from objectives/calibration targets to visible waypoints.
        - Mappings related to rovers, cameras, and stores (equipment, onboard cameras,
          camera support, calibration targets, store ownership).
        - Builds a waypoint graph based on 'visible' and 'can_traverse' (for any rover).
        - Computes all-pairs shortest paths between waypoints using BFS.
        - Identifies waypoints visible from the lander.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for all unachieved goal facts.

    For each goal `(communicated_soil_data ?p)`:
    1. If already communicated, cost is 0.
    2. If not communicated:
       - Check if any rover currently has `(have_soil_analysis ?r ?p)`.
       - If yes: Cost = 1 (communicate) + (shortest path from rover's location to closest lander-visible waypoint, if not already visible).
       - If no: Cost = 1 (sample) + 1 (communicate) + (1 for drop if store is full) + (shortest path from best soil-equipped rover's location to `?p`) + (shortest path from `?p` to closest lander-visible waypoint). The "best" rover is the one minimizing this total cost.

    For each goal `(communicated_rock_data ?p)`:
    - Similar logic as soil data, using rock predicates and rock-equipped rovers.

    For each goal `(communicated_image_data ?o ?m)`:
    1. If already communicated, cost is 0.
    2. If not communicated:
       - Check if any rover currently has `(have_image ?r ?o ?m)`.
       - If yes: Cost = 1 (communicate) + (shortest path from rover's location to closest lander-visible waypoint, if not already visible).
       - If no: Cost = 1 (take_image) + 1 (communicate) + (1 for calibrate if needed) + navigation cost. The navigation cost depends on whether calibration is needed:
         - If calibrated: Shortest path from best imaging-equipped rover's location to closest image waypoint for `?o`, plus shortest path from closest image waypoint to closest lander-visible waypoint.
         - If not calibrated: Shortest path from best imaging-equipped rover's location to closest calibration waypoint for the camera, plus shortest path from closest calibration waypoint to closest image waypoint for `?o`, plus shortest path from closest image waypoint to closest lander-visible waypoint. The "best" rover/camera combination is the one minimizing this total cost.

    The total heuristic value is the sum of these estimated costs for all unachieved goals.
    Returns float('inf') if any required waypoint is unreachable.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self._lander_loc = None
        self._soil_wps = set()
        self._rock_wps = set()
        self._obj_to_wps = {} # objective -> set of visible waypoints
        self._caltarget_to_wps = {} # caltarget -> set of visible waypoints
        self._camera_to_caltarget = {} # camera -> caltarget
        self._rover_to_cameras = {} # rover -> set of cameras on board
        self._camera_to_modes = {} # camera -> set of supported modes
        self._rover_to_store = {} # rover -> store
        self._equipped_for_soil = set()
        self._equipped_for_rock = set()
        self._equipped_for_imaging = set()
        self._waypoints = set()
        self._visible_graph = {} # waypoint -> set of visible waypoints
        self._can_traverse_graph = {} # waypoint -> set of reachable waypoints (for any rover)

        # Collect all objects of relevant types to ensure all waypoints are known
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            if pred == 'at_lander':
                self._lander_loc = parts[2]
                all_landers.add(parts[1])
                self._waypoints.add(parts[2])
            elif pred == 'at_soil_sample':
                self._soil_wps.add(parts[1])
                self._waypoints.add(parts[1])
            elif pred == 'at_rock_sample':
                self._rock_wps.add(parts[1])
                self._waypoints.add(parts[1])
            elif pred == 'equipped_for_soil_analysis':
                self._equipped_for_soil.add(parts[1])
                all_rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                self._equipped_for_rock.add(parts[1])
                all_rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                self._equipped_for_imaging.add(parts[1])
                all_rovers.add(parts[1])
            elif pred == 'store_of':
                self._rover_to_store[parts[2]] = parts[1]
                all_stores.add(parts[1])
                all_rovers.add(parts[2])
            elif pred == 'calibration_target':
                self._camera_to_caltarget[parts[1]] = parts[2]
                all_cameras.add(parts[1])
                all_objectives.add(parts[2]) # Calibration targets are objectives
            elif pred == 'on_board':
                rover, camera = parts[2], parts[1]
                self._rover_to_cameras.setdefault(rover, set()).add(camera)
                all_rovers.add(rover)
                all_cameras.add(camera)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self._camera_to_modes.setdefault(camera, set()).add(mode)
                all_cameras.add(camera)
                all_modes.add(mode)
            elif pred == 'visible':
                w1, w2 = parts[1], parts[2]
                self._waypoints.add(w1)
                self._waypoints.add(w2)
                self._visible_graph.setdefault(w1, set()).add(w2)
                self._visible_graph.setdefault(w2, set()).add(w1) # Visibility is symmetric
            elif pred == 'can_traverse':
                 # Build a combined graph for reachability for any rover
                r, w1, w2 = parts[1], parts[2], parts[3]
                self._waypoints.add(w1)
                self._waypoints.add(w2)
                self._can_traverse_graph.setdefault(w1, set()).add(w2)
                self._can_traverse_graph.setdefault(w2, set()).add(w1) # Assume traverse is symmetric
            elif pred == 'visible_from':
                obj, wp = parts[1], parts[2]
                self._waypoints.add(wp)
                all_objectives.add(obj)
                # Check if this objective is a calibration target
                is_cal_target = False
                for cam, target in self._camera_to_caltarget.items():
                    if target == obj:
                        self._caltarget_to_wps.setdefault(obj, set()).add(wp)
                        is_cal_target = True
                        break
                if not is_cal_target:
                     self._obj_to_wps.setdefault(obj, set()).add(wp)
            # Ensure all declared objects of relevant types are considered
            elif pred == 'rover': all_rovers.add(parts[1])
            elif pred == 'waypoint': self._waypoints.add(parts[1])
            elif pred == 'store': all_stores.add(parts[1])
            elif pred == 'camera': all_cameras.add(parts[1])
            elif pred == 'mode': all_modes.add(parts[1])
            elif pred == 'lander': all_landers.add(parts[1])
            elif pred == 'objective': all_objectives.add(parts[1])


        # Ensure all waypoints mentioned in visible/can_traverse are in the graph keys
        for w in list(self._waypoints): # Iterate over a copy in case waypoints are added
             self._can_traverse_graph.setdefault(w, set())
             self._visible_graph.setdefault(w, set())

        # Combine visible and can_traverse for the full navigation graph
        # An edge w1 -> w2 exists if (visible w1 w2) AND (can_traverse ?r w1 w2) for *any* rover ?r
        self._nav_graph = {}
        for w1 in self._waypoints:
            self._nav_graph[w1] = set()
            if w1 in self._visible_graph:
                for w2 in self._visible_graph[w1]:
                    # Check if w1->w2 traverse is possible for *any* rover
                    if w2 in self._can_traverse_graph.get(w1, set()):
                         self._nav_graph[w1].add(w2)

        # Compute all-pairs shortest paths
        self._dist = {}
        for start_wp in self._waypoints:
            self._dist[start_wp] = bfs(self._nav_graph, start_wp)

        # Identify waypoints visible from the lander
        self._lander_visible_wps = set()
        if self._lander_loc and self._lander_loc in self._visible_graph:
             self._lander_visible_wps = self._visible_graph[self._lander_loc]

        # --- Parse Goals ---
        self._required_soil_wps = set()
        self._required_rock_wps = set()
        self._required_images = set() # set of (objective, mode) tuples

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]
            if pred == 'communicated_soil_data':
                self._required_soil_wps.add(parts[1])
            elif pred == 'communicated_rock_data':
                self._required_rock_wps.add(parts[1])
            elif pred == 'communicated_image_data':
                self._required_images.add((parts[1], parts[2]))

    def _get_min_dist_to_set(self, start_wp, target_wps_set, start_wps_set=None):
        """
        Helper to find the minimum shortest distance from start_wp (or any in start_wps_set)
        to any waypoint in target_wps_set.
        Returns float('inf') if no path exists between any start and any target.
        """
        if not target_wps_set:
             return float('inf') # No targets

        min_d = float('inf')

        if start_wps_set is not None: # Min dist between two sets
             if not start_wps_set: return float('inf') # No starts
             for s_wp in start_wps_set:
                  if s_wp in self._dist:
                       for t_wp in target_wps_set:
                            if t_wp in self._dist[s_wp]:
                                 min_d = min(min_d, self._dist[s_wp][t_wp])
        elif start_wp is not None: # Min dist from single start to set
            if start_wp not in self._dist:
                 return float('inf') # Start waypoint not in graph
            for target_wp in target_wps_set:
                if target_wp in self._dist[start_wp]:
                    min_d = min(min_d, self._dist[start_wp][target_wp])
        else: # Invalid call
             return float('inf')

        return min_d

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State Facts ---
        rover_locations = {}
        have_soil = set() # set of (rover, waypoint)
        have_rock = set() # set of (rover, waypoint)
        have_image = set() # set of (rover, objective, mode)
        calibrated_cameras = set() # set of (camera, rover)
        full_stores = set() # set of (store,)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and parts[1].startswith('rover'): # Assuming objects starting with 'rover' are rovers
                rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'full':
                full_stores.add((parts[1],))

        total_cost = 0

        # --- Estimate Cost for Soil Goals ---
        for soil_wp in self._required_soil_wps:
            goal_fact = f'(communicated_soil_data {soil_wp})'
            if goal_fact in state:
                continue # Goal already achieved

            # Check if sample is already collected by any rover
            sample_held = False
            rover_with_sample_loc = None
            for r, wp in have_soil:
                if wp == soil_wp:
                    sample_held = True
                    if r in rover_locations:
                        rover_with_sample_loc = rover_locations[r]
                    break # Found a rover with the sample

            if sample_held:
                # Cost for communication action
                current_goal_cost = 1
                # Need to navigate to a communication point if not already visible
                if rover_with_sample_loc and rover_with_sample_loc not in self._lander_visible_wps:
                     nav_cost = self._get_min_dist_to_set(rover_with_sample_loc, self._lander_visible_wps)
                     if nav_cost == float('inf'): return float('inf') # Cannot reach lander visible waypoint
                     current_goal_cost += nav_cost
                total_cost += current_goal_cost

            else: # Need to sample and then communicate
                # Find the best soil-equipped rover (one that minimizes total cost for this goal)
                min_total_goal_cost = float('inf')

                for r in self._equipped_for_soil:
                    if r not in rover_locations: continue # Rover location unknown

                    r_loc = rover_locations[r]
                    sample_wp = soil_wp
                    comm_wps = self._lander_visible_wps

                    if sample_wp not in self._waypoints or not comm_wps:
                         continue # Cannot achieve this goal with this rover/sample/lander setup

                    # Base action costs
                    action_cost = 1 # communicate
                    action_cost += 1 # sample_soil

                    # Nav: current -> sample_wp -> comm_wp
                    nav_to_sample = self._get_min_dist_to_set(r_loc, {sample_wp})
                    if nav_to_sample == float('inf'): continue # Cannot reach sample waypoint

                    nav_sample_to_comm = self._get_min_dist_to_set(None, comm_wps, start_wps_set={sample_wp})
                    if nav_sample_to_comm == float('inf'): continue # Cannot reach comm waypoint from sample waypoint

                    nav_cost = nav_to_sample + nav_sample_to_comm

                    # Check if this rover's store is full
                    drop_cost = 0
                    store = self._rover_to_store.get(r)
                    if store and (store,) in full_stores:
                        drop_cost = 1 # Cost for drop action

                    current_total_cost = action_cost + drop_cost + nav_cost

                    min_total_goal_cost = min(min_total_goal_cost, current_total_cost)

                if min_total_goal_cost == float('inf'):
                    # No suitable rover or required waypoints unreachable for this goal
                    return float('inf') # Problem likely unsolvable from here
                else:
                    total_cost += min_total_goal_cost


        # --- Estimate Cost for Rock Goals ---
        for rock_wp in self._required_rock_wps:
            goal_fact = f'(communicated_rock_data {rock_wp})'
            if goal_fact in state:
                continue # Goal already achieved

            # Check if sample is already collected by any rover
            sample_held = False
            rover_with_sample_loc = None
            for r, wp in have_rock:
                if wp == rock_wp:
                    sample_held = True
                    if r in rover_locations:
                        rover_with_sample_loc = rover_locations[r]
                    break # Found a rover with the sample

            if sample_held:
                # Cost for communication action
                current_goal_cost = 1
                # Need to navigate to a communication point if not already visible
                if rover_with_sample_loc and rover_with_sample_loc not in self._lander_visible_wps:
                     nav_cost = self._get_min_dist_to_set(rover_with_sample_loc, self._lander_visible_wps)
                     if nav_cost == float('inf'): return float('inf') # Cannot reach lander visible waypoint
                     current_goal_cost += nav_cost
                total_cost += current_goal_cost

            else: # Need to sample and then communicate
                # Find the best rock-equipped rover (one that minimizes total cost for this goal)
                min_total_goal_cost = float('inf')

                for r in self._equipped_for_rock:
                    if r not in rover_locations: continue # Rover location unknown

                    r_loc = rover_locations[r]
                    sample_wp = rock_wp
                    comm_wps = self._lander_visible_wps

                    if sample_wp not in self._waypoints or not comm_wps:
                         continue # Cannot achieve this goal with this rover/sample/lander setup

                    # Base action costs
                    action_cost = 1 # communicate
                    action_cost += 1 # sample_rock

                    # Nav: current -> sample_wp -> comm_wp
                    nav_to_sample = self._get_min_dist_to_set(r_loc, {sample_wp})
                    if nav_to_sample == float('inf'): continue # Cannot reach sample waypoint

                    nav_sample_to_comm = self._get_min_dist_to_set(None, comm_wps, start_wps_set={sample_wp})
                    if nav_sample_to_comm == float('inf'): continue # Cannot reach comm waypoint from sample waypoint

                    nav_cost = nav_to_sample + nav_sample_to_comm

                    # Check if this rover's store is full
                    drop_cost = 0
                    store = self._rover_to_store.get(r)
                    if store and (store,) in full_stores:
                        drop_cost = 1 # Cost for drop action

                    current_total_cost = action_cost + drop_cost + nav_cost

                    min_total_goal_cost = min(min_total_goal_cost, current_total_cost)

                if min_total_goal_cost == float('inf'):
                    # No suitable rover or required waypoints unreachable for this goal
                    return float('inf') # Problem likely unsolvable from here
                else:
                    total_cost += min_total_goal_cost


        # --- Estimate Cost for Image Goals ---
        for obj, mode in self._required_images:
            goal_fact = f'(communicated_image_data {obj} {mode})'
            if goal_fact in state:
                continue # Goal already achieved

            # Check if image is already taken by any rover
            image_held = False
            rover_with_image_loc = None
            for r, o, m in have_image:
                if o == obj and m == mode:
                    image_held = True
                    if r in rover_locations:
                        rover_with_image_loc = rover_locations[r]
                    break # Found a rover with the image

            if image_held:
                 # Cost for communication action
                 current_goal_cost = 1
                 # Need to navigate to a communication point if not already visible
                 if rover_with_image_loc and rover_with_image_loc not in self._lander_visible_wps:
                      nav_cost = self._get_min_dist_to_set(rover_with_image_loc, self._lander_visible_wps)
                      if nav_cost == float('inf'): return float('inf') # Cannot reach lander visible waypoint
                      current_goal_cost += nav_cost
                 total_cost += current_goal_cost

            else: # Need to take image and then communicate
                # Find the best imaging-equipped rover with a suitable camera
                min_total_goal_cost = float('inf')

                for r in self._equipped_for_imaging:
                    if r not in rover_locations: continue # Rover location unknown

                    r_loc = rover_locations[r]
                    cameras_on_board = self._rover_to_cameras.get(r, set())

                    for cam in cameras_on_board:
                        supported_modes = self._camera_to_modes.get(cam, set())
                        if mode in supported_modes:
                            # Found a suitable rover and camera
                            is_calibrated = (cam, r) in calibrated_cameras
                            cal_target = self._camera_to_caltarget.get(cam)
                            cal_wps = self._caltarget_to_wps.get(cal_target, set()) if cal_target else set()
                            image_wps = self._obj_to_wps.get(obj, set())
                            comm_wps = self._lander_visible_wps

                            if not image_wps or not comm_wps or (not is_calibrated and not cal_wps):
                                # Cannot achieve this image goal with this camera/rover combination
                                continue # Try next camera/rover

                            # Base action costs
                            action_cost = 1 # communicate
                            action_cost += 1 # take_image
                            if not is_calibrated:
                                action_cost += 1 # calibrate

                            nav_cost = float('inf')

                            if is_calibrated:
                                # Nav: current -> image_wp -> comm_wp
                                nav_to_img = self._get_min_dist_to_set(r_loc, image_wps)
                                if nav_to_img == float('inf'): continue # Cannot reach image waypoint

                                nav_img_to_comm = self._get_min_dist_to_set(None, comm_wps, start_wps_set=image_wps)
                                if nav_img_to_comm == float('inf'): continue # Cannot reach comm waypoint from image waypoints

                                nav_cost = nav_to_img + nav_img_to_comm

                            else: # Not calibrated
                                # Nav: current -> cal_wp -> image_wp -> comm_wp
                                nav_to_cal = self._get_min_dist_to_set(r_loc, cal_wps)
                                if nav_to_cal == float('inf'): continue # Cannot reach cal waypoint

                                nav_cal_to_img = self._get_min_dist_to_set(None, image_wps, start_wps_set=cal_wps)
                                if nav_cal_to_img == float('inf'): continue # Cannot reach image waypoint from cal waypoints

                                nav_img_to_comm = self._get_min_dist_to_set(None, comm_wps, start_wps_set=image_wps)
                                if nav_img_to_comm == float('inf'): continue # Cannot reach comm waypoint from image waypoints

                                nav_cost = nav_to_cal + nav_cal_to_img + nav_img_to_comm

                            if nav_cost != float('inf'):
                                current_total_cost = action_cost + nav_cost
                                min_total_goal_cost = min(min_total_goal_cost, current_total_cost)

                if min_total_goal_cost == float('inf'):
                    # No suitable rover/camera or required waypoints unreachable for this goal
                    return float('inf') # Problem likely unsolvable from here
                else:
                    total_cost += min_total_goal_cost


        return total_cost
