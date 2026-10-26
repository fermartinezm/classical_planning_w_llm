import heapq
import logging
from collections import deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


def parse_fact(fact_str):
    """Parses a PDDL fact string into (predicate, [args])."""
    # Example: '(at rover1 waypoint1)' -> ('at', ['rover1', 'waypoint1'])
    parts = fact_str[1:-1].split()
    if not parts: # Handle empty string case, though unlikely for facts
        return None, []
    return parts[0], parts[1:]

def get_arg(fact_str, index):
    """Gets the argument at the specified index (0-based after predicate)."""
    # Example: get_arg('(at rover1 waypoint1)', 0) -> 'rover1'
    # Example: get_arg('(at rover1 waypoint1)', 1) -> 'waypoint1'
    parts = fact_str[1:-1].split()
    if index + 1 < len(parts):
        return parts[index + 1]
    return None # Or raise an error, depending on desired behavior

def get_shortest_path_cost_to_set(rover, start_wp, target_wp_set, nav_graph):
    """
    Finds the shortest path cost from a start waypoint to any waypoint in a target set
    for a specific rover using BFS on the navigation graph.
    """
    if start_wp in target_wp_set:
        return 0
    if not target_wp_set: # Cannot reach an empty set
        return float('inf')
    if rover not in nav_graph or start_wp not in nav_graph[rover]:
         return float('inf') # Rover cannot navigate from start_wp (or start_wp not in graph)

    queue = deque([(start_wp, 0)])
    visited = {start_wp}

    while queue:
        current_wp, cost = queue.popleft()

        if current_wp in target_wp_set:
            return cost

        # Ensure current_wp is a valid key in the graph for the rover
        if current_wp in nav_graph[rover]:
            for neighbor_wp in nav_graph[rover][current_wp]:
                if neighbor_wp not in visited:
                    visited.add(neighbor_wp)
                    queue.append((neighbor_wp, cost + 1))

    return float('inf') # No path found


def get_shortest_path_cost(rover, start_wp, end_wp, nav_graph):
    """
    Finds the shortest path cost between two specific waypoints for a rover using BFS.
    Wrapper around get_shortest_path_cost_to_set.
    """
    return get_shortest_path_cost_to_set(rover, start_wp, {end_wp}, nav_graph)


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing up
    the estimated costs for each individual unachieved goal fact. For each
    unachieved goal, it calculates the minimum estimated number of actions
    required to satisfy that specific goal, considering rover capabilities,
    locations, required resources (samples, images), and navigation costs.
    Navigation costs are estimated using Breadth-First Search (BFS) on the
    rover's traversability graph, assuming a cost of 1 per navigation step.
    The heuristic is not admissible but aims to guide a greedy best-first
    search efficiently by prioritizing states closer to satisfying multiple
    or difficult goals.

    Assumptions:
    - The input state and task objects conform to the structure provided
      (frozenset of fact strings, Task object with initial_state, goals,
      operators, and static facts).
    - PDDL facts are represented as strings like '(predicate arg1 arg2)'.
    - The navigation graph defined by `can_traverse` facts is undirected
      if `(can_traverse r x y)` implies `(can_traverse r y x)` for all r, x, y,
      otherwise it's directed. The current implementation assumes directed
      edges as specified by `can_traverse` facts.
    - The lander location is static.
    - `at_soil_sample` and `at_rock_sample` are not static and are consumed
      by sampling actions. Their presence must be checked in the current state.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic pre-processes the static
    information from `task.static` to build efficient data structures:
    - `lander_waypoint`: The location of the lander.
    - `lander_visible_waypoints`: A set of waypoints visible from the lander's location.
    - `rover_capabilities`: A dictionary mapping each rover to its equipment
      capabilities ('soil', 'rock', 'imaging').
    - `store_owners`: A dictionary mapping each store to its owning rover.
    - `objective_visible_from`: A dictionary mapping each objective to a set
      of waypoints from which it is visible.
    - `camera_info`: A dictionary mapping each camera to its properties:
      the rover it's on, the modes it supports, and its calibration target.
    - `rover_cameras`: A dictionary mapping each rover to the set of cameras
      it has on board.
    - `rover_nav_graph`: A dictionary representing the navigation graph for
      each rover. Keys are rovers, values are dictionaries mapping waypoints
      to lists of directly reachable neighbor waypoints based on `can_traverse`.
      This graph is used for efficient shortest path calculations (BFS).
    It also stores the set of goal facts (`self.goals`).

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1.  Check if the current state is a goal state (`self.goals <= state`). If yes, the heuristic is 0.
    2.  Initialize the total heuristic value `total_h` to 0.
    3.  Extract dynamic information from the current `state`:
        - Current location of each rover.
        - Which rovers have which soil/rock samples.
        - Which rovers have which images.
        - The status (empty/full) of each store.
        - The calibration status of each camera on each rover.
        - The presence of soil/rock samples at waypoints.
    4.  Iterate through each goal fact in `self.goals`.
    5.  If a goal fact is already present in the current `state`, its contribution to the heuristic is 0. Continue to the next goal.
    6.  If a goal fact is *not* in the current `state`, estimate the minimum cost to achieve it:
        -   **For `(communicated_soil_data ?w_goal)`:**
            -   Minimum cost is initialized to infinity.
            -   Check if any rover `r` currently has `(have_soil_analysis ?r ?w_goal)`. If yes, the cost is the navigation cost for `r` from its current location to *any* lander-visible waypoint plus 1 (for the communicate action). Minimize this over all such rovers `r`. Update minimum cost.
            -   If no rover has the sample *and* `(at_soil_sample ?w_goal)` is in the current state: Find rovers `r` equipped for soil analysis. For each such rover, calculate the cost: navigation from current location to `w_goal`, plus 1 (drop) if the store is full, plus 1 (sample), plus navigation from `w_goal` to *any* lander-visible waypoint, plus 1 (communicate). Minimize this over all suitable rovers `r` and lander-visible waypoints. Update minimum cost.
            -   If the sample is not at `w_goal` and no rover has it, the goal is likely unreachable by sampling from this state; the cost remains infinity.
        -   **For `(communicated_rock_data ?w_goal)`:** Similar logic to soil data, using rock analysis capabilities and `at_rock_sample`.
        -   **For `(communicated_image_data ?o_goal ?m_goal)`:**
            -   Minimum cost is initialized to infinity.
            -   Check if any rover `r` currently has `(have_image ?r ?o_goal ?m_goal)`. If yes, the cost is the navigation cost for `r` from its current location to *any* lander-visible waypoint plus 1 (communicate). Minimize this over all such rovers `r`. Update minimum cost.
            -   If no rover has the image: Find rovers `r` equipped for imaging with a camera `i` on board that supports mode `m_goal`. For each such `(r, i)` pair:
                -   Find the calibration target `t` for camera `i`.
                -   Find the set of waypoints `w_cal_set` where `t` is visible.
                -   Find the set of waypoints `w_img_set` where `o_goal` is visible.
                -   If both sets are non-empty: Calculate the cost as navigation from `r`'s current location to *any* waypoint in `w_cal_set`, plus 1 (calibrate), plus navigation from *any* waypoint in `w_cal_set` to *any* waypoint in `w_img_set`, plus 1 (take image), plus navigation from *any* waypoint in `w_img_set` to *any* lander-visible waypoint, plus 1 (communicate). Minimize this total cost over all suitable `(r, i)` pairs and waypoint combinations. Update minimum cost.
                -   Navigation costs between sets of waypoints (e.g., from `w_cal_set` to `w_img_set`) are estimated by finding the minimum shortest path cost between any waypoint in the source set and any waypoint in the target set for that rover.
            -   If no suitable rover/camera exists or required visible waypoints don't exist, the goal is likely unreachable; the cost remains infinity.
    7.  Add the calculated minimum cost for the unachieved goal to `total_h`.
    8.  After iterating through all goals, return `total_h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.task = task # Keep task reference for static info

        # --- Process Static Information ---
        self.lander_waypoint = None
        self.lander_visible_waypoints = set()
        self.rover_capabilities = {} # {rover: {'soil': bool, 'rock': bool, 'imaging': bool}}
        self.store_owners = {} # {store: rover}
        self.objective_visible_from = {} # {objective: {waypoint, ...}}
        self.camera_info = {} # {camera: {'rover': rover, 'modes': {mode, ...}, 'calibration_target': objective}}
        self.rover_cameras = {} # {rover: {camera, ...}}
        self.rover_nav_graph = {} # {rover: {wp: [neighbor_wp, ...], ...}}

        all_rovers = set()
        all_waypoints = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        # First pass to identify all objects by type
        # This is a simplification; a proper parser would get this from the problem definition
        # We infer types from predicates in static facts and initial state
        potential_objects = set()
        for fact_str in task.static | task.initial_state:
             _, args = parse_fact(fact_str)
             potential_objects.update(args)

        # Try to infer types from common predicates
        for obj in potential_objects:
            # Heuristic guess based on common usage in rovers domain
            if 'rover' in obj: all_rovers.add(obj)
            elif 'waypoint' in obj: all_waypoints.add(obj)
            elif 'camera' in obj: all_cameras.add(obj)
            elif 'objective' in obj: all_objectives.add(obj)
            elif 'mode' in obj: all_modes.add(obj)
            elif 'store' in obj: all_stores.add(obj)
            elif 'lander' in obj: all_landers.add(obj)
            # Add other potential objects not covered by simple name check
            # This is fragile, a real parser is needed for robustness

        # Initialize structures for all known objects
        for r in all_rovers:
            self.rover_capabilities[r] = {'soil': False, 'rock': False, 'imaging': False}
            self.rover_cameras[r] = set()
            self.rover_nav_graph[r] = {wp: [] for wp in all_waypoints}
        for c in all_cameras:
             self.camera_info[c] = {'rover': None, 'modes': set(), 'calibration_target': None}
        for o in all_objectives:
             self.objective_visible_from[o] = set()


        # Process static facts
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)

            if pred == 'at_lander':
                # (at_lander ?x - lander ?y - waypoint)
                lander, waypoint = args
                self.lander_waypoint = waypoint # Assuming only one lander or we care about the first one found
            elif pred == 'can_traverse':
                # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                rover, wp_from, wp_to = args
                if rover in self.rover_nav_graph and wp_from in self.rover_nav_graph[rover]:
                     self.rover_nav_graph[rover][wp_from].append(wp_to)
            elif pred == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis ?r - rover)
                rover = args[0]
                if rover in self.rover_capabilities:
                    self.rover_capabilities[rover]['soil'] = True
            elif pred == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis ?r - rover)
                rover = args[0]
                if rover in self.rover_capabilities:
                    self.rover_capabilities[rover]['rock'] = True
            elif pred == 'equipped_for_imaging':
                # (equipped_for_imaging ?r - rover)
                rover = args[0]
                if rover in self.rover_capabilities:
                    self.rover_capabilities[rover]['imaging'] = True
            elif pred == 'store_of':
                # (store_of ?s - store ?r - rover)
                store, rover = args
                self.store_owners[store] = rover
            elif pred == 'supports':
                # (supports ?c - camera ?m - mode)
                camera, mode = args
                if camera in self.camera_info:
                    self.camera_info[camera]['modes'].add(mode)
            elif pred == 'visible':
                # (visible ?w - waypoint ?p - waypoint)
                # This predicate is symmetric in the example, but not necessarily in PDDL.
                # We only need visibility from rover location to lander location for communication.
                # We'll process lander visibility separately after finding lander_waypoint.
                pass # Processed later
            elif pred == 'have_image':
                 # This is NOT static, it's dynamic. Ignore here.
                 pass
            elif pred == 'communicated_soil_data':
                 # This is NOT static, it's a goal predicate. Ignore here.
                 pass
            elif pred == 'communicated_rock_data':
                 # This is NOT static, it's a goal predicate. Ignore here.
                 pass
            elif pred == 'communicated_image_data':
                 # This is NOT static, it's a goal predicate. Ignore here.
                 pass
            elif pred == 'at_soil_sample':
                 # This is NOT static, it's dynamic (consumed). Ignore here.
                 pass
            elif pred == 'at_rock_sample':
                 # This is NOT static, it's dynamic (consumed). Ignore here.
                 pass
            elif pred == 'visible_from':
                # (visible_from ?o - objective ?w - waypoint)
                objective, waypoint = args
                if objective in self.objective_visible_from:
                    self.objective_visible_from[objective].add(waypoint)
            elif pred == 'calibration_target':
                # (calibration_target ?i - camera ?o - objective)
                camera, objective = args
                if camera in self.camera_info:
                    self.camera_info[camera]['calibration_target'] = objective
            elif pred == 'on_board':
                # (on_board ?i - camera ?r - rover)
                camera, rover = args
                if camera in self.camera_info:
                    self.camera_info[camera]['rover'] = rover
                if rover in self.rover_cameras:
                    self.rover_cameras[rover].add(camera)
            # Add other static predicates if any are relevant for precomputation

        # Process lander visibility after lander waypoint is known
        if self.lander_waypoint:
             for fact_str in task.static:
                 pred, args = parse_fact(fact_str)
                 if pred == 'visible':
                     # (visible ?w - waypoint ?p - waypoint)
                     wp1, wp2 = args
                     if wp2 == self.lander_waypoint:
                         self.lander_visible_waypoints.add(wp1)


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 3. Extract dynamic information from state
        rover_locations = {} # {rover: waypoint}
        rover_soil_samples = {} # {rover: {waypoint, ...}}
        rover_rock_samples = {} # {rover: {waypoint, ...}}
        rover_images = {} # {rover: {(objective, mode), ...}}
        rover_store_status = {} # {rover: 'empty'/'full'}
        rover_calibration_status = {} # {rover: {camera: bool}}
        current_soil_samples_at = set() # {waypoint, ...}
        current_rock_samples_at = set() # {waypoint, ...}

        # Initialize dynamic structures for known rovers/cameras/stores
        for rover in self.rover_capabilities:
             rover_soil_samples[rover] = set()
             rover_rock_samples[rover] = set()
             rover_images[rover] = set()
             rover_store_status[rover] = 'unknown' # Will be updated by 'empty' or 'full'
             rover_calibration_status[rover] = {}
             for camera in self.rover_cameras.get(rover, []):
                  rover_calibration_status[rover][camera] = False

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at':
                # (at ?x - rover ?y - waypoint)
                rover, waypoint = args
                rover_locations[rover] = waypoint
            elif pred == 'have_soil_analysis':
                # (have_soil_analysis ?r - rover ?w - waypoint)
                rover, waypoint = args
                if rover in rover_soil_samples:
                    rover_soil_samples[rover].add(waypoint)
            elif pred == 'have_rock_analysis':
                # (have_rock_analysis ?r - rover ?w - waypoint)
                rover, waypoint = args
                if rover in rover_rock_samples:
                    rover_rock_samples[rover].add(waypoint)
            elif pred == 'have_image':
                # (have_image ?r - rover ?o - objective ?m - mode)
                rover, objective, mode = args
                if rover in rover_images:
                    rover_images[rover].add((objective, mode))
            elif pred == 'empty':
                # (empty ?s - store)
                store = args[0]
                if store in self.store_owners:
                    rover = self.store_owners[store]
                    rover_store_status[rover] = 'empty'
            elif pred == 'full':
                # (full ?s - store)
                store = args[0]
                if store in self.store_owners:
                    rover = self.store_owners[store]
                    rover_store_status[rover] = 'full'
            elif pred == 'calibrated':
                # (calibrated ?c - camera ?r - rover)
                camera, rover = args
                if rover in rover_calibration_status and camera in rover_calibration_status[rover]:
                    rover_calibration_status[rover][camera] = True
            elif pred == 'at_soil_sample':
                # (at_soil_sample ?w - waypoint)
                waypoint = args[0]
                current_soil_samples_at.add(waypoint)
            elif pred == 'at_rock_sample':
                # (at_rock_sample ?w - waypoint)
                waypoint = args[0]
                current_rock_samples_at.add(waypoint)

        # 4. Calculate total heuristic
        total_h = 0
        unachieved_goals = self.goals - state

        for goal_fact_str in unachieved_goals:
            pred, args = parse_fact(goal_fact_str)
            min_goal_cost = float('inf')

            if pred == 'communicated_soil_data':
                # (communicated_soil_data ?w - waypoint)
                w_goal = args[0]

                # Option 1: Rover already has the sample
                for rover in rover_soil_samples:
                    if w_goal in rover_soil_samples[rover]:
                        if rover in rover_locations:
                            current_wp = rover_locations[rover]
                            nav_cost = get_shortest_path_cost_to_set(rover, current_wp, self.lander_visible_waypoints, self.rover_nav_graph)
                            if nav_cost < float('inf'):
                                min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                # Option 2: Sample needs to be collected
                if w_goal in current_soil_samples_at:
                    for rover in self.rover_capabilities:
                        if self.rover_capabilities[rover]['soil']:
                            if rover in rover_locations:
                                current_wp = rover_locations[rover]
                                nav_to_sample_cost = get_shortest_path_cost(rover, current_wp, w_goal, self.rover_nav_graph)

                                if nav_to_sample_cost < float('inf'):
                                    store_cost = 0
                                    # Find the store for this rover
                                    rover_store = None
                                    for s, r in self.store_owners.items():
                                        if r == rover:
                                            rover_store = s
                                            break

                                    if rover_store and rover_store_status.get(rover, 'unknown') == 'full':
                                         # If store is full, need to drop first
                                         store_cost = 1 # drop action

                                    # Check if rover can reach a communication point from sample location
                                    nav_sample_to_comm_cost = get_shortest_path_cost_to_set(rover, w_goal, self.lander_visible_waypoints, self.rover_nav_graph)

                                    if nav_sample_to_comm_cost < float('inf'):
                                        # Nav to sample + (drop if needed) + sample + Nav sample to comm + communicate
                                        cost = nav_to_sample_cost + store_cost + 1 + nav_sample_to_comm_cost + 1
                                        min_goal_cost = min(min_goal_cost, cost)


            elif pred == 'communicated_rock_data':
                # (communicated_rock_data ?w - waypoint)
                w_goal = args[0]

                # Option 1: Rover already has the sample
                for rover in rover_rock_samples:
                    if w_goal in rover_rock_samples[rover]:
                        if rover in rover_locations:
                            current_wp = rover_locations[rover]
                            nav_cost = get_shortest_path_cost_to_set(rover, current_wp, self.lander_visible_waypoints, self.rover_nav_graph)
                            if nav_cost < float('inf'):
                                min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                # Option 2: Sample needs to be collected
                if w_goal in current_rock_samples_at:
                    for rover in self.rover_capabilities:
                        if self.rover_capabilities[rover]['rock']:
                            if rover in rover_locations:
                                current_wp = rover_locations[rover]
                                nav_to_sample_cost = get_shortest_path_cost(rover, current_wp, w_goal, self.rover_nav_graph)

                                if nav_to_sample_cost < float('inf'):
                                    store_cost = 0
                                    # Find the store for this rover
                                    rover_store = None
                                    for s, r in self.store_owners.items():
                                        if r == rover:
                                            rover_store = s
                                            break

                                    if rover_store and rover_store_status.get(rover, 'unknown') == 'full':
                                         # If store is full, need to drop first
                                         store_cost = 1 # drop action

                                    # Check if rover can reach a communication point from sample location
                                    nav_sample_to_comm_cost = get_shortest_path_cost_to_set(rover, w_goal, self.lander_visible_waypoints, self.rover_nav_graph)

                                    if nav_sample_to_comm_cost < float('inf'):
                                        # Nav to sample + (drop if needed) + sample + Nav sample to comm + communicate
                                        cost = nav_to_sample_cost + store_cost + 1 + nav_sample_to_comm_cost + 1
                                        min_goal_cost = min(min_goal_cost, cost)


            elif pred == 'communicated_image_data':
                # (communicated_image_data ?o - objective ?m - mode)
                o_goal, m_goal = args

                # Option 1: Rover already has the image
                for rover in rover_images:
                    if (o_goal, m_goal) in rover_images[rover]:
                         if rover in rover_locations:
                            current_wp = rover_locations[rover]
                            nav_cost = get_shortest_path_cost_to_set(rover, current_wp, self.lander_visible_waypoints, self.rover_nav_graph)
                            if nav_cost < float('inf'):
                                min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate

                # Option 2: Image needs to be taken
                # Find rovers equipped for imaging with a camera supporting the mode
                for rover in self.rover_capabilities:
                    if self.rover_capabilities[rover]['imaging']:
                        for camera in self.rover_cameras.get(rover, []):
                            if m_goal in self.camera_info.get(camera, {}).get('modes', set()):
                                # Found a suitable rover and camera
                                cal_target = self.camera_info[camera].get('calibration_target')
                                if cal_target is None: continue # Camera has no calibration target

                                cal_wps = self.objective_visible_from.get(cal_target, set())
                                img_wps = self.objective_visible_from.get(o_goal, set())

                                if cal_wps and img_wps:
                                    if rover in rover_locations:
                                        current_wp = rover_locations[rover]

                                        # Nav from current to any calibration waypoint
                                        nav_to_cal_cost = get_shortest_path_cost_to_set(rover, current_wp, cal_wps, self.rover_nav_graph)

                                        if nav_to_cal_cost < float('inf'):
                                            # Nav from any calibration waypoint to any image waypoint
                                            min_nav_cal_to_img = float('inf')
                                            for wc in cal_wps:
                                                min_nav_cal_to_img = min(min_nav_cal_to_img, get_shortest_path_cost_to_set(rover, wc, img_wps, self.rover_nav_graph))

                                            if min_nav_cal_to_img < float('inf'):
                                                # Nav from any image waypoint to any communication waypoint
                                                min_nav_img_to_comm = float('inf')
                                                for wi in img_wps:
                                                     min_nav_img_to_comm = min(min_nav_img_to_comm, get_shortest_path_cost_to_set(rover, wi, self.lander_visible_waypoints, self.rover_nav_graph))

                                                if min_nav_img_to_comm < float('inf'):
                                                    # Nav current to cal + calibrate + Nav cal to img + take_image + Nav img to comm + communicate
                                                    cost = nav_to_cal_cost + 1 + min_nav_cal_to_img + 1 + min_nav_img_to_comm + 1
                                                    min_goal_cost = min(min_goal_cost, cost)


            # Add the minimum cost for this goal to the total heuristic
            # If min_goal_cost is inf, it means this goal is likely unreachable
            # from the current state based on this heuristic's logic.
            if min_goal_cost < float('inf'):
                 total_h += min_goal_cost
            else:
                 # If any unachieved goal is unreachable, the state is a dead end
                 # or requires a path not captured by this heuristic.
                 # Returning inf for the total heuristic is appropriate.
                 return float('inf')


        return total_h

