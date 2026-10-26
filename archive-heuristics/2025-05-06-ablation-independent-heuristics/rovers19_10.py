# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove surrounding brackets and split by space
    parts = fact_string.strip('()').split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Assuming Heuristic and Task classes are available from the planner environment
# from heuristics.heuristic_base import Heuristic
# from task import Operator, Task

from collections import deque, defaultdict
from typing import Dict, Set, List, Tuple
import math

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each uncommunicated goal fact. The cost for an uncommunicated goal
    is estimated based on whether the required data (sample or image) is
    already collected. If data is collected, the cost is 1 (communicate)
    plus the minimum path cost for the rover holding the data to reach a
    waypoint visible from the lander. If data is not collected, the cost
    includes sampling/imaging actions, potential calibration/dropping actions,
    and the minimum path costs for a suitable rover to reach the required
    locations (sample/observation/calibration waypoint and then a lander
    communication waypoint). Path costs are precomputed shortest distances
    on the navigation graph.

    Assumptions:
    - The navigation graph defined by (visible ?w1 ?w2) is undirected.
    - (can_traverse ?r ?w1 ?w2) is the same for all rovers and is equivalent
      to (visible ?w1 ?w2).
    - There is exactly one lander.
    - Initial soil/rock samples and objective visibility from waypoints are static.
    - Unreachable goals (e.g., sample requested from waypoint with no initial sample,
      objective not visible from any waypoint, camera calibration target not visible
      from any waypoint) result in an infinite heuristic value.

    Heuristic Initialization:
    - Parses static facts to identify lander location, navigation graph edges,
      rover capabilities (equipped_for_*), store-rover mapping, camera properties
      (on_board, supports, calibration_target), objective visibility from waypoints.
    - Parses initial state facts to identify initial soil/rock sample locations.
    - Collects all objects (rovers, waypoints, etc.) mentioned in static facts,
      initial state, and goals.
    - Builds the navigation graph based on (visible ?w1 ?w2).
    - Computes all-pairs shortest paths between waypoints using BFS from each waypoint.
    - Identifies lander communication waypoints (visible from lander location).
    - Computes minimum distance from every waypoint to any lander communication waypoint.
    - Stores relevant static and initial state information in class members for quick access.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h` to 0.
    2. Get the current state from the search node.
    3. Parse the current state to extract dynamic information: rover positions,
       store statuses (full/empty), collected samples (have_soil_analysis,
       have_rock_analysis), collected images (have_image), camera calibration
       statuses (calibrated).
    4. Iterate through each goal fact in `self.goals`.
    5. If a goal fact is already present in the current state, skip it.
    6. If a goal fact is not in the current state, estimate the cost to achieve it
       and add it to `h`.
    7. For a goal `(communicated_soil_data ?w)`:
        a. Check if `(have_soil_analysis ?r ?w)` is true for any rover `r` in the current state.
        b. If yes (sample is collected):
            i. Cost = 1 (communicate) + minimum path cost from the current position
               of any rover that has the sample to any lander communication waypoint.
        c. If no (sample needs to be collected):
            i. Check if `(at_soil_sample ?w)` was true in the initial state. If not,
               cost is infinity.
            ii. Find suitable rovers (equipped for soil, has store). If none, cost is infinity.
            iii. Cost = minimum over suitable rovers (
                   path cost from current position to `w` +
                   (1 if rover's store is full - for drop action) +
                   1 (sample action) +
                   path cost from `w` to any lander communication waypoint +
                   1 (communicate action)
                 ).
    8. For a goal `(communicated_rock_data ?w)`: Follow similar logic as for soil data,
       using rock-specific predicates and equipped status.
    9. For a goal `(communicated_image_data ?o ?m)`:
        a. Check if `(have_image ?r ?o ?m)` is true for any rover `r` in the current state.
        b. If yes (image is collected):
            i. Cost = 1 (communicate) + minimum path cost from the current position
               of any rover that has the image to any lander communication waypoint.
        c. If no (image needs to be taken):
            i. Find suitable observation waypoints `p` for objective `o` (from static info).
               If none, cost is infinity.
            ii. Find suitable rovers `r` and cameras `i` (equipped for imaging, on board,
                supports mode `m`). If none, cost is infinity.
            iii. Cost = minimum over suitable (rover, camera, observation waypoint `p`) combinations (
                   cost to get image at `p` +
                   path cost from `p` to any lander communication waypoint +
                   1 (communicate action)
                 ).
            iv. Cost to get image at `p` with rover `r` and camera `i`:
                - If `(calibrated ?i ?r)`: path cost from current position to `p` + 1 (take image action).
                - If not `(calibrated ?i ?r)`:
                    - Find suitable calibration waypoints `w` for camera `i`'s target. If none, cost is infinity.
                    - Cost = minimum over suitable calibration waypoints `w` (
                        path cost from current position to `w` +
                        1 (calibrate action) +
                        path cost from `w` to `p`
                      ) + 1 (take image action).
    10. Return the total heuristic value `h`. If `h` is infinity, return `math.inf`.
    """

    def __init__(self, task):
        super().__init__(task)

        # Collect all objects mentioned in the task
        self.waypoints = set()
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        def collect_objects(fact_string):
             pred, args = parse_fact(fact_string)
             for arg in args:
                 if arg.startswith('rover'): self.rovers.add(arg)
                 elif arg.startswith('waypoint'): self.waypoints.add(arg)
                 elif arg.startswith('store'): self.stores.add(arg)
                 elif arg.startswith('camera'): self.cameras.add(arg)
                 elif arg.startswith('mode'): self.modes.add(arg)
                 elif arg.startswith('lander'): self.landers.add(arg)
                 elif arg.startswith('objective'): self.objectives.add(arg)

        for fact_string in task.static:
            collect_objects(fact_string)
        for fact_string in task.initial_state:
            collect_objects(fact_string)
        for goal_string in task.goals:
            collect_objects(goal_string)


        self.lander_wp = None
        self.visible_graph = defaultdict(set) # waypoint -> set(waypoint)
        self.equipped_soil = set() # rover
        self.equipped_rock = set() # rover
        self.equipped_imaging = set() # rover
        self.store_rover = {} # store -> rover
        self.rover_store = {} # rover -> store
        self.supports_mode = defaultdict(set) # camera -> set(mode)
        self.visible_from_obj = defaultdict(set) # objective -> set(waypoint)
        self.calibration_target = {} # camera -> objective
        self.on_board = {} # camera -> rover
        self.rover_cameras = defaultdict(set) # rover -> set(camera)
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint

        # Parse static facts again to populate specific data structures
        for fact_string in task.static:
            pred, args = parse_fact(fact_string)
            if pred == 'at_lander':
                self.lander_wp = args[1] # Assuming one lander
            elif pred == 'visible':
                self.visible_graph[args[0]].add(args[1])
                self.visible_graph[args[1]].add(args[0]) # Assuming symmetric visibility
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(args[0])
            elif pred == 'store_of':
                self.stores.add(args[0]) # Ensure store object is added
                self.store_rover[args[0]] = args[1]
                self.rover_store[args[1]] = args[0]
            elif pred == 'supports':
                self.supports_mode[args[0]].add(args[1])
            elif pred == 'visible_from':
                self.visible_from_obj[args[0]].add(args[1])
            elif pred == 'calibration_target':
                self.calibration_target[args[0]] = args[1]
            elif pred == 'on_board':
                self.on_board[args[0]] = args[1]
                self.rover_cameras[args[1]].add(args[0])

        # Parse initial state facts for sample locations
        for fact_string in task.initial_state:
             pred, args = parse_fact(fact_string)
             if pred == 'at_soil_sample':
                 self.initial_soil_samples.add(args[0])
             elif pred == 'at_rock_sample':
                 self.initial_rock_samples.add(args[0])

        # Compute all-pairs shortest paths
        self.dist = self._compute_all_pairs_shortest_paths()

        # Find lander communication waypoints
        self.lander_comm_wps = set()
        if self.lander_wp:
            for wp in self.waypoints:
                if wp in self.visible_graph.get(self.lander_wp, set()):
                    self.lander_comm_wps.add(wp)

        # Compute min distance from every waypoint to any lander communication waypoint
        self.min_dist_to_lander_comm = {}
        for w1 in self.waypoints:
            min_d = math.inf
            for w2 in self.lander_comm_wps:
                if self.dist.get(w1, {}).get(w2, math.inf) != math.inf:
                    min_d = min(min_d, self.dist.get(w1, {}).get(w2, math.inf))
            self.min_dist_to_lander_comm[w1] = min_d


    def _compute_all_pairs_shortest_paths(self) -> Dict[str, Dict[str, float]]:
        """Computes shortest paths between all pairs of waypoints using BFS."""
        dist = {w: {other_w: math.inf for other_w in self.waypoints} for w in self.waypoints}

        for start_node in self.waypoints:
            dist[start_node][start_node] = 0
            queue = deque([start_node])
            # Use a set for visited for faster lookups
            visited = {start_node}

            while queue:
                u = queue.popleft()
                for v in self.visible_graph.get(u, set()):
                    if v not in visited:
                        visited.add(v)
                        dist[start_node][v] = dist[start_node][u] + 1
                        queue.append(v)
        return dist

    def __call__(self, node) -> float:
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        h = 0.0

        # Parse current state dynamic facts
        rover_pos = {} # rover -> waypoint
        store_full = {} # store -> bool
        have_soil = defaultdict(set) # rover -> set(waypoint)
        have_rock = defaultdict(set) # rover -> set(waypoint)
        have_image = defaultdict(lambda: defaultdict(set)) # rover -> objective -> set(mode)
        calibrated_cam = defaultdict(set) # rover -> set(camera)

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at':
                rover_pos[args[0]] = args[1]
            elif pred == 'full':
                store_full[args[0]] = True
            elif pred == 'empty':
                 store_full[args[0]] = False # Explicitly mark empty
            elif pred == 'have_soil_analysis':
                have_soil[args[0]].add(args[1])
            elif pred == 'have_rock_analysis':
                have_rock[args[0]].add(args[1])
            elif pred == 'have_image':
                have_image[args[0]][args[1]].add(args[2])
            elif pred == 'calibrated':
                calibrated_cam[args[1]].add(args[0]) # calibrated ?c ?r -> rover -> camera

        # Estimate cost for each uncommunicated goal
        for goal_string in self.goals:
            if goal_string in state:
                continue # Goal already achieved

            pred, args = parse_fact(goal_string)

            if pred == 'communicated_soil_data':
                w = args[0]
                # Check if sample is already collected by any rover
                sample_collected = any(w in have_soil[r] for r in self.rovers)

                if sample_collected:
                    # Cost: Communicate + move to comm point
                    min_comm_move_cost = math.inf
                    for r in self.rovers:
                        if w in have_soil[r] and r in rover_pos:
                             if self.min_dist_to_lander_comm.get(rover_pos[r], math.inf) != math.inf:
                                min_comm_move_cost = min(min_comm_move_cost, self.min_dist_to_lander_comm.get(rover_pos[r], math.inf))

                    if min_comm_move_cost == math.inf:
                         h += math.inf # Cannot reach lander comm point from any rover with sample
                    else:
                         h += 1 + min_comm_move_cost # 1 for communicate action
                else:
                    # Cost: Sample + Communicate + movement
                    # Need to sample at w
                    if w not in self.initial_soil_samples:
                        h += math.inf # Cannot sample if no sample initially
                        continue

                    min_total_cost_for_goal = math.inf

                    for rover in self.equipped_soil:
                        if rover not in rover_pos: continue # Rover not in current state?

                        store = self.rover_store.get(rover)
                        if not store: continue # Rover has no store?

                        store_is_full = store_full.get(store, False) # Default to not full if status unknown

                        # Cost to sample: move to w + (drop if full) + sample action
                        cost_to_w = self.dist.get(rover_pos[rover], {}).get(w, math.inf)
                        if cost_to_w == math.inf: continue # Cannot reach sample waypoint

                        drop_action_cost = 1 if store_is_full else 0
                        sample_action_cost = 1

                        # Cost to communicate: move from w to comm point + communicate action
                        cost_from_w_to_comm = self.min_dist_to_lander_comm.get(w, math.inf)
                        if cost_from_w_to_comm == math.inf: continue # Cannot reach comm point from sample waypoint

                        comm_action_cost = 1

                        total_cost = cost_to_w + drop_action_cost + sample_action_cost + cost_from_w_to_comm + comm_action_cost
                        min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost)

                    if min_total_cost_for_goal == math.inf:
                         h += math.inf # No suitable rover can achieve this
                    else:
                         h += min_total_cost_for_goal


            elif pred == 'communicated_rock_data':
                w = args[0]
                # Check if sample is already collected by any rover
                sample_collected = any(w in have_rock[r] for r in self.rovers)

                if sample_collected:
                    # Cost: Communicate + move to comm point
                    min_comm_move_cost = math.inf
                    for r in self.rovers:
                        if w in have_rock[r] and r in rover_pos:
                             if self.min_dist_to_lander_comm.get(rover_pos[r], math.inf) != math.inf:
                                min_comm_move_cost = min(min_comm_move_cost, self.min_dist_to_lander_comm.get(rover_pos[r], math.inf))

                    if min_comm_move_cost == math.inf:
                         h += math.inf # Cannot reach lander comm point from any rover with sample
                    else:
                         h += 1 + min_comm_move_cost # 1 for communicate action
                else:
                    # Cost: Sample + Communicate + movement
                    # Need to sample at w
                    if w not in self.initial_rock_samples:
                        h += math.inf # Cannot sample if no sample initially
                        continue

                    min_total_cost_for_goal = math.inf

                    for rover in self.equipped_rock:
                        if rover not in rover_pos: continue

                        store = self.rover_store.get(rover)
                        if not store: continue

                        store_is_full = store_full.get(store, False)

                        # Cost to sample: move to w + (drop if full) + sample action
                        cost_to_w = self.dist.get(rover_pos[rover], {}).get(w, math.inf)
                        if cost_to_w == math.inf: continue

                        drop_action_cost = 1 if store_is_full else 0
                        sample_action_cost = 1

                        # Cost to communicate: move from w to comm point + communicate action
                        cost_from_w_to_comm = self.min_dist_to_lander_comm.get(w, math.inf)
                        if cost_from_w_to_comm == math.inf: continue

                        comm_action_cost = 1

                        total_cost = cost_to_w + drop_action_cost + sample_action_cost + cost_from_w_to_comm + comm_action_cost
                        min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost)

                    if min_total_cost_for_goal == math.inf:
                         h += math.inf # No suitable rover can achieve this
                    else:
                         h += min_total_cost_for_goal


            elif pred == 'communicated_image_data':
                o = args[0]
                m = args[1]

                # Check if image is already collected by any rover
                image_collected = any(m in have_image[r].get(o, set()) for r in self.rovers)

                if image_collected:
                    # Cost: Communicate + move to comm point
                    min_comm_move_cost = math.inf
                    for r in self.rovers:
                        if m in have_image[r].get(o, set()) and r in rover_pos:
                             if self.min_dist_to_lander_comm.get(rover_pos[r], math.inf) != math.inf:
                                min_comm_move_cost = min(min_comm_move_cost, self.min_dist_to_lander_comm.get(rover_pos[r], math.inf))

                    if min_comm_move_cost == math.inf:
                         h += math.inf # Cannot reach lander comm point from any rover with image
                    else:
                         h += 1 + min_comm_move_cost # 1 for communicate action
                else:
                    # Cost: Take Image + Communicate + movement + (Calibrate if needed)
                    suitable_obs_wps = self.visible_from_obj.get(o, set())
                    if not suitable_obs_wps:
                        h += math.inf # Cannot observe objective from anywhere
                        continue

                    min_total_cost_for_goal = math.inf

                    for rover in self.equipped_imaging:
                        if rover not in rover_pos: continue

                        for camera in self.rover_cameras.get(rover, set()):
                            if m in self.supports_mode.get(camera, set()):
                                # This rover+camera can potentially take the image in this mode

                                is_calibrated = camera in calibrated_cam[rover]
                                cal_target = self.calibration_target.get(camera)
                                suitable_cal_wps = self.visible_from_obj.get(cal_target, set()) if cal_target else set()

                                for p in suitable_obs_wps:
                                    # Cost to get image at p with this rover/camera
                                    cost_to_get_image_at_p = math.inf

                                    if is_calibrated:
                                        # Cost: move to p + take image action
                                        cost_to_p = self.dist.get(rover_pos[rover], {}).get(p, math.inf)
                                        if cost_to_p != math.inf:
                                            cost_to_get_image_at_p = cost_to_p + 1 # 1 for take_image action
                                    else:
                                        # Cost: move to cal_wp + calibrate + move to p + take image action
                                        if not suitable_cal_wps: continue # Cannot calibrate this camera

                                        min_cal_path_cost = math.inf
                                        for w in suitable_cal_wps:
                                            # Path: current_pos -> w -> p
                                            if self.dist.get(rover_pos[rover], {}).get(w, math.inf) != math.inf and self.dist.get(w, {}).get(p, math.inf) != math.inf:
                                                min_cal_path_cost = min(min_cal_path_cost, self.dist[rover_pos[rover]][w] + self.dist[w][p])

                                        if min_cal_path_cost != math.inf:
                                             cost_to_get_image_at_p = min_cal_path_cost + 1 + 1 # 1 for calibrate, 1 for take_image

                                    if cost_to_get_image_at_p == math.inf: continue # Cannot get image at p

                                    # Cost to communicate: move from p to comm point + communicate action
                                    cost_from_p_to_comm = self.min_dist_to_lander_comm.get(p, math.inf)
                                    if cost_from_p_to_comm == math.inf: continue # Cannot reach comm point from observation waypoint

                                    comm_action_cost = 1

                                    total_cost = cost_to_get_image_at_p + cost_from_p_to_comm + comm_action_cost
                                    min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost)

                    if min_total_cost_for_goal == math.inf:
                         h += math.inf # No suitable rover/camera/waypoint combination can achieve this
                    else:
                         h += min_total_cost_for_goal

        # Return infinity if any goal is unreachable, otherwise return the sum
        return h
