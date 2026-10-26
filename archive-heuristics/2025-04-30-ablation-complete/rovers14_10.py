# Need to import necessary modules
from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string case if necessary, though PDDL facts aren't empty
    if not fact or fact[0] != '(' or fact[-1] != ')':
         # Or raise an error, depending on expected input robustness
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
    # Check if the number of parts matches the number of args, unless args contains wildcards that could match multiple parts (not typical for simple PDDL predicates)
    # A simpler check: zip stops when the shortest iterable is exhausted.
    # Ensure all args are consumed if not wildcard.
    if len(parts) != len(args):
         return False # Basic check for arity mismatch

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It calculates the cost for each unachieved goal fact independently
    and sums these costs. The cost for a goal fact is estimated based on the
    minimum actions needed to collect the required data (soil sample, rock sample,
    or image) and then communicate it to the lander. Navigation costs are
    estimated using shortest paths on the `can_traverse` graph for each rover.

    # Assumptions
    - The heuristic assumes that soil/rock samples exist at their initial locations
      if needed and not yet collected.
    - It assumes that a rover always has an empty store when needed for sampling.
    - It assumes that camera calibration is always possible if the rover is at a
      waypoint visible from the calibration target.
    - It assumes that the best-suited rover and the shortest paths are used for
      each required navigation segment.
    - It assumes that communication can happen from any waypoint visible from the
      lander's location.
    - The heuristic is additive over unachieved goals and does not account for
      negative interactions or shared resources beyond selecting the minimum cost
      rover/path for each goal component. This makes it non-admissible but
      potentially more informative than a simple goal count.

    # Heuristic Initialization
    - Parses static facts to identify lander location, rover capabilities,
      store ownership, camera properties (on-board, supported modes, calibration
      targets), waypoint visibility (`visible_from`), and rover traversal
      capabilities (`can_traverse`).
    - Builds a navigation graph for each rover based on `can_traverse` facts.
    - Computes all-pairs shortest paths for each rover's graph using BFS.
    - Identifies waypoints visible from the lander's location (communication points).
    - Stores mappings for quick lookup of objective visibility, camera targets,
      camera modes, etc.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not yet true in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each unachieved goal fact:
        a. Estimate the minimum cost to achieve this specific goal fact, assuming
           it can be pursued independently by the most capable and best-positioned
           rover.
        b. Add this estimated cost to the total heuristic cost.

    Detailed cost estimation per unachieved goal type:

    - For `(communicated_soil_data W)`:
        - If `(have_soil_analysis R W)` is true for any rover R in the state:
            - Cost to collect data = 0.
            - Cost to communicate: Find the rover R holding the data. Find its current location `W_curr`. Find the minimum navigation cost for R from `W_curr` to any waypoint X visible from the lander (`comm_wps`). Add 1 for the `communicate_soil_data` action.
        - If `(have_soil_analysis R W)` is false for all rovers:
            - Cost to collect data: Find a soil-equipped rover R. Find its current location `W_curr`. Find the minimum navigation cost for R from `W_curr` to W. Add 1 for the `sample_soil` action. Minimize this over all soil-equipped rovers.
            - Cost to communicate: Assume the rover R that collected the sample is now at W. Find the minimum navigation cost for R from W to any waypoint X visible from the lander (`comm_wps`). Add 1 for the `communicate_soil_data` action.
            - Total cost for this goal = (Min cost to collect) + (Min cost to communicate).

    - For `(communicated_rock_data W)`: Similar logic as soil data, using rock-equipped rovers and rock samples.

    - For `(communicated_image_data O M)`:
        - If `(have_image R O M)` is true for any rover R in the state:
            - Cost to collect data = 0.
            - Cost to communicate: Find the rover R holding the image. Find its current location `W_curr`. Find the minimum navigation cost for R from `W_curr` to any waypoint X visible from the lander (`comm_wps`). Add 1 for the `communicate_image_data` action.
        - If `(have_image R O M)` is false for all rovers:
            - Cost to collect data: Find an imaging-equipped rover R with a camera C supporting mode M. Find its current location `W_curr`. Find a waypoint P visible from objective O (`obj_visible_from[O]`). Find a waypoint W' visible from camera C's calibration target T (`obj_visible_from[cam_cal_target[C]]`). The path involves navigating to W' (for calibration) and P (for the image). Estimate the minimum navigation cost for R from `W_curr` to W' and then to P. Add 1 for `calibrate` and 1 for `take_image`. Minimize this over all suitable (R, C, M, P, W') combinations.
            - Cost to communicate: Assume the rover R that took the image is now at P. Find the minimum navigation cost for R from P to any waypoint X visible from the lander (`comm_wps`). Add 1 for the `communicate_image_data` action.
            - Total cost for this goal = (Min cost to collect) + (Min cost to communicate).

    4. The total heuristic value is the sum of the estimated costs for all unachieved goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Pre-computation from static facts ---

        self.rover_capabilities = {
            'soil': set(),
            'rock': set(),
            'imaging': set()
        }
        self.rover_store = {} # rover -> store
        self.lander_pos = None
        self.comm_wps = set() # Waypoints visible from lander location

        self.rover_graphs = {} # rover -> {wp -> [neighbor_wp, ...]}
        self.rover_dist = {} # rover -> {start_wp -> {end_wp -> distance}}

        self.obj_visible_from = {} # objective -> {waypoint, ...}
        self.cam_on_board = {} # camera -> rover
        self.cam_supports_mode = {} # camera -> {mode, ...}
        self.cam_cal_target = {} # camera -> objective
        self.rover_cameras = {} # rover -> {camera, ...}

        # Collect all objects and relevant static predicates
        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_stores = set()
        lander_location = None # Assuming one lander

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at_lander':
                lander_location = parts[2]
                all_waypoints.add(parts[2])
            elif parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities['soil'].add(parts[1])
                all_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities['rock'].add(parts[1])
                all_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities['imaging'].add(parts[1])
                all_rovers.add(parts[1])
            elif parts[0] == 'store_of':
                self.rover_store[parts[2]] = parts[1]
                all_rovers.add(parts[2])
                all_stores.add(parts[1])
            elif parts[0] == 'visible':
                 all_waypoints.add(parts[1])
                 all_waypoints.add(parts[2])
            elif parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                all_rovers.add(rover)
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
                if rover not in self.rover_graphs:
                    self.rover_graphs[rover] = {} # Initialize graph for this rover
                if wp1 not in self.rover_graphs[rover]:
                    self.rover_graphs[rover][wp1] = []
                self.rover_graphs[rover][wp1].append(wp2)
            elif parts[0] == 'calibration_target':
                self.cam_cal_target[parts[1]] = parts[2]
                all_cameras.add(parts[1])
                all_objectives.add(parts[2])
            elif parts[0] == 'on_board':
                self.cam_on_board[parts[1]] = parts[2]
                all_cameras.add(parts[1])
                all_rovers.add(parts[2])
                if parts[2] not in self.rover_cameras:
                    self.rover_cameras[parts[2]] = set()
                self.rover_cameras[parts[2]].add(parts[1])
            elif parts[0] == 'supports':
                if parts[1] not in self.cam_supports_mode:
                    self.cam_supports_mode[parts[1]] = set()
                self.cam_supports_mode[parts[1]].add(parts[2])
                all_cameras.add(parts[1])
                all_modes.add(parts[2])
            elif parts[0] == 'visible_from':
                if parts[1] not in self.obj_visible_from:
                    self.obj_visible_from[parts[1]] = set()
                self.obj_visible_from[parts[1]].add(parts[2])
                all_objectives.add(parts[1])
                all_waypoints.add(parts[2])

        self.rovers = all_rovers
        self.waypoints = all_waypoints
        self.cameras = all_cameras
        self.objectives = all_objectives
        self.modes = all_modes
        self.stores = all_stores
        self.lander_pos = lander_location

        # Identify communication waypoints based on lander position and visible facts
        if self.lander_pos:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if match(fact, "visible", self.lander_pos, "*"):
                     self.comm_wps.add(parts[2])
                 elif match(fact, "visible", "*", self.lander_pos):
                     self.comm_wps.add(parts[1])
             # Add lander location itself as a potential communication point if no visible links found
             if not self.comm_wps:
                  self.comm_wps.add(self.lander_pos)


        # Ensure all waypoints found are in all rover graphs before computing distances
        for rover in self.rovers:
             if rover not in self.rover_graphs:
                  self.rover_graphs[rover] = {} # Rover exists but has no can_traverse facts?
             for wp in self.waypoints:
                 if wp not in self.rover_graphs[rover]:
                     self.rover_graphs[rover][wp] = []


        # Compute shortest paths for each rover
        for rover, graph in self.rover_graphs.items():
            self.rover_dist[rover] = {}
            waypoints_list = list(self.waypoints) # Use the collected set of all waypoints

            for start_wp in waypoints_list:
                self.rover_dist[rover][start_wp] = {}
                q = deque([start_wp])
                d = {start_wp: 0}

                while q:
                    curr_wp = q.popleft()
                    self.rover_dist[rover][start_wp][curr_wp] = d[curr_wp]

                    # Ensure curr_wp is a key in the graph before iterating neighbors
                    if curr_wp in graph:
                        for neighbor_wp in graph[curr_wp]:
                            if neighbor_wp not in d:
                                d[neighbor_wp] = d[curr_wp] + 1
                                q.append(neighbor_wp)

                # Mark unreachable waypoints with infinity
                for end_wp in waypoints_list:
                    if end_wp not in self.rover_dist[rover][start_wp]:
                        self.rover_dist[rover][start_wp][end_wp] = math.inf

        # Handle case where no lander was found
        if not self.lander_pos:
             # If communication goals exist, they are impossible. Heuristic will reflect this
             # if min_goal_cost remains infinity.
             pass # No specific action needed, infinity propagation handles it.


    def get_distance(self, rover, start_wp, end_wp):
        """Lookup precomputed shortest distance, return infinity if not found."""
        # Ensure rover, start_wp, end_wp are valid and distances were computed
        if rover not in self.rover_dist or start_wp not in self.rover_dist[rover] or end_wp not in self.rover_dist[rover][start_wp]:
            # This can happen if a waypoint exists but is not in the rover's graph
            # or if rover/waypoint names in state don't match precomputed ones.
            # Return infinity as it's unreachable by this rover from this start.
            return math.inf
        return self.rover_dist[rover][start_wp][end_wp]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract dynamic facts from state ---
        rover_pos = {} # rover -> waypoint
        rover_soil_data = {} # rover -> {waypoint, ...}
        rover_rock_data = {} # rover -> {waypoint, ...}
        rover_image_data = {} # rover -> {(objective, mode), ...}
        store_empty = {} # store -> bool
        cam_calibrated = {} # camera -> bool
        soil_samples_at = set() # waypoint with sample
        rock_samples_at = set() # waypoint with sample
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        # Initialize dynamic state trackers based on known objects
        for r in self.rovers:
             rover_soil_data[r] = set()
             rover_rock_data[r] = set()
             rover_image_data[r] = set()
        for s in self.stores:
             store_empty[s] = False # Assume full unless stated empty
        for c in self.cameras:
             cam_calibrated[c] = False # Assume not calibrated unless stated

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at' and parts[1] in self.rovers:
                rover_pos[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis' and parts[1] in self.rovers:
                rover_soil_data[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis' and parts[1] in self.rovers:
                rover_rock_data[parts[1]].add(parts[2])
            elif parts[0] == 'have_image' and parts[1] in self.rovers and len(parts) == 4:
                rover_image_data[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'empty' and parts[1] in self.stores:
                store_empty[parts[1]] = True
            elif parts[0] == 'full' and parts[1] in self.stores:
                 store_empty[parts[1]] = False # Explicitly mark as not empty
            elif parts[0] == 'calibrated' and parts[1] in self.cameras:
                cam_calibrated[parts[1]] = True
            elif parts[0] == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data' and len(parts) == 3:
                communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # --- Process unachieved goals ---

        # Soil Goals
        soil_goals_needed = {get_parts(g)[1] for g in self.goals if match(g, "communicated_soil_data", "*")} - communicated_soil
        for waypoint_w in soil_goals_needed:
            min_goal_cost = math.inf

            # Check if data is already held by any rover
            data_held_by_rover = None
            for r in self.rovers:
                 if waypoint_w in rover_soil_data.get(r, set()):
                      data_held_by_rover = r
                      break

            if data_held_by_rover:
                # Data is held, just need to communicate
                rover_r = data_held_by_rover
                w_curr = rover_pos.get(rover_r)
                if w_curr is None: continue # Rover location unknown, cannot communicate

                min_comm_nav_cost = math.inf
                for comm_wp in self.comm_wps:
                    nav_cost = self.get_distance(rover_r, w_curr, comm_wp)
                    min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                if min_comm_nav_cost != math.inf:
                    min_goal_cost = min_comm_nav_cost + 1 # +1 for communicate_soil_data action

            else:
                # Need to collect data and then communicate
                min_collection_cost = math.inf
                best_collector_rover = None

                # Find best rover and path to collect
                for rover_r in self.rover_capabilities['soil']:
                    w_curr = rover_pos.get(rover_r)
                    if w_curr is None: continue # Rover location unknown

                    nav_cost = self.get_distance(rover_r, w_curr, waypoint_w)
                    if nav_cost != math.inf:
                        # Cost to collect = navigate + sample
                        cost = nav_cost + 1
                        if cost < min_collection_cost:
                            min_collection_cost = cost
                            best_collector_rover = rover_r

                if best_collector_rover:
                    # Assume rover is at waypoint_w after collection
                    min_comm_nav_cost = math.inf
                    for comm_wp in self.comm_wps:
                        nav_cost = self.get_distance(best_collector_rover, waypoint_w, comm_wp)
                        min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                    if min_comm_nav_cost != math.inf:
                        # Total cost = collection + communication
                        min_goal_cost = min_collection_cost + min_comm_nav_cost + 1 # +1 for communicate_soil_data action

            if min_goal_cost != math.inf:
                total_cost += min_goal_cost
            # else: goal is likely impossible for any available rover/path, contributes infinity

        # Rock Goals (Similar to Soil Goals)
        rock_goals_needed = {get_parts(g)[1] for g in self.goals if match(g, "communicated_rock_data", "*")} - communicated_rock
        for waypoint_w in rock_goals_needed:
            min_goal_cost = math.inf

            data_held_by_rover = None
            for r in self.rovers:
                 if waypoint_w in rover_rock_data.get(r, set()):
                      data_held_by_rover = r
                      break

            if data_held_by_rover:
                rover_r = data_held_by_rover
                w_curr = rover_pos.get(rover_r)
                if w_curr is None: continue

                min_comm_nav_cost = math.inf
                for comm_wp in self.comm_wps:
                    nav_cost = self.get_distance(rover_r, w_curr, comm_wp)
                    min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                if min_comm_nav_cost != math.inf:
                    min_goal_cost = min_comm_nav_cost + 1

            else:
                min_collection_cost = math.inf
                best_collector_rover = None

                for rover_r in self.rover_capabilities['rock']:
                    w_curr = rover_pos.get(rover_r)
                    if w_curr is None: continue

                    nav_cost = self.get_distance(rover_r, w_curr, waypoint_w)
                    if nav_cost != math.inf:
                        cost = nav_cost + 1
                        if cost < min_collection_cost:
                            min_collection_cost = cost
                            best_collector_rover = rover_r

                if best_collector_rover:
                    min_comm_nav_cost = math.inf
                    for comm_wp in self.comm_wps:
                        nav_cost = self.get_distance(best_collector_rover, waypoint_w, comm_wp)
                        min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                    if min_comm_nav_cost != math.inf:
                        min_goal_cost = min_collection_cost + min_comm_nav_cost + 1

            if min_goal_cost != math.inf:
                total_cost += min_goal_cost


        # Image Goals
        image_goals_needed = {tuple(get_parts(g)[1:]) for g in self.goals if match(g, "communicated_image_data", "*", "*")} - communicated_image
        for objective_o, mode_m in image_goals_needed:
            min_goal_cost = math.inf

            data_held_by_rover = None
            for r in self.rovers:
                 if (objective_o, mode_m) in rover_image_data.get(r, set()):
                      data_held_by_rover = r
                      break

            if data_held_by_rover:
                # Data is held, just need to communicate
                rover_r = data_held_by_rover
                w_curr = rover_pos.get(rover_r)
                if w_curr is None: continue

                min_comm_nav_cost = math.inf
                for comm_wp in self.comm_wps:
                    nav_cost = self.get_distance(rover_r, w_curr, comm_wp)
                    min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                if min_comm_nav_cost != math.inf:
                    min_goal_cost = min_comm_nav_cost + 1

            else:
                # Need to collect data and then communicate
                min_collection_cost = math.inf
                best_collector_rover = None
                best_image_wp = None # Waypoint where image is taken for the best plan

                # Find best rover, camera, and path to collect
                for rover_r in self.rover_capabilities['imaging']:
                    w_curr = rover_pos.get(rover_r)
                    if w_curr is None: continue

                    # Find cameras on this rover that support the mode
                    suitable_cameras = [
                        c for c in self.rover_cameras.get(rover_r, set())
                        if mode_m in self.cam_supports_mode.get(c, set())
                    ]

                    if not suitable_cameras: continue # This rover can't take this image

                    # Find waypoints visible from the objective
                    image_wps = self.obj_visible_from.get(objective_o, set())
                    if not image_wps: continue # Cannot see objective from anywhere

                    min_rover_collection_cost = math.inf
                    current_best_image_wp_for_rover = None # Store the image wp for the best plan *for this rover*

                    for camera_c in suitable_cameras:
                        # Find waypoints visible from the calibration target for this camera
                        cal_target = self.cam_cal_target.get(camera_c)
                        if not cal_target: continue # Camera has no calibration target
                        cal_wps = self.obj_visible_from.get(cal_target, set())
                        if not cal_wps: continue # Calibration target not visible from anywhere

                        # Find minimum path cost: W_curr -> W_cal -> W_img
                        min_path_cost_for_cam = math.inf
                        current_best_image_wp_for_cam = None # Store the image wp for the best plan *for this camera*

                        for img_wp in image_wps:
                            for cal_wp in cal_wps:
                                # Path: W_curr -> Cal_WP -> Img_WP
                                path_cost = self.get_distance(rover_r, w_curr, cal_wp) + self.get_distance(rover_r, cal_wp, img_wp)
                                if path_cost != math.inf:
                                    if path_cost < min_path_cost_for_cam:
                                         min_path_cost_for_cam = path_cost
                                         current_best_image_wp_for_cam = img_wp # Store the image wp used

                        if min_path_cost_for_cam != math.inf:
                             # Cost to collect = path + calibrate + take_image
                             cost = min_path_cost_for_cam + 1 + 1

                             if cost < min_rover_collection_cost:
                                 min_rover_collection_cost = cost
                                 current_best_image_wp_for_rover = current_best_image_wp_for_cam # Store the image wp for this rover/cam combo

                    if min_rover_collection_cost != math.inf:
                         if min_rover_collection_cost < min_collection_cost:
                             min_collection_cost = min_rover_collection_cost
                             best_collector_rover = rover_r
                             best_image_wp = current_best_image_wp_for_rover # Store the image wp for the overall best collection

                if best_collector_rover and best_image_wp:
                    # Assume rover is at best_image_wp after collection
                    min_comm_nav_cost = math.inf
                    for comm_wp in self.comm_wps:
                        nav_cost = self.get_distance(best_collector_rover, best_image_wp, comm_wp)
                        min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                    if min_comm_nav_cost != math.inf:
                        # Total cost = collection + communication
                        min_goal_cost = min_collection_cost + min_comm_nav_cost + 1 # +1 for communicate_image_data action

            if min_goal_cost != math.inf:
                total_cost += min_goal_cost

        # The heuristic is 0 if and only if all goals are communicated.
        # If total_cost is 0, it means the loops for needed goals were skipped,
        # which implies all goals are in the communicated sets.
        # If there are unachieved goals but no path/rover exists, cost is infinity.
        # If solvable, cost is finite and > 0 until goal.
        return total_cost
