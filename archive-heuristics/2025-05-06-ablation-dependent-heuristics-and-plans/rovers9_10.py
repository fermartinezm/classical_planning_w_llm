from fnmatch import fnmatch
import math

# Assume heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example
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
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated cost for each unachieved goal fact independently.
    The cost for each goal fact includes navigation steps and the core actions
    (sample, calibrate, take_image, communicate). Navigation cost is estimated
    using shortest path distances on the waypoint graph.

    # Assumptions
    - The waypoint graph defined by `(visible ?w1 ?w2)` is undirected and traversable
      by any rover where `(can_traverse ?r ?w1 ?w2)` exists (we assume `can_traverse`
      mirrors `visible` for simplicity). Shortest paths are precomputed.
    - Store capacity and dropping samples are ignored for soil/rock sampling costs.
    - Camera calibration state is partially considered: calibration is needed before
      taking an image, and taking an image uncalibrates. However, the heuristic
      simplifies by assuming a rover equipped for imaging can perform calibration
      and imaging steps if it can reach the necessary waypoints, without tracking
      specific camera calibration states across multiple image goals.
    - The heuristic sums costs for goals independently, ignoring potential synergies
      (e.g., collecting multiple samples at one waypoint, communicating multiple
      data items from one communication point).
    - It assumes all necessary static conditions (like sample presence, objective visibility,
      rover equipment, camera presence/support/target) are met if they existed in the
      initial state and are required by a goal. Unreachable goals due to missing
      static conditions might result in infinite heuristic values.

    # Heuristic Initialization
    - Parses static facts and initial state to identify objects and build relevant data structures.
    - Builds the waypoint graph based on `(visible ...)` facts.
    - Computes all-pairs shortest paths between waypoints.
    - Identifies lander locations, communication waypoints, initial sample locations,
      objective/target visibility, rover capabilities, and camera details.
    - Stores the set of goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not yet true in the current state.
    2. For each unachieved goal fact, estimate the minimum actions required to achieve it:
       - **Communicated Soil/Rock Data (W):**
         - If any rover currently has `(have_X_analysis R W)`:
           - Cost = 1 (communicate) + minimum navigation cost for any rover with the data to reach any communication waypoint.
         - If no rover has the data:
           - Cost = 1 (sample) + 1 (communicate) + minimum navigation cost for any equipped rover to reach waypoint W, then from W to any communication waypoint. (Assumes `(at_X_sample W)` was true initially).
       - **Communicated Image Data (O M):**
         - If any rover currently has `(have_image R O M)`:
           - Cost = 1 (communicate) + minimum navigation cost for any rover with the image to reach any communication waypoint.
         - If no rover has the data:
           - Find rovers equipped for imaging with a camera supporting mode M and having a calibration target T.
           - Find waypoints visible from T (`Vis_from_T`) and waypoints visible from O (`Vis_from_O`).
           - Cost = 1 (calibrate) + 1 (take_image) + 1 (communicate) + minimum navigation cost for a suitable rover to travel from its current location to a waypoint in `Vis_from_T`, then to a waypoint in `Vis_from_O`, then to a waypoint in `Comm_WPs`. This navigation cost is estimated as `dist(R_loc, Vis_from_T) + dist(Vis_from_T, Vis_from_O) + dist(Vis_from_O, Comm_WPs)`.
    3. The total heuristic value is the sum of the estimated costs for all unachieved goal facts.
    4. If any required navigation step is impossible (e.g., no path exists, or required waypoints don't exist/are not visible), the cost for that goal component is infinite, making the total heuristic infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Precompute Static Information ---

        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        # Extract objects by type from relevant predicates in static facts, initial state, and goals
        # This is a simplified approach assuming typical predicate structures.
        all_relevant_facts = set(static_facts) | set(self.goals) | set(initial_state)
        for fact_str in all_relevant_facts:
             parts = get_parts(fact_str)
             if not parts: continue # Skip empty facts if any

             predicate = parts[0]
             # Waypoints appear in many predicates
             if len(parts) > 1:
                 if predicate in ['at', 'at_lander', 'can_traverse', 'visible', 'at_soil_sample', 'at_rock_sample', 'visible_from', 'have_soil_analysis', 'have_rock_analysis', 'communicated_soil_data', 'communicated_rock_data']:
                     for i in range(1, len(parts)):
                         if parts[i] != '*': self.waypoints.add(parts[i])

             # Rovers
             if len(parts) > 1:
                 if predicate in ['at', 'can_traverse', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'store_of', 'on_board', 'calibrated', 'have_soil_analysis', 'have_rock_analysis', 'have_image']:
                     self.rovers.add(parts[1])

             # Landers
             if predicate == 'at_lander' and len(parts) > 1: self.landers.add(parts[1])

             # Stores
             if predicate in ['store_of', 'empty', 'full'] and len(parts) > 1: self.stores.add(parts[1])

             # Cameras
             if predicate in ['on_board', 'calibrated', 'supports', 'calibration_target'] and len(parts) > 1: self.cameras.add(parts[1])

             # Modes
             if predicate in ['supports', 'have_image', 'communicated_image_data'] and len(parts) > 2: self.modes.add(parts[2])

             # Objectives
             if predicate in ['calibration_target', 'visible_from', 'have_image', 'communicated_image_data']:
                 if len(parts) > 1: self.objectives.add(parts[1])
                 if len(parts) > 2: self.objectives.add(parts[2]) # Calibration target is an objective


        # Build waypoint graph from (visible ?w1 ?w2)
        self.graph = {wp: set() for wp in self.waypoints}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                _, w1, w2 = get_parts(fact)
                if w1 in self.graph and w2 in self.graph: # Ensure waypoints are known
                    self.graph[w1].add(w2)
                    self.graph[w2].add(w1) # Assume symmetric visibility implies symmetric traversability

        # Compute all-pairs shortest paths using Floyd-Warshall
        wp_list = list(self.waypoints)
        num_wps = len(wp_list)
        wp_to_idx = {wp: i for i, wp in enumerate(wp_list)}
        self.distances = [[float('inf')] * num_wps for _ in range(num_wps)]

        for i in range(num_wps):
            self.distances[i][i] = 0

        for w1, neighbors in self.graph.items():
            if w1 in wp_to_idx: # Ensure waypoint is in our list
                i = wp_to_idx[w1]
                for w2 in neighbors:
                    if w2 in wp_to_idx: # Ensure neighbor is in our list
                        j = wp_to_idx[w2]
                        self.distances[i][j] = 1

        for k in range(num_wps):
            for i in range(num_wps):
                for j in range(num_wps):
                    self.distances[i][j] = min(self.distances[i][j], self.distances[i][k] + self.distances[k][j])

        # Store lander locations
        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}

        # Identify communication waypoints (visible from any lander location)
        self.comm_wps = set()
        for lander_loc in self.lander_locations:
             if lander_loc in self.graph: # Ensure lander location is a known waypoint
                self.comm_wps.update(self.graph[lander_loc])

        # Store sample locations (initial state)
        self.soil_wps = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_soil_sample", "*")}
        self.rock_wps = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_rock_sample", "*")}


        # Store objective/target visibility
        self.obj_visible_from = {}
        self.target_visible_from = {} # Calibration targets are objectives
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                _, obj_or_target, wp = get_parts(fact)
                # Add to obj_visible_from if it's a known objective
                if obj_or_target in self.objectives:
                     self.obj_visible_from.setdefault(obj_or_target, set()).add(wp)


        # Store rover capabilities
        self.rover_capabilities = {r: set() for r in self.rovers}
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]].add('soil')
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rover_capabilities[get_parts(fact)[1]].add('rock')
            elif match(fact, "equipped_for_imaging", "*"):
                self.rover_capabilities[get_parts(fact)[1]].add('imaging')

        # Store camera details
        self.camera_info = {} # camera -> { 'rover': rover, 'target': target, 'modes': set() }
        # First pass to find rover and target
        for fact in static_facts:
             if match(fact, "on_board", "*", "*"):
                 _, camera, rover = get_parts(fact)
                 self.camera_info.setdefault(camera, {'rover': None, 'target': None, 'modes': set()})['rover'] = rover
             elif match(fact, "calibration_target", "*", "*"):
                 _, camera, target = get_parts(fact)
                 self.camera_info.setdefault(camera, {'rover': None, 'target': None, 'modes': set()})['target'] = target
                 # Add target visibility to target_visible_from, using obj_visible_from data
                 if target in self.obj_visible_from:
                      self.target_visible_from[target] = self.obj_visible_from[target]


        # Second pass to find supported modes
        for fact in static_facts:
             if match(fact, "supports", "*", "*"):
                 _, camera, mode = get_parts(fact)
                 if camera in self.camera_info:
                     self.camera_info[camera]['modes'].add(mode)

        # Store goal facts
        self.goal_facts = set(self.goals)

        # Store waypoint index mapping for distance lookup
        self._wp_to_idx = wp_to_idx
        self._wp_list = wp_list


    def _get_dist(self, wp1, wp2):
        """Get shortest distance between two waypoints."""
        if wp1 not in self._wp_to_idx or wp2 not in self._wp_to_idx:
            return float('inf')
        idx1 = self._wp_to_idx[wp1]
        idx2 = self._wp_to_idx[wp2]
        return self.distances[idx1][idx2]

    def _dist_to_set(self, wp, wp_set):
        """Get minimum distance from a waypoint to any waypoint in a set."""
        if not wp_set or wp not in self._wp_to_idx:
            return float('inf')
        min_d = float('inf')
        for target_wp in wp_set:
             d = self._get_dist(wp, target_wp)
             if d != float('inf'):
                  min_d = min(min_d, d)
        return min_d

    def _dist_set_to_set(self, set1, set2):
        """Get minimum distance between any waypoint in set1 and any in set2."""
        if not set1 or not set2:
            return float('inf')
        min_d = float('inf')
        for wp1 in set1:
            for wp2 in set2:
                 d = self._get_dist(wp1, wp2)
                 if d != float('inf'):
                      min_d = min(min_d, d)
        return min_d


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_cost = 0
        current_rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, rover, wp = get_parts(fact)
                current_rover_locations[rover] = wp

        # Iterate through unachieved goal facts
        unachieved_goals = self.goal_facts - state

        for goal in unachieved_goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint_goal = parts[1]
                # Check if data exists
                rovers_with_data = {r for r in self.rovers if f'(have_soil_analysis {r} {waypoint_goal})' in state}

                if rovers_with_data:
                    # Data exists, need to communicate
                    min_nav_to_comm = float('inf')
                    for r in rovers_with_data:
                        if r in current_rover_locations:
                            r_loc = current_rover_locations[r]
                            min_nav_to_comm = min(min_nav_to_comm, self._dist_to_set(r_loc, self.comm_wps))

                    if min_nav_to_comm != float('inf'):
                         total_cost += 1 + min_nav_to_comm # 1 for communicate + navigation
                    else:
                         # Cannot reach communication point from any rover with data
                         total_cost += float('inf') # This goal component is likely impossible from this state
                         continue # Move to next goal

                else:
                    # Data does not exist, need to sample and communicate
                    if waypoint_goal not in self.soil_wps:
                         # No sample exists at this waypoint initially, goal is impossible
                         total_cost += float('inf')
                         continue # Move to next goal

                    min_total_goal_cost = float('inf')
                    equipped_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}

                    for r in equipped_rovers:
                        if r in current_rover_locations:
                            r_loc = current_rover_locations[r]
                            nav_to_sample = self._get_dist(r_loc, waypoint_goal)
                            nav_sample_to_comm = self._dist_to_set(waypoint_goal, self.comm_wps)

                            if nav_to_sample != float('inf') and nav_sample_to_comm != float('inf'):
                                min_total_goal_cost = min(min_total_goal_cost, nav_to_sample + 1 + nav_sample_to_comm + 1) # +1 sample, +1 communicate

                    total_cost += min_total_goal_cost # Add the minimum cost over suitable rovers


            elif predicate == "communicated_rock_data":
                waypoint_goal = parts[1]
                # Check if data exists
                rovers_with_data = {r for r in self.rovers if f'(have_rock_analysis {r} {waypoint_goal})' in state}

                if rovers_with_data:
                    # Data exists, need to communicate
                    min_nav_to_comm = float('inf')
                    for r in rovers_with_data:
                        if r in current_rover_locations:
                            r_loc = current_rover_locations[r]
                            min_nav_to_comm = min(min_nav_to_comm, self._dist_to_set(r_loc, self.comm_wps))

                    if min_nav_to_comm != float('inf'):
                         total_cost += 1 + min_nav_to_comm # 1 for communicate + navigation
                    else:
                         total_cost += float('inf')
                         continue

                else:
                    # Data does not exist, need to sample and communicate
                    if waypoint_goal not in self.rock_wps:
                         total_cost += float('inf')
                         continue

                    min_total_goal_cost = float('inf')
                    equipped_rovers = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}

                    for r in equipped_rovers:
                        if r in current_rover_locations:
                            r_loc = current_rover_locations[r]
                            nav_to_sample = self._get_dist(r_loc, waypoint_goal)
                            nav_sample_to_comm = self._dist_to_set(waypoint_goal, self.comm_wps)

                            if nav_to_sample != float('inf') and nav_sample_to_comm != float('inf'):
                                min_total_goal_cost = min(min_total_goal_cost, nav_to_sample + 1 + nav_sample_to_comm + 1) # +1 sample, +1 communicate

                    total_cost += min_total_goal_cost


            elif predicate == "communicated_image_data":
                objective_goal = parts[1]
                mode_goal = parts[2]

                # Check if data exists
                rovers_with_data = {r for r in self.rovers if f'(have_image {r} {objective_goal} {mode_goal})' in state}

                if rovers_with_data:
                    # Data exists, need to communicate
                    min_nav_to_comm = float('inf')
                    for r in rovers_with_data:
                        if r in current_rover_locations:
                            r_loc = current_rover_locations[r]
                            min_nav_to_comm = min(min_nav_to_comm, self._dist_to_set(r_loc, self.comm_wps))

                    if min_nav_to_comm != float('inf'):
                         total_cost += 1 + min_nav_to_comm # 1 for communicate + navigation
                    else:
                         total_cost += float('inf')
                         continue

                else:
                    # Data does not exist, need to calibrate, take image, and communicate
                    min_total_goal_cost = float('inf')
                    equipped_imaging_rovers = {r for r, caps in self.rover_capabilities.items() if 'imaging' in caps}

                    for r in equipped_imaging_rovers:
                        if r in current_rover_locations:
                            r_loc = current_rover_locations[r]

                            # Find cameras on this rover supporting the mode and having a target
                            suitable_cameras = [
                                cam for cam, info in self.camera_info.items()
                                if info['rover'] == r and mode_goal in info['modes'] and info['target'] is not None
                            ]

                            for cam in suitable_cameras:
                                target = self.camera_info[cam]['target']
                                w_cal_set = self.target_visible_from.get(target, set())
                                w_img_set = self.obj_visible_from.get(objective_goal, set())

                                if w_cal_set and w_img_set:
                                    # Estimate navigation: R_loc -> W_cal -> W_img -> Comm_WP
                                    nav_r_to_cal = self._dist_to_set(r_loc, w_cal_set)
                                    nav_cal_to_img = self._dist_set_to_set(w_cal_set, w_img_set)
                                    nav_img_to_comm = self._dist_set_to_set(w_img_set, self.comm_wps)

                                    if nav_r_to_cal != float('inf') and nav_cal_to_img != float('inf') and nav_img_to_comm != float('inf'):
                                         # +1 calibrate, +1 take_image, +1 communicate
                                         min_total_goal_cost = min(min_total_goal_cost, nav_r_to_cal + 1 + nav_cal_to_img + 1 + nav_img_to_comm + 1)

                    total_cost += min_total_goal_cost

            # If total_cost became infinity for any goal, the overall heuristic is infinity
            if total_cost == float('inf'):
                 return float('inf')


        return total_cost
