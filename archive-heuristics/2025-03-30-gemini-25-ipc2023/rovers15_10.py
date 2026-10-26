import itertools
import math
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts against patterns
def match(fact_parts, pattern):
    """
    Check if a parsed PDDL fact matches a given pattern.
    - `fact_parts`: The components of the fact (list of strings).
    - `pattern`: The expected pattern (list of strings, wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    if len(fact_parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all goals
    from the current state. It calculates the estimated cost for each unachieved goal
    individually and sums these costs. The cost for each goal is estimated by finding
    the best-suited rover (and camera, if applicable) and calculating the sequence
    of actions (navigation, sampling/calibration/imaging, communication) needed.
    Navigation costs are based on precomputed shortest paths.

    # Assumptions
    - The heuristic assumes rovers can operate somewhat independently, although it tries
      to select the "best" rover for each task based on current state and proximity.
    - It ignores potential negative interactions like resource contention (e.g., only
      one rover can sample a specific location if the sample is consumed) or the
      need for repeated calibration if one rover takes multiple images.
    - It assumes that if `can_traverse(r, wp1, wp2)` exists, the `visible(wp1, wp2)`
      precondition for navigation between them is generally satisfiable or requires
      minimal detour (though the BFS for pathfinding *does* check visibility).
    - Stores are simple (empty/full), and a `drop` action (cost 1) is assumed needed
      before sampling if the rover's store is full.
    - If a goal seems impossible (e.g., no equipped rover, unreachable location),
      a large penalty is added.

    # Heuristic Initialization
    - Parses static facts to build knowledge about the map, rover capabilities,
      camera properties, visibility, traversability, lander location, etc.
    - Precomputes all-pairs shortest paths for each rover between waypoints using BFS,
      considering both `can_traverse` and `visible` predicates. Stores these distances.
    - Identifies waypoints visible from the lander for communication.
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract dynamic information like rover locations,
        store status, camera calibration, held samples/images.
    2.  **Iterate Through Goals:** For each goal predicate in the task's goals:
        a.  Check if the goal is already satisfied in the current state. If yes, cost is 0 for this goal.
        b.  If not satisfied, determine the goal type (communicate soil, rock, or image).
        c.  **Find Best Rover/Camera:** Identify all rovers (and cameras, for imaging goals)
            capable of achieving this goal (based on equipment, camera support).
        d.  **Estimate Cost per Candidate:** For each capable rover/camera combination:
            i.   **Data Acquisition Cost:** Estimate the cost to acquire the necessary data
                 (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
                 - Check if data is already held by the rover.
                 - If not, estimate cost for necessary prerequisites:
                     - **Sampling:** Check store status (add 1 for `drop` if full). Calculate
                       navigation cost to sample location. Add 1 for `sample_*` action.
                     - **Imaging:** Check calibration status. If needed, calculate cost to
                       navigate to a calibration waypoint, add 1 for `calibrate`. Calculate
                       navigation cost to imaging waypoint. Add 1 for `take_image`.
            ii.  **Communication Cost:** Estimate the cost to communicate the data.
                 - Find the closest waypoint `cp` visible to the lander from the rover's
                   (potentially hypothetical) location after data acquisition.
                 - Calculate navigation cost to `cp`.
                 - Add 1 for `communicate_*` action.
            iii. **Total Cost for Candidate:** Sum of acquisition and communication costs.
        e.  **Select Minimum Cost:** Choose the minimum estimated cost among all capable
            candidates for this goal. If no candidate can achieve it, assign a large penalty.
        f.  **Add to Total:** Add the minimum cost for this goal to the overall heuristic value `h`.
    3.  **Return Total Heuristic Value:** Return the final sum `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures for Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_wp = None

        self.rover_equipment_soil = set()
        self.rover_equipment_rock = set()
        self.rover_equipment_imaging = set()
        self.rover_stores = {} # rover -> store
        self.store_rovers = {} # store -> rover

        self.visibility = set() # (wp1, wp2)
        self.traversability = {} # rover -> set((wp1, wp2))
        self.visible_from = {} # objective -> set(waypoint)
        self.calibration_targets = {} # camera -> objective
        self.camera_on_board = {} # camera -> rover
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_supports = {} # camera -> set(mode)
        self.soil_at = set() # waypoint
        self.rock_at = set() # waypoint

        # --- Parse Static Facts ---
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            # Object types (implicit from usage) - Collect all mentioned objects
            for obj in parts[1:]:
                 # Basic type inference (can be improved if types were explicit)
                 if 'rover' in obj: self.rovers.add(obj)
                 elif 'waypoint' in obj: self.waypoints.add(obj)
                 elif 'camera' in obj: self.cameras.add(obj)
                 elif 'objective' in obj: self.objectives.add(obj)
                 elif 'mode' in obj: self.modes.add(obj) # Modes are often keywords like 'colour'
                 elif 'store' in obj: self.stores.add(obj)
                 elif 'general' in obj: self.lander = obj # Assuming 'general' is the lander name based on examples

            # Predicates
            if pred == "at_lander" and len(parts) == 3:
                self.lander, self.lander_wp = parts[1], parts[2]
                self.waypoints.add(self.lander_wp)
            elif pred == "equipped_for_soil_analysis" and len(parts) == 2:
                self.rover_equipment_soil.add(parts[1])
                self.rovers.add(parts[1])
            elif pred == "equipped_for_rock_analysis" and len(parts) == 2:
                self.rover_equipment_rock.add(parts[1])
                self.rovers.add(parts[1])
            elif pred == "equipped_for_imaging" and len(parts) == 2:
                self.rover_equipment_imaging.add(parts[1])
                self.rovers.add(parts[1])
            elif pred == "store_of" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                self.store_rovers[store] = rover
                self.stores.add(store)
                self.rovers.add(rover)
            elif pred == "visible" and len(parts) == 3:
                self.visibility.add(tuple(parts[1:]))
                self.waypoints.add(parts[1])
                self.waypoints.add(parts[2])
            elif pred == "can_traverse" and len(parts) == 4:
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.traversability:
                    self.traversability[rover] = set()
                self.traversability[rover].add((wp1, wp2))
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif pred == "calibration_target" and len(parts) == 3:
                cam, obj = parts[1], parts[2]
                self.calibration_targets[cam] = obj
                self.cameras.add(cam)
                self.objectives.add(obj)
            elif pred == "on_board" and len(parts) == 3:
                cam, rover = parts[1], parts[2]
                self.camera_on_board[cam] = rover
                if rover not in self.rover_cameras:
                    self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(cam)
                self.cameras.add(cam)
                self.rovers.add(rover)
            elif pred == "supports" and len(parts) == 3:
                cam, mode = parts[1], parts[2]
                if cam not in self.camera_supports:
                    self.camera_supports[cam] = set()
                self.camera_supports[cam].add(mode)
                self.cameras.add(cam)
                self.modes.add(mode)
            elif pred == "visible_from" and len(parts) == 3:
                obj, wp = parts[1], parts[2]
                if obj not in self.visible_from:
                    self.visible_from[obj] = set()
                self.visible_from[obj].add(wp)
                self.objectives.add(obj)
                self.waypoints.add(wp)
            elif pred == "at_soil_sample" and len(parts) == 2: # Initial locations
                 self.soil_at.add(parts[1])
                 self.waypoints.add(parts[1])
            elif pred == "at_rock_sample" and len(parts) == 2: # Initial locations
                 self.rock_at.add(parts[1])
                 self.waypoints.add(parts[1])

        # --- Precompute Shortest Paths using BFS ---
        self.distances = {} # rover -> start_wp -> end_wp -> distance
        for r in self.rovers:
            self.distances[r] = {}
            # Build rover-specific navigation graph
            rover_graph = {wp: [] for wp in self.waypoints}
            if r in self.traversability:
                for wp1, wp2 in self.traversability[r]:
                    # Navigation requires visibility from source to destination
                    if (wp1, wp2) in self.visibility:
                        rover_graph[wp1].append(wp2)

            # Run BFS from each waypoint
            for start_wp in self.waypoints:
                q = deque([(start_wp, 0)])
                dist = {wp: float('inf') for wp in self.waypoints}
                dist[start_wp] = 0
                visited = {start_wp}

                while q:
                    curr_wp, d = q.popleft()
                    for neighbor in rover_graph.get(curr_wp, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dist[neighbor] = d + 1
                            q.append((neighbor, d + 1))
                self.distances[r][start_wp] = dist

        # --- Find Communication Points ---
        self.comm_points = set()
        if self.lander_wp:
            for wp1, wp2 in self.visibility:
                if wp2 == self.lander_wp:
                    self.comm_points.add(wp1)
        # print(f"DEBUG: Lander at {self.lander_wp}, Comm points: {self.comm_points}")


    def _get_dist(self, rover, wp_start, wp_end):
        """Safely get distance, returning infinity if unreachable or args invalid."""
        if rover not in self.distances or wp_start not in self.distances[rover] or wp_end not in self.distances[rover][wp_start]:
             # print(f"WARN: Distance lookup failed for {rover} from {wp_start} to {wp_end}")
             return float('inf')
        return self.distances[rover][wp_start][wp_end]

    def _find_closest_target(self, rover, current_wp, target_wps):
        """Find the closest waypoint in target_wps from current_wp for a rover."""
        min_dist = float('inf')
        closest_wp = None
        if not target_wps:
             return None, float('inf')

        for target_wp in target_wps:
            dist = self._get_dist(rover, current_wp, target_wp)
            if dist < min_dist:
                min_dist = dist
                closest_wp = target_wp

        return closest_wp, min_dist


    def __call__(self, node):
        state = node.state
        h = 0

        # --- Parse Current State ---
        rover_locs = {}
        store_status = {} # store -> 'empty' or 'full'
        calibrated_cameras = set() # camera
        have_soil = {} # rover -> set(waypoint)
        have_rock = {} # rover -> set(waypoint)
        have_image = {} # rover -> objective -> set(mode)
        current_soil_at = set() # waypoint
        current_rock_at = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "at" and len(parts) == 3:
                rover_locs[parts[1]] = parts[2]
            elif pred == "empty" and len(parts) == 2:
                store_status[parts[1]] = 'empty'
            elif pred == "full" and len(parts) == 2:
                store_status[parts[1]] = 'full'
            elif pred == "calibrated" and len(parts) == 3:
                calibrated_cameras.add(parts[1])
            elif pred == "have_soil_analysis" and len(parts) == 3:
                r, wp = parts[1], parts[2]
                if r not in have_soil: have_soil[r] = set()
                have_soil[r].add(wp)
            elif pred == "have_rock_analysis" and len(parts) == 3:
                r, wp = parts[1], parts[2]
                if r not in have_rock: have_rock[r] = set()
                have_rock[r].add(wp)
            elif pred == "have_image" and len(parts) == 4:
                r, o, m = parts[1], parts[2], parts[3]
                if r not in have_image: have_image[r] = {}
                if o not in have_image[r]: have_image[r][o] = set()
                have_image[r][o].add(m)
            elif pred == "at_soil_sample" and len(parts) == 2:
                 current_soil_at.add(parts[1])
            elif pred == "at_rock_sample" and len(parts) == 2:
                 current_rock_at.add(parts[1])


        # --- Evaluate Goals ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal_fact)
            goal_pred = goal_parts[0]
            min_cost_for_goal = float('inf')

            # --- Case 1: communicated_soil_data ---
            if goal_pred == "communicated_soil_data" and len(goal_parts) == 2:
                wp_target = goal_parts[1]
                # Find best rover
                for r in self.rover_equipment_soil:
                    if r not in rover_locs: continue # Rover not in current state? Skip.
                    current_rover_loc = rover_locs[r]
                    cost_rover = 0
                    hypothetical_loc = current_rover_loc

                    # 1. Acquire data if needed
                    if not have_soil.get(r, set()).__contains__(wp_target):
                        # Check store
                        rover_store = self.rover_stores.get(r)
                        if rover_store and store_status.get(rover_store) == 'full':
                            cost_rover += 1 # Drop action

                        # Navigate to sample
                        dist_to_sample = self._get_dist(r, current_rover_loc, wp_target)
                        if dist_to_sample == float('inf'): continue # Cannot reach sample
                        cost_rover += dist_to_sample
                        cost_rover += 1 # Sample action
                        hypothetical_loc = wp_target # Rover is now at sample location

                    # 2. Communicate data
                    closest_comm_point, dist_to_comm = self._find_closest_target(r, hypothetical_loc, self.comm_points)
                    if dist_to_comm == float('inf'): continue # Cannot reach comm point
                    cost_rover += dist_to_comm
                    cost_rover += 1 # Communicate action

                    min_cost_for_goal = min(min_cost_for_goal, cost_rover)

            # --- Case 2: communicated_rock_data ---
            elif goal_pred == "communicated_rock_data" and len(goal_parts) == 2:
                wp_target = goal_parts[1]
                # Find best rover
                for r in self.rover_equipment_rock:
                    if r not in rover_locs: continue
                    current_rover_loc = rover_locs[r]
                    cost_rover = 0
                    hypothetical_loc = current_rover_loc

                    # 1. Acquire data if needed
                    if not have_rock.get(r, set()).__contains__(wp_target):
                        # Check store
                        rover_store = self.rover_stores.get(r)
                        if rover_store and store_status.get(rover_store) == 'full':
                            cost_rover += 1 # Drop action

                        # Navigate to sample
                        dist_to_sample = self._get_dist(r, current_rover_loc, wp_target)
                        if dist_to_sample == float('inf'): continue
                        cost_rover += dist_to_sample
                        cost_rover += 1 # Sample action
                        hypothetical_loc = wp_target

                    # 2. Communicate data
                    closest_comm_point, dist_to_comm = self._find_closest_target(r, hypothetical_loc, self.comm_points)
                    if dist_to_comm == float('inf'): continue
                    cost_rover += dist_to_comm
                    cost_rover += 1 # Communicate action

                    min_cost_for_goal = min(min_cost_for_goal, cost_rover)

            # --- Case 3: communicated_image_data ---
            elif goal_pred == "communicated_image_data" and len(goal_parts) == 4:
                o_target, m_target = goal_parts[2], goal_parts[3]
                # Find best rover and camera
                for r in self.rover_equipment_imaging:
                    if r not in rover_locs: continue
                    current_rover_loc = rover_locs[r]

                    for cam in self.rover_cameras.get(r, set()):
                        # Check if camera supports mode
                        if m_target not in self.camera_supports.get(cam, set()):
                            continue

                        cost_rover_cam = 0
                        hypothetical_loc = current_rover_loc
                        is_cam_calibrated = cam in calibrated_cameras

                        # 1. Acquire image if needed
                        if not have_image.get(r, {}).get(o_target, set()).__contains__(m_target):
                            # 1a. Calibrate if needed
                            if not is_cam_calibrated:
                                cal_obj = self.calibration_targets.get(cam)
                                if not cal_obj: continue # No calibration target for this camera

                                # Find waypoints where calibration target is visible
                                cal_wps = self.visible_from.get(cal_obj, set())
                                if not cal_wps: continue # Cannot calibrate this camera

                                # Find closest calibration waypoint
                                closest_cal_wp, dist_to_cal = self._find_closest_target(r, hypothetical_loc, cal_wps)
                                if dist_to_cal == float('inf'): continue # Cannot reach calibration spot
                                cost_rover_cam += dist_to_cal
                                cost_rover_cam += 1 # Calibrate action
                                hypothetical_loc = closest_cal_wp
                                # Assume calibration successful for rest of this goal's estimate

                            # 1b. Take image
                            # Find waypoints where objective is visible
                            img_wps = self.visible_from.get(o_target, set())
                            if not img_wps: continue # Cannot see objective from anywhere

                            # Find closest imaging waypoint
                            closest_img_wp, dist_to_img = self._find_closest_target(r, hypothetical_loc, img_wps)
                            if dist_to_img == float('inf'): continue # Cannot reach imaging spot
                            cost_rover_cam += dist_to_img
                            cost_rover_cam += 1 # Take_image action
                            hypothetical_loc = closest_img_wp
                            # Assume take_image successful, camera becomes uncalibrated (ignored for next goals)

                        # 2. Communicate image
                        closest_comm_point, dist_to_comm = self._find_closest_target(r, hypothetical_loc, self.comm_points)
                        if dist_to_comm == float('inf'): continue # Cannot reach comm point
                        cost_rover_cam += dist_to_comm
                        cost_rover_cam += 1 # Communicate action

                        min_cost_for_goal = min(min_cost_for_goal, cost_rover_cam)


            # Add cost for this goal (or penalty if unreachable)
            if min_cost_for_goal == float('inf'):
                # print(f"WARN: Goal {goal_fact} seems unreachable from state.")
                h += 1000 # Large penalty for potentially unreachable goals
            else:
                h += min_cost_for_goal

        # Ensure heuristic is 0 only for goal states
        if h == 0 and not self.goals <= state:
             # This can happen if all goals individually seem achievable with 0 cost
             # but the combined state isn't the goal (e.g., initial state satisfies some parts).
             # Or if goals are contradictory / impossible. Add a small value.
             # A better approach might be to count unsatisfied goals.
             num_unsatisfied = sum(1 for g in self.goals if g not in state)
             return num_unsatisfied # Return count of unsatisfied goals if calculated sum is 0 but not goal

        return h

