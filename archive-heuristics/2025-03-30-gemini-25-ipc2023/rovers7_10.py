import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq # Not used in current implementation, but potentially useful for extensions


# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
    """
    return fact[1:-1].split()

# Helper function to match facts - takes already split parts
def match(fact_parts, *pattern):
    """
    Checks if parts of a fact match a pattern (with '*' wildcards).
    Example: match(["at", "rover1", "wp1"], "at", "rover*", "*") -> True
    """
    return len(fact_parts) == len(pattern) and all(
        fnmatch(part, pat) for part, pat in zip(fact_parts, pattern)
    )


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for achieving each unsatisfied goal predicate independently.
    The cost for each goal considers the necessary steps like navigation, sampling/imaging,
    calibration, and communication. It prioritizes using rovers that are already equipped
    or closer to the required locations, but uses a simplified navigation cost (0 or 1 per
    required move segment). It is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - Each goal (`communicated_...`) can be achieved independently, ignoring positive interactions
      (e.g., a single navigation action helping multiple goals) but implicitly considering some
      negative ones (e.g., needing to re-calibrate after taking an image).
    - Navigation cost between required locations (e.g., current rover location to sample site,
      sample site to communication site) is estimated as 1 action if the rover is not already
      at a suitable location, and 0 otherwise. This avoids complex pathfinding.
    - If a required sample (`at_soil_sample`, `at_rock_sample`) is not present in the current state,
      but the corresponding `communicated_...` goal is not met, the heuristic assumes some rover
      already holds the data (`have_..._analysis` or `have_image`). This is a relaxation to avoid
      infinite estimates when samples are depleted but goals remain.
    - When multiple rovers could achieve a goal, the heuristic selects the one estimated to have
      the lowest cost based on current location, equipment, and state (e.g., store empty, camera calibrated).
      This selection is greedy and local to the specific goal being evaluated.
    - Dropping samples to free a store costs 1 action if needed before sampling.
    - The heuristic returns a large penalty (1000) for goals deemed impossible to achieve from the
      current state (e.g., no equipped rover, objective not visible from any waypoint), rather than
      infinity, to allow comparisons between states with different impossible goals.

    # Heuristic Initialization
    - Parses static facts from the task definition (`task.static`) to build knowledge about the environment:
        - Lander location (`at_lander`).
        - Waypoint-to-waypoint visibility (`visible`), used for navigation and communication checks.
        - Waypoints within communication range of the lander (derived from `visible` and `at_lander`).
        - Objective visibility from waypoints (`visible_from`), used for imaging and calibration.
        - Rover capabilities (`equipped_for_...`).
        - Rover-store associations (`store_of`).
        - Rover-camera associations (`on_board`).
        - Camera capabilities (`supports`).
        - Camera calibration targets (`calibration_target`) and the waypoints they are visible from
          (derived from `calibration_target` and `visible_from`).
    - Stores the goal predicates defined in the task (`task.goals`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check Goal Completion: If the current state (`node.state`) satisfies all goals (`self.goals`),
       the heuristic value is 0.
    2. Extract Dynamic State: Parse the current state (`node.state`) to get current rover locations (`at`),
       store statuses (`empty`/`full`), camera calibration statuses (`calibrated`), which rovers hold which
       data/images (`have_...`), and locations of available samples (`at_..._sample`). Store this in a
       `dynamic_info` dictionary.
    3. Identify Unsatisfied Goals: Find the goal predicates in `self.goals` that are not present in the
       current state `node.state`.
    4. Iterate Through Unsatisfied Goals: For each unsatisfied `communicated_...` goal:
       a. Estimate Cost: Call a helper function `_estimate_cost(goal, state, dynamic_info)` to determine
          the minimum number of actions needed to achieve this single goal from the current state.
       b. Cost Estimation Logic (`_estimate_cost`):
          i. Check if Data Already Held: See if any rover `r` already has the required `have_..._analysis`
             or `have_image` fact in the `dynamic_info`.
             - If yes: Estimate cost = (navigate to communication range) + (communicate action).
               Navigation cost is 1 if rover `r` is not currently at a waypoint visible to the lander
               (`self.comm_waypoints`), 0 otherwise.
             - If no: Proceed to estimate the cost of acquiring the data.
          ii. Estimate Data Acquisition Cost (if data not held):
             - **Soil/Rock Sample (`communicated_soil_data`/`communicated_rock_data`):**
                - Check if the physical sample (`at_soil_sample`/`at_rock_sample`) exists at the target waypoint `wp` in `dynamic_info`.
                - If not: Apply relaxation - assume an equipped rover already holds the data and calculate cost as in step i. If no rover is equipped, return penalty.
                - If yes: Find the "best" rover `r` (must be equipped for the task and have a store). The "best" is the one yielding the minimum estimated cost below:
                   - Cost = 0
                   - Cost += 1 (drop) if `r`'s store is full (`dynamic_info`).
                   - Cost += 1 (navigate) if `r`'s current location (`dynamic_info`) is not `wp`.
                   - Cost += 1 (sample action).
                   - Cost += 1 (navigate) if sample waypoint `wp` is not in communication range (`self.comm_waypoints`).
                   - Cost += 1 (communicate action).
                   - Sum these costs. Iterate through all suitable rovers and take the minimum cost. If no suitable rover, return penalty.
             - **Image Data (`communicated_image_data`):**
                - Find the "best" rover `r` (must be equipped for imaging and have a camera `i` that supports the required `mode`). The "best" yields the minimum cost below:
                - Find a waypoint `w_img` from which the objective `obj` is visible (`self.objective_visibility`). If none, return penalty. (Simplification: picks the first such waypoint found).
                - Check if camera `i` is calibrated for rover `r` (`dynamic_info`).
                - Cost = 0
                - If Not Calibrated:
                    - Find a calibration target `t` for `i` (`self.camera_calibration_targets`) and a waypoint `w_cal` where `t` is visible (`self.calibration_target_visibility`). If none, continue to next camera/rover or return penalty if none work. (Simplification: picks the first `w_cal` found).
                    - Cost += 1 (navigate) if `r`'s current location (`dynamic_info`) is not `w_cal`.
                    - Cost += 1 (calibrate action).
                    - Cost += 1 (navigate) if `w_cal` is not `w_img`. (Move from calibration site to imaging site). Update hypothetical location to `w_img`.
                - If Calibrated:
                    - Cost += 1 (navigate) if `r`'s current location is not `w_img`. Update hypothetical location to `w_img`.
                - Cost += 1 (take_image action). (Camera `i` becomes uncalibrated after this). Hypothetical location is now `w_img`.
                - Cost += 1 (navigate) if `w_img` is not in communication range (`self.comm_waypoints`).
                - Cost += 1 (communicate action).
                - Sum these costs. Iterate through all suitable rover/camera pairs and take the minimum cost. If none suitable, return penalty.
       c. Accumulate Costs: Add the estimated cost (or penalty) for this goal to the `total_heuristic_cost`.
    5. Final Adjustment: If the `total_heuristic_cost` is 0 but goals remain unsatisfied (due to estimation quirks or relaxations), return 1 to ensure non-zero value for non-goal states.
    6. Return Total Cost: The sum of costs for all unsatisfied goals.
    """
    # Define a large penalty for seemingly impossible goals to avoid true infinity
    IMPOSSIBLE_GOAL_PENALTY = 1000

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute static information ---
        self.lander_location = None
        self.rover_equipment = {} # rover -> set{'soil', 'rock', 'imaging'}
        self.rover_stores = {} # rover -> store name
        self.rover_cameras = {} # rover -> set{camera name}
        self.camera_supports = {} # camera -> set{mode}
        self.camera_calibration_targets = {} # camera -> set{objective}
        self.objective_visibility = {} # objective -> set{waypoint}
        self.waypoint_visibility = {} # waypoint -> set{waypoint} # WPs visible FROM key WP
        self.calibration_target_visibility = {} # objective -> set{waypoint} # WPs where cal target is visible

        # Keep track of all known objects to ensure completeness
        all_rovers = set()
        all_waypoints = set()
        all_cameras = set()
        all_objectives = set()
        all_stores = set()
        all_modes = set()

        # Parse static facts once
        _static_predicates = {}
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            _static_predicates.setdefault(predicate, []).append(parts[1:])

            # Infer object types from usage for completeness checks
            if predicate == 'rover': all_rovers.add(parts[1])
            elif predicate == 'waypoint': all_waypoints.add(parts[1])
            elif predicate == 'camera': all_cameras.add(parts[1])
            elif predicate == 'objective': all_objectives.add(parts[1])
            elif predicate == 'store': all_stores.add(parts[1])
            elif predicate == 'mode': all_modes.add(parts[1]) # Although mode is a type, track instances if needed
            elif predicate == 'at_lander': self.lander_location = parts[2]; all_waypoints.add(parts[2])
            elif predicate == 'equipped_for_soil_analysis': all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis': all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging': all_rovers.add(parts[1])
            elif predicate == 'store_of': all_stores.add(parts[1]); all_rovers.add(parts[2])
            elif predicate == 'on_board': all_cameras.add(parts[1]); all_rovers.add(parts[2])
            elif predicate == 'supports': all_cameras.add(parts[1]); all_modes.add(parts[2])
            elif predicate == 'calibration_target': all_cameras.add(parts[1]); all_objectives.add(parts[2])
            elif predicate == 'visible_from': all_objectives.add(parts[1]); all_waypoints.add(parts[2])
            elif predicate == 'visible': all_waypoints.add(parts[1]); all_waypoints.add(parts[2])
            # 'can_traverse' is not directly used by this heuristic but helps identify objects

        # Populate data structures from parsed predicates
        for args in _static_predicates.get('equipped_for_soil_analysis', []):
            self.rover_equipment.setdefault(args[0], set()).add('soil')
        for args in _static_predicates.get('equipped_for_rock_analysis', []):
            self.rover_equipment.setdefault(args[0], set()).add('rock')
        for args in _static_predicates.get('equipped_for_imaging', []):
            self.rover_equipment.setdefault(args[0], set()).add('imaging')
        for args in _static_predicates.get('store_of', []):
            self.rover_stores[args[1]] = args[0] # Map rover -> store name
        for args in _static_predicates.get('on_board', []):
            self.rover_cameras.setdefault(args[1], set()).add(args[0]) # Map rover -> set{camera name}
        for args in _static_predicates.get('supports', []):
            self.camera_supports.setdefault(args[0], set()).add(args[1]) # Map camera -> set{mode}
        for args in _static_predicates.get('calibration_target', []):
            self.camera_calibration_targets.setdefault(args[0], set()).add(args[1]) # Map camera -> set{objective}
        for args in _static_predicates.get('visible_from', []):
            self.objective_visibility.setdefault(args[0], set()).add(args[1]) # Map objective -> set{waypoint}
        for args in _static_predicates.get('visible', []):
            self.waypoint_visibility.setdefault(args[0], set()).add(args[1]) # Map waypoint -> set{visible waypoint}

        # Derive calibration target visibility using objective visibility
        for cam, targets in self.camera_calibration_targets.items():
            for target_obj in targets:
                 visible_wps = self.objective_visibility.get(target_obj, set())
                 # Store which waypoints can see this specific calibration target
                 self.calibration_target_visibility.setdefault(target_obj, set()).update(visible_wps)

        # Ensure all known rovers have entries in equipment/cameras for safe lookups
        for r in all_rovers:
            self.rover_equipment.setdefault(r, set())
            self.rover_cameras.setdefault(r, set())
            # Note: We assume rover_stores is populated only for rovers that actually have stores via 'store_of'

        # Determine waypoints from which the lander is visible (for communication)
        self.comm_waypoints = set()
        if self.lander_location:
            # Check which waypoints have the lander's location in their visibility list
            for wp1, visible_wps in self.waypoint_visibility.items():
                if self.lander_location in visible_wps:
                    self.comm_waypoints.add(wp1)


    def _get_dynamic_info(self, state):
        """Parses the current state frozenset to extract dynamic information."""
        info = {
            'rover_locations': {}, # rover -> waypoint
            'store_status': {}, # rover -> 'empty' / 'full'
            'calibrated_cameras': set(), # set of (camera, rover) tuples
            'held_soil': {}, # rover -> set{waypoint}
            'held_rock': {}, # rover -> set{waypoint}
            'held_images': {}, # rover -> set{(objective, mode)}
            'available_soil': set(), # set{waypoint}
            'available_rock': set(), # set{waypoint}
        }
        # Pre-parse all facts in the state for efficient lookup
        current_facts_parsed = [get_parts(fact) for fact in state]

        for parts in current_facts_parsed:
            predicate = parts[0]

            # Use match for robustness, though direct indexing is often safe if structure is known
            if predicate == 'at' and len(parts) == 3 and parts[1] in self.rover_equipment: # Check if arg1 is a known rover
                info['rover_locations'][parts[1]] = parts[2]
            elif predicate == 'empty' and len(parts) == 2:
                store_name = parts[1]
                # Find rover associated with this store
                for r, s in self.rover_stores.items():
                    if s == store_name:
                        info['store_status'][r] = 'empty'
                        break
            elif predicate == 'full' and len(parts) == 2:
                 store_name = parts[1]
                 # Find rover associated with this store
                 for r, s in self.rover_stores.items():
                    if s == store_name:
                        info['store_status'][r] = 'full'
                        break
            elif predicate == 'calibrated' and len(parts) == 3:
                info['calibrated_cameras'].add((parts[1], parts[2])) # (camera, rover)
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                info['held_soil'].setdefault(parts[1], set()).add(parts[2])
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                info['held_rock'].setdefault(parts[1], set()).add(parts[2])
            elif predicate == 'have_image' and len(parts) == 4:
                info['held_images'].setdefault(parts[1], set()).add((parts[2], parts[3])) # rover -> set{(obj, mode)}
            elif predicate == 'at_soil_sample' and len(parts) == 2:
                info['available_soil'].add(parts[1])
            elif predicate == 'at_rock_sample' and len(parts) == 2:
                info['available_rock'].add(parts[1])

        # Assume stores are empty if not mentioned as full for rovers with known stores
        # This relies on the PDDL state representation (closed world for state predicates)
        for rover in self.rover_stores:
            if rover not in info['store_status']:
                 info['store_status'][rover] = 'empty'

        return info

    def _estimate_cost(self, goal, state, dynamic_info):
        """
        Estimates the cost (number of actions) to achieve a single goal predicate
        from the current state. Uses simplified navigation costs (0 or 1).
        Returns the estimated cost or IMPOSSIBLE_GOAL_PENALTY if deemed impossible.
        """
        goal_parts = get_parts(goal)
        goal_pred = goal_parts[0]
        cost = 0

        # All communication goals require knowing the lander location
        if not self.lander_location:
            return self.IMPOSSIBLE_GOAL_PENALTY

        # --- Case 1: communicated_soil_data(wp) ---
        if goal_pred == 'communicated_soil_data':
            wp = goal_parts[1]
            rover_holding = None
            # Check if any rover already holds the specific data
            for r, wps in dynamic_info['held_soil'].items():
                if wp in wps:
                    rover_holding = r
                    break

            if rover_holding:
                # Rover r already has the data, estimate communication cost
                r_loc = dynamic_info['rover_locations'].get(rover_holding)
                if r_loc is None: return 1 + 1 # Assume 1 nav + 1 comm if location unknown for some reason
                # Check if current location is suitable for communication
                if r_loc not in self.comm_waypoints:
                    cost += 1 # Navigate to communication range
                cost += 1 # Communicate action
                return cost
            else:
                # Data not held, need to sample and communicate
                # Check if sample is physically available at the waypoint
                if wp not in dynamic_info['available_soil']:
                     # Sample not available at location - Relaxation: assume it's held by *some* equipped rover
                     possible_rovers = [r for r, equip in self.rover_equipment.items() if 'soil' in equip]
                     if not possible_rovers: return self.IMPOSSIBLE_GOAL_PENALTY # No rover can do this task
                     # Estimate cost assuming one of them has it (pick first for simplicity)
                     r = possible_rovers[0]
                     r_loc = dynamic_info['rover_locations'].get(r)
                     if r_loc is None: return 1 + 1 # Assume 1 nav + 1 comm
                     if r_loc not in self.comm_waypoints: cost += 1 # Navigate
                     cost += 1 # Communicate
                     return cost

                # Sample is available, find the best rover to get it
                best_rover = None
                min_cost_rover = float('inf')

                for r, equip in self.rover_equipment.items():
                    # Rover must have equipment and an associated store
                    if 'soil' in equip and r in self.rover_stores:
                        current_rover_cost = 0
                        r_loc = dynamic_info['rover_locations'].get(r)
                        # Default store status to 'empty' if not explicitly known (from _get_dynamic_info)
                        r_store_status = dynamic_info['store_status'].get(r, 'empty')

                        if r_loc is None: continue # Cannot estimate cost if rover location unknown

                        # Cost to potentially empty store
                        if r_store_status == 'full':
                            current_rover_cost += 1 # Drop action

                        # Cost to navigate to sample site
                        if r_loc != wp:
                            current_rover_cost += 1 # Navigate action

                        # Cost of sampling
                        current_rover_cost += 1 # Sample action
                        # Hypothetical location after sampling is wp

                        # Cost to navigate from sample site to communication site
                        if wp not in self.comm_waypoints:
                            current_rover_cost += 1 # Navigate action

                        # Cost of communicating
                        current_rover_cost += 1 # Communicate action

                        # Check if this rover is better than the current best
                        if current_rover_cost < min_cost_rover:
                            min_cost_rover = current_rover_cost
                            best_rover = r

                # Return cost for the best rover found, or penalty if none could do it
                return min_cost_rover if best_rover else self.IMPOSSIBLE_GOAL_PENALTY

        # --- Case 2: communicated_rock_data(wp) ---
        elif goal_pred == 'communicated_rock_data':
            # Logic is identical to soil, just replace 'soil' with 'rock'
            wp = goal_parts[1]
            rover_holding = None
            for r, wps in dynamic_info['held_rock'].items():
                if wp in wps:
                    rover_holding = r
                    break

            if rover_holding:
                r_loc = dynamic_info['rover_locations'].get(rover_holding)
                if r_loc is None: return 1 + 1
                if r_loc not in self.comm_waypoints: cost += 1
                cost += 1
                return cost
            else:
                if wp not in dynamic_info['available_rock']:
                     # Relaxation: assume held
                     possible_rovers = [r for r, equip in self.rover_equipment.items() if 'rock' in equip]
                     if not possible_rovers: return self.IMPOSSIBLE_GOAL_PENALTY
                     r = possible_rovers[0]
                     r_loc = dynamic_info['rover_locations'].get(r)
                     if r_loc is None: return 1 + 1
                     if r_loc not in self.comm_waypoints: cost += 1
                     cost += 1
                     return cost

                # Sample available
                best_rover = None
                min_cost_rover = float('inf')
                for r, equip in self.rover_equipment.items():
                    if 'rock' in equip and r in self.rover_stores:
                        current_rover_cost = 0
                        r_loc = dynamic_info['rover_locations'].get(r)
                        r_store_status = dynamic_info['store_status'].get(r, 'empty')
                        if r_loc is None: continue

                        if r_store_status == 'full': current_rover_cost += 1
                        if r_loc != wp: current_rover_cost += 1
                        current_rover_cost += 1 # Sample
                        # Location after sampling is wp
                        if wp not in self.comm_waypoints: current_rover_cost += 1
                        current_rover_cost += 1 # Communicate

                        if current_rover_cost < min_cost_rover:
                            min_cost_rover = current_rover_cost
                            best_rover = r
                return min_cost_rover if best_rover else self.IMPOSSIBLE_GOAL_PENALTY

        # --- Case 3: communicated_image_data(obj, mode) ---
        elif goal_pred == 'communicated_image_data':
            obj, mode = goal_parts[1], goal_parts[2]
            rover_holding = None
            # Check if any rover already holds the specific image
            for r, images in dynamic_info['held_images'].items():
                if (obj, mode) in images:
                    rover_holding = r
                    break

            if rover_holding:
                # Rover has the image, estimate communication cost
                r_loc = dynamic_info['rover_locations'].get(rover_holding)
                if r_loc is None: return 1 + 1
                if r_loc not in self.comm_waypoints: cost += 1
                cost += 1
                return cost
            else:
                # Need to take image and communicate
                best_rover_cam = None
                min_cost_rover = float('inf')

                # Find waypoints from which the objective is visible
                visible_wps_for_obj = self.objective_visibility.get(obj)
                if not visible_wps_for_obj:
                    return self.IMPOSSIBLE_GOAL_PENALTY # Cannot see objective from anywhere

                # Simplification: Choose one potential imaging waypoint.
                # A simple choice is the first one found. Could be improved later.
                # Using list() to handle potential empty set gracefully if needed, then check length
                potential_w_imgs = list(visible_wps_for_obj)
                if not potential_w_imgs: return self.IMPOSSIBLE_GOAL_PENALTY # Should not happen based on above check
                w_img = potential_w_imgs[0] # Pick the first one

                # Iterate through rovers equipped for imaging
                for r, equip in self.rover_equipment.items():
                    if 'imaging' in equip:
                        r_loc = dynamic_info['rover_locations'].get(r)
                        if r_loc is None: continue # Skip rover if location unknown

                        # Iterate through cameras on this rover
                        for cam in self.rover_cameras.get(r, set()):
                            # Check if camera supports the required mode
                            if mode in self.camera_supports.get(cam, set()):
                                # This rover/camera combo can potentially do the job
                                current_rover_cost = 0
                                is_calibrated = (cam, r) in dynamic_info['calibrated_cameras']
                                temp_loc = r_loc # Track hypothetical location through steps

                                # --- Calibration Step (if needed) ---
                                if not is_calibrated:
                                    # Find a calibration target for this camera
                                    cal_targets = self.camera_calibration_targets.get(cam)
                                    if not cal_targets: continue # Camera cannot be calibrated, try next camera/rover

                                    w_cal = None
                                    # Find *any* waypoint where *any* of these calibration targets is visible
                                    for target_obj in cal_targets:
                                        cal_visible_wps = self.calibration_target_visibility.get(target_obj)
                                        if cal_visible_wps:
                                            # Simplification: pick first calibration waypoint found
                                            potential_w_cals = list(cal_visible_wps)
                                            if potential_w_cals:
                                                w_cal = potential_w_cals[0]
                                                break # Found a possible calibration spot
                                    if w_cal is None: continue # No visible calibration target waypoint found for this camera

                                    # Cost to navigate to calibration site
                                    if temp_loc != w_cal:
                                        current_rover_cost += 1 # Navigate action
                                        temp_loc = w_cal # Update hypothetical location

                                    # Cost of calibration action
                                    current_rover_cost += 1 # Calibrate action

                                    # Cost to navigate from calibration site to imaging site (if different)
                                    if temp_loc != w_img:
                                        current_rover_cost += 1 # Navigate action
                                        temp_loc = w_img # Update hypothetical location
                                else:
                                    # Already calibrated, just need to navigate to imaging site
                                    if temp_loc != w_img:
                                        current_rover_cost += 1 # Navigate action
                                        temp_loc = w_img # Update hypothetical location

                                # --- Imaging Step ---
                                current_rover_cost += 1 # Take image action
                                # Hypothetical location after taking image is w_img

                                # --- Communication Step ---
                                # Cost to navigate from imaging site to communication site
                                if temp_loc not in self.comm_waypoints:
                                    current_rover_cost += 1 # Navigate action

                                # Cost of communication action
                                current_rover_cost += 1 # Communicate action

                                # Update best cost found so far for this goal
                                if current_rover_cost < min_cost_rover:
                                    min_cost_rover = current_rover_cost
                                    best_rover_cam = (r, cam) # Keep track of which combo was best

                # Return cost for the best rover/camera found, or penalty if none could do it
                return min_cost_rover if best_rover_cam else self.IMPOSSIBLE_GOAL_PENALTY

        # Return 0 if the goal predicate is not recognized (should not happen in valid PDDL)
        return 0


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state
        # Goal check: If all goal predicates are in the current state, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract dynamic information from the current state
        dynamic_info = self._get_dynamic_info(state)

        total_heuristic_cost = 0
        # Find goals not yet satisfied
        unsatisfied_goals = self.goals - state

        # If lander location wasn't found during initialization, communication is impossible
        # This check might be redundant if _estimate_cost handles it, but safe to have.
        if not self.lander_location and unsatisfied_goals:
             # Assign penalty based on number of goals requiring communication
             num_comm_goals = sum(1 for g in unsatisfied_goals if get_parts(g)[0].startswith('communicated_'))
             return self.IMPOSSIBLE_GOAL_PENALTY * num_comm_goals if num_comm_goals > 0 else 0


        # Calculate cost for each unsatisfied goal and sum them up
        for goal in unsatisfied_goals:
            cost = self._estimate_cost(goal, state, dynamic_info)
            total_heuristic_cost += cost

        # Final adjustment: Ensure heuristic is non-zero if goals are not met
        # This handles cases where the estimation might incorrectly yield 0 for a non-goal state.
        if total_heuristic_cost == 0 and unsatisfied_goals:
             return 1

        # Avoid returning excessively large numbers if penalties accumulate
        # Use a large number instead of float('inf')
        if total_heuristic_cost >= 1e9:
            return 1e9

        return int(total_heuristic_cost) # Return integer cost
