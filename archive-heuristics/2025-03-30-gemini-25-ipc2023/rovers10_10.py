import sys
from fnmatch import fnmatch
# Assuming heuristic_base.py is accessible.
# If not, you might need to adjust the import path based on your project structure.
# e.g., sys.path.append("path/to/directory/containing/heuristics")
try:
    # Assumes the heuristic is used within a framework where Heuristic is defined
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Simple fallback if the import structure is different or for standalone testing
    class Heuristic:
        """ Base class placeholder if the actual Heuristic class is not found. """
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

def get_parts(fact: str):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at rover1 waypoint2)" -> ["at", "rover1", "waypoint2"]
    """
    # Remove leading/trailing whitespace and parentheses, then split
    return fact.strip()[1:-1].split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    Estimates the cost to reach the goal state in the PDDL Rovers domain for use
    with Greedy Best-First Search. It calculates the sum of estimated costs for
    achieving each individual goal predicate that is not yet true in the current state.
    The cost for each goal is estimated based on a simplified count of necessary
    actions (navigation, sampling/imaging, communication), assuming each major
    step (like navigate, sample, communicate) costs 1 action. It considers the
    rover's current state (e.g., store full/empty, camera calibrated) to refine
    the action count.

    # Assumptions
    - Navigation between any two required waypoints costs 1 action if the rover
      is not already there. It does not compute actual shortest paths or consider
      specific rover traversal capabilities (`can_traverse`).
    - Visibility constraints for communication (`visible(rover_wp, lander_wp)`)
      are checked to determine if direct communication (cost 1) or
      navigation+communication (cost 2) is needed.
    - Visibility for calibration (`visible_from(cal_target, cal_wp)`) and
      imaging (`visible_from(obj, image_wp)`) is implicitly required by the
      actions, and the heuristic accounts for the actions (calibrate, take_image)
      but not the navigation cost to specific visible waypoints.
    - Lander location and rover equipment are static.
    - When multiple rovers can achieve a goal, the heuristic calculates the
      minimum estimated cost among the capable rovers.
    - The heuristic assumes problems are solvable; if a required capability
      (e.g., a rover equipped for soil analysis) doesn't exist for a goal,
      it adds a large penalty (1000) but doesn't return infinity, allowing search
      to continue (though likely away from this state).

    # Heuristic Initialization (`__init__`)
    - Stores the goal predicates (`self.goals`).
    - Parses static facts (`task.static`) provided by the planner framework
      to build internal data structures about the fixed elements of the problem:
        - Lander location (`self.lander_location`).
        - Rover equipment: capabilities (soil, rock, imaging), associated store,
          and cameras with supported modes and calibration targets (`self.rover_equipment`).
        - Waypoint visibility graph: which waypoints are visible from each other
          (`self.waypoint_visibility`). Assumes visibility is symmetric based on examples.
        - Objective visibility: which waypoints objectives are visible from
          (`self.objective_visibility`). This is used by actions but not directly in heuristic cost calculation.
        - Camera info: mapping cameras to rovers, modes, and calibration targets
          (`self.camera_info`).
        - Store mapping: which rover a store belongs to (`self.store_to_rover`).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize heuristic value `h = 0`.
    2. Parse the current `state` (`node.state`) to determine dynamic information:
        - Rover locations (`rover_locations`).
        - Store status (empty/full) (`store_status`).
        - Currently calibrated cameras (`calibrated_cameras`).
        - Data currently held by rovers: soil analysis (`held_soil_data`),
          rock analysis (`held_rock_data`), images (`held_image_data`).
        - Currently available samples at waypoints (`available_soil`, `available_rock`).
    3. For each `goal_fact` in `self.goals`:
        a. If `goal_fact` is already present in the `state`, skip it (cost is 0).
        b. Determine the goal type (`communicated_soil_data`, `communicated_rock_data`,
           `communicated_image_data`).
        c. **Estimate the minimum cost (`min_cost_for_goal`) to achieve this single goal:**
           This involves considering different ways the goal might be achieved
           (e.g., data already held vs. needing acquisition) and finding the minimum
           estimated cost among capable rovers.
           i. **Check if data is already held:** If `have_..._analysis(R, wp)` or
              `have_image(R, obj, mode)` exists for *any* rover `R`:
              - Calculate cost: 1 if `R` is at a location visible from the lander,
                otherwise 2 (navigate + communicate). Find the minimum cost across
                all rovers holding the data. Store this as `cost_if_held`.
           ii. **Check if data needs to be acquired:**
              - **For Soil/Rock:** Check if the sample `wp` is available (`at_..._sample wp`).
                 - If yes: Find all rovers `R` equipped for this sample type. If none,
                   set cost to 1000. Otherwise, calculate cost for each `R`: 4 actions
                   if store is empty (nav_sample + sample + nav_comm + comm), 6 actions
                   if store is full (nav_comm + drop + nav_sample + sample + nav_comm + comm).
                   Find the minimum cost among equipped rovers. Store as `cost_if_sample`.
              - **For Image:** Find all rovers `R` equipped for imaging with a camera `I`
                 supporting the required `mode` and having a calibration target. If none,
                 set cost to 1000. Otherwise, calculate cost for each `(R, I)`: 4 actions
                 if camera `I` is currently calibrated (nav_img + take + nav_comm + comm),
                 6 actions if not calibrated (nav_cal + cal + nav_img + take + nav_comm + comm).
                 Find the minimum cost among suitable pairs. Store as `cost_if_take`.
           iii. **Determine fallback cost:** If data isn't held and cannot be acquired
                (e.g., sample gone), estimate a fallback cost (e.g., 2 for nav+comm,
                assuming it was acquired earlier).
           iv. **Combine costs:** `min_cost_for_goal` is the minimum of the applicable
                costs calculated (e.g., `min(cost_if_held, cost_if_sample)` or
                `min(cost_if_held, cost_if_take)`). If no path seems possible
                (all costs are infinity or 1000), use the high penalty value (1000).
        d. Add `min_cost_for_goal` to the total heuristic value `h`.
    4. Return the total accumulated heuristic value `h`. It should be 0 if and only
       if all goals are met in the state.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        # rover -> {'soil': bool, 'rock': bool, 'imaging': bool, 'store': store, 'cameras': {cam: {'modes': set(), 'cal_target': obj}}}
        self.rover_equipment = {}
        self.waypoint_visibility = {} # wp -> set(visible_wp)
        self.objective_visibility = {} # obj -> set(visible_wp)
        # cam -> {'rover': r, 'modes': set(), 'cal_target': obj}
        self.camera_info = {}
        self.store_to_rover = {} # store -> rover
        self.all_waypoints = set() # Collect all known waypoints

        # --- Parse Static Facts ---
        # Temporary storage for camera properties before associating with rovers
        temp_on_board = {} # cam -> rover
        temp_supports = {} # cam -> set(modes)
        temp_cal_target = {} # cam -> obj

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            pred = parts[0]

            # Basic environment structure
            if pred == "at_lander" and len(parts) == 3:
                self.lander_location = parts[2]
                self.all_waypoints.add(parts[2])
            elif pred == "visible" and len(parts) == 3:
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility.setdefault(wp1, set()).add(wp2)
                self.waypoint_visibility.setdefault(wp2, set()).add(wp1) # Assume symmetry
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif pred == "visible_from" and len(parts) == 3:
                 obj, wp = parts[1], parts[2]
                 self.objective_visibility.setdefault(obj, set()).add(wp)
                 self.all_waypoints.add(wp)

            # Rover static properties
            elif pred == "store_of" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                # Ensure rover entry exists before accessing keys
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False, 'store': None, 'cameras': {}})
                self.rover_equipment[rover]['store'] = store
            elif pred == "equipped_for_soil_analysis" and len(parts) == 2:
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False, 'store': None, 'cameras': {}})
                self.rover_equipment[rover]['soil'] = True
            elif pred == "equipped_for_rock_analysis" and len(parts) == 2:
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False, 'store': None, 'cameras': {}})
                self.rover_equipment[rover]['rock'] = True
            elif pred == "equipped_for_imaging" and len(parts) == 2:
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False, 'store': None, 'cameras': {}})
                self.rover_equipment[rover]['imaging'] = True

            # Camera static properties (store temporarily)
            elif pred == "on_board" and len(parts) == 3:
                temp_on_board[parts[1]] = parts[2] # cam -> rover
            elif pred == "supports" and len(parts) == 3:
                cam, mode = parts[1], parts[2]
                temp_supports.setdefault(cam, set()).add(mode)
            elif pred == "calibration_target" and len(parts) == 3:
                temp_cal_target[parts[1]] = parts[2] # cam -> obj

        # --- Consolidate Camera Info ---
        # Combine info from on_board, supports, calibration_target
        all_cameras = set(temp_on_board.keys()) | set(temp_supports.keys()) | set(temp_cal_target.keys())
        for cam in all_cameras:
            rover = temp_on_board.get(cam)
            if rover:
                # Add to camera_info dictionary
                self.camera_info[cam] = {
                    'rover': rover,
                    'modes': temp_supports.get(cam, set()),
                    'cal_target': temp_cal_target.get(cam)
                }
                # Add camera details to the rover's equipment dict if rover exists and is equipped for imaging
                if rover in self.rover_equipment and self.rover_equipment[rover].get('imaging', False):
                     # Ensure 'cameras' dict exists
                     self.rover_equipment[rover].setdefault('cameras', {})
                     self.rover_equipment[rover]['cameras'][cam] = {
                         'modes': temp_supports.get(cam, set()),
                         'cal_target': temp_cal_target.get(cam)
                     }

        # --- Sanity Checks ---
        if self.lander_location is None:
             # This might be critical for the heuristic calculation
             print("Warning: Lander location not found in static facts. Heuristic might be inaccurate.", file=sys.stderr)


    def _match(self, fact_str, *pattern):
        """Checks if a fact string matches a pattern using fnmatch."""
        parts = get_parts(fact_str)
        if len(parts) != len(pattern):
            return False
        # Compare parts to pattern using fnmatch for wildcard support
        return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

    def __call__(self, node):
        state = node.state
        h = 0
        IMPOSSIBLE_COST = 1000 # Penalty for seemingly unachievable goals

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        calibrated_cameras = set() # (camera, rover) tuples
        held_soil_data = {} # waypoint -> set(rovers)
        held_rock_data = {} # waypoint -> set(rovers)
        held_image_data = {} # (objective, mode) -> set(rovers)
        available_soil = set() # waypoints
        available_rock = set() # waypoints

        for fact in state:
            # Use _match for robust parsing relative to known object types
            if self._match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Check if first arg is a rover we know about
                if parts[1] in self.rover_equipment:
                    rover_locations[parts[1]] = parts[2]
            elif self._match(fact, "empty", "*"):
                parts = get_parts(fact)
                # Check if it's a known store
                if parts[1] in self.store_to_rover:
                     store_status[parts[1]] = "empty"
            elif self._match(fact, "full", "*"):
                 parts = get_parts(fact)
                 if parts[1] in self.store_to_rover:
                     store_status[parts[1]] = "full"
            elif self._match(fact, "calibrated", "*", "*"):
                parts = get_parts(fact)
                # Check if it's a known camera
                if parts[1] in self.camera_info:
                    calibrated_cameras.add((parts[1], parts[2])) # (camera, rover)
            elif self._match(fact, "have_soil_analysis", "*", "*"):
                parts = get_parts(fact)
                # parts[1] is rover, parts[2] is waypoint
                held_soil_data.setdefault(parts[2], set()).add(parts[1])
            elif self._match(fact, "have_rock_analysis", "*", "*"):
                parts = get_parts(fact)
                held_rock_data.setdefault(parts[2], set()).add(parts[1])
            elif self._match(fact, "have_image", "*", "*", "*"):
                parts = get_parts(fact)
                # parts[1] rover, parts[2] objective, parts[3] mode
                held_image_data.setdefault((parts[2], parts[3]), set()).add(parts[1])
            elif self._match(fact, "at_soil_sample", "*"):
                available_soil.add(get_parts(fact)[1])
            elif self._match(fact, "at_rock_sample", "*"):
                available_rock.add(get_parts(fact)[1])

        # --- Goal Achievement Estimation ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            if not parts: continue # Skip invalid goal facts
            goal_type = parts[0]
            min_cost_for_goal = float('inf') # Min cost among possible ways/rovers for this goal

            # --- Estimate cost based on goal type ---

            if goal_type == "communicated_soil_data" and len(parts) == 2:
                wp = parts[1]
                rovers_with_data = held_soil_data.get(wp, set())
                cost_if_held = float('inf')
                cost_if_sample = float('inf')

                # Case 1: Data is already held by some rover
                if rovers_with_data:
                    current_min = 2 # Default: nav + comm
                    for rover in rovers_with_data:
                        if rover in rover_locations:
                            rover_loc = rover_locations[rover]
                            # Check visibility - requires lander location
                            if self.lander_location and rover_loc in self.waypoint_visibility and \
                               self.lander_location in self.waypoint_visibility.get(rover_loc, set()):
                                current_min = 1 # Direct comm possible
                                break # Found best case for held data
                    cost_if_held = current_min

                # Case 2: Data needs to be sampled
                if wp in available_soil:
                    current_min = IMPOSSIBLE_COST # Default if no suitable rover
                    suitable_rovers_found = False
                    for r, equip in self.rover_equipment.items():
                        if equip.get('soil', False):
                            suitable_rovers_found = True
                            store = equip.get('store')
                            current_rover_cost = 4 # Assume empty store
                            # Check store status, default to 'empty' if unknown
                            if store and store_status.get(store, 'empty') == 'full':
                                current_rover_cost = 6 # Store is full
                            current_min = min(current_min, current_rover_cost)
                    # Only update cost_if_sample if a suitable rover was found
                    if suitable_rovers_found:
                        cost_if_sample = current_min

                # Case 3: Fallback (sample gone, data not held)
                cost_fallback = 2 # Assume nav + comm needed

                # Determine minimum cost for this goal
                min_cost_for_goal = min(cost_if_held, cost_if_sample)
                # Only use fallback if neither case 1 nor 2 applies (both are inf)
                if min_cost_for_goal == float('inf'):
                     min_cost_for_goal = cost_fallback


            elif goal_type == "communicated_rock_data" and len(parts) == 2:
                wp = parts[1]
                rovers_with_data = held_rock_data.get(wp, set())
                cost_if_held = float('inf')
                cost_if_sample = float('inf')

                if rovers_with_data:
                    current_min = 2
                    for rover in rovers_with_data:
                        if rover in rover_locations:
                             rover_loc = rover_locations[rover]
                             if self.lander_location and rover_loc in self.waypoint_visibility and \
                                self.lander_location in self.waypoint_visibility.get(rover_loc, set()):
                                 current_min = 1
                                 break
                    cost_if_held = current_min

                if wp in available_rock:
                    current_min = IMPOSSIBLE_COST
                    suitable_rovers_found = False
                    for r, equip in self.rover_equipment.items():
                        if equip.get('rock', False):
                            suitable_rovers_found = True
                            store = equip.get('store')
                            current_rover_cost = 4
                            if store and store_status.get(store, 'empty') == 'full':
                                current_rover_cost = 6
                            current_min = min(current_min, current_rover_cost)
                    if suitable_rovers_found:
                        cost_if_sample = current_min

                cost_fallback = 2
                min_cost_for_goal = min(cost_if_held, cost_if_sample)
                if min_cost_for_goal == float('inf'):
                     min_cost_for_goal = cost_fallback


            elif goal_type == "communicated_image_data" and len(parts) == 4:
                obj, mode = parts[1], parts[2] # parts[3] is usually the mode again? No, check domain.
                # communicate_image_data ?r ?l ?o ?m ?x ?y -> goal is (communicated_image_data ?o ?m)
                # So parts are [pred, obj, mode]
                obj, mode = parts[1], parts[2]
                goal_tuple = (obj, mode)
                rovers_with_data = held_image_data.get(goal_tuple, set())
                cost_if_held = float('inf')
                cost_if_take = float('inf')

                if rovers_with_data:
                    current_min = 2
                    for rover in rovers_with_data:
                        if rover in rover_locations:
                            rover_loc = rover_locations[rover]
                            if self.lander_location and rover_loc in self.waypoint_visibility and \
                               self.lander_location in self.waypoint_visibility.get(rover_loc, set()):
                                current_min = 1
                                break
                    cost_if_held = current_min

                # Check if image can be taken
                current_min_take = IMPOSSIBLE_COST
                suitable_rover_cam_found = False
                for r, equip in self.rover_equipment.items():
                    # Check if rover is equipped for imaging
                    if equip.get('imaging', False):
                        # Iterate through cameras on this rover
                        for cam_name, cam_details in equip.get('cameras', {}).items():
                            # Check if camera supports mode and has a calibration target
                            if mode in cam_details.get('modes', set()) and cam_details.get('cal_target') is not None:
                                suitable_rover_cam_found = True
                                current_pair_cost = 6 # Assume needs calibration
                                if (cam_name, r) in calibrated_cameras:
                                    current_pair_cost = 4 # Already calibrated
                                current_min_take = min(current_min_take, current_pair_cost)
                # Update cost_if_take only if a suitable setup was found
                if suitable_rover_cam_found:
                    cost_if_take = current_min_take

                # Determine minimum cost: min of having it or being able to take it
                min_cost_for_goal = min(cost_if_held, cost_if_take)
                # If neither is possible, min_cost_for_goal remains float('inf')


            # Add cost for this goal to total heuristic value
            if min_cost_for_goal != float('inf'):
                h += min_cost_for_goal
            else:
                # Goal seems unachievable from current analysis (e.g., no capable rover/camera)
                # Add a large penalty to indicate this state is likely bad.
                h += IMPOSSIBLE_COST

        return h
