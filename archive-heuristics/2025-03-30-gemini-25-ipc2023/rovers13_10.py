import re
from fnmatch import fnmatch
# Assuming heuristic_base.py containing the Heuristic base class is in the python path
# If not, you might need to define a placeholder or adjust the import.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Placeholder if the base class structure is not available
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper function to parse facts: remove parentheses and split
def get_parts(fact_str):
    """
    Extracts the components of a PDDL fact string.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    # Remove leading '(' and trailing ')' before splitting
    return fact_str[1:-1].split()


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated costs
    for achieving each individual unsatisfied goal predicate. The cost for each
    goal is estimated by finding the minimum cost sequence of actions (navigate,
    sample, calibrate, take_image, communicate, drop) required by any capable
    rover, considering rover capabilities, locations, and current state (store,
    calibration). Navigation cost between waypoints is simplified to 1 action
    if movement is needed, 0 otherwise.

    # Assumptions
    - Action costs are uniform (1).
    - The heuristic focuses on achieving each goal independently and sums the costs.
      It finds the minimum cost sequence for *each* goal, potentially selecting
      different rovers for different sub-tasks or goals based on minimal estimated cost.
    - It considers the need for `drop` (cost 1) if a rover's store is full before sampling.
    - It considers the need for `calibrate` (cost 1 + nav cost) if a camera is not
      calibrated before imaging.
    - Navigation cost is simplified: 0 if already at the target waypoint, 1 otherwise.
      It does not compute shortest paths or consider traversability constraints beyond
      basic reachability assumptions.
    - Visibility between waypoints and from objectives/targets is checked based on
      static facts.
    - Assumes that if a sample goal `communicated_soil/rock_data(w)` is unmet and
      no rover `has_soil/rock_analysis(r, w)`, then the sample `at_soil/rock_sample(w)`
      is still available at the waypoint. If not, the goal is considered unreachable
      from the current path (assigned infinite cost).

    # Heuristic Initialization
    - Parses static facts (`task.static`) to build lookups for:
        - Lander location (`at_lander`).
        - Waypoint visibility graph (`visible`).
        - Objective visibility (`visible_from`).
        - Calibration target visibility (uses `visible_from`).
        - Rover equipment (`equipped_for_...`).
        - Rover-store mapping (`store_of`).
        - Cameras on board each rover (`on_board`).
        - Camera capabilities (`supports`, `calibration_target`).
    - Stores the goal predicates (`task.goals`) in a set for quick lookup.
    - Precomputes the set of waypoints visible from the lander's location (`comm_waypoints`)
      for efficient communication checks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state (`node.state`) to determine dynamic information:
        - Rover locations (`at ?r ?wp`).
        - Store status (`empty ?s` / `full ?s`).
        - Calibration status (`calibrated ?c ?r`).
        - Samples/images currently held (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
        - Available samples (`at_soil_sample`, `at_rock_sample`).
        - Already communicated data (`communicated_...`).
    2. Initialize `total_heuristic_cost = 0`.
    3. For each goal predicate `g` in the task goals:
        a. Check if `g` is already satisfied in the current state based on `communicated_...` predicates. If yes, continue to the next goal.
        b. If `g` is `(communicated_soil_data ?w)`:
            i. Check if any rover `r` has `(have_soil_analysis ?r ?w)`.
            ii. If yes: Estimate cost = cost_navigate(rover_with_data to comm_wp) + 1 (communicate). Find the minimum such cost over all rovers holding the analysis.
            iii. If no: Check if `(at_soil_sample ?w)` exists in the state.
                - If yes: Find the minimum cost among all capable rovers `r_can` to achieve the goal:
                    Cost(r_can) = cost_drop(r_can) + cost_navigate(r_can to w) + 1 (sample) + cost_navigate(r_can from w to comm_wp) + 1 (communicate).
                    Add the minimum Cost(r_can) found to `total_heuristic_cost`.
                - If no: The goal seems unreachable (sample needed but not available). Add a large penalty (INFINITE_COST).
        c. If `g` is `(communicated_rock_data ?w)`: Similar logic to soil data, using rock predicates and capabilities.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            i. Check if any rover `r` has `(have_image ?r ?o ?m)`.
            ii. If yes: Estimate cost = cost_navigate(rover_with_image to comm_wp) + 1 (communicate). Find the minimum such cost over all rovers holding the image.
            iii. If no: Find the minimum cost among all capable rovers `r_can` with a suitable camera `i`:
                - Calculate cost to calibrate if needed: `cost_cal = cost_navigate(r_can to w_cal) + 1 (calibrate)`. Update conceptual location.
                - Calculate cost to take image: `cost_img = cost_navigate(r_can from cal_loc to w_img) + 1 (take_image)`. Update conceptual location.
                - Calculate cost to communicate: `cost_comm = cost_navigate(r_can from img_loc to comm_wp) + 1 (communicate)`.
                - Cost(r_can) = cost_cal + cost_img + cost_comm.
                Add the minimum Cost(r_can) found to `total_heuristic_cost`.
    4. Return `total_heuristic_cost`. If any goal was deemed unreachable (cost is INFINITE_COST), the total heuristic value will reflect this large penalty, guiding the search away from such states. If `total_heuristic_cost` is 0, it implies all goals were already met in the state.
    """
    INFINITE_COST = 100000 # A large number representing infinity

    def __init__(self, task):
        super().__init__(task) # Ensure base class init is called if needed
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute static information ---
        self.lander_location = None
        self.waypoint_visibility = {} # wp -> set(visible_wp)
        self.objective_visibility = {} # objective -> set(wp)
        self.calibration_target_visibility = {} # objective -> set(wp)
        self.rover_equipment = {} # rover -> set{'soil', 'rock', 'imaging'}
        self.rover_stores = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.cameras_onboard = {} # rover -> set(camera)
        self.camera_supports = {} # camera -> set(mode)
        self.camera_calibration_target = {} # camera -> objective
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.cameras = set()
        self.stores = set()
        self.modes = set()

        # Helper to add to dict of sets
        def add_to_dict_set(d, key, value):
            d.setdefault(key, set()).add(value)

        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at_lander':
                self.lander_location = parts[2]
                self.waypoints.add(parts[2])
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                add_to_dict_set(self.waypoint_visibility, wp1, wp2)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif pred == 'visible_from':
                obj, wp = parts[1], parts[2]
                add_to_dict_set(self.objective_visibility, obj, wp)
                self.objectives.add(obj)
                self.waypoints.add(wp)
                # Assume calibration targets use the same visibility predicate
                add_to_dict_set(self.calibration_target_visibility, obj, wp)
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                add_to_dict_set(self.rover_equipment, rover, 'soil')
                self.rovers.add(rover)
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                add_to_dict_set(self.rover_equipment, rover, 'rock')
                self.rovers.add(rover)
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                add_to_dict_set(self.rover_equipment, rover, 'imaging')
                self.rovers.add(rover)
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                self.store_rover[store] = rover
                self.stores.add(store)
                self.rovers.add(rover)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                add_to_dict_set(self.cameras_onboard, rover, camera)
                self.cameras.add(camera)
                self.rovers.add(rover)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                add_to_dict_set(self.camera_supports, camera, mode)
                self.cameras.add(camera)
                self.modes.add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
                self.cameras.add(camera)
                self.objectives.add(objective)

        # Find waypoints visible from the lander's location
        self.comm_waypoints = set()
        if self.lander_location:
            # A rover at waypoint X can communicate if X is visible from the lander's waypoint Y
            # i.e., (visible X Y) where Y is self.lander_location
            for wp1, visible_wps in self.waypoint_visibility.items():
                if self.lander_location in visible_wps:
                    self.comm_waypoints.add(wp1)
            # Check if lander location is visible from itself
            if self.lander_location in self.waypoint_visibility.get(self.lander_location, set()):
                 self.comm_waypoints.add(self.lander_location)


        # Parse goals into a set for easier checking
        self.parsed_goals = set(task.goals)


    def _get_nav_cost(self, current_wp, target_wp):
        """Simplified navigation cost: 0 if already there, 1 otherwise."""
        return 0 if current_wp == target_wp else 1

    def _find_closest_comm_wp_cost(self, current_wp):
        """Finds minimum cost (0 or 1) to reach any comm waypoint."""
        if not self.comm_waypoints:
            return self.INFINITE_COST # Cannot communicate
        if current_wp in self.comm_waypoints:
            return 0 # Already at a comm waypoint
        else:
            # Simplification: assume 1 nav action reaches *some* comm waypoint if any exist
            return 1 if self.comm_waypoints else self.INFINITE_COST

    def _find_closest_visible_from_wp_cost(self, current_wp, target_obj, visibility_map):
        """
        Finds minimum cost (0 or 1) to reach a waypoint seeing the target object.
        Returns (waypoint, cost). Waypoint is one such waypoint (e.g., the first found).
        """
        visible_wps = visibility_map.get(target_obj, set())
        if not visible_wps:
            return None, self.INFINITE_COST # Target not visible from anywhere

        if current_wp in visible_wps:
            return current_wp, 0 # Already at a suitable waypoint

        # Simplification: assume 1 nav action reaches *some* suitable waypoint
        # Return the first one found and cost 1
        return list(visible_wps)[0], 1

    def __call__(self, node):
        state = node.state
        heuristic_value = 0

        # --- Parse current state ---
        rover_locations = {} # rover -> wp
        store_status = {} # store -> 'empty' / 'full'
        calibrated_cameras = set() # (camera, rover) tuples
        have_soil = {} # wp -> set(rover)
        have_rock = {} # wp -> set(rover)
        have_image = {} # (obj, mode) -> set(rover)
        at_soil_sample = set() # wp
        at_rock_sample = set() # wp
        communicated_soil = set() # wp
        communicated_rock = set() # wp
        communicated_image = set() # (obj, mode) tuples

        # Initialize store status (assume empty unless 'full' is stated)
        for store in self.stores:
            store_status[store] = 'empty'

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at':
                rover_locations[parts[1]] = parts[2]
            elif pred == 'full': # Only need to know if full
                store_status[parts[1]] = 'full'
            elif pred == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                have_soil.setdefault(wp, set()).add(rover)
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                have_rock.setdefault(wp, set()).add(rover)
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                key = (obj, mode)
                have_image.setdefault(key, set()).add(rover)
            elif pred == 'at_soil_sample':
                at_soil_sample.add(parts[1])
            elif pred == 'at_rock_sample':
                at_rock_sample.add(parts[1])
            elif pred == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        # --- Evaluate unsatisfied goals ---
        for goal_fact in self.parsed_goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            min_cost_for_goal = self.INFINITE_COST # Assume infinity until proven otherwise

            # Check if goal is already achieved and skip if so
            if pred == 'communicated_soil_data':
                wp_goal = parts[1]
                if wp_goal in communicated_soil: continue
                # --- Calculate cost ---
                rovers_with_data = have_soil.get(wp_goal, set())
                if rovers_with_data:
                    # Need to communicate
                    current_min = self.INFINITE_COST
                    for r_has in rovers_with_data:
                        r_loc = rover_locations.get(r_has)
                        if r_loc:
                            nav_cost = self._find_closest_comm_wp_cost(r_loc)
                            if nav_cost < self.INFINITE_COST:
                                current_min = min(current_min, nav_cost + 1) # navigate + communicate
                    min_cost_for_goal = min(min_cost_for_goal, current_min)
                elif wp_goal in at_soil_sample:
                    # Need to sample then communicate
                    possible_rovers = [r for r, equips in self.rover_equipment.items()
                                       if 'soil' in equips and r in self.rover_stores]
                    current_min = self.INFINITE_COST
                    for r_can in possible_rovers:
                        r_loc = rover_locations.get(r_can)
                        store = self.rover_stores.get(r_can)
                        if not r_loc or not store: continue # Rover location or store unknown? Skip.
                        s_status = store_status.get(store, 'empty')

                        cost_drop = 1 if s_status == 'full' else 0
                        cost_nav_to_sample = self._get_nav_cost(r_loc, wp_goal)
                        cost_sample = 1
                        # Assume after sampling, rover is at wp_goal
                        cost_nav_to_comm = self._find_closest_comm_wp_cost(wp_goal)
                        cost_comm = 1

                        if cost_nav_to_comm < self.INFINITE_COST:
                             total_rover_cost = cost_drop + cost_nav_to_sample + cost_sample + cost_nav_to_comm + cost_comm
                             current_min = min(current_min, total_rover_cost)
                    min_cost_for_goal = min(min_cost_for_goal, current_min)
                # else: Goal requires sample at wp_goal, but it's not there. min_cost_for_goal remains INFINITE_COST

            elif pred == 'communicated_rock_data':
                wp_goal = parts[1]
                if wp_goal in communicated_rock: continue
                # --- Calculate cost ---
                rovers_with_data = have_rock.get(wp_goal, set())
                if rovers_with_data:
                    # Need to communicate
                    current_min = self.INFINITE_COST
                    for r_has in rovers_with_data:
                        r_loc = rover_locations.get(r_has)
                        if r_loc:
                            nav_cost = self._find_closest_comm_wp_cost(r_loc)
                            if nav_cost < self.INFINITE_COST:
                                current_min = min(current_min, nav_cost + 1)
                    min_cost_for_goal = min(min_cost_for_goal, current_min)
                elif wp_goal in at_rock_sample:
                    # Need to sample then communicate
                    possible_rovers = [r for r, equips in self.rover_equipment.items()
                                       if 'rock' in equips and r in self.rover_stores]
                    current_min = self.INFINITE_COST
                    for r_can in possible_rovers:
                        r_loc = rover_locations.get(r_can)
                        store = self.rover_stores.get(r_can)
                        if not r_loc or not store: continue
                        s_status = store_status.get(store, 'empty')

                        cost_drop = 1 if s_status == 'full' else 0
                        cost_nav_to_sample = self._get_nav_cost(r_loc, wp_goal)
                        cost_sample = 1
                        cost_nav_to_comm = self._find_closest_comm_wp_cost(wp_goal)
                        cost_comm = 1

                        if cost_nav_to_comm < self.INFINITE_COST:
                            total_rover_cost = cost_drop + cost_nav_to_sample + cost_sample + cost_nav_to_comm + cost_comm
                            current_min = min(current_min, total_rover_cost)
                    min_cost_for_goal = min(min_cost_for_goal, current_min)
                # else: Goal requires sample at wp_goal, but it's not there. min_cost_for_goal remains INFINITE_COST

            elif pred == 'communicated_image_data':
                obj_goal, mode_goal = parts[1], parts[2]
                goal_key = (obj_goal, mode_goal)
                if goal_key in communicated_image: continue
                # --- Calculate cost ---
                rovers_with_data = have_image.get(goal_key, set())
                if rovers_with_data:
                    # Need to communicate
                    current_min = self.INFINITE_COST
                    for r_has in rovers_with_data:
                        r_loc = rover_locations.get(r_has)
                        if r_loc:
                            nav_cost = self._find_closest_comm_wp_cost(r_loc)
                            if nav_cost < self.INFINITE_COST:
                                current_min = min(current_min, nav_cost + 1)
                    min_cost_for_goal = min(min_cost_for_goal, current_min)
                else:
                    # Need to calibrate (maybe), take image, then communicate
                    possible_rovers_cameras = []
                    for r, equips in self.rover_equipment.items():
                        if 'imaging' in equips and r in self.cameras_onboard:
                            for cam in self.cameras_onboard.get(r, set()):
                                supported_modes = self.camera_supports.get(cam, set())
                                cal_target = self.camera_calibration_target.get(cam)
                                if mode_goal in supported_modes and cal_target:
                                     possible_rovers_cameras.append((r, cam))

                    current_min = self.INFINITE_COST
                    for r_can, cam_can in possible_rovers_cameras:
                        r_loc = rover_locations.get(r_can)
                        if not r_loc: continue

                        cal_target = self.camera_calibration_target[cam_can]
                        is_calibrated = (cam_can, r_can) in calibrated_cameras

                        cost_cal = 0
                        loc_after_cal = r_loc

                        if not is_calibrated:
                            # Find waypoint for calibration
                            w_cal_found, nav_cost_cal = self._find_closest_visible_from_wp_cost(r_loc, cal_target, self.calibration_target_visibility)
                            if nav_cost_cal == self.INFINITE_COST: continue # Cannot calibrate
                            cost_cal = nav_cost_cal + 1 # navigate + calibrate
                            loc_after_cal = w_cal_found # Conceptually rover moves here

                        # Find waypoint for taking image
                        # Rover is conceptually at loc_after_cal (either original location or w_cal)
                        w_img_found, nav_cost_img = self._find_closest_visible_from_wp_cost(loc_after_cal, obj_goal, self.objective_visibility)
                        if nav_cost_img == self.INFINITE_COST: continue # Cannot see objective

                        cost_img = nav_cost_img + 1 # navigate + take_image
                        loc_after_img = w_img_found # Conceptually rover moves here

                        # Find waypoint for communication
                        # Rover is conceptually at loc_after_img
                        cost_nav_to_comm = self._find_closest_comm_wp_cost(loc_after_img)
                        if cost_nav_to_comm == self.INFINITE_COST: continue # Cannot communicate

                        cost_comm = cost_nav_to_comm + 1 # navigate + communicate

                        total_rover_cost = cost_cal + cost_img + cost_comm
                        current_min = min(current_min, total_rover_cost)

                    min_cost_for_goal = min(min_cost_for_goal, current_min)

            # Accumulate cost for this unsatisfied goal
            # If min_cost_for_goal is still INFINITE_COST, the goal seems unreachable
            heuristic_value += min_cost_for_goal


        # Check for overall potential unreachability indicated by INFINITE_COST contribution
        # If heuristic_value >= INFINITE_COST, it means at least one goal was deemed unreachable.
        # Return the large value to guide search away.
        if heuristic_value >= self.INFINITE_COST:
            return self.INFINITE_COST

        # The heuristic value is the sum of minimum costs for each unsatisfied goal.
        # If heuristic_value is 0, it means no goals were unsatisfied in the loop.
        return heuristic_value
