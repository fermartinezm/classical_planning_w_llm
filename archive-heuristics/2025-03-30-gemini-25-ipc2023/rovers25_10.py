import sys
import os

# Ensure the heuristics directory is in the Python path
# This might be needed if the heuristic is run standalone or from a different directory structure
# Adjust the path ('../') as necessary based on your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    return fact[1:-1].split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for achieving each individual unmet goal.
    It focuses on the core actions needed: sampling, calibrating, taking images,
    dropping samples, and communicating data, while largely ignoring navigation costs
    and resource contention for efficiency. This makes it suitable for greedy best-first
    search where admissibility is not required.

    # Assumptions
    - The problem instance is solvable from the current state.
    - Navigation costs are simplified or ignored to keep computation fast. The heuristic
      counts the essential non-navigation actions.
    - Resource contention (multiple goals needing the same rover/equipment simultaneously)
      is ignored; costs are summed independently per goal.
    - Visibility constraints for navigation, communication, calibration, and imaging
      are assumed to be eventually satisfiable without adding explicit cost, focusing
      instead on the action prerequisites (having data, being calibrated, store empty etc.).
    - Each action has a uniform cost of 1.

    # Heuristic Initialization
    - Stores the goal conditions (`task.goals`).
    - Parses static facts (`task.static`) to build efficient lookups for:
        - Lander location.
        - Rover capabilities (soil analysis, rock analysis, imaging).
        - Rover-store associations.
        - Cameras on board each rover, their supported modes, and calibration targets.
        - Initial sample locations (used to check if a sample *can* be taken).
    - Static information about waypoint visibility or objective visibility is not explicitly
      stored or used in this version to maintain simplicity, relying on the assumption
      that necessary visibility conditions can be met when needed.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all goal predicates defined in the problem (`self.goals`).
    2.  For the given `state`, determine which goal predicates are *not* yet satisfied.
    3.  For each unsatisfied goal predicate:
        a.  **`communicated_soil_data(wp)`:**
            i.  Estimate +1 action for the `communicate_soil_data` action itself. This is always needed if the goal is unmet.
            ii. If no rover currently `have_soil_analysis(rover, wp)`, estimate +1 action for `sample_soil`. This prerequisite must be met before communication.
            iii.If sampling is needed (step ii) AND no capable rover currently has an `empty` store AND the sample `at_soil_sample(wp)` still exists in the current state, estimate +1 action for a potential `drop` action needed to free up a store.
        b.  **`communicated_rock_data(wp)`:**
            i.  Estimate +1 action for `communicate_rock_data`.
            ii. If no rover `have_rock_analysis(rover, wp)`, estimate +1 action for `sample_rock`.
            iii.If sampling is needed (step ii) AND no capable rover has an `empty` store AND `at_rock_sample(wp)` exists, estimate +1 for `drop`.
        c.  **`communicated_image_data(obj, mode)`:**
            i.  Estimate +1 action for `communicate_image_data`.
            ii. If no rover `have_image(rover, obj, mode)`, estimate +1 action for `take_image`.
            iii.If taking an image is needed (step ii) AND no suitable rover/camera (rover equipped for imaging, camera on board supporting the mode) is currently `calibrated`, estimate +1 action for `calibrate`. We check if *any* suitable camera is calibrated; if not, we assume one calibration action is needed.
    4.  Sum the estimated costs calculated for all unsatisfied goals. This sum is the heuristic value.
    5.  The heuristic returns 0 if and only if all goal predicates are present in the state.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Pre-process Static Information ---
        self.lander_location = None
        # rover -> {'store': str, 'capabilities': set, 'cameras': {cam -> {'modes': set, 'target': str}}}
        self.rover_data = {}
        # Stores initial sample locations, not updated with state
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        all_rovers = set()
        all_stores = set()
        all_cameras = set()
        all_waypoints = set()
        all_objectives = set()
        all_modes = set()

        # Dynamically discover objects from facts if type predicates are missing/incomplete
        # Also useful for cross-referencing later
        for fact in static_facts.union(task.initial_state).union(task.goals):
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            try:
                if pred == 'rover': all_rovers.add(args[0])
                elif pred == 'store': all_stores.add(args[0])
                elif pred == 'camera': all_cameras.add(args[0])
                elif pred == 'waypoint': all_waypoints.add(args[0])
                elif pred == 'objective': all_objectives.add(args[0])
                elif pred == 'mode': all_modes.add(args[0])
                elif pred == 'at': all_rovers.add(args[0]); all_waypoints.add(args[1])
                elif pred == 'at_lander': all_waypoints.add(args[1])
                elif pred == 'can_traverse': all_rovers.add(args[0]); all_waypoints.add(args[1]); all_waypoints.add(args[2])
                elif pred == 'equipped_for_soil_analysis': all_rovers.add(args[0])
                elif pred == 'equipped_for_rock_analysis': all_rovers.add(args[0])
                elif pred == 'equipped_for_imaging': all_rovers.add(args[0])
                elif pred == 'store_of': all_stores.add(args[0]); all_rovers.add(args[1])
                elif pred == 'on_board': all_cameras.add(args[0]); all_rovers.add(args[1])
                elif pred == 'supports': all_cameras.add(args[0]); all_modes.add(args[1])
                elif pred == 'calibration_target': all_cameras.add(args[0]); all_objectives.add(args[1])
                elif pred == 'at_soil_sample': all_waypoints.add(args[0])
                elif pred == 'at_rock_sample': all_waypoints.add(args[0])
                # Add other predicates if they introduce new objects
            except IndexError:
                 # Ignore malformed facts silently during discovery
                 pass


        # Initialize rover_data structure for all discovered rovers
        for r in all_rovers:
            self.rover_data[r] = {'store': None, 'capabilities': set(), 'cameras': {}}

        # Populate static info from static_facts
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            try:
                if pred == 'at_lander':
                    self.lander_location = args[1]
                elif pred == 'store_of':
                    store, rover = args
                    if rover in self.rover_data:
                        self.rover_data[rover]['store'] = store
                elif pred == 'equipped_for_soil_analysis':
                    if args[0] in self.rover_data: self.rover_data[args[0]]['capabilities'].add('soil')
                elif pred == 'equipped_for_rock_analysis':
                    if args[0] in self.rover_data: self.rover_data[args[0]]['capabilities'].add('rock')
                elif pred == 'equipped_for_imaging':
                    if args[0] in self.rover_data: self.rover_data[args[0]]['capabilities'].add('imaging')
                elif pred == 'on_board':
                    camera, rover = args
                    if rover in self.rover_data:
                         if camera not in self.rover_data[rover]['cameras']:
                              # Initialize camera info structure
                              self.rover_data[rover]['cameras'][camera] = {'modes': set(), 'target': None}
                elif pred == 'supports':
                    camera, mode = args
                    # Find which rover has this camera and add the mode
                    for r_data in self.rover_data.values():
                        if camera in r_data['cameras']:
                            r_data['cameras'][camera]['modes'].add(mode)
                            break # Assume camera is on only one rover
                elif pred == 'calibration_target':
                    camera, objective = args
                     # Find which rover has this camera and set the target
                    for r_data in self.rover_data.values():
                        if camera in r_data['cameras']:
                            r_data['cameras'][camera]['target'] = objective
                            break # Assume camera is on only one rover
                # Store initial sample locations from static facts if available
                # (More robustly handled by checking initial state, but can be here too)
                elif pred == 'at_soil_sample':
                     self.initial_soil_samples.add(args[0])
                elif pred == 'at_rock_sample':
                     self.initial_rock_samples.add(args[0])

            except IndexError:
                 print(f"Warning: Malformed static fact ignored during processing: {fact}")
            except KeyError:
                 # This might happen if discovery missed an object, e.g., rover not in all_rovers
                 print(f"Warning: Static fact refers to potentially unknown object: {fact}")

        # Ensure initial sample locations are captured from the initial state as well,
        # as they might not be static.
        for fact in task.initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             args = parts[1:]
             if pred == 'at_soil_sample':
                 self.initial_soil_samples.add(args[0])
             elif pred == 'at_rock_sample':
                 self.initial_rock_samples.add(args[0])

        if self.lander_location is None:
            # Try finding lander location in initial state if not static
            for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'at_lander':
                     self.lander_location = parts[2]
                     break
            if self.lander_location is None:
                 print("CRITICAL WARNING: Lander location could not be determined from static facts or initial state.")


    # Helper method for checking fact patterns in the state
    def _state_contains_pattern(self, state, pred, *args):
        """Checks if any fact in the state matches the predicate and arguments (wildcard '*' allowed)."""
        for fact in state:
            parts = fact[1:-1].split() # Remove brackets and split
            if parts[0] == pred and len(parts) - 1 == len(args):
                match = True
                # Compare arguments, respecting wildcards
                for i, arg_pattern in enumerate(args):
                    if arg_pattern != '*' and parts[i+1] != arg_pattern:
                        match = False
                        break
                if match:
                    return True # Found at least one match
        return False

    # Helper method for checking exact fact existence
    def _state_contains_fact(self, state, fact_str):
        """Checks if the exact fact string exists in the state."""
        return fact_str in state


    def __call__(self, node):
        state = node.state
        heuristic_value = 0

        # --- Calculate heuristic value ---
        for goal_fact in self.goals:
            if self._state_contains_fact(state, goal_fact):
                continue # Goal already achieved

            goal_parts = get_parts(goal_fact)
            goal_pred = goal_parts[0]
            cost_for_goal = 0

            if goal_pred == 'communicated_soil_data':
                wp = goal_parts[1]
                # 1. Cost for communication action (always needed if goal unmet)
                cost_for_goal += 1

                # 2. Check if sampling is needed (prerequisite for communication)
                if not self._state_contains_pattern(state, 'have_soil_analysis', '*', wp):
                    cost_for_goal += 1 # Need sample_soil action

                    # 3. Check if drop is needed before sampling
                    # Condition: No capable rover has an empty store AND the sample still exists at the location
                    can_sample_directly = False
                    for r_name, r_info in self.rover_data.items():
                        # Check capability and if rover has a store assigned
                        if 'soil' in r_info['capabilities'] and r_info['store']:
                            if self._state_contains_fact(state, f'(empty {r_info["store"]})'):
                                can_sample_directly = True
                                break # Found a rover that can sample

                    # Add drop cost only if needed and possible
                    if not can_sample_directly and self._state_contains_pattern(state, 'at_soil_sample', wp):
                         # Check if *any* capable rover has a full store (implying drop is possible)
                         drop_possible = False
                         for r_name, r_info in self.rover_data.items():
                             if 'soil' in r_info['capabilities'] and r_info['store']:
                                 if self._state_contains_fact(state, f'(full {r_info["store"]})'):
                                     drop_possible = True
                                     break
                         if drop_possible:
                            cost_for_goal += 1 # Estimate 1 drop action needed

            elif goal_pred == 'communicated_rock_data':
                wp = goal_parts[1]
                # 1. Cost for communication action
                cost_for_goal += 1

                # 2. Check if sampling is needed
                if not self._state_contains_pattern(state, 'have_rock_analysis', '*', wp):
                    cost_for_goal += 1 # Need sample_rock action

                    # 3. Check if drop is needed before sampling
                    can_sample_directly = False
                    for r_name, r_info in self.rover_data.items():
                        if 'rock' in r_info['capabilities'] and r_info['store']:
                             if self._state_contains_fact(state, f'(empty {r_info["store"]})'):
                                can_sample_directly = True
                                break
                    if not can_sample_directly and self._state_contains_pattern(state, 'at_rock_sample', wp):
                         drop_possible = False
                         for r_name, r_info in self.rover_data.items():
                             if 'rock' in r_info['capabilities'] and r_info['store']:
                                 if self._state_contains_fact(state, f'(full {r_info["store"]})'):
                                     drop_possible = True
                                     break
                         if drop_possible:
                            cost_for_goal += 1 # Estimate 1 drop action needed

            elif goal_pred == 'communicated_image_data':
                obj, mode = goal_parts[1], goal_parts[2]
                # 1. Cost for communication action
                cost_for_goal += 1

                # 2. Check if imaging is needed
                if not self._state_contains_pattern(state, 'have_image', '*', obj, mode):
                    cost_for_goal += 1 # Need take_image action

                    # 3. Check if calibration is needed before imaging
                    # Requires finding a suitable rover/camera that is already calibrated
                    found_calibrated_suitable_camera = False
                    for r_name, r_info in self.rover_data.items():
                        # Check if rover has imaging capability
                        if 'imaging' in r_info['capabilities']:
                            for cam_name, cam_info in r_info['cameras'].items():
                                # Check if this camera supports the required mode
                                if mode in cam_info['modes']:
                                    # Check if this specific camera is calibrated on this rover
                                    if self._state_contains_fact(state, f'(calibrated {cam_name} {r_name})'):
                                        found_calibrated_suitable_camera = True
                                        break # Found one, no need to check other cameras on this rover
                            if found_calibrated_suitable_camera:
                                break # Found one, no need to check other rovers

                    # Add calibration cost only if no suitable camera is currently calibrated
                    if not found_calibrated_suitable_camera:
                        # Check if calibration is *possible* (i.e. exists a suitable rover/camera/target)
                        # This check prevents adding cost if the goal is impossible due to lack of equipment
                        calibration_possible = False
                        for r_name, r_info in self.rover_data.items():
                            if 'imaging' in r_info['capabilities']:
                                for cam_name, cam_info in r_info['cameras'].items():
                                    # Check if camera supports mode and has a calibration target defined
                                    if mode in cam_info['modes'] and cam_info.get('target'):
                                        calibration_possible = True
                                        break
                            if calibration_possible:
                                break
                        if calibration_possible:
                            cost_for_goal += 1 # Estimate 1 calibrate action needed

            heuristic_value += cost_for_goal

        # The heuristic should naturally be 0 if all goals are met, because the loop over
        # unmet goals will not run. If the heuristic is 0 but the state is not a goal state,
        # it implies a potential issue in goal checking or calculation logic.
        # The current logic ensures cost_for_goal >= 1 for every unmet goal.
        return heuristic_value
