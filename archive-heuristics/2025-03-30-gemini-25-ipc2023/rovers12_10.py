import fnmatch
import os
import sys
from collections import defaultdict

# Ensure the heuristics directory is in the Python path
# (Adjust path logic as needed for the specific environment)
try:
    # Assumes the script is run in an environment where 'heuristics' package is available
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback: Add the parent directory of the current script's directory to the path
    # This works if the script is inside a 'heuristics' directory and run directly.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    try:
        from heuristics.heuristic_base import Heuristic
    except ImportError:
        # If still not found, raise a clear error
        raise ImportError(
            "Cannot find 'heuristics.heuristic_base'. "
            "Make sure the script is run from a location where the 'heuristics' module is accessible, "
            "or adjust the Python path accordingly."
        )


# Helper functions
def get_parts(fact_string):
    """
    Extracts predicate and arguments from a PDDL fact string like '(pred arg1 arg2)'.
    Removes parentheses and splits by space.
    """
    return fact_string[1:-1].split()

def match(fact_string, *pattern):
    """
    Checks if a PDDL fact string matches a given pattern using fnmatch for wildcard support.

    Args:
        fact_string (str): The PDDL fact string (e.g., "(at rover1 wp1)").
        *pattern: A sequence of strings representing the pattern (e.g., "at", "rover*", "*").

    Returns:
        bool: True if the fact matches the pattern, False otherwise.

    Example:
        match("(at rover1 wp1)", "at", "rover*", "*") -> True
        match("(visible waypoint1 waypoint2)", "visible", "waypoint?", "waypoint?") -> True
        match("(at rover1 wp1)", "at", "camera*", "*") -> False
    """
    parts = get_parts(fact_string)
    if len(parts) != len(pattern):
        return False
    # Check if each part of the fact matches the corresponding pattern element
    return all(fnmatch.fnmatch(part, pat) for part, pat in zip(parts, pattern))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for achieving each unsatisfied goal predicate independently.
    It considers the necessary actions like sampling, calibrating, taking images,
    communicating, and estimates the need for navigation (cost 1 if needed, 0 otherwise)
    and dropping samples based on the current state and rover capabilities. It aims for
    computational efficiency over admissibility, suitable for greedy best-first search.

    # Assumptions
    - The cost of navigation between waypoints is simplified: 0 if a suitable rover is
      already at a required location (or type of location, e.g., visible from lander),
      and 1 otherwise. It doesn't calculate actual path lengths.
    - Resource contention (e.g., multiple goals needing the same rover or camera)
      is ignored; costs are summed as if goals can be pursued independently or
      with minimal interference.
    - If a required sample (soil/rock) is not present at its waypoint (`at_soil/rock_sample` is false),
      the heuristic assumes the corresponding `sample_*` action cannot be performed for that goal,
      effectively ignoring that part of the cost (assuming it might be achieved via another path
      or is impossible, without explicitly calculating reachability).
    - The heuristic assumes at least one rover/camera combination exists that is capable
      of achieving each required task (sampling, imaging).

    # Heuristic Initialization (`__init__`)
    - Parses the task's static facts to build data structures for efficient lookups:
        - Lander location.
        - Rover capabilities (equipment for soil, rock, imaging).
        - Rover-store mapping.
        - Camera details (on which rover, supported modes, calibration targets).
        - Visibility information (objective <-> waypoint, waypoint <-> waypoint, especially lander visibility).
        - Pre-computed sets of capable rovers for different task types.
        - Pre-computed sets of waypoints suitable for communication, calibration, and imaging specific objectives.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize heuristic value `h = 0`.
    2. Parse the current state (`node.state`) to get dynamic information:
       - Rover locations.
       - Store status (which stores are empty).
       - Camera calibration status.
       - Data currently held by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
       - Availability of samples at waypoints (`at_soil_sample`, `at_rock_sample`).
    3. Iterate through each goal predicate `g` defined in `task.goals`.
    4. If `g` is not satisfied in the current state:
       a. Identify the goal type (`communicated_soil_data`, `communicated_rock_data`, `communicated_image_data`).
       b. **Communication Cost:** Add 1 for the `communicate_*` action. Add 1 for navigation cost *unless* any capable rover is already at a waypoint visible from the lander.
       c. **Data Acquisition Cost:**
          i. **If Soil/Rock Data:** Check if the required `have_*_analysis` fact exists for the target waypoint `w`.
             - If NO:
               - Add 1 for the `sample_*` action.
               - Check if the physical sample exists (`at_*_sample w`).
                 - If YES:
                   - Add 1 for navigation to `w` *unless* any capable rover is already at `w`.
                   - Add 1 for `drop` action *if* all capable rovers currently have full stores (no empty stores available among them).
          ii. **If Image Data:** Check if the required `have_image o m` fact exists for the objective `o` and mode `m`.
             - If NO:
               - Add 1 for the `take_image` action.
               - Add 1 for navigation to a waypoint `p` where `o` is visible *unless* any capable rover with a suitable camera is already at such a waypoint `p`.
               - Check if calibration is needed: Is there *any* capable rover `r` with a suitable camera `i` (onboard, supports mode `m`) such that `(calibrated i r)` is true?
                 - If NO (calibration needed):
                   - Add 1 for the `calibrate` action.
                   - Add 1 for navigation to a calibration waypoint `w` (where the camera's target `t` is visible) *unless* any capable rover is already at such a waypoint `w`.
    5. Sum the costs calculated for all unsatisfied goals to get the total heuristic value `h`.
    6. **Final Adjustment:** If `h` is 0 but the goal state is not reached (e.g., due to unachievable goals not detected by the heuristic), return 1 to avoid termination at non-goal states. If the goal is reached, return 0. Otherwise, return `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Initialize data structures using defaultdict for convenience
        self.lander_location = None
        self.rover_stores = {} # rover -> store
        self.rover_equipment = defaultdict(set) # rover -> set{'soil', 'rock', 'image'}
        self.rover_cameras = defaultdict(set) # rover -> set{camera}
        self.camera_supports = defaultdict(set) # camera -> set{mode}
        self.camera_calib_targets = {} # camera -> objective
        self.obj_visible_from = defaultdict(set) # objective -> set{waypoint}
        self.visibility_map = defaultdict(set) # wp -> set{visible wp}

        # Populate data structures from static facts
        for fact in static_facts:
            # Use the match helper function for clarity and consistency
            if match(fact, "at_lander", "?l", "?wp"):
                self.lander_location = get_parts(fact)[2]
            elif match(fact, "store_of", "?s", "?r"):
                self.rover_stores[get_parts(fact)[2]] = get_parts(fact)[1] # rover -> store
            elif match(fact, "equipped_for_soil_analysis", "?r"):
                self.rover_equipment[get_parts(fact)[1]].add('soil')
            elif match(fact, "equipped_for_rock_analysis", "?r"):
                self.rover_equipment[get_parts(fact)[1]].add('rock')
            elif match(fact, "equipped_for_imaging", "?r"):
                self.rover_equipment[get_parts(fact)[1]].add('image')
            elif match(fact, "on_board", "?c", "?r"):
                parts = get_parts(fact)
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> set{camera}
            elif match(fact, "supports", "?c", "?m"):
                parts = get_parts(fact)
                self.camera_supports[parts[1]].add(parts[2]) # camera -> set{mode}
            elif match(fact, "calibration_target", "?c", "?o"):
                parts = get_parts(fact)
                self.camera_calib_targets[parts[1]] = parts[2] # camera -> objective
            elif match(fact, "visible_from", "?o", "?wp"):
                parts = get_parts(fact)
                self.obj_visible_from[parts[1]].add(parts[2]) # objective -> set{waypoint}
            elif match(fact, "visible", "?wp1", "?wp2"):
                parts = get_parts(fact)
                self.visibility_map[parts[1]].add(parts[2])
                # Assume visibility is symmetric based on domain definition and examples
                self.visibility_map[parts[2]].add(parts[1])

        # Precompute sets of capable rovers for faster access during heuristic calculation
        self.soil_capable_rovers = {r for r, eq in self.rover_equipment.items() if 'soil' in eq}
        self.rock_capable_rovers = {r for r, eq in self.rover_equipment.items() if 'rock' in eq}
        self.image_capable_rovers = {r for r, eq in self.rover_equipment.items() if 'image' in eq}

        # Precompute waypoints visible from the lander
        self.waypoints_visible_from_lander = set()
        if self.lander_location:
            self.waypoints_visible_from_lander = self.visibility_map.get(self.lander_location, set())

        # Precompute locations where calibration targets are visible
        self.calib_target_locations = defaultdict(set) # objective -> set{waypoint} where visible
        for cam, target_obj in self.camera_calib_targets.items():
            # Update the set of waypoints where this calibration target objective is visible
            self.calib_target_locations[target_obj].update(self.obj_visible_from.get(target_obj, set()))


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state
        h = 0 # Initialize heuristic value

        # --- Parse current state into efficient lookup structures ---
        rover_locations = {} # rover -> current waypoint
        empty_stores = set() # set of store names that are empty
        calibrated_cameras = set() # set of (camera, rover) tuples that are calibrated
        have_soil = set() # set of waypoints 'w' for which (have_soil_analysis r w) exists
        have_rock = set() # set of waypoints 'w' for which (have_rock_analysis r w) exists
        have_image = set() # set of (objective, mode) tuples for which (have_image r o m) exists
        at_soil_sample = set() # set of waypoints 'w' where (at_soil_sample w) exists
        at_rock_sample = set() # set of waypoints 'w' where (at_rock_sample w) exists

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            # Use direct predicate matching after splitting for efficiency
            if pred == "at" and len(parts) == 3 and parts[1].startswith("rover"):
                rover_locations[parts[1]] = parts[2]
            elif pred == "empty" and len(parts) == 2:
                empty_stores.add(parts[1])
            elif pred == "calibrated" and len(parts) == 3:
                calibrated_cameras.add((parts[1], parts[2])) # (camera, rover)
            elif pred == "have_soil_analysis" and len(parts) == 3:
                have_soil.add(parts[2]) # waypoint
            elif pred == "have_rock_analysis" and len(parts) == 3:
                have_rock.add(parts[2]) # waypoint
            elif pred == "have_image" and len(parts) == 4:
                have_image.add((parts[2], parts[3])) # (objective, mode)
            elif pred == "at_soil_sample" and len(parts) == 2:
                at_soil_sample.add(parts[1]) # waypoint
            elif pred == "at_rock_sample" and len(parts) == 2:
                at_rock_sample.add(parts[1]) # waypoint

        # --- Iterate through goals and estimate cost if unsatisfied ---
        for goal_fact in self.goals:
            if goal_fact not in state:
                goal_parts = get_parts(goal_fact)
                goal_type = goal_parts[0]

                # Determine the set of rovers potentially relevant for this goal type
                relevant_rovers = set()
                if goal_type == "communicated_soil_data":
                    relevant_rovers = self.soil_capable_rovers
                elif goal_type == "communicated_rock_data":
                    relevant_rovers = self.rock_capable_rovers
                elif goal_type == "communicated_image_data":
                    relevant_rovers = self.image_capable_rovers

                # --- Estimate Communication Cost ---
                # Cost = 1 (communicate action) + navigation cost (0 or 1)
                comm_nav_needed = 1 # Assume navigation is needed
                for r in relevant_rovers:
                    # If any relevant rover is currently at a waypoint visible from the lander
                    if r in rover_locations and rover_locations[r] in self.waypoints_visible_from_lander:
                         comm_nav_needed = 0 # No navigation needed for communication step
                         break
                h += 1 # Add cost for the communicate_* action itself
                h += comm_nav_needed # Add estimated navigation cost (0 or 1)

                # --- Estimate Data Acquisition Cost ---
                if goal_type == "communicated_soil_data":
                    w = goal_parts[1] # Target waypoint for soil data
                    if w not in have_soil: # Check if we already have the analysis data
                        h += 1 # Add cost for sample_soil action
                        if w in at_soil_sample: # Check if the sample physically exists at the waypoint
                            # Estimate drop cost: 1 if no capable rover has an empty store
                            needs_drop = True
                            for r in self.soil_capable_rovers:
                                store = self.rover_stores.get(r) # Get store associated with rover r
                                if store and store in empty_stores:
                                    needs_drop = False # Found a capable rover with an empty store
                                    break
                            if needs_drop: h += 1 # Add cost for drop action

                            # Estimate navigation to sample cost: 1 unless a capable rover is already there
                            sample_nav_needed = 1
                            for r in self.soil_capable_rovers:
                                 if r in rover_locations and rover_locations[r] == w:
                                     sample_nav_needed = 0 # Found a capable rover at the sample location
                                     break
                            h += sample_nav_needed # Add estimated navigation cost (0 or 1)

                elif goal_type == "communicated_rock_data":
                    w = goal_parts[1] # Target waypoint for rock data
                    if w not in have_rock: # Check if we already have the analysis data
                        h += 1 # Add cost for sample_rock action
                        if w in at_rock_sample: # Check if the sample physically exists
                            # Estimate drop cost
                            needs_drop = True
                            for r in self.rock_capable_rovers:
                                store = self.rover_stores.get(r)
                                if store and store in empty_stores:
                                    needs_drop = False
                                    break
                            if needs_drop: h += 1

                            # Estimate navigation to sample cost
                            sample_nav_needed = 1
                            for r in self.rock_capable_rovers:
                                 if r in rover_locations and rover_locations[r] == w:
                                     sample_nav_needed = 0
                                     break
                            h += sample_nav_needed

                elif goal_type == "communicated_image_data":
                    o, m = goal_parts[1], goal_parts[2] # Target objective and mode
                    if (o, m) not in have_image: # Check if we already have the image data
                        h += 1 # Add cost for take_image action

                        # Estimate navigation to image cost: 1 unless a capable rover with a
                        # suitable camera is already at a waypoint where the objective is visible.
                        image_nav_needed = 1
                        possible_image_wps = self.obj_visible_from.get(o, set()) # Waypoints where 'o' is visible
                        rover_at_image_loc = False
                        for r in self.image_capable_rovers:
                            # Check if rover is at a possible imaging location
                            if r in rover_locations and rover_locations[r] in possible_image_wps:
                                # Check if this rover has a camera that supports the required mode
                                for cam in self.rover_cameras.get(r, set()):
                                    if m in self.camera_supports.get(cam, set()):
                                        rover_at_image_loc = True # Found suitable rover/camera at location
                                        break
                            if rover_at_image_loc: break # Exit outer loop once found
                        if rover_at_image_loc: image_nav_needed = 0 # No navigation needed if already there
                        h += image_nav_needed # Add estimated navigation cost (0 or 1)

                        # Estimate calibration cost
                        needs_calibration = True # Assume calibration is needed initially
                        rover_at_calib_loc = False # Tracks if any capable rover is at a potential calibration spot

                        # Check if any suitable camera on a capable rover is already calibrated
                        for r in self.image_capable_rovers:
                            for cam in self.rover_cameras.get(r, set()):
                                # Check if this camera supports the required mode
                                if m in self.camera_supports.get(cam, set()):
                                    # Check if this specific camera/rover pair is calibrated
                                    if (cam, r) in calibrated_cameras:
                                        needs_calibration = False # Found a calibrated suitable camera
                                        break # No need to check further cameras for this rover

                                    # If not calibrated, check if rover is at a location to calibrate this camera
                                    target_obj = self.camera_calib_targets.get(cam) # Get calibration target for this camera
                                    if target_obj:
                                        # Find waypoints where this target is visible
                                        wps_for_this_calib = self.calib_target_locations.get(target_obj, set())
                                        # Check if the current rover is at one of these waypoints
                                        if r in rover_locations and rover_locations[r] in wps_for_this_calib:
                                            rover_at_calib_loc = True
                                            # Note: We don't break here, as another camera might already be calibrated.
                                            # rover_at_calib_loc just means *a* calibration *could* happen without moving.

                            if not needs_calibration: break # Exit outer loop if a calibrated camera was found

                        # If no suitable calibrated camera was found across all capable rovers
                        if needs_calibration:
                            h += 1 # Add cost for the calibrate action
                            # Estimate navigation cost for calibration: 1 unless a rover is already positioned
                            calib_nav_needed = 0 if rover_at_calib_loc else 1
                            h += calib_nav_needed # Add estimated navigation cost (0 or 1)


        # --- Final Adjustment ---
        # The heuristic value `h` should be 0 if and only if the goal state is reached.
        if self.goals <= state:
            # If all goal predicates are present in the current state set
            return 0
        elif h == 0:
            # If goals are not met, but the heuristic calculated 0 (e.g., due to perceived
            # unreachability based on simple checks like missing samples), return 1.
            # This ensures the search doesn't prematurely stop if the state is not truly a goal,
            # and helps explore other paths if this one seems blocked.
            return 1
        else:
            # Otherwise, return the calculated heuristic value
            return h
