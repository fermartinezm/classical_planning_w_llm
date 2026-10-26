from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It counts the core actions (sample, take_image, calibrate,
    communicate) needed for each unachieved goal and adds a fixed cost for
    associated navigation and preparation steps. It specifically tracks the
    need for image calibration based on the number of images still to be taken
    versus the number of currently calibrated suitable cameras.

    # Assumptions
    - All goal conditions are reachable in principle (e.g., required equipment exists,
      objectives are visible from some waypoint, calibration targets exist).
    - Navigation and preparation (like emptying a store) can typically be done
      in a small, fixed number of actions (estimated as 1 per major step).
    - Any equipped rover can perform the necessary actions (sampling, imaging,
      communicating) if at the right location with the right tools/state.
    - A calibrated camera can be used for any image it supports, regardless of
      which specific image goal triggered the calibration need.

    # Heuristic Initialization
    - Extracts static information such as lander location, waypoint visibility
      (used to find communication waypoints), camera calibration targets,
      camera-mode support, rover equipment, and camera assignments to rovers.
      This information is stored in dictionaries and sets for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic iterates through each goal condition that is not yet satisfied
    in the current state and adds an estimated cost for achieving it.

    1.  **Identify Unachieved Goals:** Determine which `communicated_soil_data`,
        `communicated_rock_data`, and `communicated_image_data` facts from the
        problem's goals are not present in the current state.

    2.  **Process Soil and Rock Goals:** For each unachieved `(communicated_soil_data ?w)`
        or `(communicated_rock_data ?w)` goal:
        -   Add 1 for the `communicate_soil_data` or `communicate_rock_data` action.
        -   Check if the corresponding `(have_soil_analysis ?r ?w)` or
            `(have_rock_analysis ?r ?w)` fact exists for any rover `?r` in the state.
        -   If the `have_analysis` fact is missing:
            -   Add 1 for the `sample_soil` or `sample_rock` action.
            -   Add 1 for the estimated navigation and preparation (e.g., moving to the waypoint, ensuring store is empty) needed before sampling. This cost is added only if the sample is still physically present at the waypoint, implying it needs to be sampled.

    3.  **Process Image Goals:** For each unachieved `(communicated_image_data ?o ?m)` goal:
        -   Add 1 for the `communicate_image_data` action.
        -   Check if the corresponding `(have_image ?r ?o ?m)` fact exists for any rover `?r` in the state.
        -   If the `have_image` fact is missing:
            -   Increment a counter for needed `take_image` actions.
            -   Add 1 for the estimated navigation and preparation needed before taking the image (moving to a waypoint visible from the objective).

    4.  **Calculate Calibration Costs:** After processing all image goals, determine the total number of `take_image` actions needed (`needed_take_image_count`).
        -   Count the number of suitable cameras that are *currently* calibrated in the state. A camera `(i, r)` is suitable if rover `r` is equipped for imaging, camera `i` is on board `r`, camera `i` supports the mode `m` of *any* unachieved image goal `(o, m)`, and objective `o` is visible from at least one waypoint.
        -   The number of *additional* calibrations required is `max(0, needed_take_image_count - num_currently_calibrated_suitable_cams)`.
        -   For each additional calibration needed, add 1 for the `calibrate` action and 1 for the estimated navigation and preparation needed before calibrating (moving to a waypoint visible from the calibration target).

    5.  **Sum Costs:** The total heuristic value is the sum of all costs accumulated in steps 2, 3, and 4. The heuristic is 0 if and only if all goal conditions are met.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        self.lander_loc = None
        self.obj_visible_from = {} # objective/target -> set(waypoints)
        self.cam_target = {}       # camera -> target
        self.rover_cams = {}       # rover -> set(cameras)
        self.cam_modes = {}        # camera -> set(modes)
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()

        # Although not used for navigation cost in this heuristic,
        # collecting visible links is needed to find communication waypoints.
        visible_links = set() # (wp_from, wp_to)

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_lander":
                # Assuming only one lander
                self.lander_loc = parts[2]
            elif predicate == "visible_from":
                obj, wp = parts[1], parts[2]
                self.obj_visible_from.setdefault(obj, set()).add(wp)
            elif predicate == "calibration_target":
                cam, target = parts[1], parts[2]
                self.cam_target[cam] = target
            elif predicate == "on_board":
                cam, rover = parts[1], parts[2]
                self.rover_cams.setdefault(rover, set()).add(cam)
            elif predicate == "supports":
                cam, mode = parts[1], parts[2]
                self.cam_modes.setdefault(cam, set()).add(mode)
            elif predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif predicate == "visible": # Collect visible links
                 wp1, wp2 = parts[1], parts[2]
                 visible_links.add((wp1, wp2))

        # Precompute communication waypoints (visible from lander_loc)
        self.comm_wps = set()
        if self.lander_loc:
             for wp1, wp2 in visible_links:
                 if wp1 == self.lander_loc:
                     self.comm_wps.add(wp2)
                 if wp2 == self.lander_loc:
                     self.comm_wps.add(wp1)
             # The lander location itself is also a communication waypoint
             self.comm_wps.add(self.lander_loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract Dynamic Information from State ---
        soil_samples_at = set() # waypoints with soil samples
        rock_samples_at = set() # waypoints with rock samples
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        calibrated_cams = set() # (camera, rover)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_soil_sample":
                soil_samples_at.add(parts[1])
            elif predicate == "at_rock_sample":
                rock_samples_at.add(parts[1])
            elif predicate == "have_soil_analysis":
                if len(parts) > 2:
                    have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                 if len(parts) > 2:
                    have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                 if len(parts) > 3:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "calibrated":
                 if len(parts) > 2:
                    calibrated_cams.add((parts[1], parts[2]))
            elif predicate == "communicated_soil_data":
                 if len(parts) > 1:
                    communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                 if len(parts) > 1:
                    communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                 if len(parts) > 2:
                    communicated_image.add((parts[1], parts[2]))

        # --- Compute Heuristic ---
        h = 0

        # Identify unachieved goals
        soil_goals_wps = set()
        rock_goals_wps = set()
        image_goals_om = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "communicated_soil_data":
                if len(parts) > 1: # Ensure waypoint exists
                    wp = parts[1]
                    if wp not in communicated_soil:
                        soil_goals_wps.add(wp)
            elif predicate == "communicated_rock_data":
                 if len(parts) > 1: # Ensure waypoint exists
                    wp = parts[1]
                    if wp not in communicated_rock:
                        rock_goals_wps.add(wp)
            elif predicate == "communicated_image_data":
                 if len(parts) > 2: # Ensure objective and mode exist
                    o, m = parts[1], parts[2]
                    if (o, m) not in communicated_image:
                        image_goals_om.add((o, m))

        # Cost for Soil Goals
        for w in soil_goals_wps:
            h += 1 # communicate action
            # Check if any equipped rover has the analysis data
            if not any((r, w) in have_soil for r in self.equipped_soil):
                 h += 1 # sample action
                 # Add cost for navigation/prep only if sample is still available to be sampled
                 if w in soil_samples_at:
                     h += 1 # nav/prep for sample

        # Cost for Rock Goals
        for w in rock_goals_wps:
            h += 1 # communicate action
            # Check if any equipped rover has the analysis data
            if not any((r, w) in have_rock for r in self.equipped_rock):
                 h += 1 # sample action
                 # Add cost for navigation/prep only if sample is still available to be sampled
                 if w in rock_samples_at:
                     h += 1 # nav/prep for sample

        # Cost for Image Goals
        needed_take_image_count = 0
        for (o, m) in image_goals_om:
            h += 1 # communicate action
            # Check if any equipped imaging rover has the image
            if not any((r, o, m) in have_image for r in self.equipped_imaging):
                needed_take_image_count += 1
                h += 1 # nav/prep for take_image (nav to image location)

        # Calculate Calibration Costs based on needed images vs calibrated cameras
        currently_calibrated_suitable_cams = set()
        for (cam, r) in calibrated_cams:
            if r in self.equipped_imaging:
                # Check if this camera/rover can potentially satisfy any remaining image goal
                is_suitable = False
                for (o, m) in image_goals_om:
                    # Check if camera supports mode AND objective is visible from anywhere
                    if m in self.cam_modes.get(cam, set()):
                         if self.obj_visible_from.get(o):
                             is_suitable = True
                             break
                if is_suitable:
                    currently_calibrated_suitable_cams.add((cam, r))

        num_calibrated_suitable = len(currently_calibrated_suitable_cams)
        num_calibrations_needed = max(0, needed_take_image_count - num_calibrated_suitable)

        h += num_calibrations_needed * (1 + 1) # 1 for calibrate action, 1 for nav/prep for calibrate

        return h
