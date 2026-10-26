from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    # Handle potential empty strings after split
    return [part for part in fact[1:-1].split(' ') if part]

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.
    Pattern arguments can include wildcards '*'.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    each uncommunicated goal (soil data, rock data, images). It sums
    the estimated costs for each goal independently, ignoring potential
    synergies or negative interactions between goals.

    # Assumptions
    - Navigation between any two waypoints is possible with a cost of 1
      for heuristic purposes, if the rover is not already at the target.
    - For sampling goals, a suitable equipped rover and an available store
      exist. If the store is full, one 'drop' action is needed.
    - For imaging goals, a suitable equipped rover with a camera supporting
      the required mode exists. Calibration is needed before taking an image
      if the camera is not calibrated. A calibration target and visible
      waypoint for calibration exist. An image waypoint visible from the
      objective exists.
    - A lander exists and is at a fixed location. There is always a waypoint
      visible from the lander that the rover can reach.
    - The initial soil/rock samples exist at their locations if required by goals.
      (The heuristic doesn't check if `at_soil_sample` or `at_rock_sample`
       are still true in the current state, assuming the goal is achievable
       via sampling if the data is not yet collected).

    # Heuristic Initialization
    The heuristic extracts static information from the task definition,
    including lander locations, rover capabilities, store ownership,
    camera properties (support, on-board, calibration targets), waypoint
    visibility, and objective visibility from waypoints. This information
    is stored in dictionaries and sets for quick lookup during heuristic
    computation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic iterates through each goal predicate that is not yet
    satisfied in the current state. For each unachieved goal, it estimates
    the minimum number of actions required to achieve it, assuming ideal
    conditions (e.g., availability of rovers, stores, visible waypoints)
    and sums these estimates.

    For an unachieved `(communicated_soil_data ?w)` goal:
    1. Check if `(have_soil_analysis ?r ?w)` is true for any rover `r`.
    2. If YES (data collected):
       - Add 1 for `communicate_soil_data`.
       - Find a rover `r` with the data. Find its current location `L`.
       - Check if `L` is visible from any lander location. If not, add 1 for navigation.
    3. If NO (data not collected):
       - Find an equipped rover `r`.
       - Find its current location `L`. If `L != ?w`, add 1 for navigation to `?w`.
       - Check if `r`'s store is full. If yes, add 1 for `drop`.
       - Add 1 for `sample_soil`.
       - The rover is now at `?w`. Check if `?w` is visible from any lander location. If not, add 1 for navigation from `?w` to a lander-visible waypoint.
       - Add 1 for `communicate_soil_data`.

    For an unachieved `(communicated_rock_data ?w)` goal:
    Similar logic as soil data, replacing soil-specific predicates/equipment.

    For an unachieved `(communicated_image_data ?o ?m)` goal:
    1. Check if `(have_image ?r ?o ?m)` is true for any rover `r`.
    2. If YES (image taken):
       - Add 1 for `communicate_image_data`.
       - Find a rover `r` with the image. Find its current location `L`.
       - Check if `L` is visible from any lander location. If not, add 1 for navigation.
    3. If NO (image not taken):
       - Find an equipped rover `r` with a camera `i` supporting mode `m`.
       - Find its current location `L`.
       - Check if `(calibrated ?i ?r)` is true.
         - If NO (needs calibration):
           - Find calibration target `t` for `i`. Find a waypoint `W_cal` visible from `t`.
           - If `L != W_cal`, add 1 for navigation to `W_cal`.
           - Add 1 for `calibrate`.
           - The rover is now at `W_cal`. Find a waypoint `W_img` visible from `o`.
           - If `W_cal != W_img`, add 1 for navigation from `W_cal` to `W_img`.
           - The rover is now at `W_img`.
         - If YES (calibrated):
           - Find a waypoint `W_img` visible from `o`.
           - If `L != W_img`, add 1 for navigation to `W_img`.
           - The rover is now at `W_img`.
       - Add 1 for `take_image`.
       - The rover is now at `W_img`. Check if `W_img` is visible from any lander location. If not, add 1 for navigation from `W_img` to a lander-visible waypoint.
       - Add 1 for `communicate_image_data`.

    The total heuristic value is the sum of these estimated costs for all
    unachieved goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_locations = {get_parts(fact)[2] for fact in self.static if match(fact, 'at_lander', '*', '*')}
        self.visible_pairs = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in self.static if match(fact, 'visible', '*', '*')}
        self.objective_visible_from = {}
        for fact in self.static:
            if match(fact, 'visible_from', '*', '*'):
                obj, wp = get_parts(fact)[1:3]
                if obj not in self.objective_visible_from:
                    self.objective_visible_from[obj] = set()
                self.objective_visible_from[obj].add(wp)

        self.calibration_targets = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, 'calibration_target', '*', '*')}
        self.camera_supports = {}
        for fact in self.static:
             if match(fact, 'supports', '*', '*'):
                 cam, mode = get_parts(fact)[1:3]
                 if cam not in self.camera_supports:
                     self.camera_supports[cam] = set()
                 self.camera_supports[cam].add(mode)

        self.camera_on_board = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, 'on_board', '*', '*')}
        self.rover_equipped_soil = {get_parts(fact)[1] for fact in self.static if match(fact, 'equipped_for_soil_analysis', '*')}
        self.rover_equipped_rock = {get_parts(fact)[1] for fact in self.static if match(fact, 'equipped_for_rock_analysis', '*')}
        self.rover_equipped_imaging = {get_parts(fact)[1] for fact in self.static if match(fact, 'equipped_for_imaging', '*')}
        self.store_owners = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, 'store_of', '*', '*')}

        # Pre-calculate a set of waypoints visible from any lander
        self.lander_visible_wps = set()
        for wp1, wp2 in self.visible_pairs:
            if wp2 in self.lander_locations:
                self.lander_visible_wps.add(wp1)
            if wp1 in self.lander_locations: # Visibility is often symmetric
                 self.lander_visible_wps.add(wp2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Define simplified costs
        NAV_COST = 1
        SAMPLE_COST = 1
        DROP_COST = 1
        CALIBRATE_COST = 1
        TAKE_IMAGE_COST = 1
        COMMUNICATE_COST = 1

        # Find current rover locations
        rover_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')}

        # Iterate through unachieved goals
        unachieved_goals = self.goals - state

        for goal in unachieved_goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                # Check if data is already collected by *any* rover
                has_data = any(match(fact, 'have_soil_analysis', '*', waypoint) for fact in state)

                if has_data:
                    # Data exists, need to communicate
                    h += COMMUNICATE_COST
                    # Find a rover with the data (approximate, just need one)
                    rover_with_data = next((r for fact in state if match(fact, 'have_soil_analysis', r, waypoint)), None)
                    if rover_with_data and rover_with_data in rover_locations:
                        rover_loc = rover_locations[rover_with_data]
                        # Check if rover is at a waypoint visible from a lander
                        if rover_loc not in self.lander_visible_wps:
                            h += NAV_COST # Navigate to a lander-visible waypoint
                    # If rover_with_data or its location is unknown, assume nav is needed
                    else:
                         h += NAV_COST


                else:
                    # Data does not exist, need to sample and communicate
                    # Find an equipped rover (approximate, just need one)
                    equipped_rover = next((r for r in self.rover_equipped_soil), None)
                    if equipped_rover and equipped_rover in rover_locations:
                        rover_loc = rover_locations[equipped_rover]

                        # Cost to sample
                        if rover_loc != waypoint:
                            h += NAV_COST # Navigate to sample waypoint

                        # Check if store is full for this rover
                        rover_store = next((s for s, r in self.store_owners.items() if r == equipped_rover), None)
                        if rover_store and match(f'(full {rover_store})', '*', '*') in state:
                             h += DROP_COST # Drop current sample

                        h += SAMPLE_COST # Sample soil

                        # Cost to communicate (rover is now at waypoint, or estimated)
                        # Check if sample waypoint is visible from a lander
                        if waypoint not in self.lander_visible_wps:
                            h += NAV_COST # Navigate from sample waypoint to lander-visible waypoint

                        h += COMMUNICATE_COST # Communicate
                    # If no equipped rover found or its location is unknown, skip adding cost for this goal


            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                # Check if data is already collected by *any* rover
                has_data = any(match(fact, 'have_rock_analysis', '*', waypoint) for fact in state)

                if has_data:
                    # Data exists, need to communicate
                    h += COMMUNICATE_COST
                    rover_with_data = next((r for fact in state if match(fact, 'have_rock_analysis', r, waypoint)), None)
                    if rover_with_data and rover_with_data in rover_locations:
                        rover_loc = rover_locations[rover_with_data]
                        if rover_loc not in self.lander_visible_wps:
                            h += NAV_COST
                    else:
                         h += NAV_COST

                else:
                    # Data does not exist, need to sample and communicate
                    equipped_rover = next((r for r in self.rover_equipped_rock), None)
                    if equipped_rover and equipped_rover in rover_locations:
                        rover_loc = rover_locations[equipped_rover]

                        # Cost to sample
                        if rover_loc != waypoint:
                            h += NAV_COST

                        rover_store = next((s for s, r in self.store_owners.items() if r == equipped_rover), None)
                        if rover_store and match(f'(full {rover_store})', '*', '*') in state:
                             h += DROP_COST

                        h += SAMPLE_COST

                        # Cost to communicate (rover is now at waypoint, or estimated)
                        if waypoint not in self.lander_visible_wps:
                            h += NAV_COST

                        h += COMMUNICATE_COST
                    # If no equipped rover found or its location is unknown, skip adding cost for this goal


            elif predicate == 'communicated_image_data':
                objective, mode = parts[1:3]
                # Check if image is already taken by *any* rover
                has_image = any(match(fact, 'have_image', '*', objective, mode) for fact in state)

                if has_image:
                    # Image exists, need to communicate
                    h += COMMUNICATE_COST
                    rover_with_image = next((r for fact in state if match(fact, 'have_image', r, objective, mode)), None)
                    if rover_with_image and rover_with_image in rover_locations:
                        rover_loc = rover_locations[rover_with_image]
                        if rover_loc not in self.lander_visible_wps:
                            h += NAV_COST
                    else:
                         h += NAV_COST

                else:
                    # Image does not exist, need to take image and communicate
                    # Find an equipped rover with a suitable camera (approximate)
                    suitable_rover_cam = next(((r, c) for r in self.rover_equipped_imaging for c, owner in self.camera_on_board.items() if owner == r and mode in self.camera_supports.get(c, set())), None)

                    if suitable_rover_cam and suitable_rover_cam[0] in rover_locations:
                        equipped_rover, camera = suitable_rover_cam
                        rover_loc = rover_locations[equipped_rover]

                        # Need to be at a waypoint visible from the objective
                        image_wps = self.objective_visible_from.get(objective, set())
                        # Pick an arbitrary image waypoint if multiple exist
                        target_image_wp = next(iter(image_wps), None)

                        if target_image_wp: # Only proceed if an image waypoint exists
                            # Need camera calibrated
                            is_calibrated = match(f'(calibrated {camera} {equipped_rover})', '*', '*') in state

                            current_rover_wp = rover_loc # Start location for this sequence

                            if not is_calibrated:
                                # Need to calibrate first
                                cal_target_obj = self.calibration_targets.get(camera)
                                if cal_target_obj:
                                    cal_wps = self.objective_visible_from.get(cal_target_obj, set())
                                    # Pick an arbitrary calibration waypoint
                                    target_cal_wp = next(iter(cal_wps), None)

                                    if target_cal_wp: # Only proceed if a calibration waypoint exists
                                        if current_rover_wp != target_cal_wp:
                                            h += NAV_COST # Nav L -> W_cal
                                        h += CALIBRATE_COST # Calibrate
                                        current_rover_wp = target_cal_wp # Rover is now at cal waypoint

                                    # If no cal target or cal waypoint, calibration path is blocked, skip calibration costs

                            # After potential calibration, rover is at current_rover_wp (either original L or W_cal)
                            # Need to navigate to image waypoint
                            if current_rover_wp != target_image_wp:
                                 h += NAV_COST # Nav current -> W_img
                            # Rover is now at target_image_wp (for heuristic purposes)

                            h += TAKE_IMAGE_COST # Take image

                            # Cost to communicate (rover is now at target_image_wp)
                            if target_image_wp not in self.lander_visible_wps:
                                h += NAV_COST # Nav W_img -> LanderView

                            h += COMMUNICATE_COST # Communicate
                        # If no image waypoint found, skip adding cost for this goal
                    # If no suitable rover/camera found or its location is unknown, skip adding cost for this goal


        # Ensure heuristic is 0 only at goal state
        if h == 0 and unachieved_goals:
             # This indicates unachieved goals exist, but the heuristic calculated 0.
             # This might happen if required resources/waypoints were not found
             # during the static lookup, causing costs to be skipped.
             # Return a small value to indicate it's not a goal state.
             return 1
        elif h == 0 and not unachieved_goals:
             # Goal state reached
             return 0
        else:
             return h
