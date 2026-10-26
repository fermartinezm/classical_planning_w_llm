from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint2)".
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
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It sums the estimated costs for each unsatisfied goal
    independently. The cost for each goal is estimated by considering the
    sequence of actions needed (e.g., sample/image, communicate) and adding
    a cost of 1 for each necessary navigation step or prerequisite action
    (like dropping a sample or calibrating a camera) that is not currently met.
    It finds the minimum cost over available rovers and cameras for each goal.

    # Assumptions
    - The heuristic assumes that necessary equipment (soil/rock/imaging),
      stores, cameras, and calibration targets exist on rovers as defined
      in the static facts.
    - Navigation between waypoints costs a fixed amount (1 action) if the
      rover is not already at a suitable location for the next required step.
      This ignores actual path length and graph connectivity beyond simple
      visibility/reachability checks derived from static facts.
    - Dropping a sample costs 1 action if the store is full and a new sample
      needs to be collected.
    - Calibrating a camera costs 1 action if the camera is not calibrated
      and an image needs to be taken with it.
    - The heuristic does not account for resource contention (e.g., multiple
      rovers needing the same store or camera simultaneously) or fuel/power
      constraints if they existed.
    - Goals are treated independently; achieving one goal might help another,
      but this is only partially captured by checking for existing data/images.

    # Heuristic Initialization
    The heuristic constructor extracts static information from the task,
    including:
    - Which rovers are equipped for soil, rock, and imaging analysis.
    - Which store belongs to which rover.
    - Which cameras are on board which rovers.
    - Which modes each camera supports.
    - Which objective is the calibration target for each camera.
    - Which waypoints are visible from which objectives/calibration targets.
    - Which waypoints are visible from other waypoints (used for communication location checks).
    - The location of the lander.
    - Sets of all rovers, cameras, stores, waypoints, objectives, and modes
      mentioned in the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Parse the current state to identify:
        - The current location of each rover.
        - Which rovers have collected soil data for which waypoints.
        - Which rovers have collected rock data for which waypoints.
        - Which rovers have taken which images (objective, mode).
        - Which stores are currently full.
        - Which cameras are currently calibrated on which rovers.
    2.  Initialize the total heuristic cost to 0.
    3.  Iterate through each goal condition specified in the task.
    4.  For each goal condition:
        - If the goal condition is already true in the current state, add 0 to the total cost and continue to the next goal.
        - If the goal is `(communicated_soil_data ?w)`:
            - Calculate the minimum cost to achieve this goal across all suitable rovers.
            - A suitable rover is one equipped for soil analysis.
            - The cost depends on whether the soil data `(have_soil_analysis ?r ?w)` already exists for that rover.
            - If data exists: Cost is 1 (communicate) + 1 (if rover's current location is not visible from the lander).
            - If data does not exist: Cost is 1 (sample) + 1 (communicate) + 1 (if rover's store is full, for drop) + 1 (if rover's current location is not `?w`, for navigation to sample location) + 1 (if location after sampling - assumed to be `?w` - is not visible from the lander, for navigation to communication location).
            - Take the minimum cost over all suitable rovers. Add this minimum cost to the total.
        - If the goal is `(communicated_rock_data ?w)`:
            - Calculate the minimum cost similarly to soil data, using rock analysis equipment and rock data facts.
            - Add this minimum cost to the total.
        - If the goal is `(communicated_image_data ?o ?m)`:
            - Calculate the minimum cost to achieve this goal across all suitable rover-camera pairs.
            - A suitable pair is a rover equipped for imaging with a camera on board that supports mode `?m`.
            - The cost depends on whether the image `(have_image ?r ?o ?m)` already exists for that rover.
            - If image exists: Cost is 1 (communicate) + 1 (if rover's current location is not visible from the lander).
            - If image does not exist: Cost is 1 (take_image) + 1 (communicate) + 1 (if camera is not calibrated, for calibrate) + 1 (if calibrate needed and rover's current location is not visible from the camera's calibration target, for navigation to calibration location) + 1 (if rover's current location - or location after calibration nav - is not visible from objective `?o`, for navigation to image location) + 1 (if location after imaging - assumed to be the image location - is not visible from the lander, for navigation to communication location).
            - Take the minimum cost over all suitable rover-camera pairs. Add this minimum cost to the total.
    5.  Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts into lookup structures.
        """
        self.goals = task.goals
        static_facts = task.static

        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = {} # {rover: set(camera)}
        self.camera_modes = {} # {camera: set(mode)}
        self.camera_calibration_target = {} # {camera: target_objective}
        self.objective_visible_from_wp = {} # {objective: set(waypoint)}
        self.waypoint_visible_wp = {} # {waypoint: set(waypoint)}
        self.lander_location = None

        # Collect all object names mentioned in static facts for type checking in __call__
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_stores = set()
        self.all_waypoints = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_landers = set()


        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == 'equipped_for_soil_analysis':
                if len(parts) > 1: self.equipped_soil.add(parts[1])
                if len(parts) > 1: self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                if len(parts) > 1: self.equipped_rock.add(parts[1])
                if len(parts) > 1: self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                if len(parts) > 1: self.equipped_imaging.add(parts[1])
                if len(parts) > 1: self.all_rovers.add(parts[1])
            elif predicate == 'store_of':
                if len(parts) > 2:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store
                    self.all_stores.add(store)
                    self.all_rovers.add(rover)
            elif predicate == 'on_board':
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    self.rover_cameras.setdefault(rover, set()).add(camera)
                    self.all_cameras.add(camera)
                    self.all_rovers.add(rover)
            elif predicate == 'supports':
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    self.camera_modes.setdefault(camera, set()).add(mode)
                    self.all_cameras.add(camera)
                    self.all_modes.add(mode)
            elif predicate == 'calibration_target':
                if len(parts) > 2:
                    camera, objective = parts[1], parts[2]
                    self.camera_calibration_target[camera] = objective
                    self.all_cameras.add(camera)
                    self.all_objectives.add(objective)
            elif predicate == 'visible_from':
                if len(parts) > 2:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visible_from_wp.setdefault(objective, set()).add(waypoint)
                    self.all_objectives.add(objective)
                    self.all_waypoints.add(waypoint)
            elif predicate == 'visible':
                if len(parts) > 2:
                    wp1, wp2 = parts[1], parts[2]
                    self.waypoint_visible_wp.setdefault(wp1, set()).add(wp2)
                    self.waypoint_visible_wp.setdefault(wp2, set()).add(wp1) # Visibility is symmetric
                    self.all_waypoints.add(wp1)
                    self.all_waypoints.add(wp2)
            elif predicate == 'at_lander':
                if len(parts) > 2:
                    lander, waypoint = parts[1], parts[2]
                    self.lander_location = waypoint
                    self.all_landers.add(lander)
                    self.all_waypoints.add(waypoint)
            # Collect waypoints from can_traverse too
            elif predicate == 'can_traverse':
                 if len(parts) > 3:
                     r, wp1, wp2 = parts[1], parts[2], parts[3]
                     self.all_rovers.add(r)
                     self.all_waypoints.add(wp1)
                     self.all_waypoints.add(wp2)


        # Ensure all mentioned waypoints in visible_from and visible are in waypoint_visible_wp keys
        # even if they have no outgoing visible links listed explicitly.
        # This was handled by adding symmetric links above.
        # Also add lander location if not already in waypoints
        if self.lander_location and self.lander_location not in self.all_waypoints:
             self.all_waypoints.add(self.lander_location)
             self.waypoint_visible_wp.setdefault(self.lander_location, set())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state
        total_cost = 0

        # Build current state lookup structures
        current_rover_locations = {} # {rover: waypoint}
        current_soil_data = set() # {(rover, waypoint)}
        current_rock_data = set() # {(rover, waypoint)}
        current_image_data = set() # {(rover, objective, mode)}
        current_full_stores = set() # {store}
        current_calibrated_cameras = set() # {(camera, rover)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at' and len(parts) > 2 and parts[1] in self.all_rovers:
                 current_rover_locations[parts[1]] = parts[2]
            elif predicate == 'have_soil_analysis' and len(parts) > 2:
                 current_soil_data.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis' and len(parts) > 2:
                 current_rock_data.add((parts[1], parts[2]))
            elif predicate == 'have_image' and len(parts) > 3:
                 current_image_data.add((parts[1], parts[2], parts[3]))
            elif predicate == 'full' and len(parts) > 1:
                 current_full_stores.add(parts[1])
            elif predicate == 'calibrated' and len(parts) > 2:
                 current_calibrated_cameras.add((parts[1], parts[2]))

        # Pre-calculate waypoints visible from the lander
        lander_comm_wps = self.waypoint_visible_wp.get(self.lander_location, set()) if self.lander_location else set()


        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) > 1:
                w = parts[1]
                min_goal_cost = math.inf

                # Case 1: Data already collected by some rover
                rovers_with_data = {r for r, wp in current_soil_data if wp == w}
                if rovers_with_data:
                    for r in rovers_with_data:
                        cost = 1 # communicate
                        loc = current_rover_locations.get(r)
                        if loc is None: continue # Rover location unknown? Skip this rover.

                        # Navigation to communication point if needed
                        if not lander_comm_wps or loc not in lander_comm_wps:
                             cost += 1 # navigate to communication point
                        min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Data needs to be collected
                for r in self.equipped_soil:
                    loc = current_rover_locations.get(r)
                    if loc is None: continue # Rover location unknown? Skip this rover.

                    cost = 1 + 1 # sample + communicate

                    # Drop if store is full
                    store = self.rover_stores.get(r)
                    if store and store in current_full_stores:
                        cost += 1 # drop

                    # Navigation to sample location (w)
                    loc_after_sample_nav = loc
                    if loc != w:
                        cost += 1 # navigate to w
                        loc_after_sample_nav = w # Assume rover is at w after navigation

                    # Navigation to communication location from location after sampling
                    if not lander_comm_wps or loc_after_sample_nav not in lander_comm_wps:
                         cost += 1 # navigate to communication point

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != math.inf:
                     total_cost += min_goal_cost
                # else: goal is unreachable (e.g., no equipped rover), min_goal_cost remains inf, doesn't add to total_cost.

            elif predicate == 'communicated_rock_data' and len(parts) > 1:
                w = parts[1]
                min_goal_cost = math.inf

                # Case 1: Data already collected
                rovers_with_data = {r for r, wp in current_rock_data if wp == w}
                if rovers_with_data:
                     for r in rovers_with_data:
                        cost = 1 # communicate
                        loc = current_rover_locations.get(r)
                        if loc is None: continue

                        if not lander_comm_wps or loc not in lander_comm_wps:
                             cost += 1 # navigate to communication point
                        min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Data needs to be collected
                for r in self.equipped_rock:
                    loc = current_rover_locations.get(r)
                    if loc is None: continue

                    cost = 1 + 1 # sample + communicate

                    store = self.rover_stores.get(r)
                    if store and store in current_full_stores:
                        cost += 1 # drop

                    # Navigation to sample location (w)
                    loc_after_sample_nav = loc
                    if loc != w:
                        cost += 1 # navigate to w
                        loc_after_sample_nav = w

                    # Navigation to communication location from location after sampling
                    if not lander_comm_wps or loc_after_sample_nav not in lander_comm_wps:
                         cost += 1 # navigate to communication point

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != math.inf:
                     total_cost += min_goal_cost

            elif predicate == 'communicated_image_data' and len(parts) > 2:
                o, m = parts[1], parts[2]
                min_goal_cost = math.inf

                # Case 1: Image already taken
                rovers_with_image = {r for r, obj, mode in current_image_data if obj == o and mode == m}
                if rovers_with_image:
                     for r in rovers_with_image:
                        cost = 1 # communicate
                        loc = current_rover_locations.get(r)
                        if loc is None: continue

                        if not lander_comm_wps or loc not in lander_comm_wps:
                             cost += 1 # navigate to communication point
                        min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Image needs to be taken
                for r in self.equipped_imaging:
                    loc = current_rover_locations.get(r)
                    if loc is None: continue

                    for i in self.rover_cameras.get(r, set()):
                        if m in self.camera_modes.get(i, set()): # Camera supports mode
                            cost = 1 + 1 # take_image + communicate

                            loc_after_cal_nav = loc # Assume location doesn't change unless calibration nav happens
                            loc_after_img_nav = loc_after_cal_nav # Assume location doesn't change unless image nav happens

                            # Step 1: Calibrate (if needed)
                            cal_needed = (i, r) not in current_calibrated_cameras
                            if cal_needed:
                                cost += 1 # calibrate
                                # Navigation to calibration point
                                target = self.camera_calibration_target.get(i)
                                cal_wps = self.objective_visible_from_wp.get(target, set()) if target else set()

                                if not cal_wps: # No waypoints visible from target -> impossible to calibrate via nav
                                     cost = math.inf # Mark this path as impossible
                                else:
                                    if loc not in cal_wps:
                                        cost += 1 # navigate to calibration point
                                        # Assume rover is now at *a* CalWP for subsequent steps
                                        loc_after_cal_nav = next(iter(cal_wps)) # Pick an arbitrary one
                                    else:
                                        loc_after_cal_nav = loc # Already at a CalWP

                            if cost == math.inf: # If calibration step was impossible, skip rest for this rover/camera
                                min_goal_cost = min(min_goal_cost, cost)
                                continue

                            # Step 2: Take Image
                            # Navigation to image point from loc_after_cal_nav
                            img_wps = self.objective_visible_from_wp.get(o, set())

                            if not img_wps: # No waypoints visible from objective -> impossible to image via nav
                                cost = math.inf # Mark this path as impossible
                            else:
                                if loc_after_cal_nav not in img_wps:
                                    cost += 1 # navigate to image point
                                    # Assume rover is now at *an* ImgWP for subsequent steps
                                    loc_after_img_nav = next(iter(img_wps)) # Pick an arbitrary one
                                else:
                                    loc_after_img_nav = loc_after_cal_nav # Already at an ImgWP

                            if cost == math.inf: # If image step was impossible, skip rest for this rover/camera
                                min_goal_cost = min(min_goal_cost, cost)
                                continue

                            # Step 3: Communicate
                            # Navigation to communication point from loc_after_img_nav
                            if not lander_comm_wps: # No waypoints visible from lander -> impossible to communicate
                                cost = math.inf # Mark this path as impossible
                            else:
                                if loc_after_img_nav not in lander_comm_wps:
                                    cost += 1 # navigate to communication point

                            min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != math.inf:
                     total_cost += min_goal_cost

        return total_cost
