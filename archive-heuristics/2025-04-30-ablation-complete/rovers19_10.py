from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys # Import sys to use sys.maxsize for unreachable goals

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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It breaks down each goal (communicating soil, rock,
    or image data) into a sequence of necessary steps (sampling/imaging,
    calibration, dropping samples, navigation, and communication) and sums
    the estimated costs for each unachieved goal independently.

    # Assumptions
    - The heuristic assumes that for each unachieved goal, there exists at
      least one equipped rover capable of performing the necessary tasks
      (sampling, imaging).
    - Navigation cost is estimated simply as 1 action if the required rover
      is not currently at *any* suitable location for the next step (e.g.,
      sample location, image location, calibration location, communication
      location). It does not calculate shortest paths.
    - Resource constraints like store capacity (beyond needing an empty store
      to sample) or camera calibration state persisting across actions are
      simplified or ignored. Calibration is assumed needed if the camera is
      not currently calibrated.
    - The heuristic sums costs for each unachieved goal independently, ignoring
      potential synergies (e.g., one navigation action serving multiple goals)
      or conflicts (e.g., multiple rovers needing the same resource).
    - Assumes solvable instances, so necessary equipped rovers/cameras exist.

    # Heuristic Initialization
    The heuristic extracts the following static information from the task:
    - The set of all goal facts.
    - Sets of rovers equipped for soil analysis, rock analysis, and imaging.
    - A mapping from rovers to their stores.
    - Information about cameras: which rover they are on, which modes they
      support, and their calibration target objective.
    - A mapping from objectives to the set of waypoints from which they are visible.
    - The lander's location and the set of waypoints visible from the lander
      (communication points).
    - The initial locations of soil and rock samples (though the heuristic
      primarily checks the current state for samples).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the estimated cost as follows:

    1.  Check if the state is a goal state. If yes, return 0.
    2.  Initialize `total_cost` to 0.
    3.  Parse the current state to create efficient lookups for rover locations,
        store statuses, held samples/images, and camera calibration statuses.
    4.  Iterate through each goal fact defined in the problem:
        a.  If the goal fact is already true in the current state, skip it.
        b.  If the goal is `(communicated_soil_data ?w)`:
            - Add 1 for the `communicate_soil_data` action.
            - Check if `(have_soil_analysis ?r ?w)` is true for *any* equipped rover `?r`.
            - If not (sampling is needed):
                - Add 1 for the `sample_soil` action.
                - Pick *one* equipped soil rover (`rover_for_sample`).
                - If this rover is not at waypoint `?w`, add 1 for navigation.
                - Find the rover's store. If the store is full, add 1 for the `drop` action.
                - The `rover_for_sample` is now considered the rover that will have the data.
            - If sampling was not needed, find *any* equipped soil rover that *currently* has `(have_soil_analysis ?r ?w)`. This rover is considered the one with the data (`rover_with_data`).
            - If a `rover_with_data` was identified, check if its current location is one of the communication waypoints. If not, add 1 for navigation to a communication point.
            - Add the calculated cost for this goal to `total_cost`.
        c.  If the goal is `(communicated_rock_data ?w)`: Follow the same logic as soil data, using rock-specific predicates and equipped rovers.
        d.  If the goal is `(communicated_image_data ?o ?m)`:
            - Add 1 for the `communicate_image_data` action.
            - Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r`.
            - If not (taking image is needed):
                - Add 1 for the `take_image` action.
                - Find *one* suitable rover `?r_img` with a camera `?i` that is on board, equipped for imaging, and supports mode `?m`.
                - If `?r_img` is not at *any* waypoint visible from objective `?o`, add 1 for navigation to an image spot.
                - Check if camera `?i` on rover `?r_img` is calibrated. If not:
                    - Add 1 for the `calibrate` action.
                    - Find the calibration target `?t` for camera `?i`.
                    - If `?r_img` is not at *any* waypoint visible from objective `?t`, add 1 for navigation to a calibration spot.
                - The `rover_for_image` is now considered the rover that will have the data.
            - If taking the image was not needed, find *any* rover that *currently* has `(have_image ?r ?o ?m)`. This rover is considered the one with the data (`rover_with_data`).
            - If a `rover_with_data` was identified, check if its current location is one of the communication waypoints. If not, add 1 for navigation to a communication point.
            - Add the calculated cost for this goal to `total_cost`.
    5.  Return `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Heuristic Initialization ---
        # Store goal facts for quick lookup
        self.goal_facts = set(self.goals)

        # Extract static information
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m1, ...}, 'cal_target': t}
        self.visible_from_objective = {} # objective -> {waypoint1, ...}
        self.lander_location = None
        self.communication_waypoints = set()
        # Initial sample locations are not strictly needed for this state-based
        # heuristic, as we check the current state for samples.

        # First pass to identify lander location and initial samples
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at_lander':
                 self.lander_location = parts[2]
             # Initial samples are dynamic, so we don't store them here.
             # We check the current state for at_soil_sample/at_rock_sample.

        # Second pass for static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['rover'] = rover
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = objective
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if objective not in self.visible_from_objective:
                    self.visible_from_objective[objective] = set()
                self.visible_from_objective[objective].add(waypoint)
            elif parts[0] == 'visible' and self.lander_location is not None:
                 wp1, wp2 = parts[1], parts[2]
                 if wp1 == self.lander_location:
                     self.communication_waypoints.add(wp2)
                 if wp2 == self.lander_location:
                     self.communication_waypoints.add(wp1)

        # Add the lander location itself to communication waypoints
        if self.lander_location:
             self.communication_waypoints.add(self.lander_location)


    def __call__(self, node):
        state = node.state

        # Goal check
        if self.goals <= state:
            return 0

        # --- Step-By-Step Thinking for Computing Heuristic ---
        # Parse current state for quick lookups
        current_rover_location = {} # rover -> waypoint
        current_store_status = {} # store -> 'empty' or 'full'
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_communicated_soil = set() # waypoint
        current_communicated_rock = set() # waypoint
        current_communicated_image = set() # (objective, mode)
        current_at_soil_sample = set() # waypoint
        current_at_rock_sample = set() # waypoint


        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                current_rover_location[parts[1]] = parts[2]
            elif predicate == 'empty':
                current_store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                current_store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis':
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'calibrated':
                current_calibrated.add((parts[1], parts[2]))
            elif predicate == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample':
                 current_at_soil_sample.add(parts[1])
            elif predicate == 'at_rock_sample':
                 current_at_rock_sample.add(parts[1])


        total_cost = 0

        # Process each unachieved goal
        for goal in self.goal_facts:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint in current_communicated_soil:
                    continue # Goal already achieved

                cost = 1 # communicate_soil_data action
                rover_with_data = None # Rover that either has the data or will get it

                # Check if soil analysis is needed
                need_sample = True
                # Check if any equipped rover already has the data
                for rover in self.equipped_soil:
                    if (rover, waypoint) in current_have_soil:
                        need_sample = False
                        rover_with_data = rover # Found a rover with the data
                        break

                if need_sample:
                    # Need to sample. Check if sample exists.
                    if waypoint not in current_at_soil_sample:
                         # Sample is gone and no rover has it. Goal might be impossible.
                         # For non-admissible, we can return a large value or just add a base cost.
                         # Let's assume solvable instances mean samples aren't permanently gone unless collected.
                         # If it's not in current_at_soil_sample, it must have been collected.
                         # If no equipped rover has it, it was dropped or collected by a non-equipped rover (not possible).
                         # This case implies the data was lost after sampling. Re-sampling is not possible.
                         # A simple heuristic might just add a base cost or return inf.
                         # Let's assume for simplicity that if need_sample is true, a sample exists or can be obtained.
                         # The simplest is to just add the sampling cost.
                         pass # Assume sample exists or can be obtained

                    cost += 1 # sample_soil action
                    # Assume one equipped rover does the job
                    if self.equipped_soil:
                        rover_for_sample = next(iter(self.equipped_soil)) # Pick one equipped rover
                        rover_with_data = rover_for_sample # This rover will get the data after sampling
                        # Navigation to sample location
                        if current_rover_location.get(rover_for_sample) != waypoint:
                            cost += 1
                        # Drop if store is full
                        store = self.rover_stores.get(rover_for_sample)
                        if store and current_store_status.get(store) == 'full':
                             cost += 1
                    else:
                         # Should not happen in solvable instances, but handle defensively
                         # If no equipped rover, sampling is impossible.
                         # Return a large cost.
                         total_cost += sys.maxsize # Indicate very high cost/impossibility
                         continue # Move to next goal

                # Check if navigation to communication point is needed for the rover with data
                if rover_with_data is not None and current_rover_location.get(rover_with_data) not in self.communication_waypoints:
                     cost += 1

                total_cost += cost

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint in current_communicated_rock:
                    continue # Goal already achieved

                cost = 1 # communicate_rock_data action
                rover_with_data = None # Rover that either has the data or will get it

                # Check if rock analysis is needed
                need_sample = True
                # Check if any equipped rover already has the data
                for rover in self.equipped_rock:
                    if (rover, waypoint) in current_have_rock:
                        need_sample = False
                        rover_with_data = rover # Found a rover with the data
                        break

                if need_sample:
                     # Check if sample exists (similar logic to soil)
                     if waypoint not in current_at_rock_sample:
                          pass # Assume sample exists or can be obtained

                     cost += 1 # sample_rock action
                     # Assume one equipped rover does the job
                     if self.equipped_rock:
                         rover_for_sample = next(iter(self.equipped_rock)) # Pick one equipped rover
                         rover_with_data = rover_for_sample # This rover will get the data after sampling
                         # Navigation to sample location
                         if current_rover_location.get(rover_for_sample) != waypoint:
                             cost += 1
                         # Drop if store is full
                         store = self.rover_stores.get(rover_for_sample)
                         if store and current_store_status.get(store) == 'full':
                              cost += 1
                     else:
                          total_cost += sys.maxsize # Indicate very high cost/impossibility
                          continue # Move to next goal

                # Check if navigation to communication point is needed for the rover with data
                if rover_with_data is not None and current_rover_location.get(rover_with_data) not in self.communication_waypoints:
                     cost += 1

                total_cost += cost

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) in current_communicated_image:
                    continue # Goal already achieved

                cost = 1 # communicate_image_data action
                rover_with_data = None # Rover that either has the data or will get it

                # Check if image is needed
                need_image = True
                # Check if any rover already has the image
                for (r, o, m) in current_have_image:
                    if o == objective and m == mode:
                        need_image = False
                        rover_with_data = r # Found a rover with the image
                        break

                if need_image:
                    cost += 1 # take_image action
                    # Find a suitable rover/camera pair to take the image
                    suitable_rover_camera = None
                    for camera, info in self.camera_info.items():
                        rover = info.get('rover')
                        modes = info.get('modes', set())
                        if rover in self.equipped_imaging and mode in modes:
                            suitable_rover_camera = (rover, camera)
                            break

                    if suitable_rover_camera:
                        rover_for_image, camera_for_image = suitable_rover_camera
                        rover_with_data = rover_for_image # This rover will get the data
                        # Navigation to image location
                        image_wps = self.visible_from_objective.get(objective, set())
                        if image_wps and current_rover_location.get(rover_for_image) not in image_wps:
                             cost += 1 # Navigate to *any* visible waypoint

                        # Check if calibration is needed
                        if (camera_for_image, rover_for_image) not in current_calibrated:
                            cost += 1 # calibrate action
                            # Navigation to calibration location
                            cal_target = self.camera_info.get(camera_for_image, {}).get('cal_target')
                            if cal_target:
                                cal_wps = self.visible_from_objective.get(cal_target, set())
                                if cal_wps and current_rover_location.get(rover_for_image) not in cal_wps:
                                     cost += 1 # Navigate to *any* visible waypoint
                            # else: Camera has no calibration target, cannot calibrate? Assume solvable means target exists if needed.
                    else:
                         # Should not happen in solvable instances
                         # If no suitable rover/camera, taking image is impossible.
                         # Return a large cost.
                         total_cost += sys.maxsize # Indicate very high cost/impossibility
                         continue # Move to next goal


                # Check if navigation to communication point is needed for the rover with data
                if rover_with_data is not None and current_rover_location.get(rover_with_data) not in self.communication_waypoints:
                     cost += 1

                total_cost += cost

        # If total_cost is sys.maxsize, it means at least one goal was deemed impossible
        # by the heuristic's simplified logic. Return a large number.
        # Otherwise, return the calculated sum.
        return total_cost if total_cost < sys.maxsize else sys.maxsize

