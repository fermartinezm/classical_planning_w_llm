from fnmatch import fnmatch
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided by the environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal fact independently. The cost for each goal fact is estimated
    by counting the number of necessary actions (sample/take_image, calibrate, drop,
    communicate) and associated navigation steps (to sample/image location, to calibration
    target location, to lander location) required to achieve that specific goal fact,
    ignoring interactions between goals and resource constraints beyond a basic check
    for empty stores or calibrated cameras.

    # Assumptions
    - Each unachieved goal fact can be pursued independently.
    - Navigation between any two required waypoints costs a fixed amount (estimated as 1).
    - The presence of a sample/image on *any* equipped rover satisfies the 'have' condition
      for that goal type.
    - The need for a 'drop' action is estimated simply by checking if *any* equipped rover
      that needs to sample has *any* full store.
    - The need for 'calibrate' is estimated by checking if *any* suitable camera for the
      required image mode is not calibrated for the rover carrying it.
    - Initial samples (`at_soil_sample`, `at_rock_sample`) are present if needed for a goal.
    - Equipped status (`equipped_for_...`) and camera properties (`on_board`, `supports`,
      `calibration_target`) are static and correctly specified.

    # Heuristic Initialization
    The constructor extracts static information from the task, including:
    - The lander's location.
    - Lists of rovers, stores, cameras, objectives, waypoints, and modes (inferred from facts).
    - Which rovers are equipped for soil, rock, and imaging.
    - Mapping between stores and the rovers they belong to.
    - Information about cameras (which rover they are on, modes supported, calibration target).
    - Mapping between objectives and waypoints from which they are visible.
    - The set of waypoints visible from the lander's location (communication waypoints).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:

    1. Initialize the total heuristic cost `h` to 0.
    2. Extract dynamic information from the current state:
       - Current location of each rover.
       - Which stores are full.
       - Which cameras are calibrated for which rovers.
       - Which soil samples, rock samples, and images each rover possesses.
    3. Iterate through each goal fact specified in the task.
    4. If a goal fact is already present in the current state, it contributes 0 to the heuristic.
    5. If a goal fact is *not* present in the current state, estimate the cost to achieve it:
       - **For `(communicated_soil_data ?w)`:**
         - Add 1 for the final `communicate_soil_data` action.
         - Check if any equipped rover currently holds `(have_soil_analysis ?r ?w)`.
         - If no rover holds the sample:
           - Add 1 for the `sample_soil` action.
           - Check if any equipped rover is currently at waypoint `?w`. If not, add 1 for navigation to `?w`.
           - Check if *any* equipped rover has *any* full store (a simplified check for needing a `drop` action before sampling). If so, add 1 for the `drop` action.
         - If a rover holds the sample:
           - Check if that rover is currently at a waypoint visible from the lander (a communication waypoint). If not, add 1 for navigation to a communication waypoint.
       - **For `(communicated_rock_data ?w)`:** Follow the same logic as soil data, substituting rock analysis and rock samples.
       - **For `(communicated_image_data ?o ?m)`:**
         - Add 1 for the final `communicate_image_data` action.
         - Check if any equipped imaging rover currently holds `(have_image ?r ?o ?m)`.
         - If no rover holds the image:
           - Add 1 for the `take_image` action.
           - Check if any equipped imaging rover is currently at a waypoint visible from objective `?o`. If not, add 1 for navigation to such a waypoint.
           - Find a camera on an equipped imaging rover that supports mode `?m`. Check if this camera is calibrated *for that specific rover*. If not:
             - Add 1 for the `calibrate` action.
             - Find the calibration target for this camera. Check if the rover carrying this camera is currently at a waypoint visible from the calibration target. If not, add 1 for navigation to such a waypoint.
         - If a rover holds the image:
           - Check if that rover is currently at a waypoint visible from the lander (a communication waypoint). If not, add 1 for navigation to a communication waypoint.
    6. The total heuristic value is the sum of costs for all unachieved goal facts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract static information
        self.lander_location = None
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.objectives = set()
        self.waypoints = set()
        self.modes = set()

        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()

        self.store_to_rover = {}
        self.rover_to_stores = {} # Helper for drop check

        self.camera_info = {} # cam -> {'rover': r, 'modes': set(), 'cal_target': t}
        self.objective_visible_from = {} # obj -> set(waypoints)

        visible_pairs = set()
        all_facts = task.initial_state | task.static # Consider initial state for object types too

        # First pass to collect all mentioned objects and some basic static info
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == 'at_lander': self.lander_location = parts[2]
             elif pred == 'at': self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == 'can_traverse': self.rovers.add(parts[1]); self.waypoints.add(parts[2]); self.waypoints.add(parts[3])
             elif pred == 'equipped_for_soil_analysis': self.rovers.add(parts[1]); self.equipped_soil.add(parts[1])
             elif pred == 'equipped_for_rock_analysis': self.rovers.add(parts[1]); self.equipped_rock.add(parts[1])
             elif pred == 'equipped_for_imaging': self.rovers.add(parts[1]); self.equipped_imaging.add(parts[1])
             elif pred == 'empty': self.stores.add(parts[1])
             elif pred == 'full': self.stores.add(parts[1])
             elif pred == 'have_rock_analysis': self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == 'have_soil_analysis': self.rovers.add(parts[1]); self.waypoints.add(parts[2])
             elif pred == 'calibrated': self.cameras.add(parts[1]); self.rovers.add(parts[2])
             elif pred == 'supports': self.cameras.add(parts[1]); self.modes.add(parts[2])
             elif pred == 'visible': self.waypoints.add(parts[1]); self.waypoints.add(parts[2]); visible_pairs.add((parts[1], parts[2]))
             elif pred == 'have_image': self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3])
             elif pred == 'communicated_soil_data': self.waypoints.add(parts[1])
             elif pred == 'communicated_rock_data': self.waypoints.add(parts[1])
             elif pred == 'communicated_image_data': self.objectives.add(parts[1]); self.modes.add(parts[2])
             elif pred == 'at_soil_sample': self.waypoints.add(parts[1])
             elif pred == 'at_rock_sample': self.waypoints.add(parts[1])
             elif pred == 'visible_from': self.objectives.add(parts[1]); self.waypoints.add(parts[2]); self.objective_visible_from.setdefault(parts[1], set()).add(parts[2])
             elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.stores.add(store); self.rovers.add(rover)
                self.store_to_rover[store] = rover
                self.rover_to_stores.setdefault(rover, set()).add(store)
             elif pred == 'calibration_target':
                camera, target = parts[1], parts[2]
                self.cameras.add(camera); self.objectives.add(target)
                self.camera_info.setdefault(camera, {}).update({'cal_target': target})
             elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera); self.rovers.add(rover)
                self.camera_info.setdefault(camera, {}).update({'rover': rover, 'modes': set()}) # Ensure modes key exists

        # Populate modes for cameras (needs a second pass as camera_info might be populated in first pass)
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'modes': set()}) # Ensure modes key exists
                self.camera_info[camera]['modes'].add(mode)


        # Determine communication waypoints (visible from lander location)
        self.comm_waypoints = set()
        if self.lander_location:
             # Waypoints from which the lander location is visible
             self.comm_waypoints = {w1 for w1, w2 in visible_pairs if w2 == self.lander_location}
             # Assume rover can communicate if at the lander location itself
             self.comm_waypoints.add(self.lander_location)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Extract dynamic information from the state
        rover_at = {} # rover -> waypoint
        store_full = set() # set of stores
        camera_calibrated = set() # set of (camera, rover) tuples
        rover_has_soil = {} # rover -> set(waypoints)
        rover_has_rock = {} # rover -> set(waypoints)
        rover_has_image = {} # rover -> set((objective, mode))

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at':
                rover_at[parts[1]] = parts[2]
            elif pred == 'full':
                store_full.add(parts[1])
            elif pred == 'calibrated':
                camera_calibrated.add((parts[1], parts[2])) # Store (camera, rover) tuple
            elif pred == 'have_soil_analysis':
                rover, waypoint = parts[1], parts[2]
                rover_has_soil.setdefault(rover, set()).add(waypoint)
            elif pred == 'have_rock_analysis':
                rover, waypoint = parts[1], parts[2]
                rover_has_rock.setdefault(rover, set()).add(waypoint)
            elif pred == 'have_image':
                rover, objective, mode = parts[1], parts[2], parts[3]
                rover_has_image.setdefault(rover, set()).add((objective, mode))

        # Check each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                h += 1 # Cost for communicate action

                # Check if sample is held by any equipped rover
                has_sample = any(waypoint in rover_has_soil.get(r, set()) for r in self.equipped_soil)

                if not has_sample:
                    h += 1 # Cost for sample action

                    # Check if any equipped rover is at the sample location
                    at_sample_loc = any(rover_at.get(r) == waypoint for r in self.equipped_soil)
                    if not at_sample_loc:
                        h += 1 # Cost for navigation to sample location

                    # Check if any equipped rover needs to drop a sample
                    # Simplified: Check if *any* equipped rover has *any* full store
                    needs_drop = any(store_full.intersection(self.rover_to_stores.get(r, set())) for r in self.equipped_soil)
                    if needs_drop:
                         h += 1 # Cost for drop action

                else: # Has sample
                    # Find *a* rover that has the sample (simplification)
                    rover_with_sample = next((r for r in self.equipped_soil if waypoint in rover_has_soil.get(r, set())), None)
                    if rover_with_sample: # Should be true if has_sample is true
                        # Check if the rover with the sample is at a communication waypoint
                        at_comm_loc = rover_at.get(rover_with_sample) in self.comm_waypoints
                        if not at_comm_loc:
                            h += 1 # Cost for navigation to lander

            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                h += 1 # Cost for communicate action

                # Check if sample is held by any equipped rover
                has_sample = any(waypoint in rover_has_rock.get(r, set()) for r in self.equipped_rock)

                if not has_sample:
                    h += 1 # Cost for sample action

                    # Check if any equipped rover is at the sample location
                    at_sample_loc = any(rover_at.get(r) == waypoint for r in self.equipped_rock)
                    if not at_sample_loc:
                        h += 1 # Cost for navigation to sample location

                    # Check if any equipped rover needs to drop a sample
                    # Simplified: Check if *any* equipped rover has *any* full store
                    needs_drop = any(store_full.intersection(self.rover_to_stores.get(r, set())) for r in self.equipped_rock)
                    if needs_drop:
                         h += 1 # Cost for drop action

                else: # Has sample
                    # Find *a* rover that has the sample (simplification)
                    rover_with_sample = next((r for r in self.equipped_rock if waypoint in rover_has_rock.get(r, set())), None)
                    if rover_with_sample: # Should be true if has_sample is true
                        # Check if the rover with the sample is at a communication waypoint
                        at_comm_loc = rover_at.get(rover_with_sample) in self.comm_waypoints
                        if not at_comm_loc:
                            h += 1 # Cost for navigation to lander

            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                h += 1 # Cost for communicate action

                # Check if image is held by any equipped imaging rover
                has_image = any((objective, mode) in rover_has_image.get(r, set()) for r in self.equipped_imaging)

                if not has_image:
                    h += 1 # Cost for take_image action

                    # Check if any equipped imaging rover is at a waypoint visible from the objective
                    visible_wps = self.objective_visible_from.get(objective, set())
                    at_image_loc = any(rover_at.get(r) in visible_wps for r in self.equipped_imaging)
                    if not at_image_loc:
                        h += 1 # Cost for navigation to image location

                    # Check if calibration is needed for a suitable camera
                    needs_calibration = False
                    camera_rover_pair = None
                    # Find a camera/rover that can take this image and check calibration
                    for cam, info in self.camera_info.items():
                        rover = info.get('rover')
                        if rover in self.equipped_imaging and mode in info.get('modes', set()):
                            camera_rover_pair = (cam, rover)
                            if (cam, rover) not in camera_calibrated: # Check specific (camera, rover) pair
                                 needs_calibration = True
                            break # Found one suitable camera/rover

                    if needs_calibration:
                        h += 1 # Cost for calibrate action
                        if camera_rover_pair: # Should be true if needs_calibration is true
                            cam, r = camera_rover_pair
                            cal_target = self.camera_info.get(cam, {}).get('cal_target')
                            if cal_target:
                                cal_visible_wps = self.objective_visible_from.get(cal_target, set())
                                at_cal_loc = rover_at.get(r) in cal_visible_wps
                                if not at_cal_loc:
                                    h += 1 # Cost for navigation to calibration location
                            # else: Calibration target not found? Problem definition issue.

                else: # Has image
                    # Find *a* rover that has the image (simplification)
                    rover_with_image = next((r for r in self.equipped_imaging if (objective, mode) in rover_has_image.get(r, set())), None)
                    if rover_with_image: # Should be true if has_image is true
                        # Check if the rover with the image is at a communication waypoint
                        at_comm_loc = rover_at.get(rover_with_image) in self.comm_waypoints
                        if not at_comm_loc:
                            h += 1 # Cost for navigation to lander

        return h
