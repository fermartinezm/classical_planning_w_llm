from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    for each uncommunicated goal fact. It counts the number of core actions
    (sample/take_image, calibrate, communicate) required, ignoring navigation costs
    and some resource constraints (like store availability).

    # Assumptions
    - Each uncommunicated goal fact (soil data, rock data, image data) contributes
      independently to the heuristic value (additive heuristic).
    - Navigation costs between waypoints are ignored.
    - Resource constraints like empty stores for sampling are mostly ignored
      (except for checking if equipment exists).
    - It is assumed that for any goal requiring sampling or imaging, a suitable
      equipped rover and necessary resources (like soil/rock samples at the waypoint,
      visible objectives/calibration targets) exist in the problem instance,
      making the goal reachable in principle.
    - The cost of achieving a prerequisite (like sampling or taking an image)
      is added only once per goal fact, regardless of which rover achieves it.

    # Heuristic Initialization
    - Extracts static information from the task:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Cameras on board each rover and modes they support.
        - Calibration targets for each camera.
        - Waypoints visible from objectives (used for imaging and calibration).
        - Waypoints visible from the lander (communication points).
    - Extracts goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each uncommunicated goal fact.

    For each goal fact `(communicated_soil_data ?w)` that is not in the current state:
    1. Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `?r`.
    2. If yes: Add 1 to the heuristic (cost for communication).
    3. If no: Add 2 to the heuristic (cost for sampling + cost for communication).
       (Assumes a soil-equipped rover exists and can reach the location).

    For each goal fact `(communicated_rock_data ?w)` that is not in the current state:
    1. Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `?r`.
    2. If yes: Add 1 to the heuristic (cost for communication).
    3. If no: Add 2 to the heuristic (cost for sampling + cost for communication).
       (Assumes a rock-equipped rover exists and can reach the location).

    For each goal fact `(communicated_image_data ?o ?m)` that is not in the current state:
    1. Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r`.
    2. If yes: Add 1 to the heuristic (cost for communication).
    3. If no:
       a. Check if there exists *any* combination of rover `?r` and camera `?i` such that:
          - `?r` is equipped for imaging.
          - `?i` is on board `?r`.
          - `?i` supports mode `?m`.
          (Assumes such a combination exists for solvable problems).
       b. If such a combination exists, check if `(calibrated ?i ?r)` is true for *any* such combination.
       c. If *any* suitable camera/rover is calibrated: Add 2 to the heuristic (cost for take_image + cost for communication).
       d. If *no* suitable camera/rover is calibrated: Add 3 to the heuristic (cost for calibrate + cost for take_image + cost for communication).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal conditions.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.lander_location = None
        self.rover_equipment = {} # {rover_name: set(equipment_types)}
        self.rover_cameras = {}   # {rover_name: set(camera_names)}
        self.camera_modes = {}    # {camera_name: set(mode_names)}
        self.camera_calibration_target = {} # {camera_name: objective_name}
        self.visible_from_objective = {} # {objective_name: set(waypoint_names)}
        self.communication_waypoints = set() # set of waypoints visible from lander

        # Initialize rover equipment sets
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'equipped_for_soil_analysis':
                 rover = parts[1]
                 self.rover_equipment.setdefault(rover, set()).add('soil')
             elif parts[0] == 'equipped_for_rock_analysis':
                 rover = parts[1]
                 self.rover_equipment.setdefault(rover, set()).add('rock')
             elif parts[0] == 'equipped_for_imaging':
                 rover = parts[1]
                 self.rover_equipment.setdefault(rover, set()).add('imaging')
             elif parts[0] == 'on_board':
                 camera, rover = parts[1], parts[2]
                 self.rover_cameras.setdefault(rover, set()).add(camera)
             elif parts[0] == 'supports':
                 camera, mode = parts[1], parts[2]
                 self.camera_modes.setdefault(camera, set()).add(mode)
             elif parts[0] == 'calibration_target':
                 camera, target = parts[1], parts[2]
                 self.camera_calibration_target[camera] = target
             elif parts[0] == 'visible_from':
                 objective, waypoint = parts[1], parts[2]
                 self.visible_from_objective.setdefault(objective, set()).add(waypoint)
             elif parts[0] == 'at_lander':
                 # Assuming only one lander
                 self.lander_location = parts[2]

        # Determine communication waypoints based on lander location
        if self.lander_location:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if not parts: continue
                 # We need waypoints ?x such that (visible ?x lander_location) is true
                 if parts[0] == 'visible':
                     w1, w2 = parts[1], parts[2]
                     if w2 == self.lander_location:
                         self.communication_waypoints.add(w1)


        # Extract goal information
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_images = set() # set of (objective, mode) tuples

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'communicated_soil_data':
                self.goal_soil_waypoints.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                self.goal_rock_waypoints.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                self.goal_images.add((parts[1], parts[2]))

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        h = 0

        # Extract relevant state information
        have_soil = set() # set of (rover, waypoint)
        have_rock = set() # set of (rover, waypoint)
        have_image = set() # set of (rover, objective, mode)
        calibrated_cameras = set() # set of (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))


        # Calculate heuristic based on uncommunicated goals

        # Soil Goals
        for w in self.goal_soil_waypoints:
            # Check if goal is already communicated
            if f'(communicated_soil_data {w})' not in state:
                # Check if sample is already collected by any rover equipped for soil
                sample_collected = False
                for r in self.rover_equipment:
                    if 'soil' in self.rover_equipment.get(r, set()):
                        if (r, w) in have_soil:
                            sample_collected = True
                            break # Found one, no need to check other rovers

                if sample_collected:
                    # Need to communicate
                    h += 1
                else:
                    # Need to sample and communicate
                    # Assumes sample exists at waypoint and an equipped rover exists
                    h += 2

        # Rock Goals
        for w in self.goal_rock_waypoints:
            # Check if goal is already communicated
            if f'(communicated_rock_data {w})' not in state:
                # Check if sample is already collected by any rover equipped for rock
                sample_collected = False
                for r in self.rover_equipment:
                    if 'rock' in self.rover_equipment.get(r, set()):
                        if (r, w) in have_rock:
                            sample_collected = True
                            break # Found one, no need to check other rovers

                if sample_collected:
                    # Need to communicate
                    h += 1
                else:
                    # Need to sample and communicate
                    # Assumes sample exists at waypoint and an equipped rover exists
                    h += 2

        # Image Goals
        for o, m in self.goal_images:
            # Check if goal is already communicated
            if f'(communicated_image_data {o} {m})' not in state:
                # Check if image is already taken by any rover equipped for imaging
                image_taken = False
                for r in self.rover_equipment:
                     if 'imaging' in self.rover_equipment.get(r, set()):
                         if (r, o, m) in have_image:
                             image_taken = True
                             break # Found one, no need to check other rovers

                if image_taken:
                    # Need to communicate
                    h += 1
                else:
                    # Need to take image and communicate
                    # Check if any suitable camera/rover exists and is calibrated
                    suitable_calibrated_exists = False
                    # suitable_exists = False # Not strictly needed for this heuristic logic
                    for r in self.rover_equipment:
                        if 'imaging' in self.rover_equipment.get(r, set()):
                            for cam in self.rover_cameras.get(r, set()):
                                if m in self.camera_modes.get(cam, set()):
                                    # This combination (r, cam, m) is suitable for taking the image
                                    # suitable_exists = True
                                    if (cam, r) in calibrated_cameras:
                                        suitable_calibrated_exists = True
                                        break # Found a calibrated suitable one, no need to check other cameras for this rover
                            if suitable_calibrated_exists:
                                break # Found a calibrated suitable one, no need to check other rovers for this goal

                    # Assume suitable_exists is True for solvable problems

                    if suitable_calibrated_exists:
                        # Need to take image and communicate
                        h += 2
                    else:
                        # Need to calibrate, take image, and communicate
                        h += 3

        return h
