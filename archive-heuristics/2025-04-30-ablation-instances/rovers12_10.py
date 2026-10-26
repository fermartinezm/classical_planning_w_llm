from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
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
    This heuristic estimates the number of actions required to achieve each
    unmet goal condition independently and sums these estimates. It considers
    the main steps needed for each type of goal (communicating soil, rock,
    or image data): sampling/imaging, processing (implicitly covered by
    having the data), and communicating.

    # Assumptions
    - Action costs are uniform (e.g., 1 per action).
    - Navigation between any two connected waypoints takes a fixed cost (e.g., 1).
    - Samples and objectives required by the goal are initially present and
      do not reappear once collected/imaged.
    - The heuristic provides a finite value for solvable states and a large
      value for states where a required sample/objective is missing or
      statically impossible to obtain.
    - The lander location is static.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Which rovers are equipped for soil, rock, and imaging.
    - Which store belongs to which rover.
    - Which cameras are on board which rovers and which modes they support.
    - Calibration targets for each camera.
    - Waypoints from which objectives and calibration targets are visible.
    - The lander's location and waypoints visible from it (for communication).
    It also pre-calculates which image goals are statically possible based on
    the existence of suitable rovers, cameras, and visible waypoints for imaging
    and calibration.
    This information is stored in dictionaries and sets for efficient lookup
    during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of estimated costs for each
    goal fact that is not yet true in the state.

    For each goal fact G:
    1.  If G is already true in the current state, the cost for this goal is 0.
    2.  If G is `(communicated_soil_data W)` and not true:
        a.  Check if `(have_soil_analysis R W)` is true for *any* rover R.
        b.  If yes: The analysis is ready. The cost is estimated as 2 actions:
            Navigate to a waypoint visible from the lander + Communicate soil data.
        c.  If no: The analysis needs to be sampled.
            i.  Check if `(at_soil_sample W)` is true in the state.
            ii. If yes: The sample is available. The cost is estimated as 4 actions:
                Navigate to W + Sample soil + Navigate to a waypoint visible from the lander + Communicate soil data. (This simplifies potential drop actions and assumes a suitable equipped rover/store exists).
            iii. If no: The sample is gone and no rover has the analysis. This goal is likely impossible from this state. Assign a large cost (e.g., 1000).
    3.  If G is `(communicated_rock_data W)` and not true:
        a.  Check if `(have_rock_analysis R W)` is true for *any* rover R.
        b.  If yes: Cost is 2 (Navigate + Communicate).
        c.  If no:
            i.  Check if `(at_rock_sample W)` is true in the state.
            ii. If yes: Cost is 4 (Navigate to W + Sample rock + Navigate to comm + Communicate).
            iii. If no: Cost is 1000.
    4.  If G is `(communicated_image_data O M)` and not true:
        a.  Check if `(have_image R O M)` is true for *any* rover R.
        b.  If yes: Cost is 2 (Navigate + Communicate).
        c.  If no: The image needs to be taken.
            i.  Check if the goal (O, M) was determined to be statically possible during initialization.
            ii. If yes: The cost is estimated as 6 actions:
                Navigate to W (cal spot) + Calibrate camera I + Navigate to P (image spot) + Take image + Navigate to a waypoint visible from the lander + Communicate image data. (This assumes recalibration is needed before taking the image).
            iii. If no: The image goal is statically impossible (e.g. objective not visible from anywhere, or no suitable camera/rover). Cost is 1000.

    The total heuristic value is the sum of these individual goal costs.
    A heuristic value of 0 is returned only when all goal facts are true.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # The set of facts that must hold in goal states.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract static information
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_owner = {} # {store: rover}
        self.camera_on_board = {} # {camera: rover}
        self.camera_supports = {} # {camera: set of modes}
        self.calibration_target = {} # {camera: objective}
        self.visible_from_objective = {} # {objective: set of waypoints}
        self.visible_from_target = {} # {objective: set of waypoints} (target is an objective)
        self.lander_location = None
        self.communication_waypoints = set() # Waypoints visible from lander

        # Build lookup structures from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "equipped_for_soil_analysis":
                if len(parts) == 2: self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                if len(parts) == 2: self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                if len(parts) == 2: self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                if len(parts) == 3: self.store_owner[parts[1]] = parts[2]
            elif predicate == "on_board":
                if len(parts) == 3: self.camera_on_board[parts[1]] = parts[2]
            elif predicate == "supports":
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    if camera not in self.camera_supports:
                        self.camera_supports[camera] = set()
                    self.camera_supports[camera].add(mode)
            elif predicate == "calibration_target":
                if len(parts) == 3: self.calibration_target[parts[1]] = parts[2]
            elif predicate == "visible_from":
                if len(parts) == 3:
                    objective, waypoint = parts[1], parts[2]
                    if objective not in self.visible_from_objective:
                        self.visible_from_objective[objective] = set()
                    self.visible_from_objective[objective].add(waypoint)
                    # Calibration targets are objectives, store their visible_from too
                    # Check if this objective is a calibration target for any camera
                    if objective in self.calibration_target.values():
                         if objective not in self.visible_from_target:
                             self.visible_from_target[objective] = set()
                         self.visible_from_target[objective].add(waypoint)
            elif predicate == "at_lander":
                 # Assuming only one lander and its location is static
                 if len(parts) == 3: self.lander_location = parts[2]

        # Find waypoints visible from the lander location (for communication)
        if self.lander_location:
             for fact in static_facts:
                 parts = get_parts(fact)
                 if match(fact, "visible", "*", self.lander_location):
                     if len(parts) == 3: self.communication_waypoints.add(parts[1])


        # Pre-calculate if image goals are statically possible
        self.statically_possible_image_goals = set() # Stores (objective, mode) tuples

        # Collect all unique image goals from the task goals
        image_goals_in_task = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "communicated_image_data":
                if len(parts) == 3:
                     image_goals_in_task.add((parts[1], parts[2]))
                # else: malformed goal, ignore or warn

        # Check static possibility for each image goal found
        for obj, mode in image_goals_in_task:
            is_possible = False
            # Need a waypoint P visible from objective O
            waypoints_img = self.visible_from_objective.get(obj, set())
            if not waypoints_img: continue # Cannot image this objective if not visible from anywhere

            # Need a rover R equipped for imaging
            for rover in self.equipped_imaging:
                # Need a camera I on board R that supports mode M
                for camera, cam_rover in self.camera_on_board.items():
                    if cam_rover != rover: continue # Camera must be on this rover
                    supported_modes = self.camera_supports.get(camera, set())
                    if mode not in supported_modes: continue # Camera must support the mode

                    # Need a calibration target T for camera I
                    target = self.calibration_target.get(camera)
                    if not target: continue # Camera must have a calibration target

                    # Need a waypoint W visible from target T
                    waypoints_cal = self.visible_from_target.get(target, set())
                    if not waypoints_cal: continue # Target must be visible from somewhere

                    # If we reached here, we found a valid chain:
                    # Rover R (equipped_imaging) -> Camera I (on_board R, supports M)
                    # Objective O (visible_from P) -> Target T (calibration_target I, visible_from W)
                    # This image goal (O, M) is statically possible.
                    is_possible = True
                    break # Found one possibility for this camera/rover, no need to check others for this (obj, mode)

                if is_possible:
                    break # Found a possibility for this (obj, mode), move to next goal

            if is_possible:
                self.statically_possible_image_goals.add((obj, mode))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0
        large_cost = 1000 # Penalty for seemingly impossible subgoals

        # Pre-compute current data holdings and sample locations for quick lookup
        have_soil = {} # {waypoint: set of rovers}
        have_rock = {} # {waypoint: set of rovers}
        have_image = {} # {(objective, mode): set of rovers}
        at_soil_sample = set() # {waypoint}
        at_rock_sample = set() # {waypoint}
        # calibrated_cameras = set() # {(camera, rover)} # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "have_soil_analysis":
                if len(parts) == 3:
                    rover, waypoint = parts[1], parts[2]
                    if waypoint not in have_soil: have_soil[waypoint] = set()
                    have_soil[waypoint].add(rover)
            elif predicate == "have_rock_analysis":
                 if len(parts) == 3:
                    rover, waypoint = parts[1], parts[2]
                    if waypoint not in have_rock: have_rock[waypoint] = set()
                    have_rock[waypoint].add(rover)
            elif predicate == "have_image":
                 if len(parts) == 4:
                    rover, objective, mode = parts[1], parts[2], parts[3]
                    key = (objective, mode)
                    if key not in have_image: have_image[key] = set()
                    have_image[key].add(rover)
            elif predicate == "at_soil_sample":
                 if len(parts) == 2:
                    at_soil_sample.add(parts[1])
            elif predicate == "at_rock_sample":
                 if len(parts) == 2:
                    at_rock_sample.add(parts[1])
            # elif predicate == "calibrated": # Not used in this heuristic's cost logic
            #      if len(parts) == 3:
            #         calibrated_cameras.add((parts[1], parts[2]))


        # Check each goal condition
        for goal in self.goals:
            # Ensure goal fact is well-formed before processing
            goal_parts = get_parts(goal)
            if not goal_parts:
                 # Handle malformed goal? Or assume valid goals? Assume valid for now.
                 continue

            if goal in state:
                continue # Goal already achieved

            predicate = goal_parts[0]

            if predicate == "communicated_soil_data":
                if len(goal_parts) != 2: continue # Malformed goal

                waypoint = goal_parts[1]
                # Check if any rover has the analysis
                if waypoint in have_soil and len(have_soil[waypoint]) > 0:
                    # Analysis exists, need to communicate
                    total_cost += 2 # Navigate to comm spot + Communicate
                else:
                    # Analysis missing, need to sample and communicate
                    if waypoint in at_soil_sample:
                        # Sample is available
                        total_cost += 4 # Nav to sample + Sample + Nav to comm + Communicate
                    else:
                        # Sample is gone and no one has it - likely impossible
                        total_cost += large_cost

            elif predicate == "communicated_rock_data":
                if len(goal_parts) != 2: continue # Malformed goal

                waypoint = goal_parts[1]
                # Check if any rover has the analysis
                if waypoint in have_rock and len(have_rock[waypoint]) > 0:
                    # Analysis exists, need to communicate
                    total_cost += 2 # Navigate to comm spot + Communicate
                else:
                    # Analysis missing, need to sample and communicate
                    if waypoint in at_rock_sample:
                        # Sample is available
                        total_cost += 4 # Nav to sample + Sample + Nav to comm + Communicate
                    else:
                        # Sample is gone and no one has it - likely impossible
                        total_cost += large_cost

            elif predicate == "communicated_image_data":
                if len(goal_parts) != 3: continue # Malformed goal

                objective, mode = goal_parts[1], goal_parts[2]
                key = (objective, mode)
                # Check if any rover has the image
                if key in have_image and len(have_image[key]) > 0:
                    # Image exists, need to communicate
                    total_cost += 2 # Navigate to comm spot + Communicate
                else:
                    # Image missing, need to take image and communicate
                    if key in self.statically_possible_image_goals:
                         # Image is statically possible to take
                         # Estimate cost: Calibrate + Take Image + Communicate
                         # Full sequence: NavCal + Cal + NavImg + Take + NavComm + Comm = 6
                         total_cost += 6
                    else:
                         # Image goal is statically impossible
                         total_cost += large_cost

            # Add checks for other potential goal types if necessary, though the domain file only shows these three.

        # Safety check: If total_cost is 0, it must be a goal state.
        # If it's 0 but not a goal state, return a small positive value.
        # If it's > 0 but is a goal state, return 0.
        if total_cost == 0 and not self.goals <= state:
             # This implies there's an unachieved goal that wasn't assigned a cost > 0
             # based on the logic above. This shouldn't happen if all goal types
             # are covered and impossible subgoals get large_cost.
             # However, as a safeguard, return 1.
             return 1
        elif total_cost > 0 and self.goals <= state:
             # This implies a goal was counted as unachieved but is actually in the state.
             # This shouldn't happen with the 'if goal in state: continue' check.
             # Return 0 as it is a goal state.
             return 0

        return total_cost
