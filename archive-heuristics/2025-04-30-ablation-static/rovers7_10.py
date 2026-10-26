from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions required to achieve each uncommunicated goal fact.
    It sums up estimated costs for each goal based on whether the required data/image has been collected
    but not yet communicated, or if it still needs to be collected and then communicated.
    The heuristic is non-admissible and designed to guide a greedy best-first search.

    # Assumptions
    - The heuristic provides a fixed cost estimate for reaching intermediate states (having data/image)
      and then communicating it, ignoring actual navigation distances and complex resource constraints
      (like specific rover locations, store availability beyond a simple 'drop if needed',
      camera calibration state beyond the initial image capture need, or specific rover/camera capabilities
      beyond assuming *a* capable agent exists if needed).
    - All goals are assumed to be achievable from the initial state in principle.
    - The costs are simplified action counts representing the remaining steps:
        - Communicate data/image (if already collected): 2 actions (estimated move to lander + communicate).
        - Collect soil/rock sample and communicate: 5 actions (estimated move to sample + drop (if needed) + sample + move to lander + communicate).
        - Take image and communicate: 6 actions (estimated move to calibrate + calibrate + move to image + take image + move to lander + communicate).

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the task.
    - Static facts are parsed to potentially identify lander locations, communication waypoints,
      rover capabilities, camera details, calibration targets, and imaging locations.
      While the simplified heuristic doesn't strictly *use* all these for cost calculation,
      parsing them fulfills the requirement and allows for potential future refinement or debugging.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Iterate through each goal fact specified in the task (`self.goals`).
    3. For the current goal fact:
       a. Check if the goal fact is already present in the current state. If yes, this goal is achieved; add 0 to the total cost and proceed to the next goal.
       b. If the goal fact is not in the state, determine its type (soil, rock, or image communication) by parsing the predicate name.
       c. If the goal is `(communicated_soil_data ?w)`:
          - Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `?r` in the current state. This indicates the sample has been collected.
          - If yes (sample collected): Add 2 to the total cost (estimated cost for moving the rover to a lander-visible waypoint and performing the communication action).
          - If no (sample not collected): Add 5 to the total cost (estimated cost for moving to the sample location, potentially dropping a full store, sampling, moving to a lander-visible waypoint, and communicating).
       d. If the goal is `(communicated_rock_data ?w)`:
          - Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `?r` in the current state. This indicates the sample has been collected.
          - If yes (sample collected): Add 2 to the total cost (estimated cost for moving the rover to a lander-visible waypoint and performing the communication action).
          - If no (sample not collected): Add 5 to the total cost (estimated cost for moving to the sample location, potentially dropping a full store, sampling, moving to a lander-visible waypoint, and communicating).
       e. If the goal is `(communicated_image_data ?o ?m)`:
          - Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r` in the current state. This indicates the image has been taken.
          - If yes (image taken): Add 2 to the total cost (estimated cost for moving the rover to a lander-visible waypoint and performing the communication action).
          - If no (image not taken): Add 6 to the total cost (estimated cost for moving to a calibration target, calibrating the camera, moving to the objective's imaging location, taking the image, moving to a lander-visible waypoint, and communicating).
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Set of goal facts

        # Extract static facts into suitable data structures as required,
        # even if the simplified heuristic doesn't use all of them for cost calculation.
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_targets = {} # camera -> calibration target objective
        self.objective_imaging_wps = {} # objective -> set of visible waypoints
        self.caltarget_visible_wps = {} # cal_target_objective -> set of visible waypoints
        self.rover_stores = {} # rover -> store
        self.lander_location = None # Assuming one lander for simplicity if multiple exist
        self.comm_wps = set() # Waypoints visible from lander

        lander_wp = None
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_cal_targets[camera] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                # Distinguish between calibration targets and other objectives
                is_cal_target = objective in self.camera_cal_targets.values()
                if is_cal_target:
                     self.caltarget_visible_wps.setdefault(objective, set()).add(waypoint)
                else:
                     self.objective_imaging_wps.setdefault(objective, set()).add(waypoint)
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store # Assuming one store per rover
            elif predicate == "at_lander":
                 lander, waypoint = parts[1], parts[2]
                 self.lander_location = waypoint # Store the lander location

        # Find communication waypoints (visible from lander location)
        if self.lander_location:
             for fact in task.static:
                 parts = get_parts(fact)
                 if not parts: continue
                 # Check for waypoints visible *to* the lander location or *from* it
                 if parts[0] == "visible" and (parts[1] == self.lander_location or parts[2] == self.lander_location):
                     self.comm_wps.add(parts[1])
                     self.comm_wps.add(parts[2])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state
        total_cost = 0

        # Helper to check for existence of a predicate pattern in the state
        def state_has_pattern(state, *args):
             return any(match(fact, *args) for fact in state)

        for goal in self.goals:
            # If goal is already achieved, cost is 0 for this goal
            if goal in state:
                continue

            # Parse the goal predicate
            parts = get_parts(goal)
            if not parts:
                 continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                if len(parts) < 2: continue # Malformed goal
                waypoint = parts[1]
                # Check if sample is collected by any rover
                if state_has_pattern(state, "have_soil_analysis", "*", waypoint):
                    # Sample collected, need to communicate
                    total_cost += 2 # move to lander + communicate
                else:
                    # Sample not collected, need to sample and communicate
                    total_cost += 5 # move to sample + drop + sample + move to lander + communicate

            elif predicate == "communicated_rock_data":
                if len(parts) < 2: continue # Malformed goal
                waypoint = parts[1]
                # Check if sample is collected by any rover
                if state_has_pattern(state, "have_rock_analysis", "*", waypoint):
                    # Sample collected, need to communicate
                    total_cost += 2 # move to lander + communicate
                else:
                    # Sample not collected, need to sample and communicate
                    total_cost += 5 # move to sample + drop + sample + move to lander + communicate

            elif predicate == "communicated_image_data":
                if len(parts) < 3: continue # Malformed goal
                objective, mode = parts[1], parts[2]
                # Check if image is taken by any rover
                if state_has_pattern(state, "have_image", "*", objective, mode):
                    # Image taken, need to communicate
                    total_cost += 2 # move to lander + communicate
                else:
                    # Image not taken, need to take image and communicate
                    total_cost += 6 # move to calibrate + calibrate + move to image + take image + move to lander + communicate

        return total_cost
