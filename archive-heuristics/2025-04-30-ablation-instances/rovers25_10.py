from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the execution environment
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy the goal
    conditions, which involve communicating data (soil, rock, images) back to
    the lander. It counts the necessary steps for each unachieved goal, including
    sampling/imaging, calibration, dropping samples, and navigation, while
    attempting to avoid double-counting shared prerequisites like calibration
    or navigation to communication points.

    # Assumptions
    - The problem instance is solvable. Required locations (sample sites,
      image viewpoints, calibration viewpoints, lander location) exist and
      are reachable/visible as needed.
    - Navigation between waypoints costs a fixed amount (e.g., 1 action)
      regardless of distance or path complexity.
    - Resource constraints (like limited store capacity or camera availability)
      are simplified: a drop action is counted if *any* equipped rover needs
      to sample but no equipped rover has an empty store. Calibration is
      counted once per camera/rover pair if needed for any image goal.

    # Heuristic Initialization
    - Extracts static information from the task, such as rover capabilities,
      store ownership, camera properties (on-board, supports, calibration target),
      waypoint visibility, and lander location/visibility. This information
      is stored in data structures for efficient lookup during heuristic computation.
      It also identifies all objects of relevant types (rovers, waypoints, etc.)
      from the task facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is computed by summing estimated costs for unachieved
    goal conditions. The process involves multiple passes:

    1.  **Identify Missing Goals and Immediate Communication Cost:**
        Iterate through each goal predicate defined in the task. If a goal
        predicate is not present in the current state, add 1 to the total
        heuristic cost, representing the final 'communicate' action required
        for that goal. Set a flag indicating that communication is needed.
        Based on the type of communication goal (soil, rock, or image),
        identify the corresponding 'have_X' fact (e.g., `have_soil_analysis`)
        that is immediately prerequisite for communication. If this 'have_X'
        fact is not present in the state for any rover, mark the required
        waypoint (for soil/rock) or (objective, mode) pair (for image) as 'needed'.

    2.  **Process Needed 'Have_X' Facts and Sampling/Imaging Costs:**
        Iterate through the set of 'have_X' facts identified as 'needed' in Pass 1.
        For each needed `have_soil_analysis` or `have_rock_analysis` fact (identified by waypoint W):
        - Add 1 for the 'sample' action (soil or rock).
        - Add 1 for navigation to the sample location (waypoint W).
        - Check if any rover equipped for this type of sampling has an empty store. If not, add 1 for a 'drop' action (simplification: assumes a drop is needed somewhere to free up a store).
        For each needed `have_image` fact (identified by objective O and mode M):
        - Add 1 for the 'take_image' action.
        - Add 1 for navigation to a suitable image location (a waypoint visible from the objective O).
        - Check if any camera I on an equipped imaging rover R supporting the required mode M is currently calibrated in the state. If not, identify all specific camera/rover pairs (I, R) that are suitable (equipped R, I on board R, I supports M) but uncalibrated in the state, and add them to a set of `needed_calibrated` pairs.

    3.  **Process Needed 'Calibrated' Facts and Calibration Costs:**
        Iterate through the set of `needed_calibrated` (camera, rover) pairs identified in Pass 2. This set ensures each unique calibration need is counted only once.
        For each unique (camera, rover) pair in this set:
        - Add 1 for the 'calibrate' action.
        - Add 1 for navigation to a suitable calibration location (a waypoint visible from the camera's calibration target).

    4.  **Add Navigation Cost for Communication:**
        If the flag `communication_needed` was set in Pass 1 (meaning at least one communication goal is missing), check if any rover is currently located at a waypoint that is visible from the lander. If no rover is at such a location, add 1 to the total cost, representing the navigation needed for *some* rover to reach a communication point.

    The final sum represents the estimated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        all_facts = task.facts # Assuming this contains all ground atoms including types

        # Data structures to store static information and object names
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.waypoints = set()
        self.landers = set()

        # Populate object names by type from all_facts
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if len(parts) == 2: # Might be a type declaration like (rover rover1)
                obj_type, obj_name = parts
                if obj_type == 'rover': self.rovers.add(obj_name)
                elif obj_type == 'waypoint': self.waypoints.add(obj_name)
                elif obj_type == 'store': self.stores.add(obj_name)
                elif obj_type == 'camera': self.cameras.add(obj_name)
                elif obj_type == 'mode': self.modes.add(obj_name)
                elif obj_type == 'lander': self.landers.add(obj_name)
                elif obj_type == 'objective': self.objectives.add(obj_name)

        # Now parse static facts for relationships
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()
        self.rover_stores = {} # {rover: [store1, ...]}
        self.rover_cameras = {} # {rover: [camera1, ...]}
        self.camera_modes = set() # {(camera, mode), ...}
        self.camera_calibration_target = {} # {camera: target_objective}
        self.objective_visible_waypoints = {} # {objective: [waypoint1, ...]}
        self.calibration_target_visible_waypoints = {} # {target_objective: [waypoint1, ...]}
        self.lander_waypoint = None
        self.lander_visible_waypoints = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                # Assuming only one lander
                # Check if parts[1] is actually a lander object name
                if parts[1] in self.landers:
                     self.lander_waypoint = parts[2]
            elif predicate == "equipped_for_soil_analysis":
                if parts[1] in self.rovers:
                    self.equipped_soil_rovers.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                 if parts[1] in self.rovers:
                    self.equipped_rock_rovers.add(parts[1])
            elif predicate == "equipped_for_imaging":
                 if parts[1] in self.rovers:
                    self.equipped_imaging_rovers.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                if rover in self.rovers and store in self.stores:
                    self.rover_stores.setdefault(rover, []).append(store)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                if camera in self.cameras and mode in self.modes:
                    self.camera_modes.add((camera, mode))
            elif predicate == "visible":
                w1, w2 = parts[1], parts[2]
                # Store visibility for lander check later
                # No need to store full graph for this heuristic
                pass # Handled below after lander_waypoint is found
            elif predicate == "visible_from":
                obj_or_target, waypoint = parts[1], parts[2]
                # Could be objective or calibration target (which is an objective)
                # Assuming obj_or_target is in self.objectives
                if obj_or_target in self.objectives and waypoint in self.waypoints:
                    self.objective_visible_waypoints.setdefault(obj_or_target, []).append(waypoint)
            elif predicate == "calibration_target":
                camera, target = parts[1], parts[2]
                if camera in self.cameras and target in self.objectives:
                    self.camera_calibration_target[camera] = target
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                if camera in self.cameras and rover in self.rovers:
                    self.rover_cameras.setdefault(rover, []).append(camera)

        # Populate lander_visible_waypoints after finding lander_waypoint
        if self.lander_waypoint:
             for fact in self.static_facts:
                 if match(fact, "visible", "*", self.lander_waypoint):
                     self.lander_visible_waypoints.add(get_parts(fact)[1])
                 # The communicate precondition is (visible ?x ?y) where ?x is rover, ?y is lander
                 # So we need waypoints ?x such that (visible ?x lander_waypoint)
                 # The pattern "*", lander_waypoint covers this.

        # Populate calibration_target_visible_waypoints
        self.calibration_target_visible_waypoints = {
             target: self.objective_visible_waypoints.get(target, [])
             for target in self.camera_calibration_target.values()
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        cost = 0

        needed_have_soil = set() # waypoints W
        needed_have_rock = set() # waypoints W
        needed_have_image = set() # (objective O, mode M)
        needed_calibrated = set() # (camera I, rover R)

        # Pass 1: Identify missing terminal goals and required immediate prerequisites
        communication_needed = False
        for goal in self.goals:
            if goal in state:
                continue

            communication_needed = True
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                cost += 1 # Communicate action
                # Check if (have_soil_analysis R waypoint) exists for any R
                if not any(f'(have_soil_analysis {r} {waypoint})' in state for r in self.rovers):
                    needed_have_soil.add(waypoint)

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                cost += 1 # Communicate action
                # Check if (have_rock_analysis R waypoint) exists for any R
                if not any(f'(have_rock_analysis {r} {waypoint})' in state for r in self.rovers):
                    needed_have_rock.add(waypoint)

            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                cost += 1 # Communicate action
                # Check if (have_image R objective mode) exists for any R
                if not any(f'(have_image {r} {objective} {mode})' in state for r in self.rovers):
                    needed_have_image.add((objective, mode))

        # Pass 2: Process needed have_X facts and their prerequisites
        for waypoint in needed_have_soil:
            cost += 1 # Sample soil action
            cost += 1 # Navigation to waypoint
            # Check if any equipped soil rover has an empty store
            has_empty_soil_store = any(
                f'(empty {s})' in state
                for r in self.equipped_soil_rovers
                for s in self.rover_stores.get(r, [])
            )
            if not has_empty_soil_store:
                cost += 1 # Drop action

        for waypoint in needed_have_rock:
            cost += 1 # Sample rock action
            cost += 1 # Navigation to waypoint
            # Check if any equipped rock rover has an empty store
            has_empty_rock_store = any(
                f'(empty {s})' in state
                for r in self.equipped_rock_rovers
                for s in self.rover_stores.get(r, [])
            )
            if not has_empty_rock_store:
                cost += 1 # Drop action

        for objective, mode in needed_have_image:
            cost += 1 # Take image action
            # Check if there is any waypoint visible from the objective.
            # Assuming solvable, at least one exists.
            cost += 1 # Navigation to image location (waypoint visible from objective)

            # Check if any camera I on any equipped imaging rover R supporting mode M is calibrated
            # Find suitable camera/rover pairs: (I, R) where R is equipped_imaging, I is on_board R, I supports M.
            suitable_camera_rover_pairs = [
                (i, r)
                for r in self.equipped_imaging_rovers
                for i in self.rover_cameras.get(r, [])
                if (i, mode) in self.camera_modes
            ]

            is_calibrated_for_image = any(f'(calibrated {i} {r})' in state for i, r in suitable_camera_rover_pairs)

            if not is_calibrated_for_image:
                # Need to calibrate *some* suitable camera/rover pair.
                # Add *all* suitable but uncalibrated pairs to needed_calibrated set.
                # The cost for calibration will be added once per unique (camera, rover) pair in the next pass.
                for i, r in suitable_camera_rover_pairs:
                     if f'(calibrated {i} {r})' not in state:
                         needed_calibrated.add((i, r))

        # Pass 3: Process needed calibrated facts
        for camera, rover in needed_calibrated:
            cost += 1 # Calibrate action
            # Need to ensure the camera has a calibration target defined and visible from somewhere.
            # Assuming solvable, target exists and is visible from somewhere.
            cost += 1 # Navigation to calibration location (waypoint visible from calibration target)


        # Pass 4: Add navigation cost for communication if needed
        # Communication is needed if any goal was not in the state (communication_needed flag).
        if communication_needed:
            rover_at_lander_visible_waypoint = any(
                match(fact, "at", r, w)
                for fact in state
                for r in self.rovers # Iterate through all rovers
                for w in self.lander_visible_waypoints
            )
            if not rover_at_lander_visible_waypoint:
                cost += 1 # Navigation to lander-visible waypoint

        return cost
