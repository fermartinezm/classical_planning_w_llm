import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing parentheses
    and splitting by space.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    return fact[1:-1].split()

def match(fact_parts, *pattern):
    """
    Check if the parts of a PDDL fact match a given pattern.
    Wildcards (*) can be used in the pattern.
    Example: match(["at", "rover1", "waypoint1"], "at", "*", "waypoint1") -> True
             match(["at", "rover1", "waypoint1"], "at", "rover2", "*") -> False
    """
    if len(fact_parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting the minimum number of essential actions (sampling, calibration,
    imaging, communication) needed to satisfy each goal predicate that is not
    currently true. It considers the dependencies between actions (e.g., communication
    requires data, imaging requires calibration) and approximates the need for
    calibration based on the effect of the take_image action consuming the
    calibrated state. It ignores navigation costs and complex resource allocation
    (rover assignment, store management) for computational efficiency, aiming for
    an informative estimate to guide greedy best-first search.

    # Assumptions
    - The primary cost drivers are the core actions: sample_soil, sample_rock,
      calibrate, take_image, communicate_soil_data, communicate_rock_data,
      communicate_image_data.
    - Navigation costs (navigate action) are ignored to simplify the calculation
      and ensure efficient computability.
    - The cost of the 'drop' action is ignored, assuming the store management
      is a secondary factor compared to achieving the main goals.
    - Any capable rover/camera can be used for a task; the heuristic does not
      perform optimal resource assignment.
    - The heuristic assumes goals are potentially reachable if the necessary
      equipment (rovers, cameras, capabilities) exists according to static facts.
      It returns infinity if a required capability is entirely missing.

    # Heuristic Initialization (`__init__`)
    - Stores the goal predicates defined in the task.
    - Parses static facts provided in the task object to build efficient lookup tables:
        - `equipped_soil`, `equipped_rock`, `equipped_image`: Sets of rovers with
          specific analysis or imaging capabilities.
        - `camera_supports`: Dictionary mapping each camera to the set of modes it supports.
        - `camera_on_board`: Dictionary mapping each rover to a list of cameras it carries.
        - `camera_calibration_target`: Dictionary mapping cameras to their specific
          calibration target objective.
        - `objective_visible_from`: Dictionary mapping objectives to the set of
          waypoints from which they are visible (needed for calibrate and take_image).
    - Precomputes sets of goal conditions for faster lookup during heuristic evaluation:
        - `goal_comm_soil`: Set of waypoints for which soil data must be communicated.
        - `goal_comm_rock`: Set of waypoints for which rock data must be communicated.
        - `goal_comm_image`: Set of (objective, mode) tuples for which image data
          must be communicated.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Check Goal Completion: If the current state satisfies all goal predicates,
        the heuristic value is 0.
    2.  Parse Current State: Extract relevant information from the current state's facts,
        such as which data has been collected (`have_soil_analysis`, `have_rock_analysis`,
        `have_image`), which cameras are calibrated (`calibrated`), and which goals
        have already been achieved (`communicated_*`). Store these in sets for
        efficient lookup.
    3.  Identify Unsatisfied Communication Goals: Determine the sets of soil waypoints,
        rock waypoints, and (objective, mode) pairs for which communication is
        still required by comparing the goal sets with the communicated sets from the
        current state. Add the count of these unsatisfied communication goals to the
        heuristic value (1 action per communication).
    4.  Identify Necessary Data Acquisition (Sampling/Imaging): For each unsatisfied
        communication goal, check if the prerequisite data (soil analysis, rock analysis,
        or image) is present in the current state (i.e., held by *any* rover).
        - If the data is missing, increment the heuristic value by 1 for the required
          data acquisition action (`sample_soil`, `sample_rock`, or `take_image`).
        - Keep track of the specific samples (`required_sample_soil`, `required_sample_rock`)
          and images (`required_take_image`) that need to be acquired.
        - Perform a basic reachability check: if a required action needs equipment
          (e.g., soil sampler) and no rover has it, return infinity.
    5.  Estimate Necessary Calibration Actions: The `take_image` action requires a
        calibrated camera and consumes the calibration.
        - Identify cameras that are currently calibrated and capable of taking *any*
          of the required images (`required_take_image`).
        - Simulate the process: For each required image, try to assign a currently
          calibrated camera. If found, "consume" that calibration for the estimate
          (remove it from a temporary set of available calibrated cameras).
        - If no suitable calibrated camera is available for a required image, check if
          *any* camera *could* be calibrated and take the image. If yes, increment
          a counter for necessary calibration actions (`num_calibrations_needed`)
          by 1. This estimates one `calibrate` action is needed before that `take_image`.
        - If a required image cannot be taken even with calibration (e.g., no camera
          supports the mode), return infinity.
        - Add `num_calibrations_needed` to the total heuristic value.
    6.  Final Value: The heuristic value is the sum of counts from steps 3, 4, and 5.
        Ensure the value is at least 1 if the goal state has not been reached.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals

        # --- Extract Static Information ---
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        # self.rover_stores = {} # rover -> store # Not used in this version
        self.camera_supports = collections.defaultdict(set) # camera -> set(modes)
        self.camera_on_board = collections.defaultdict(list) # rover -> list(cameras)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = collections.defaultdict(set) # objective -> set(waypoints)
        # self.lander_loc = None # Not needed in this version
        # self.visibility = collections.defaultdict(set) # wp1 -> set(wp2) # Not needed

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif pred == "equipped_for_imaging":
                self.equipped_image.add(parts[1])
            # elif pred == "store_of": # Not used
            #     self.rover_stores[parts[2]] = parts[1]
            elif pred == "supports":
                self.camera_supports[parts[1]].add(parts[2])
            elif pred == "on_board":
                self.camera_on_board[parts[2]].append(parts[1])
            elif pred == "calibration_target":
                self.camera_calibration_target[parts[1]] = parts[2]
            elif pred == "visible_from":
                self.objective_visible_from[parts[1]].add(parts[2])
            # elif pred == "at_lander": # Not used
            #     self.lander_loc = parts[2]
            # elif pred == "visible": # Not used
            #     self.visibility[parts[1]].add(parts[2])

        # --- Precompute Goal Information ---
        self.goal_comm_soil = set()
        self.goal_comm_rock = set()
        self.goal_comm_image = set() # set of (objective, mode) tuples

        for goal in self.goals:
             parts = get_parts(goal)
             pred = parts[0]
             if pred == "communicated_soil_data":
                 self.goal_comm_soil.add(parts[1])
             elif pred == "communicated_rock_data":
                 self.goal_comm_rock.add(parts[1])
             elif pred == "communicated_image_data":
                 self.goal_comm_image.add(tuple(parts[1:])) # (objective, mode)


    def __call__(self, node):
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # --- Parse Current State ---
        achieved_soil = set() # set of (rover, waypoint)
        achieved_rock = set() # set of (rover, waypoint)
        achieved_image = set() # set of (rover, objective, mode)
        calibrated = set() # set of (camera, rover)
        communicated_soil = set() # set of waypoints
        communicated_rock = set() # set of waypoints
        communicated_image = set() # set of (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "have_soil_analysis":
                achieved_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis":
                achieved_rock.add((parts[1], parts[2]))
            elif pred == "have_image":
                achieved_image.add((parts[1], parts[2], parts[3]))
            elif pred == "calibrated":
                calibrated.add((parts[1], parts[2]))
            elif pred == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif pred == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif pred == "communicated_image_data":
                communicated_image.add((parts[1], parts[2]))

        # --- Calculate Missing Steps ---
        actions_needed = 0

        # 1. Communications needed
        needed_comm_soil = self.goal_comm_soil - communicated_soil
        needed_comm_rock = self.goal_comm_rock - communicated_rock
        needed_comm_image = self.goal_comm_image - communicated_image
        # Add 1 action cost for each required communication
        actions_needed += len(needed_comm_soil) + len(needed_comm_rock) + len(needed_comm_image)

        # 2. Samples needed (if corresponding communication is needed but data absent)
        required_sample_soil = set() # Stores waypoints where soil sample is needed
        for w in needed_comm_soil:
            # Check if *any* rover currently has the analysis for this waypoint
            if not any(s_w == w for r, s_w in achieved_soil):
                 # Check if *any* rover is equipped for soil analysis (basic reachability)
                 if not self.equipped_soil:
                     return float('inf') # Goal likely unreachable if no rover can sample soil
                 required_sample_soil.add(w)

        required_sample_rock = set() # Stores waypoints where rock sample is needed
        for w in needed_comm_rock:
            # Check if *any* rover currently has the analysis for this waypoint
            if not any(r_w == w for r, r_w in achieved_rock):
                 # Check if *any* rover is equipped for rock analysis (basic reachability)
                 if not self.equipped_rock:
                     return float('inf') # Goal likely unreachable
                 required_sample_rock.add(w)

        # Add 1 action cost for each required sample action
        actions_needed += len(required_sample_soil) + len(required_sample_rock)

        # 3. Images needed (if corresponding communication is needed but data absent)
        required_take_image = set() # Stores (objective, mode) tuples for needed images
        for o, m in needed_comm_image:
            # Check if *any* rover currently has the image
            if not any(i_o == o and i_m == m for r, i_o, i_m in achieved_image):
                 # Check if *any* rover/camera can take this image mode (basic reachability)
                 can_take = False
                 for r in self.equipped_image:
                     for i in self.camera_on_board.get(r, []):
                         if m in self.camera_supports.get(i, set()):
                             can_take = True
                             break
                     if can_take: break
                 if not can_take:
                     return float('inf') # Goal likely unreachable if no rover/camera can take this image
                 required_take_image.add((o, m))

        # Add 1 action cost for each required take_image action
        actions_needed += len(required_take_image)

        # 4. Calibrations needed (estimate based on required images and current calibration state)
        num_calibrations_needed = 0
        if required_take_image:
            # Find cameras that are currently calibrated and could potentially take one of the needed images
            available_calibrated_cameras = set() # Stores (camera, rover) tuples
            for i, r in calibrated:
                 if r in self.equipped_image:
                     # Check if this camera supports *any* of the required modes
                     supported_modes = self.camera_supports.get(i, set())
                     if any(m in supported_modes for o, m in required_take_image):
                         available_calibrated_cameras.add((i, r))

            # Simulate the consumption of calibrations by required take_image actions
            temp_calibrated = available_calibrated_cameras.copy()
            for o, m in required_take_image:
                found_calibrated_cam_for_this_image = False
                cam_to_use = None
                rover_to_use = None

                # Try to find an available calibrated camera that supports the required mode
                for i, r in list(temp_calibrated): # Iterate over a copy for safe removal
                    if m in self.camera_supports.get(i, set()):
                         found_calibrated_cam_for_this_image = True
                         cam_to_use = i
                         rover_to_use = r
                         break # Found one

                if found_calibrated_cam_for_this_image:
                    # Consume this calibration for the estimate
                    temp_calibrated.remove((cam_to_use, rover_to_use))
                else:
                    # No suitable *currently* calibrated camera found. Estimate 1 calibrate action needed.
                    # Perform a check: can *any* camera be calibrated for this mode?
                    can_calibrate_for_this_mode = False
                    for r_potential in self.equipped_image:
                        for i_potential in self.camera_on_board.get(r_potential, []):
                            # Check if camera supports mode AND has a calibration target defined
                            if m in self.camera_supports.get(i_potential, set()) and i_potential in self.camera_calibration_target:
                                # Check if the calibration target objective is visible from *any* waypoint
                                target_obj = self.camera_calibration_target[i_potential]
                                if target_obj in self.objective_visible_from:
                                     can_calibrate_for_this_mode = True
                                     break
                        if can_calibrate_for_this_mode: break

                    if can_calibrate_for_this_mode:
                        num_calibrations_needed += 1
                        # We don't add to temp_calibrated because the take_image consumes
                        # the calibration immediately in this simplified model.
                    else:
                        # If no camera can be calibrated for this required image mode, goal is unreachable
                        return float('inf')

        # Add the estimated calibration actions cost
        actions_needed += num_calibrations_needed

        # Ensure heuristic is at least 1 for non-goal states
        # This prevents the heuristic from being 0 when actions are still needed
        # (e.g., only communication actions left)
        if actions_needed == 0 and self.goals > state:
             return 1

        return actions_needed
