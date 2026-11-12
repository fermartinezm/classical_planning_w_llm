from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (rover-2) domain.

    Summary
    -------
    This heuristic estimates the number of actions remaining to achieve the goal(s) 
    primarily by counting how many modes need to become full (the goal is to have all 
    goal modes be full), and estimating the minimal number of necessary steps to 
    communicate the soil or rock data to them.

    Assumptions
    -----------
    - The goal facts require certain modes to be full.
    - To fill a mode (make it full), a sequence of actions involving:
      sampling soil, taking images, calibrating, and communicating data is needed.
    - The rover can traverse landers that support required equipment.
    - Communication of data and calibration require certain preconditions (e.g., 
      equipped modes, visibility from waypoints).
    - The heuristic does not account for complicated traversal sequences but assumes 
      that traversal steps are available if "can_traverse" facts exist.
    - Heuristic is not admissible but aims to minimize expanded nodes by a realistic 
      estimate of required steps.

    Heuristic Initialization
    -------------------------
    - Extract from static facts:
      * Which modes support soil analysis (supports).
      * Which modes are initially equipped for soil analysis.
      * Which cameras have communicated soil data.
      * Which objectives have communicated rock data.
      * The location of modes with respect to rovers (at facts).
    - From goals:
      * Extract all modes required to be full.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. Identify which modes the goal requires to be full.
    2. For each mode still not full:
       - Check if the mode is currently full in the state → no cost needed.
       - Otherwise:
         a. Check if the mode is equipped for soil analysis or rock analysis.
         b. If not equipped for soil analysis:
            - One calibrate or communicate_soil_data action will be needed to produce full.
         c. Each mode likely requires at least one communication calibrate or communication action.
         d. Modes that are equipped but not full will require one final communicate or calibrate action.
    3. Consider the availability of images and soil analyses for objectives and waypoints:
       - If have_image and have_soil_analysis are available for corresponding objects, 
         less preliminary step is needed.
    4. The heuristic estimates the sum of:
       - Number of modes not full × minimal required actions (usually 1 or 2).
       - Number of cameras or objectives that still have images or empty data and must be processed.
    5. Visibility and traversal conditions are assumed to hold or cost 1 action each.
    6. This heuristic counts mostly the "full mode" conversions not yet done, ignoring the exact detailed chain for efficiency, but influenced by the domain structure.

    Returns 0 if the goal is satisfied (all required modes full).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract modes that must be full from goals: (full modeX)
        self.goal_full_modes = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "full" and len(parts) == 2:
                self.goal_full_modes.add(parts[1])

        # Extract modes that support soil analysis (supports modeX)
        self.supported_modes = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "supports" and len(parts) == 2:
                self.supported_modes.add(parts[1])

        # Extract modes currently equipped for soil analysis (equipped_for_soil_analysis modeX)
        self.equipped_soil_modes = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_soil_analysis" and len(parts) == 2:
                self.equipped_soil_modes.add(parts[1])

        # Extract cameras that have communicated soil data (communicated_soil_data cameraX)
        self.communicated_soil_cameras = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "communicated_soil_data" and len(parts) == 2:
                self.communicated_soil_cameras.add(parts[1])

        # Extract objectives that have communicated rock data (communicated_rock_data objectiveX)
        self.communicated_rock_objectives = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "communicated_rock_data" and len(parts) == 2:
                self.communicated_rock_objectives.add(parts[1])

        # Extract modes location: which rover they are at (at modeX roverY)
        self.mode_at_rover = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                mode_or_other = parts[1]
                vehicle = parts[2]
                # Only modes here, skip if not mode type
                # Our domain declares modes as a type and goals concern modes
                self.mode_at_rover[mode_or_other] = vehicle

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goal full modes are full in current state -> heuristic = 0
        if all(f"(full {mode})" in state for mode in self.goal_full_modes):
            return 0

        # Count how many modes must be full but are not full yet
        not_full_modes = [
            mode for mode in self.goal_full_modes if f"(full {mode})" not in state
        ]

        # Count how many modes are currently equipped for soil analysis in state
        equipped_soil_modes_in_state = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "equipped_for_soil_analysis" and len(parts) == 2
        }

        # Count how many modes are equipped for rock analysis in state
        equipped_rock_modes_in_state = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "equipped_for_rock_analysis" and len(parts) == 3
        }

        # Count how many cameras currently have images
        cameras_with_images = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "have_image" and len(parts) == 2
        }

        # Count how many objectives currently are empty
        empty_objectives = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "empty" and len(parts) == 2
        }

        # Count how many waypoints currently have soil analysis
        waypoints_with_soil_analysis = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "have_soil_analysis" and len(parts) == 2
        }

        # Count how many waypoints are visible
        visible_waypoints = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "visible" and len(parts) == 2
        }

        # Count how many waypoints are visible_from
        visible_from_waypoints = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "visible_from" and len(parts) == 2
        }

        # Count how many landers can be traversed by rover1 in state
        can_traverse_landers = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "can_traverse" and len(parts) == 3 and parts[2] == "rover1"
        }

        # Estimation strategy:
        # Each mode not full is estimated as requiring:
        # - 1 action to calibrate or communicate (finalize filling)
        # - If not equipped for soil or rock analysis, at least 1 more action to get equipped before
        # Calibrate or communicate requires equipment (equipped_for_soil_analysis or rock_analysis).
        #
        # We also consider the necessary steps to process images and objectives:
        # - Each camera with images still must be processed (sample_rock or communicate_image_data),
        #   costs at least 1 action.
        # - Each empty objective likely requires processing via sample_rock or communicate_image_data.
        #
        # We penalize more if modes are not equipped.
        #
        # Also consider the soil analyses that are present or missing.

        heuristic = 0

        # Cost per mode not full
        for mode in not_full_modes:
            # Check if mode is currently equipped for soil analysis or rock analysis
            equipped_for_soil = mode in equipped_soil_modes_in_state
            equipped_for_rock = any(
                (mode == p[1] and p[0] == "equipped_for_rock_analysis")
                for fact in state
                for p in [get_parts(fact)]
                if p[0] == "equipped_for_rock_analysis" and len(p) == 3
            )
            equipped_for_rock |= any(mode == p[1] for p in equipped_rock_modes_in_state)

            # Estimate 2 steps if not equipped (equip + communicate/calibrate), else 1 step to finalize
            if equipped_for_soil or equipped_for_rock:
                heuristic += 1
            else:
                # Mode might need to be calibrated or communicated after getting equipped
                heuristic += 2

        # Each camera with images likely needs an action to sample or communicate
        heuristic += len(cameras_with_images)

        # Each empty objective likely needs to be filled via sample_rock or communication
        heuristic += len(empty_objectives)

        # Adding minimal costs related to soil analyses at waypoints that help processing
        # We subtract the number of waypoints with soil analyses as those are ready, but not below zero
        # because these indicate pre-processing done.
        heuristic -= len(waypoints_with_soil_analysis) * 0.5  # partial credit, weighted less

        # Ensure heuristic is non-negative
        heuristic = max(heuristic, 0)

        return int(heuristic)
