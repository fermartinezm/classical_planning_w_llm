from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(have_image camera1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rover-2 (secret_domain) domain.

    Summary:
    This heuristic estimates the number of required actions to reach a goal state by
    analyzing key facts related to modes being full at the rover, images and objectives
    processed, and soil/rock analysis. It focuses on modeling the main steps needed:
    taking images, sampling rocks or soil, communicating data, calibrating modes,
    and completing the modes to be full as required by the goal.

    Assumptions:
    - Goal facts require some or all modes to be full (as seen in example goals).
    - Important supporting actions involve: 
      - having images on cameras,
      - objectives being empty or not,
      - wayspoints having soil analysis,
      - modes equipped for soil analysis,
      - equipped rock analysis in combination with landers,
      - and communication of soil and rock data.
    - The rover(s) can move on landers that they can traverse.
    - The heuristic doesn’t need to be admissible but should guide search efficiently by counting expected needed actions.

    Heuristic Initialization:
    - Extract modes involved in the goal that need to be full.
    - Record which modes are supported (from static facts).
    - Collect facts from static information to quickly reference properties like supported modes and which cameras or objectives have communicated data.
    - Keep track of landers that rovers can traverse.
    - Collect "communicated_soil_data" and "communicated_rock_data" cameras and objectives from static info for estimating communication steps.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify number of modes still not full that are needed by the goal.
    2. Count how many modes are equipped for soil analysis (likely need to be full).
    3. Count how many cameras still have images (need communication to clear).
    4. Count how many objectives are still empty (samples not yet taken).
    5. Count waypoints that still have soil analysis (likely need to take images or sample).
    6. Check how many equipped_for_rock_analysis facts exist (these need communicating or calibrating).
    7. Assess how many modes are currently full.
    8. Estimate actions by combining:
       - Taking images for waypoints that are visible,
       - Sampling rocks or soils where possible,
       - Communicating data for equipped modes,
       - Calibrating modes to become full,
       - Sampling soil with rovers traversing landers,
       - Accounting for action precondition dependencies (e.g., visibility, traversal).
    - Sum these counts with a small adjustment factor for possible overlaps to produce an estimated cost.
    - Return 0 if all goal conditions are met (all modes full as required).

    This heuristic aims to estimate the "work left" as a sum of:
    - missing mode completions (full modes),
    - unprocessed images,
    - unfinished objectives,
    - uncommunicated data (equipped rock analysis),
    - necessary calibrations,
    - and rover traversals.
    """

    def __init__(self, task):
        """
        Initialize by extracting:
        - modes that must be full from goal facts,
        - supported modes,
        - landers with rover traversal capability,
        - cameras and objectives with communicated data,
        - static states that will guide heuristic computation.
        """
        self.goals = task.goals
        self.static = task.static

        # Modes required to be full in the goal
        self.goal_modes = set()
        for goal_fact in self.goals:
            if match(goal_fact, "full", "*"):
                mode = get_parts(goal_fact)[1]
                self.goal_modes.add(mode)

        # Supported modes from statics
        self.supported_modes = {get_parts(fact)[1] for fact in self.static if match(fact, "supports", "*")}

        # Landers and their traversal capabilities (lander -> rover1)
        # We only know from can_traverse facts which landers rover1 can traverse
        # and presumably rover1 is the main rover.
        self.landers_rover1_can_traverse = {get_parts(fact)[1] for fact in self.static if match(fact, "can_traverse", "*", "rover1")}

        # Cameras with communicated soil data
        self.cameras_with_comm_soil_data = {get_parts(fact)[1] for fact in self.static if match(fact, "communicated_soil_data", "*")}
        # Objectives with communicated rock data
        self.objectives_with_comm_rock_data = {get_parts(fact)[1] for fact in self.static if match(fact, "communicated_rock_data", "*")}

        # Modes equipped for soil analysis (static info, though they can also change; use static if available)
        self.equipped_soil_modes = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_soil_analysis", "*")}

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the given state."""
        state = node.state

        # Quick goal check
        if self.goals <= state:
            return 0

        # Helper counters of relevant facts in the current state:

        # Modes currently full
        full_modes = {get_parts(fact)[1] for fact in state if match(fact, "full", "*")}

        # Modes that should be full but are not yet
        modes_needed_full = self.goal_modes - full_modes

        # Number of modes that support soil analysis and are equipped for soil analysis
        # (checking equipped_for_soil_analysis fact since it appears as a predicate in static and might be in state)
        equipped_for_soil = {get_parts(fact)[1] for fact in state if match(fact, "equipped_for_soil_analysis", "*")}
        # Also include those supported modes (to identify modes that may become full)
        modes_controlling = (equipped_for_soil | self.equipped_soil_modes)

        # Cameras that have images currently
        cameras_with_images = {get_parts(fact)[1] for fact in state if match(fact, "have_image", "*")}

        # Objectives empty (i.e., need sampling)
        empty_objectives = {get_parts(fact)[1] for fact in state if match(fact, "empty", "*")}

        # Waypoints with soil analysis (have_soil_analysis - means soil sampled or analyzed)
        waypoints_with_soil_analysis = {get_parts(fact)[1] for fact in state if match(fact, "have_soil_analysis", "*")}

        # Equipped for rock analysis (fact includes waypoint and lander)
        equipped_rock_analysis = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "equipped_for_rock_analysis", "*", "*")}

        # Visible waypoints (might be needed for taking images)
        visible_waypoints = {get_parts(fact)[1] for fact in state if match(fact, "visible", "*")}

        # Landers rovers can traverse in current state (dynamic can_traverse facts)
        # We focus on rover1 as the rover in problem statements.
        landers_rover1_can_traverse_now = {get_parts(fact)[1] for fact in state if match(fact, "can_traverse", "*", "rover1")}

        # Modes at rovers (mode -> rover)
        modes_at_rovers = { (get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "at", "*", "*")}

        # Number of calibrate-relevant modes:
        # A mode can be calibrated if it is supported and at a rover
        calibratable_modes = {mode for mode in self.supported_modes if any(m == mode for (m, _) in modes_at_rovers)}

        ##############################
        # Heuristic estimation logic #
        ##############################

        # Important observations about necessary actions:
        #
        # 1) To complete a mode (make it full), it requires that:
        #    - The mode is equipped for soil analysis or rock analysis.
        #    - Calibrate or communicate actions have been applied depending on the mode's state.
        #
        # 2) Cameras with images need their image data to be communicated before related objectives become non-empty.
        #
        # 3) Objectives empty mean samples are not yet taken, requiring sample_rock or sample_soil actions.
        #
        # 4) Equipped for rock analysis facts indicate data needing communication or calibration.
        #
        # 5) Visible waypoints may need images to be taken.
        #
        # 6) Rover traversal on landers might be needed for soil sampling or calibrations.

        # Estimate cost roughly as the sum of:
        # - Modes still not full (one calibrate or communicate per mode)
        # - Cameras that still have images (at least one communication to clear)
        # - Empty objectives (samples to be taken)
        # - Equipped rock analysis facts to communicate or calibrate (one action per)
        # - Visible waypoints that might need images taken (one per visible waypoint)
        # - Traversals needed if landers can_traverse actions appear relevant (one per lander in current state?)

        # Count modes needed full (each likely needing 1 action: calibrate or communicate)
        cost_modes_not_full = len(modes_needed_full)

        # Cost for communicating images (one per camera with images)
        cost_cameras_images = len(cameras_with_images)

        # Cost for sampling empty objectives (one action per empty objective)
        cost_empty_objectives = len(empty_objectives)

        # Cost for handling equipped_for_rock_analysis (communicate_soil_data or communicate_rock_data or calibrate)
        cost_equipped_rock = len(equipped_rock_analysis)

        # Cost for taking images of visible waypoints (to equip for rock analysis)
        cost_visible_waypoints = len(visible_waypoints)

        # Estimated cost for rover traversals on landers (count landers rover1 can traverse currently)
        cost_rover_traversals = len(landers_rover1_can_traverse_now)

        # Weights to emphasize common bottlenecks (tune to guide greedy best-first search efficiently)
        w_mode_full = 2
        w_camera_image = 2
        w_empty_objective = 3
        w_rock_analysis = 2
        w_visible_waypoint = 1
        w_rover_traversal = 1

        # Summation gives estimated actions left (not guaranteed admissible, prioritizes progress)
        heuristic_value = (
            w_mode_full * cost_modes_not_full
            + w_camera_image * cost_cameras_images
            + w_empty_objective * cost_empty_objectives
            + w_rock_analysis * cost_equipped_rock
            + w_visible_waypoint * cost_visible_waypoints
            + w_rover_traversal * cost_rover_traversals
        )

        # Defensive: heuristic must never be zero if goal not achieved (encourage expansion)
        return heuristic_value if heuristic_value > 0 else 1
