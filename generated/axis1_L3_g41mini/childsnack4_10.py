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
    A domain-dependent heuristic for the secret_domain (rover-2) domain.

    Summary:
    --------
    This heuristic estimates the number of actions required to reach the goal state (where certain modes are 'full').
    The domain involves managing cameras, objectives, modes (instruments), waypoints, landers, rovers, and analyses data.
    The heuristic focuses on the progress of filling modes to 'full' (goal condition), which requires communication actions,
    calibration, and correct equipment handling.

    Assumptions:
    ------------
    - The goal is always a conjunction of (full modeX) facts.
    - Actions revolve primarily around obtaining images, conducting analyses, and communicating/calibrating modes to become full.
    - The heuristic counts how many modes still need to become full and estimates the approximate minimal number of actions to achieve that.
    - It ignores detailed movement costs and some dependencies but accounts for critical sequential steps.
    - Each mode that must become full requires at least some combination of calibration or communication actions.
    - The domain's static facts help to identify supported modes, mode ownership (which rover), and other static relations.

    Heuristic Initialization:
    -------------------------
    - Extract the set of modes that must be full to achieve the goal.
    - Extract static information:
      * Which modes support analysis (supports ?mode).
      * Which modes have been equipped for soil analysis (equipped_for_soil_analysis ?mode).
      * Which modes are currently at which rover.
      * Which cameras have communicated soil data.
      * Which objectives have communicated rock data.
      * Landers that rovers can traverse.
    - Index these for fast retrieval during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify which modes are required to be full but are not yet marked as full in the current state.
    2. For each such mode:
       - Check if the mode supports analysis and is equipped for soil analysis (a precondition for communication/calibration).
       - Check if the mode is currently at a rover that can traverse a lander required for equipped_for_rock_analysis waypoints.
       - Consider that to fill a mode, it must undergo either calibration or communication_soil_data actions, which require:
         * The mode to be at a rover.
         * A visible waypoint equipped for rock analysis and visible_from predicates.
       - If the mode is already full, no action is needed.
       - If not full, estimate a minimal number of necessary actions:
         a) If mode is not equipped for soil analysis but "supports" it, at least one action to equip it.
         b) At least one communication or calibration action is needed per mode to make it full.
    3. For goals that require multiple modes full, sum up the minimal estimated action counts per mode.
    4. Additionally, count needed image or sample actions if relevant facts (have_image, empty objectives) indicate these steps have yet to be done.
    5. The heuristic is 0 if all modes required to be full are full (goal reached).
    6. Otherwise, return the sum of actions estimated for the remaining modes.

    This heuristic is not guaranteed admissible but tries to approximate the necessary efforts per mode
    so that greedy best-first searches are guided effectively by reducing remaining modes to fill.
    """

    def __init__(self, task):
        self.goals = task.goals  # Set of goal facts
        self.static = task.static

        # Extract modes that must be full (goal conditions)
        self.goal_full_modes = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "full":
                self.goal_full_modes.add(parts[1])

        # Extract supported modes (supports ?mode)
        self.supported_modes = set()
        for fact in self.static:
            if match(fact, "supports", "*"):
                mode = get_parts(fact)[1]
                self.supported_modes.add(mode)

        # Extract modes currently equipped for soil analysis (equipped_for_soil_analysis ?mode)
        self.modes_equipped_for_soil_analysis = set()
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                mode = get_parts(fact)[1]
                self.modes_equipped_for_soil_analysis.add(mode)

        # Extract which mode is at which rover (at ?mode ?rover)
        self.mode_at_rover = {}
        for fact in self.static:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                self.mode_at_rover[parts[1]] = parts[2]

        # Extract which rovers can traverse which landers (can_traverse ?lander ?rover)
        self.lander_rover_traversable = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*"):
                parts = get_parts(fact)
                lander, rover = parts[1], parts[2]
                self.lander_rover_traversable.setdefault(rover, set()).add(lander)

        # Extract waypoints equipped for rock analysis on landers (equipped_for_rock_analysis ?waypoint ?lander)
        self.waypoint_equipped_lander = {}
        for fact in self.static:
            if match(fact, "equipped_for_rock_analysis", "*", "*"):
                parts = get_parts(fact)
                self.waypoint_equipped_lander[parts[1]] = parts[2]

        # Static set of rovers (from domain constants)
        self.rovers = {"rover1"}

    def __call__(self, node):
        state = node.state

        # Quick helper functions for checking facts in state
        def has_fact(pattern):
            """Return True if any fact in state matches the pattern (predicate and wildcards)"""
            for fact in state:
                if match(fact, *pattern):
                    return True
            return False

        # Detect modes currently full
        modes_full_in_state = set()
        for fact in state:
            if match(fact, "full", "*"):
                mode = get_parts(fact)[1]
                modes_full_in_state.add(mode)

        # If all goal modes are full, heuristic is zero
        if self.goal_full_modes <= modes_full_in_state:
            return 0

        # Detect which modes are not full but must be
        modes_to_fill = self.goal_full_modes - modes_full_in_state

        # Detect modes currently equipped for soil analysis in current state
        modes_equipped = set()
        for fact in state:
            if match(fact, "equipped_for_soil_analysis", "*"):
                mode = get_parts(fact)[1]
                modes_equipped.add(mode)

        # Detect which cameras currently have images
        cameras_with_images = set()
        for fact in state:
            if match(fact, "have_image", "*"):
                cam = get_parts(fact)[1]
                cameras_with_images.add(cam)

        # Detect which objectives are empty (needed for sampling actions)
        objectives_empty = set()
        for fact in state:
            if match(fact, "empty", "*"):
                obj = get_parts(fact)[1]
                objectives_empty.add(obj)

        # Detect visible waypoints in state
        visible_waypoints = set()
        for fact in state:
            if match(fact, "visible", "*"):
                wp = get_parts(fact)[1]
                visible_waypoints.add(wp)

        # Detect waypoints that are visible_from in state
        visible_from_waypoints = set()
        for fact in state:
            if match(fact, "visible_from", "*"):
                wp = get_parts(fact)[1]
                visible_from_waypoints.add(wp)

        # Detect which waypoints have soil analysis (have_soil_analysis)
        waypoints_with_soil_analysis = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*"):
                wp = get_parts(fact)[1]
                waypoints_with_soil_analysis.add(wp)

        # Detect which landers each rover can traverse from current static info plus state
        # (Though can_traverse can also appear in state, we consider stable)
        rover_to_landers = {r: set() for r in self.rovers}
        for fact in state | self.static:
            if match(fact, "can_traverse", "*", "*"):
                parts = get_parts(fact)
                lander, rover = parts[1], parts[2]
                if rover in rover_to_landers:
                    rover_to_landers[rover].add(lander)

        # Heuristic estimation:
        # For each mode to be filled:
        #  - If mode supports soil analysis and is not equipped for soil analysis,
        #    estimate 1 action to equip it.
        #  - Need at least one communication (communicate_soil_data) or calibration action to fill mode.
        #  - For communication/calibration, mode must be at rover able to traverse a lander related to equipped_for_rock_analysis waypoint.
        #  - Assume at least one action to take image if any visible waypoint not equipped for rock analysis.
        #  - Account for sampling rock or soil if images or objectives are still empty or cameras have images.
        # We approximate by summing minimal required sets of these actions per incomplete mode.

        est_actions = 0
        for mode in modes_to_fill:

            # Base minimal actions to fill a mode: 1 (communication or calibration)
            mode_actions = 1

            # Check if mode supports soil analysis and not equipped_for_soil_analysis -> +1 action to equip
            if mode in self.supported_modes and mode not in modes_equipped:
                mode_actions += 1

            # Check if mode is at some rover
            if mode in self.mode_at_rover:
                rover = self.mode_at_rover[mode]
            else:
                # Unknown location for mode: must move to rover first (count as +1)
                # but since we ignore movement costs explicitly, add 0.5 as soft cost
                mode_actions += 1
                rover = None

            # Check if rover can traverse at least one lander associated with waypoints equipped for rock analysis
            # to enable communication/calibration
            traversable_landers = rover_to_landers.get(rover, set()) if rover else set()

            # If no lander traversable, extra action likely needed to reposition mode or rover - add 1 cost
            if not traversable_landers:
                mode_actions += 1

            est_actions += mode_actions

        # Additional rough estimation:
        # If cameras still have images -> at least one communicate_image_data or sample_rock action needed
        if cameras_with_images:
            est_actions += len(cameras_with_images)  # pessimistic: one action per camera image

        # If there exist empty objectives, sampling might still be required
        # Add estimation proportional to number of empty objectives
        est_actions += len(objectives_empty)

        # If visible waypoints exist but not all are equipped for rock analysis, at least one take_image action
        # might be needed per such waypoint
        waypoints_not_equipped = visible_waypoints - set(self.waypoint_equipped_lander.keys())
        est_actions += len(waypoints_not_equipped)

        # If visible_from waypoints are few and equipping/calibration needed, adding small cost
        # to encourage achieving visibility-related preconditions
        if visible_from_waypoints:
            est_actions += 1

        # Avoid returning zero if goal not reached
        if est_actions == 0:
            est_actions = 1

        return est_actions
