from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The full fact string, e.g., "(have_image camera1)".
    - `args`: The pattern to match, supporting wildcards '*'.
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (rover-2) domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state.
    The goal usually requires all specified modes to be "full". This means that each
    mode supporting soil/rock analysis must be "filled" via certain sequences of actions
    involving image capture, soil analysis, rock sampling, and communication through landers.
    The heuristic counts, for each mode that must become full, how many important supporting actions
    remain to fulfill that goal, summing up partial progress indicators from the current state.
    It thus guides the search by estimating required communication/calibration/capturing actions.

    Assumptions:
    - The goal predicates always require certain modes to be full.
    - Only provided static and current state facts are used; no numeric costs or movement distances are known.
    - The heuristic is not necessarily admissible but aims to be informative and efficient.
    - The rover and lander assignments are static and do not change the minimal action counts.
    - The heuristic assumes all relevant cameras, objectives, waypoints, and modes involved must be handled.
    - Operator preconditions and effects provide insight into the prerequisite facts for achieving "full" modes.

    Heuristic Initialization:
    - Extract the set of modes that must be "full" in the goal.
    - From static facts:
      * Identify modes that support soil analysis.
      * Identify which cameras have communicated soil data.
      * Identify which objectives have communicated rock data.
      * Identify which modes are currently equipped for soil analysis.
      * Extract which modes are at which rovers.
      * Extract which landers can be traversed by rover1 (main rover).
    - Store these sets and maps to efficiently evaluate state facts during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the goal is already satisfied (all full modes present in state), heuristic is 0.

    2. For each mode that needs to be full:
       - If the mode is already full in the state, no cost for that mode.

       Otherwise:
       - Check if the mode is currently equipped for soil analysis.
         If yes, this means it is ready to communicate soil/rock data or calibrate.
         If no, another action is needed to equip it first (simplified estimate: add 1).

       - Count how many cameras with soil data communicated are available.
         - If none are "communicated_soil_data", soil communication actions may be needed.

       - Count how many objectives have rock data communicated.
         - If none are available, rock communication actions might be needed.

       - Consider interaction with waypoints:
         - Waypoints with soil analysis available (have_soil_analysis),
         - Waypoints that are visible or visible_from, and those equipped for rock analysis.

       We approximate the required missing actions by penalizing lacking equipped fact,
       lacking communicated data facts, and lacking soil analysis availability.

    3. Sum up the estimated minimal actions across all modes that have to become full.

    4. Return the sum as the heuristic value. This heuristic is always finite for solvable states,
       zero only if at goal, and guides greedy best-first search effectively.

    Note: The heuristic is domain-dependent, leveraging static knowledge about relations between
          modes, cameras, objectives, waypoints, landers, and rovers encoded in facts.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting static information and goals."""
        self.goals = task.goals
        self.static = task.static

        # Extract modes that must be full according to goal conditions
        self.goal_full_modes = {
            get_parts(goal)[1]
            for goal in self.goals
            if goal.startswith("(full ")
        }

        # Sets from static facts
        self.supports_modes = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "supports", "*")
        }

        self.communicated_soil_data_cameras = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "communicated_soil_data", "*")
        }

        self.communicated_rock_data_objectives = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "communicated_rock_data", "*")
        }

        self.equipped_for_soil_analysis_modes = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "equipped_for_soil_analysis", "*")
        }

        # We also remember which landers can be traversed by rover1
        # which affects feasibility of some actions.
        self.landers_that_can_traverse_rover1 = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "can_traverse", "*", "rover1")
        }

    def __call__(self, node):
        """Estimate the minimal remaining number of actions to reach a full mode goal."""
        state = node.state

        # Goal check: if all required modes are full, cost is zero
        if all(f"(full {mode})" in state for mode in self.goal_full_modes):
            return 0

        # Extract current facts relevant for heuristic
        # Modes that are currently full
        full_modes = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "full", "*")
        }

        # Modes currently equipped for soil analysis
        equipped_soil_modes = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "equipped_for_soil_analysis", "*")
        }

        # Waypoints which have soil analysis already
        have_soil_analysis_waypoints = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "have_soil_analysis", "*")
        }

        # Cameras currently having images
        have_image_cameras = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "have_image", "*")
        }

        # Objectives currently empty or not (empty = true means can be used in sampling)
        empty_objectives = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "empty", "*")
        }

        # Waypoints visible and visible_from
        visible_waypoints = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "visible", "*")
        }
        visible_from_waypoints = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "visible_from", "*")
        }

        # Equipped for rock analysis (waypoint, lander)
        equipped_for_rock_analysis_pairs = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in state
            if match(fact, "equipped_for_rock_analysis", "*", "*")
        }

        # Lander and rover facts from static and dynamic (mainly rovers assumed rover1 fixed)
        # We ignore changes in can_traverse during the heuristic; assume lander-rover1 connectivity is static here.

        heuristic_value = 0

        for mode in self.goal_full_modes:
            # Skip if mode already full
            if mode in full_modes:
                continue

            # Initialize cost for this mode
            cost_mode = 0

            # Check if mode supports soil analysis (from static)
            supports_soil = mode in self.supports_modes

            # Check if mode is equipped for soil analysis in current state
            is_equipped = mode in equipped_soil_modes

            # If not equipped, it will take at least one action to equip/calibrate
            if supports_soil and not is_equipped:
                cost_mode += 1  # Action needed to equip mode for soil analysis (e.g. "calibrate")

            # To become full -> need communication or calibration

            # Check if any cameras have communicated soil data (either from static or state facts)
            # Here we combine static info with current facts as presence is a precondition
            cameras_with_soil = self.communicated_soil_data_cameras.intersection(have_image_cameras)
            if not cameras_with_soil:
                # If no camera currently has soil data communicated, add cost to acquire soil data
                cost_mode += 1

            # Check if any objectives have communicated rock data, either static or current state
            objectives_with_rock = self.communicated_rock_data_objectives.intersection(empty_objectives)
            if not objectives_with_rock:
                cost_mode += 1  # Cost to sample or communicate rock data likely needed

            # Also consider if there are waypoints with soil analysis available
            if not have_soil_analysis_waypoints:
                cost_mode += 1  # Soil analysis data needs to be obtained from somewhere

            # If waypoint visibility is required (some actions need visible or visible_from waypoints)
            # Count missing visible waypoints to guide taking images or sampling
            if not visible_waypoints:
                cost_mode += 1

            if not visible_from_waypoints:
                cost_mode += 1

            # If no equipped_for_rock_analysis pair is present, sampling or image capturing needed
            if not equipped_for_rock_analysis_pairs:
                cost_mode += 1

            heuristic_value += cost_mode

        # Return the combined heuristic estimate for all modes needing to be full.
        # The heuristic is zero only if all full modes are present.
        # The heuristic is finite and informative otherwise.
        return heuristic_value
