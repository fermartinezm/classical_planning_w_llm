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
    A domain-dependent heuristic for the 'secret_domain' domain (rover-2).

    Summary:
    This heuristic estimates the number of actions required to reach the goal of having
    all specified modes `full`. The main focus is on the completion of mode filling,
    which appears to be the goal in provided instances. The heuristic estimates the cost by
    combining the number of modes not yet full with the operations needed to get them full,
    considering soil and rock analysis prerequisites and communication. It balances between
    imaging, analysis, and communication actions according to the domain dynamics.

    Assumptions:
    - Goal facts require certain modes to be `full`.
    - To get a mode `full`, it must be equipped properly and have communications processed.
    - Cameras provide images that are prerequisite for sampling.
    - Soil analysis must be available on waypoints; some actions require soil analysis.
    - Landers and rovers play a role in traversability but movement costs aren't explicitly modeled here.
    - Communication and sampling actions rely on specific states of images, objectives, and analyses.
    - Heuristic is not required to be admissible, so it uses a simplified but informative count of remaining relevant goals.

    Heuristic Initialization:
    - Extract all modes that must be full in the goal.
    - Extract static facts about which modes are supported.
    - Extract "communicated_soil_data" cameras.
    - Extract "communicated_rock_data" objectives.
    - Index the initial environment static info on modes and support for modes.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which modes in the goal set are not yet `full`.
    2. For each mode not full:
       - If the mode is already `equipped_for_soil_analysis` in the state, that means partial progress.
       - Otherwise, count as needing preparation steps.
    3. Count the number of `have_image` cameras: indicates images still to be taken or used.
    4. Count the number of empty objectives: those that still need sampling or completing.
    5. Count the number of waypoints with soil analysis available.
    6. Evaluate communication prerequisites:
       - Number of communicated soil data cameras.
       - Number of communicated rock data objectives.
    7. The heuristic sums:
       - The number of modes not full (each implies at least one calibrate or communicate action).
       - Number of images needing to be taken or processed.
       - Number of empty objectives to be filled.
       - Number of soil analyses to be done on waypoints.
    8. Because the domain cycles through imaging, sampling, and communication,
       each counted element (mode, image, objective, soil analysis) roughly corresponds to
       at least one action.
    9. A small additional constant or linear sum is included to guide search better.
    10. The heuristic returns 0 if the state satisfies the goal (all required modes full).
    
    This heuristic is efficient to compute, uses only state facts and static info,
    and balances domain knowledge with computational simplicity.
    """

    def __init__(self, task):
        """
        Extract relevant static and goal information.

        Parameters:
        - task: The planning task instance containing goals, static facts, and operators.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract the modes that the goal requires to be full.
        self.goal_full_modes = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "full" and len(parts) == 2:
                self.goal_full_modes.add(parts[1])

        # Extract modes that are supported per static facts for reference (not strictly needed here).
        self.supported_modes = {
            get_parts(fact)[1] for fact in self.static if match(fact, "supports", "*")
        }

        # Extract cameras with communicated soil data (static info).
        self.communicated_soil_cameras = {
            get_parts(fact)[1] for fact in self.static if match(fact, "communicated_soil_data", "*")
        }

        # Extract objectives with communicated rock data (static info).
        self.communicated_rock_objectives = {
            get_parts(fact)[1] for fact in self.static if match(fact, "communicated_rock_data", "*")
        }

    def __call__(self, node):
        """
        Compute the heuristic value estimating the number of remaining actions until the goal.

        Parameters:
        - node: The search node containing the current state.

        Returns:
        - An estimated nonnegative integer heuristic value.
        """
        state = node.state

        # Quick check: if goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # 1) Modes required to be full but not yet full:
        modes_not_full = self.goal_full_modes - {
            get_parts(fact)[1] for fact in state if match(fact, "full", "*")
        }

        # 2) Modes equipped for soil analysis (partial progress):
        modes_equipped_soil = {
            get_parts(fact)[1] for fact in state if match(fact, "equipped_for_soil_analysis", "*")
        }

        # 3) Cameras that currently have images:
        cameras_with_images = {
            get_parts(fact)[1] for fact in state if match(fact, "have_image", "*")
        }

        # 4) Objectives that are still empty:
        empty_objectives = {
            get_parts(fact)[1] for fact in state if match(fact, "empty", "*")
        }

        # 5) Waypoints that currently have soil analysis:
        waypoints_with_soil_analysis = {
            get_parts(fact)[1] for fact in state if match(fact, "have_soil_analysis", "*")
        }

        # 6) Cameras with communicated soil data (from state, may differ from static):
        cameras_communicated_soil = {
            get_parts(fact)[1] for fact in state if match(fact, "communicated_soil_data", "*")
        }

        # 7) Objectives with communicated rock data (from state):
        objectives_communicated_rock = {
            get_parts(fact)[1] for fact in state if match(fact, "communicated_rock_data", "*")
        }

        # Heuristic reasoning:
        # - Every mode not full will need at least 1 calibrate or communicate_soil_data action.
        # - Modes equipped for soil analysis are partially ready, so count them less.
        # - Images need to be processed (sampled or communicated).
        # - Empty objectives require sample_rock or communicate_image_data actions.
        # - Soil analyses on waypoints may enable further communication action.
        # Penalize the lack of communication on soil and rock data.

        # Count mode-related work:
        mode_work = len(modes_not_full) * 2  # Estimated 2 actions per mode for finalizing communication/calibration.

        # Cameras with images require at least one sample or communicate step per image.
        image_work = len(cameras_with_images)  

        # Empty objectives need sampling or communication.
        empty_objective_work = len(empty_objectives)

        # Waypoints with soil analysis can participate in communication actions:
        # count how many soil analyses not yet communicated (heuristic measure).
        soil_analysis_uncommunicated = max(0, len(waypoints_with_soil_analysis) - len(cameras_communicated_soil))

        # Similarly, consider rock data objectives not communicated (heuristic measure).
        rock_data_uncommunicated = max(0, len(empty_objectives.intersection(self.communicated_rock_objectives)) - len(objectives_communicated_rock))

        # Combine all relevant costs.
        heuristic_value = (
            mode_work
            + image_work
            + empty_objective_work
            + soil_analysis_uncommunicated
            + rock_data_uncommunicated
        )

        # Ensure heuristic is at least 1 if not solved.
        return max(heuristic_value, 1)
