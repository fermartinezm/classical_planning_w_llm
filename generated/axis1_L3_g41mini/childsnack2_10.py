from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches the given pattern with wildcards '*' allowed.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (rover-2).

    Summary:
    ----------
    Estimates the number of actions required to reach a goal state where all modes are "full".
    Modes become full by performing soil and rock analysis sequences involving image capture,
    sample taking, calibration, and communication.

    The heuristic focuses on:
    - Counting how many modes still need to become full.
    - Checking which modes already are full.
    - Counting how many objectives remain empty.
    - Counting how many cameras still have images that need to be communicated.
    - Considering if soil analyses at waypoints exist or need to be done.
    - Considering visible waypoints and equipped states relevant for analysis and communication.
    - Considering whether actions like take_image, sample_rock, calibrate, and communicate_* need to be performed.

    Assumptions:
    ------------
    - The main goals are that all "full" predicates on modes are true.
    - The rover1 can traverse all landers that matter; traversal constraints are assumed to be satisfied.
    - Actions that remove "have_image" or set "full" on modes are key milestones.
    - Having soil analysis done at waypoints and the communication of soil/rock data are necessary steps.
    - The heuristic does not guarantee admissibility but aims to minimize expanded nodes by approximating remaining cost.

    Heuristic Initialization:
    --------------------------
    - Extract which modes support soil analysis and their initial full/empty states.
    - Extract which cameras have images to communicate.
    - Extract which objectives are empty or already communicated.
    - Extract which waypoints have soil analysis.
    - Extract sets of landers and map equipped_for_rock_analysis facts per waypoint and lander.
    - Extract communications status related to soil and rock data.
    - Extract which modes are currently equipped for soil analysis.
    - Extract which modes are currently full.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Count how many modes need to become full (modes_goal_not_full).
    2. For each mode that supports soil analysis, if the mode is not full, estimate the cost to get it full:
       - At minimum, one calibrate or communicate_soil_data action is required.
    3. Count how many cameras still have images not communicated and correlate with empty objectives.
       Each (have_image, empty) pair likely requires a "take_image" and some sample or communication action.
    4. Count empty objectives that need to be filled by sampling.
    5. Count waypoints that still need soil analysis or have soil analysis that need communicating.
    6. Sum approximated minimal actions: 
        - One action to process each mode (calibrate/communicate_soil_data),
        - Actions to take images,
        - Actions to sample rock or soil,
        - Actions to communicate data,
        - Actions to take image for visible waypoints.
    7. Return 0 if the goal is reached (all modes full), else return the sum.

    This heuristic approximates the minimal number of key processing actions left, 
    ignoring detailed navigation or ordering constraints for computational efficiency.
    """

    def __init__(self, task):
        # Store the goal: modes that must be full (from goals)
        self.goal_modes_full = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "full":
                self.goal_modes_full.add(parts[1])

        # Extract modes that support soil analysis
        self.modes_supporting_soil = set()
        # Extract modes equipped for soil analysis (initial state might help, but static facts too)
        self.supports_modes = set()
        for f in task.static:
            if match(f, "supports", "*"):
                self.supports_modes.add(get_parts(f)[1])

        # Extract the set of cameras from static info or facts: not directly given, 
        # so the heuristic will dynamically find cameras in state.

        # Extract communicated soil data cameras (static knowledge useful for assumptions)
        self.communicated_soil_cameras = set()
        for f in task.static:
            if match(f, "communicated_soil_data", "*"):
                self.communicated_soil_cameras.add(get_parts(f)[1])

        # Extract communicated rock data objectives (static knowledge)
        self.communicated_rock_objectives = set()
        for f in task.static:
            if match(f, "communicated_rock_data", "*"):
                self.communicated_rock_objectives.add(get_parts(f)[1])

        # Static landers collection
        self.static_landers = set()
        for f in task.static:
            # Landers appear as objects (extract from predicates or as types)
            # Although static facts may not tell us landers directly,
            # use "equipped_for_rock_analysis ? waypoint ? lander" from static in task.static if any
            parts = get_parts(f)
            if len(parts) == 3 and (parts[0] == "equipped_for_rock_analysis"):
                self.static_landers.add(parts[2])

    def __call__(self, node):
        state = node.state

        # Helper functions for matching facts in current state
        def find_all(predicate):
            """Return set of objects for predicate with arity 1 or tuples for >=2."""
            results = set()
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == predicate:
                    if len(parts) == 2:
                        results.add(parts[1])
                    elif len(parts) > 2:
                        results.add(tuple(parts[1:]))
            return results

        # 1) Find which modes are full now
        modes_full = set()
        for f in state:
            if match(f, "full", "*"):
                modes_full.add(get_parts(f)[1])

        # 2) Modes that must become full but are not yet
        modes_needed = self.goal_modes_full - modes_full

        # 3) Modes supporting soil analysis (from static supports)
        # From static they are 'supports' facts, modes supporting soil analysis
        modes_support_soil = self.supports_modes

        # 4) Cameras that currently have images (need processing)
        cameras_with_image = set()
        for fact in state:
            if match(fact, "have_image", "*"):
                cameras_with_image.add(get_parts(fact)[1])

        # 5) Objectives that are empty (need fill/sampling)
        empty_objectives = set()
        for fact in state:
            if match(fact, "empty", "*"):
                empty_objectives.add(get_parts(fact)[1])

        # 6) Waypoints that have soil analysis available
        waypoints_soil_analysis = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*"):
                waypoints_soil_analysis.add(get_parts(fact)[1])

        # 7) Waypoints equipped for rock analysis (pairs waypoint, lander)
        equipped_for_rock = set()
        for fact in state:
            if match(fact, "equipped_for_rock_analysis", "*", "*"):
                parts = get_parts(fact)
                equipped_for_rock.add( (parts[1], parts[2]) )

        # 8) Modes currently equipped for soil analysis
        equipped_for_soil_analysis_modes = set()
        for fact in state:
            if match(fact, "equipped_for_soil_analysis", "*"):
                equipped_for_soil_analysis_modes.add(get_parts(fact)[1])

        # 9) Cameras that have communicated soil data (from static and state)
        communicated_soil_data_cameras = set()
        for fact in state:
            if match(fact, "communicated_soil_data", "*"):
                communicated_soil_data_cameras.add(get_parts(fact)[1])

        # 10) Objectives with communicated rock data
        communicated_rock_data_objectives = set()
        for fact in state:
            if match(fact, "communicated_rock_data", "*"):
                communicated_rock_data_objectives.add(get_parts(fact)[1])

        # 11) Waypoints visible / visible_from (used for take_image preconditions)
        visible_waypoints = set()
        for fact in state:
            if match(fact, "visible", "*"):
                visible_waypoints.add(get_parts(fact)[1])
        visible_from_waypoints = set()
        for fact in state:
            if match(fact, "visible_from", "*"):
                visible_from_waypoints.add(get_parts(fact)[1])

        # Count heuristic estimate components:

        # ACTIONS to get modes full:
        # Each mode in modes_needed will at least need 1 calibrate or communicate_soil_data action.
        # So add +1 per mode needed
        actions_for_modes = len(modes_needed)

        # ACTIONS for cameras with images and empty objectives: 
        # Each camera with image must be either sampled or communicated.
        num_cameras_images_pending = len(cameras_with_image)

        # ACTIONS for empty objectives: Need sample_rock or communicate_image_data
        # Each empty objective must be filled once (if not already communicated rock data).
        objectives_to_fill = empty_objectives - communicated_rock_data_objectives
        num_objectives_to_fill = len(objectives_to_fill)

        # ACTIONS for soil analysis at waypoints: 
        # Each waypoint having soil analysis can require communicate_image_data or sampling actions.
        # The number of waypoints with soil analysis that are not lost yet.
        waypoints_with_soil_but_not_visible = waypoints_soil_analysis - visible_waypoints

        # Visible waypoints often require "take_image" action to be done once to equip for rock analysis.
        num_take_image_needed = len(visible_waypoints)

        # Sum approx minimal actions:
        # actions_for_modes (calibrate/communicate_soil_data)
        # + num_cameras_images_pending (sample_rock or communicate_image_data)
        # + num_objectives_to_fill (fill each empty objective once)
        # + num_take_image_needed (take_image on visible waypoints)
        # Ignoring exact dependencies or sequences for efficiency.

        heuristic_estimate = (
            actions_for_modes
            + num_cameras_images_pending
            + num_objectives_to_fill
            + num_take_image_needed
        )

        # Return 0 if at goal (all modes full), else heuristic estimate.
        return 0 if not modes_needed else heuristic_estimate
