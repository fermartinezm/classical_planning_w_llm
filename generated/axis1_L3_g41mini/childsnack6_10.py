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
    A domain-dependent heuristic for the secret_domain ("rover-2") domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state (where 
    all modes are full) from the current state by analyzing the key predicates related 
    to modes being equipped for soil analysis or rock analysis, images availability, 
    soil analyses at waypoints, and the presence of empty objectives. The heuristic 
    guides a greedy best-first search by approximating how many actions remain to 
    "fill" all modes, based on simplifying assumptions and domain knowledge.

    Assumptions
    -----------
    - The goal states require all modes to be in the "full" predicate.
    - The domain's main progress is measured by "filling" modes, which happens when 
      certain data communications and samples are taken.
    - Images must be taken and communicated; soil and rock data must be gathered, 
      communicated, and objectives that start empty become full during the process.
    - Cameras that "have_image" must be processed.
    - Soil analyses at waypoints are prerequisites for sampling rocks or communicating.
    - The heuristic is non-admissible but aims for accuracy and efficient computability.
    - Movement/traversal facts are not explicitly counted here; we approximate costs 
      mostly by objects needing processing and "full" modes left.

    Heuristic Initialization
    ------------------------
    - Extract statically known sets of:
      * Modes that support analysis (supports ?mode).
      * Cameras that have communicated soil data.
      * Objectives that have communicated rock data.
      * Modes initially equipped for soil analysis or marked as full.
      * The mapping of modes to rovers, useful for validation (not used heavily here).
    - Store these sets for quick membership checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the set of modes mentioned in the static information that support analysis.
    2. Count how many modes are currently full (indicating completion).
    3. Find how many modes are currently equipped for soil analysis.
    4. Count how many cameras still have images (must be processed).
    5. Count how many objectives are still empty (waiting to be filled).
    6. Count how many waypoints still have soil analysis information.
    7. The minimum number of actions required will at least include:
       - Communicating and processing images for all cameras that have images.
       - Filling all empty objectives by communication/sample actions.
       - Calibrating or communicating to full all modes that are equipped (not full).
       - Taking images at visible waypoints.
    8. Costs estimation outline:
       - Each "have_image camera" likely corresponds to at least two actions: sampling or communicating image data.
       - Each empty objective will require at least one action to fill.
       - Each mode equipped for soil analysis but not full needs calibration/communication action.
       - Each visible waypoint likely needs an image taken to trigger further data collection.
    9. Combine these estimates into a weighted sum that approximates the actions left, 
       taking care not to underestimate, but keeping the calculation efficient.
    10. If all modes are full, heuristic returns 0.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting relevant static information.

        Arguments:
        - task: The planning task object, containing static facts and goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract modes that support analysis
        self.support_modes = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "supports", "*")
        }

        # Extract modes that have communicated soil data (static knowledge; might be none)
        self.comm_soil_data_cameras = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "communicated_soil_data", "*")
        }

        # Extract objectives with communicated rock data
        self.comm_rock_data_objectives = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "communicated_rock_data", "*")
        }

        # Modes initially at a location / assigned to a rover (optional, not heavily used)
        self.mode_to_rover = {}
        for fact in static_facts:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                self.mode_to_rover[parts[1]] = parts[2]

    def __call__(self, node):
        """
        Calculate the heuristic estimate for the given search node.

        Arguments:
        - node: The search node with a current state attribute (set of facts).

        Returns:
        - An integer heuristic estimate of the minimal number of actions remaining
          to achieve the goal from the node's state.
        """
        state = node.state

        # Goal check: if all modes required to be full are full, heuristic is zero.
        # We check the goal "full mode" predicates presence in state.
        goal_full_modes = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "full", "*")
        }
        current_full_modes = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "full", "*")
        }
        if goal_full_modes <= current_full_modes:
            return 0

        # Count how many modes still need to be filled according to static-supported modes and goals
        modes_to_fill = goal_full_modes - current_full_modes
        num_modes_to_fill = len(modes_to_fill)

        # Modes currently equipped for soil analysis but not yet full
        modes_equipped_soil = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "equipped_for_soil_analysis", "*")
        }
        # Modes currently full (already counted)
        modes_full = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "full", "*")
        }

        # Cameras that still "have_image" (must be processed)
        cameras_with_images = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "have_image", "*")
        }

        # Objectives that are still empty (need to be filled by sample_rock/communicate)
        empty_objectives = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "empty", "*")
        }

        # Waypoints that currently have soil analysis - necessary for communicating data?
        waypoints_with_soil_analysis = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "have_soil_analysis", "*")
        }

        # Visible waypoints (may require images to be taken)
        visible_waypoints = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "visible", "*")
        }

        # Visible_from waypoints (required for communication actions)
        visible_from_waypoints = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "visible_from", "*")
        }

        # Weighted estimations of steps to finish tasks:

        # 1. Cameras with images: each requires at least 2 actions: 'sample_rock' or 'communicate_image_data' to process image and fill objectives
        #    (approximate slightly: 2 actions per camera with images)
        cam_image_cost = 2 * len(cameras_with_images)

        # 2. Empty objectives: each requires at least 1 action to fill (via sampling or communication)
        empty_obj_cost = len(empty_objectives)

        # 3. Modes equipped for soil analysis need calibration/communication; assume 2 actions each (calibration and communication)
        mode_soil_eq_cost = 2 * len(modes_equipped_soil - modes_full)

        # 4. Waypoints that have soil_analysis but need to be emptied by communication, count roughly 1 action per such waypoint
        soil_analysis_wp_cost = len(waypoints_with_soil_analysis)

        # 5. Visible waypoints require images to be taken (take_image action) before further sampling or communication
        visible_wp_cost = len(visible_waypoints)

        # 6. Modes yet to be full beyond equipped soil analysis modes estimated as 1 action overhead each (final full state)
        modes_left_cost = max(0, num_modes_to_fill - len(modes_equipped_soil))

        # Aggregate estimation
        total_estimated_actions = (
            cam_image_cost
            + empty_obj_cost
            + mode_soil_eq_cost
            + soil_analysis_wp_cost
            + visible_wp_cost
            + modes_left_cost
        )

        # Ensure heuristic is never zero unless goal reached
        if total_estimated_actions == 0:
            total_estimated_actions = 1

        return total_estimated_actions
