from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.

    Example:
    '(have_image camera1)' -> ['have_image', 'camera1']
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain 'rover-2'.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state, 
    which specifies certain modes to be 'full'. It guides a greedy best-first search by 
    evaluating the current state's progress towards having all required modes full.

    Assumptions:
    - The goal is always a conjunction of facts of the form (full ?mode).
    - Each mode must be filled by communicating soil or rock data through a chain of actions.
    - Actions that lead to fulfilling the mode's 'full' predicate require certain preconditions,
      involving equipped soil/rock analysis, visible waypoints, rover and lander connectivity.
    - The heuristic estimates the minimal number of mode completions remaining.
    - The heuristic ignores rover movements and communication intricacies beyond mode fullfilling.
    - It assumes there are no unreachable 'full' modes given correct preconditions.

    Heuristic Initialization:
    - Extract all modes that must be full from the goal conditions.
    - Index static facts to quickly know:
      - which modes are supported,
      - which modes are currently equipped for soil analysis,
      - which cameras have communicated soil data,
      - which objectives have communicated rock data,
      - the rovers each mode is at,
      - landers that can be traversed by rovers,
      - waypoints visible and those equipped for rock analysis with which landers,
      - visibility information necessary for completing communication actions.
      
    Step-By-Step Thinking for Computing Heuristic:
    1. Count how many modes required by the goal are not yet full in the current state.
       Since the goal is conjunction of (full ?mode), any missing 'full' mode increases heuristic.
    2. For each mode not full:
       - Check if mode is currently equipped for soil analysis.
       - Check if mode is supported (precondition for calibration or communicate_soil_data).
       - Check if mode is at a rover that can traverse a lander (required for some actions).
    3. Analyze the possible bottleneck steps left to achieve 'full ?mode':
       - Based on domain actions, a mode becomes full through `communicate_soil_data` or `calibrate`.
       - These require the mode to be equipped for soil analysis, equipped for rock analysis for some waypoint/lander, mode at a rover that can traverse that lander, and that lander-rover pair.
    4. The heuristic sums at least 1 action per mode not full to reach fullness.
       If mode is not equipped for soil analysis yet, it adds another step as a rough estimate.
    5. Additionally add cost estimates:
       - Presence of soil analysis on waypoints or images affects minimal number of needed sampling or communication actions.
       - Lack of images or empty objectives may point to additional actions needed.
    6. Finally, return the sum of these minimal action counts as heuristic.

    This heuristic is not admissible but efficiently computable and informative to reduce expanded nodes.
    """

    def __init__(self, task):
        # Extract the modes required to be full from the goal facts.
        self.goal_modes = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "full":
                self.goal_modes.add(parts[1])

        # Index static info:
        self.supports = set()
        self.equipped_for_soil_analysis = set()
        self.communicated_soil_data_cameras = set()
        self.communicated_rock_data_objectives = set()
        self.at_mode_rover = dict()  # mode -> rover
        self.can_traverse_pairs = set()  # (lander, rover)
        self.equipped_for_rock_analysis = set()  # (waypoint, lander)
        self.visible_waypoints = set()
        self.visible_from_waypoints = set()

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "supports":
                self.supports.add(parts[1])
            elif pred == "equipped_for_soil_analysis":
                self.equipped_for_soil_analysis.add(parts[1])
            elif pred == "communicated_soil_data":
                self.communicated_soil_data_cameras.add(parts[1])
            elif pred == "communicated_rock_data":
                self.communicated_rock_data_objectives.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                # (at ?mode ?rover)
                mode, rover = parts[1], parts[2]
                self.at_mode_rover[mode] = rover
            elif pred == "can_traverse" and len(parts) == 3:
                # (can_traverse ?lander ?rover)
                lander, rover = parts[1], parts[2]
                self.can_traverse_pairs.add((lander, rover))
            elif pred == "equipped_for_rock_analysis" and len(parts) == 3:
                self.equipped_for_rock_analysis.add((parts[1], parts[2]))
            elif pred == "visible" and len(parts) == 2:
                self.visible_waypoints.add(parts[1])
            elif pred == "visible_from" and len(parts) == 2:
                self.visible_from_waypoints.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Helper to check if a fact is true in the current state
        def fact_in_state(predicate, *args):
            fact = "(" + " ".join([predicate] + list(args)) + ")"
            return fact in state

        # Count how many modes required to be full are not yet full
        not_full_modes = [m for m in self.goal_modes if not fact_in_state("full", m)]
        if not not_full_modes:
            return 0  # Goal reached, heuristic is zero.

        # Count modes currently equipped for soil analysis in state
        modes_equipped_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_soil_analysis" and len(parts) == 2:
                modes_equipped_in_state.add(parts[1])

        # Count equipped_for_rock_analysis (waypoint, lander) pairs currently true in state
        equipped_rock_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_rock_analysis" and len(parts) == 3:
                equipped_rock_in_state.add((parts[1], parts[2]))

        # Count cameras that currently have images
        cameras_with_images = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "have_image" and len(parts) == 2:
                cameras_with_images.add(parts[1])

        # Count objectives that are empty
        empty_objectives = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "empty" and len(parts) == 2:
                empty_objectives.add(parts[1])

        # Count waypoints with soil analysis in current state
        waypoints_soil_analysis = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "have_soil_analysis" and len(parts) == 2:
                waypoints_soil_analysis.add(parts[1])

        # Count visible waypoints in current state
        visible_waypoints_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "visible" and len(parts) == 2:
                visible_waypoints_in_state.add(parts[1])

        # Count visible_from waypoints in current state
        visible_from_waypoints_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "visible_from" and len(parts) == 2:
                visible_from_waypoints_in_state.add(parts[1])

        # Heuristic cost initialization
        cost = 0

        # For each mode not full, estimate cost to become full
        for mode in not_full_modes:
            # Base cost: 1 action to communicate soil data or calibrate to make mode full
            # (per domain actions: 'communicate_soil_data' or 'calibrate')

            cost_mode = 1

            # Check if mode is supported (needed for calibrate and communicate_soil_data)
            if mode not in self.supports:
                # If mode not supported (static), it likely can't be made full, but heuristic avoids infinity
                # Add a penalty to avoid underestimation
                cost_mode += 2

            # Check if currently equipped for soil analysis; if not, it requires at least 1 action to equip
            if mode not in modes_equipped_in_state:
                cost_mode += 1  # Need to equip mode for soil analysis first

            # Check if mode is at a rover that can traverse a lander connected to the mode's rock analysis waypoint
            # This is a key precondition for communicating soil data or calibrating

            # Get rover for mode, from static info:
            rover = self.at_mode_rover.get(mode)
            has_valid_traversal = False

            # We try to find a (waypoint, lander) equipped for rock analysis that is possible to use
            # i.e., lander can be traversed by rover
            for (wp, lander) in self.equipped_for_rock_analysis:
                if (lander, rover) in self.can_traverse_pairs:
                    # Also check that waypoint is visible_from (required by communicate_soil_data)
                    if wp in visible_from_waypoints_in_state:
                        has_valid_traversal = True
                        break

            if not has_valid_traversal:
                # If no valid traversal found, add estimated cost for making this possible
                # We roughly add cost 2 for travel and setup actions (could be sampling or take_image)
                cost_mode += 2

            # Additional heuristic hints:
            # If there are empty objectives in the problem, likely more sampling needed before communication
            if not empty_objectives:
                # If no empty objective remains, means sampling is done, so reduce cost for sampling
                cost_mode = max(1, cost_mode - 1)

            # If no cameras currently have images, more take_image or sample_rock actions needed
            if not cameras_with_images:
                cost_mode += 1

            # If mode is supported and equipped for soil analysis, and no visible waypoints remain,
            # additional actions to get visibility may be needed
            if mode in self.supports and mode in self.equipped_for_soil_analysis:
                # Check if any visible waypoints
                if not visible_waypoints_in_state:
                    cost_mode += 1

            cost += cost_mode

        return cost
