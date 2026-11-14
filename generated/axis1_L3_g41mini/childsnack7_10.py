from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(have_image camera1)".
    - `args`: The expected pattern parts (wildcards '*' allowed).
    - Returns True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (rover-2) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state,
    which is the condition that some mode(s) are "full". Actions revolve around
    preparing modes to become full by supplementing rock and soil analysis through
    sequences of taking images, sampling rock or soil, communicating data, and calibrating.

    Assumptions
    -----------
    - The primary goal is to have certain mode objects in state "full".
    - The main way to make modes full is by communicating soil or rock data after
      preparing sample analysis through supported steps.
    - Key requirements include having images, empty objectives, soil analyses at waypoints,
      and equipped states for rock analyses.
    - "can_traverse" facts indicate accessibility required for many actions.
    - The heuristic does not consider complex movement cost explicitly but
      roughly estimates remaining steps based on unfulfilled subgoals.
    - The heuristic is not admissible but favors guiding the search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract the goal mode(s) that must become "full".
    - Extract static facts like:
      * Modes that support analysis.
      * Which modes are currently equipped for soil analysis.
      * Which landers can be traversed by rover1.
      * Which modes are currently at which rover.
      * Communications already done concerning soil and rock data.
    - This information informs how close the mode is to becoming full and what steps remain.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies the goal (all required modes full), return 0.
    2. For each mode that should be full (from the goal):
       a. If it is already full, no cost added.
       b. Otherwise:
          i. Check if the mode supports soil analysis and if it is currently equipped for soil analysis.
          ii. Count actions needed to prepare soil analysis if not equipped.
              This might include:
              - Taking images at visible waypoints.
              - Sampling soil or rocks if images are available.
          iii. Count communication actions to communicate soil and rock data for that mode:
              - This depends on having images and the empty objective facts.
          iv. Count calibration or communication actions to finalize the mode full state.
    3. Estimate number of "empty objective" facts that still need to be filled by sampling or communication.
    4. Count the number of remaining soil analyses to complete.
    5. Consider reachable "can_traverse" constraints as prerequisites to actions.
    6. Aggregate these partial counts into a heuristic value estimating the minimum sequence of steps needed to fulfill mode fullness.
    7. Return this aggregated estimate.

    This heuristic balances accuracy and computation cost by counting missing essential conditions
    and actions to reach the goal without enumerating all possible paths or movements.
    """

    def __init__(self, task):
        self.goals = task.goals   # Goal facts (e.g., (full mode1))

        # Extract all modes that must become full as per the goal.
        self.goal_modes = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "full":
                self.goal_modes.add(parts[1])

        # Extract static facts to facilitate heuristic calculations.
        self.supports_modes = set()
        self.equipped_for_soil_analysis = set()
        self.communicated_soil_cameras = set()
        self.communicated_rock_objectives = set()
        self.modes_at_rover = dict()  # mode -> rover
        self.can_traverse_pairs = set()  # pairs of (lander, rover)

        for fact in task.static:
            parts = get_parts(fact)

            if parts[0] == "supports":
                # (supports modeX)
                self.supports_modes.add(parts[1])

            elif parts[0] == "equipped_for_soil_analysis":
                # (equipped_for_soil_analysis modeX)
                self.equipped_for_soil_analysis.add(parts[1])

            elif parts[0] == "communicated_soil_data":
                # (communicated_soil_data cameraX)
                self.communicated_soil_cameras.add(parts[1])

            elif parts[0] == "communicated_rock_data":
                # (communicated_rock_data objectiveX)
                self.communicated_rock_objectives.add(parts[1])

            elif parts[0] == "at":
                # (at modeX roverY)
                if len(parts) == 3 and parts[1].startswith("mode"):
                    self.modes_at_rover[parts[1]] = parts[2]

            elif parts[0] == "can_traverse":
                # (can_traverse landerX roverY)
                if len(parts) == 3:
                    self.can_traverse_pairs.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Quickly check if goal is achieved, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract relevant facts from current state.
        full_modes = set()
        empty_objectives = set()
        have_images = set()
        have_soil_analyses = set()
        visible_waypoints = set()
        equipped_for_rock = set()  # (waypoint, lander) pairs
        visible_from = set()

        equipped_for_soil_analysis_modes_now = set()
        at_mode_rover_now = dict()
        can_traverse_now = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "full":
                full_modes.add(parts[1])
            elif parts[0] == "empty":
                empty_objectives.add(parts[1])
            elif parts[0] == "have_image":
                have_images.add(parts[1])
            elif parts[0] == "have_soil_analysis":
                have_soil_analyses.add(parts[1])
            elif parts[0] == "visible":
                visible_waypoints.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                # pair of (waypoint, lander)
                if len(parts) == 3:
                    equipped_for_rock.add((parts[1], parts[2]))
            elif parts[0] == "visible_from":
                visible_from.add(parts[1])
            elif parts[0] == "equipped_for_soil_analysis":
                equipped_for_soil_analysis_modes_now.add(parts[1])
            elif parts[0] == "at":
                if len(parts) == 3 and parts[1].startswith("mode"):
                    at_mode_rover_now[parts[1]] = parts[2]
            elif parts[0] == "can_traverse":
                if len(parts) == 3:
                    can_traverse_now.add((parts[1], parts[2]))

        # Estimate actions needed to reach goal modes full.
        heuristic_cost = 0

        # For each goal mode, estimate how many actions are left.
        for mode in self.goal_modes:
            # If already full in current state, no cost for this mode.
            if mode in full_modes:
                continue

            # Otherwise, estimate steps to get mode full.

            # 1. If mode does not support soil analysis, it cannot become full: no cost added;
            #    but since mode supporting is a static fact, we assume goal only contains supported modes.
            if mode not in self.supports_modes:
                # Possibly unsolvable or no plan to reach goal for this mode.
                # But to ensure finite heuristic, add a high cost here.
                heuristic_cost += 10
                continue

            # 2. We must reach a state where (full mode) holds,
            # which requires (full mode) added by actions calibrate or communicate_soil_data.

            # 3. Find whether mode is currently equipped_for_soil_analysis (a precondition for some actions).
            mode_equipped = mode in equipped_for_soil_analysis_modes_now

            # 4. Count remaining "empty" objectives that are needed to become "not empty" via sampling/communicating.
            # Use empty objectives that haven't been communicated rock data.
            # The goal likely requires empty objectives to be "not empty" eventually.
            # Count how many empty objectives exist (from state).
            num_empty_objectives = len(empty_objectives)

            # 5. Count cameras with images (have_image)
            num_have_images = len(have_images)

            # 6. Number of waypoints where soil analysis is done or can be done.
            num_soil_analyses = len(have_soil_analyses)

            # 7. Estimate required steps to prepare for mode full:
            #    - If mode not equipped for soil analysis, must take steps to equip it.
            #    - Actions include take_image, sample_soil, communicate_soil_data, calibrate.

            est_mode_cost = 0

            if not mode_equipped:
                # To equip mode for soil analysis we expect some actions:
                # - Take images at visible waypoints where can traverse.
                # - Possibly sample soil or rock, then communicate.
                # We count at least these actions as a base estimate.

                # Count how many waypoints are visible (for images)
                visible_waypoints_count = len(visible_waypoints)

                # Count number of landers that rover can traverse (essential to apply some actions)
                num_traversable = sum(1 for (lander, rover) in can_traverse_now if rover == "rover1")

                # Rough estimate: If no visible waypoints or no traversable landers, add penalty
                if visible_waypoints_count == 0 or num_traversable == 0:
                    est_mode_cost += 6  # Inaccessible or no visible points, cost to fix.

                # If mode supports and is not equipped for soil analysis, minimum steps to equip:
                # At least 2 steps: take_image + sample_soil/communicate or calibrate.
                est_mode_cost += 2

            # 8. Communicating soil and rock data is necessary to get mode full.
            # Check how many cameras are "have_image" and how many objectives are "empty".
            # Each such combination likely triggers 1-2 communication steps.
            # Add estimated costs for remaining communications:

            # Cameras needing to communicate (have_image and not communicated_soil_data yet)
            num_cameras_to_communicate = len(have_images - self.communicated_soil_cameras)

            # Objectives needing to communicate rock data (empty and not communicated_rock_data yet)
            num_objectives_to_communicate = len(empty_objectives - self.communicated_rock_objectives)

            # Estimate steps for communication: each requires at least 1 step, so sum counts.
            est_mode_cost += num_cameras_to_communicate + num_objectives_to_communicate

            # 9. Consider the number of soil analyses still not communicated or completed
            #    (those waypoints with have_soil_analysis but possibly not visible or visible_from)
            soil_analyses_to_process = len(have_soil_analyses - visible_from)

            est_mode_cost += soil_analyses_to_process

            # Add an estimated calibration / communication step to get mode full
            est_mode_cost += 1

            # 10. Aggregate estimated cost per mode
            heuristic_cost += est_mode_cost

        # Ensure heuristic is at least 1 if goal not reached (to avoid zero cost in non-goal)
        if heuristic_cost == 0:
            heuristic_cost = 1

        return heuristic_cost
