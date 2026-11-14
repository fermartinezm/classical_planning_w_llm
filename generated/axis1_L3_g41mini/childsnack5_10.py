from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (rover-2) domain.

    Summary:
    This heuristic estimates the minimum number of actions needed to achieve the goal, which
    consists of ensuring all specified "mode" objects become "full". The core tasks typically
    involve preparing modes (which can be soil or rock analysis equipment), collecting images,
    sampling soil/rocks, calibrating or communicating data so that modes become "full".

    Assumptions:
    - The goal is to have all specified modes "full".
    - Filling a mode requires actions that transform the mode state through specific steps,
      involving images, equipped states, soil and rock analysis data, and rover/lander interactions.
    - Modes can be equipped for soil analysis or rock analysis, and must be "at" a rover.
    - Lander-rover traversal and visibility conditions affect action applicability.
    - The heuristic is not necessarily admissible but seeks to estimate the minimal number of
      required actions robustly and efficiently.

    Heuristic Initialization:
    - Extract from static facts:
        * Modes that support soil analysis.
        * Which modes are initially equipped for soil analysis.
        * Mapping of modes to rovers.
        * Which landers can be traversed by rover1 (the constant rover).
        * Which cameras have communicated soil data.
        * Which objectives have communicated rock data.
    - Extract goal modes (those that must become "full").

    Step-By-Step Thinking for Computing Heuristic:
    1. If the state is a goal state (all goal modes are "full"), heuristic is 0.
    2. For each goal mode not full, estimate steps to fill it:
       - If mode is "full", no cost.
       - Otherwise:
         a) To make a mode "full", one of the following happens by effects of either
            calibrate or communicate_soil_data:
              - Preconditions require:
                * mode supports soil analysis,
                * mode is equipped for soil or rock analysis,
                * mode is at the rover,
                * visibility or equipped_for_rock_analysis at waypoints & landers,
                * rover can traverse lander.
              - Modes equipped for soil analysis do not need take_image or sample_rock actions,
                but may still require communication.
         b) Estimate costs for:
            - Equipping the mode for rock analysis via taking images at visible waypoints if not equipped.
            - Sampling rock (which requires having images and empty objective) if needed.
            - Communicating soil or rock data (which requires have_image or equipped states).
            - Calibrating if supported.
         c) Count the minimal number of actions logically needed to reach "full" based on the current state facts,
            focusing on whether mode is equipped, images are present, objectives empty/full, soil analysis done,
            and communication steps.
    3. Use static and current state facts to estimate counts of:
       - Images to take.
       - Modes requiring calibration or communication.
       - Sampling or communicating soil/rock data steps.
    4. Sum these estimated minimal actions weighted by typical action counts.
    5. Return the sum as heuristic estimate.

    This heuristic balances accuracy by relating key predicates and goals without fully simulating action sequences,
    ensuring efficient computation to guide greedy best-first search.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract modes that must be full (those appearing in goal as (full modeX))
        self.goal_modes = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "full":
                self.goal_modes.add(parts[1])

        # From static facts, find modes that support soil analysis
        self.modes_support_soil = {get_parts(fact)[1] for fact in self.static if fact.startswith("(supports")}
        # From static, modes initially equipped for soil analysis
        self.modes_equipped_soil = {get_parts(fact)[1] for fact in self.static if fact.startswith("(equipped_for_soil_analysis")}
        # Map modes to rovers (from (at mode rover))
        self.mode_at_rover = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1].startswith("mode"):
                self.mode_at_rover[parts[1]] = parts[2]
        # Landers that rover1 can traverse
        self.landers_rover1_can_traverse = {get_parts(fact)[1] for fact in self.static if fact.startswith("(can_traverse") and get_parts(fact)[2] == "rover1"}
        # Cameras that have communicated soil data
        self.communicated_soil_cameras = {get_parts(fact)[1] for fact in self.static if fact.startswith("(communicated_soil_data")}
        # Objectives that have communicated rock data
        self.communicated_rock_objectives = {get_parts(fact)[1] for fact in self.static if fact.startswith("(communicated_rock_data")}

    def __call__(self, node):
        state = node.state

        def fact_exists(p):
            return p in state

        def exists_pred(predicate, arg1=None, arg2=None, arg3=None):
            # Checks if any fact in state matches the predicate and given arguments (None are wildcards)
            for fact in state:
                parts = get_parts(fact)
                if parts[0] != predicate:
                    continue
                if arg1 is not None and parts[1] != arg1:
                    continue
                if arg2 is not None and (len(parts) < 3 or parts[2] != arg2):
                    continue
                if arg3 is not None and (len(parts) < 4 or parts[3] != arg3):
                    continue
                return True
            return False

        # Check if goal reached: all goal modes are full
        if all(fact_exists(f"(full {mode})") for mode in self.goal_modes):
            return 0

        # Helper sets from current state
        # Identify full modes (maybe beyond goals)
        full_modes = {get_parts(fact)[1] for fact in state if fact.startswith("(full")}
        # Modes equipped for rock analysis (pairs: mode, lander)
        equipped_rock = {(parts[1], parts[2]) for fact in state if fact.startswith("(equipped_for_rock_analysis") for parts in [get_parts(fact)]}
        # Modes equipped for soil analysis
        equipped_soil = {get_parts(fact)[1] for fact in state if fact.startswith("(equipped_for_soil_analysis")}
        # Which objectives are empty
        empty_objectives = {get_parts(fact)[1] for fact in state if fact.startswith("(empty")}
        # Which objectives are not empty (i.e. implicitly full)
        full_objectives = set()
        # We don't have explicit fact "not empty", but "empty" is false if missing. We infer this indirectly.
        # Alternatively, check communicated_rock_data from static + from state decrements "empty".
        # Approximate: objectives mentioned in static "communicated_rock_data" or not in empty_objectives.
        # So, use empty_objectives to know emptiness, otherwise assume full

        # Cameras that currently have images
        have_images = {get_parts(fact)[1] for fact in state if fact.startswith("(have_image")}
        # Cameras that communicated soil data (from static and state)
        communicated_soil_cameras = self.communicated_soil_cameras | {get_parts(fact)[1] for fact in state if fact.startswith("(communicated_soil_data")}
        # Objectives that communicated rock data (static + state)
        communicated_rock_objectives = self.communicated_rock_objectives | {get_parts(fact)[1] for fact in state if fact.startswith("(communicated_rock_data")}
        # Waypoints that have soil analysis
        have_soil_analysis_wps = {get_parts(fact)[1] for fact in state if fact.startswith("(have_soil_analysis")}
        # Waypoints visible (currently)
        visible_wps = {get_parts(fact)[1] for fact in state if fact.startswith("(visible")}
        # Waypoints visible_from (currently)
        visible_from_wps = {get_parts(fact)[1] for fact in state if fact.startswith("(visible_from")}
        # Landers rover1 can traverse (from static)
        can_traverse_landers = self.landers_rover1_can_traverse

        # Modes that support (from static)
        modes_support_soil = self.modes_support_soil

        # Now estimate cost for each goal mode that is not full
        cost = 0
        for mode in self.goal_modes:
            if mode in full_modes:
                continue

            # Steps to fill a mode roughly:
            # 1) Mode must be equipped for rock analysis with lander (equipped_for_rock_analysis (mode l))
            #    or equipped_for_soil_analysis (mode)
            # 2) Then must calibrate or communicate_soil_data (effects: full mode)
            # 3) To equip for rock analysis: take_image at a visible waypoint + rover can traverse lander
            # 4) For sampling rock or communicate image: have_image on camera, empty objective, have_soil_analysis waypoint, etc.
            # 5) May need to sample soil or communicate soil data if not already communicated

            mode_full = fact_exists(f"(full {mode})")
            if mode_full:
                continue

            # Check if mode supports soil analysis
            supports_soil = mode in modes_support_soil
            # Is mode equipped for soil analysis?
            is_equipped_soil = mode in equipped_soil or mode in self.modes_equipped_soil
            # Is mode equipped for rock analysis? Check pairs (mode, lander) in equipped_rock:
            # If equipped for rock analysis with any lander, treat as equipped_rock = True
            is_equipped_rock = any(m == mode for (m, l) in equipped_rock)

            # The rover at which this mode is located (from static)
            rover_of_mode = self.mode_at_rover.get(mode, None)

            # Costs start low. We'll add discrete action counts heuristically.
            local_cost = 0

            # Step 1. If mode is not equipped (neither soil nor rock), we must either:
            # - equip for soil analysis (already equipped_soil-like)
            # - or equip for rock analysis by taking images at visible waypoints combined with lander traversal
            # If neither equipped, must perform take_image action to equip rock analysis.

            if not (is_equipped_soil or is_equipped_rock):
                # Find visible waypoint that can equip mode for rock analysis if rover can traverse lander
                # If no visible waypoint or lander accessible, count 2 actions at least: take_image + movement/traversal
                if visible_wps and can_traverse_landers:
                    # Minimal cost: one take_image action
                    local_cost += 1
                else:
                    # If no visible waypoint or lander traversal, assume at least 2 actions.
                    local_cost += 2  # Conservative cost

            # Step 2. If mode supports soil and is not equipped for soil analysis,
            # probably must equip it first (1 action).
            if supports_soil and not is_equipped_soil:
                local_cost += 1  # One action to equip soil analysis mode (approximation)

            # Step 3. Sampling rock requires:
            # - having images on cameras
            # - empty objectives
            # - have_soil_analysis on waypoints as preconditions for some actions

            # Check if there exists some empty objective not communicated rock data
            num_empty_objectives = len(empty_objectives - communicated_rock_objectives)
            # Check if any camera has image that can be used to sample rock
            num_have_image_cams = len(have_images - communicated_soil_cameras)
            # Check have_soil_analysis waypoints count
            num_have_soil_analysis_wp = len(have_soil_analysis_wps)

            # If mode supports soil analysis and is not full, might need to sample soil and communicate soil data
            if supports_soil and mode not in full_modes:
                # If mode not equipped for soil analysis, assumed cost above
                # Otherwise, need to communicate soil data (1 action)
                local_cost += 1

            # Communicating image data requires have_image camera, empty objective, communicated soil and rock data, have_soil_analysis waypoint
            # If cameras have images, empty objectives exist, and soil data communicated, might require communication action
            if num_have_image_cams > 0 and num_empty_objectives > 0:
                local_cost += 1  # For communicating image data or sampling rock

            # Calibrate action also makes mode full if supports soil analysis and equipped for rock analysis
            # If mode supports soil and equipped for rock analysis
            if supports_soil and is_equipped_rock:
                local_cost += 1  # Calibrate action needed

            # If mode does not support soil but is not full, then communicate_soil_data cannot apply.
            # Might just need communicate_soil_data with equipped states
            if not supports_soil and is_equipped_rock:
                local_cost += 1  # Estimated communication or calibration

            cost += local_cost

        # To ensure finite heuristic, if cost zero but goal not reached, assign minimal positive estimate
        if cost == 0:
            cost = 1

        return cost
