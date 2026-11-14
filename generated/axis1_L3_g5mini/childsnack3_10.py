from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'rover-2' (secret_domain) PDDL domain.
    The primary planning goals in this domain are typically of the form (full <mode>).
    This heuristic estimates how many actions are still required to make each goal
    mode 'full'. It is intentionally optimistic but domain-aware: it inspects which
    supporting static facts are present (e.g. whether a mode supports calibration or
    is already equipped for soil analysis) and which current state facts exist
    (e.g. equipped_for_rock_analysis, visible, visible_from, can_traverse). It then
    estimates a small number of actions required to establish any missing preconditions
    and execute the one action that produces (full <mode>).

    Assumptions
    -----------
    - Goals are predominantly (full <mode>) facts. If other goals exist they are ignored
      by the domain-specific parts of the heuristic, but the heuristic still returns a
      finite integer (sum of estimates for identified mode goals).
    - The static information about modes (supports, equipped_for_soil_analysis, at <mode>
      <rover>) is available in task.static. Many 'at' facts for modes are static in the
      typical encoding; if an 'at' fact is missing entirely the heuristic gives a
      conservative (higher) estimate for that mode.
    - The heuristic is not required to be admissible. It aims to be informative and
      cheaply computable to guide greedy best-first search (minimize node expansion).
    - Some domain dynamics (which specific waypoint/lander pair will be used) are
      abstracted/aggregated: the heuristic reasons about existence of helpful fact
      patterns rather than exact object-level sequences. This yields a small integer
      estimate per mode rather than an exact plan length.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts and stores:
    - The set of goal facts (self.goals).
    - Static facts (self.static) and convenient derived indices:
      * supports_modes: set of modes that have (supports modeX) in static or initial facts.
      * equipped_for_soil_modes: set of modes that are already equipped for soil analysis
        (could allow using communicate_soil_data).
      * mode_to_rover: mapping mode -> rover from any (at mode rover) fact found either
        in static or in the initial state (the heuristic uses static+state at evaluation
        time, but having static knowledge up-front is useful).
    These are used to quickly decide which final action (calibrate or communicate_soil_data)
    is likely to be applicable for each target mode.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given node/state the heuristic works as follows (high-level):
    1. If the state's facts already satisfy all goals, return 0.
    2. Combine the current state's facts with static facts (static facts are treated as
       always available for precondition checks where appropriate). Build quick indexes:
       - visible_waypoints, visible_from_waypoints
       - equipped_for_rock_analysis pairs (waypoint, lander)
       - can_traverse pairs (lander, rover)
       - presence of have_image / have_soil_analysis / communicated_* etc (for conservative
         visibility estimates)
    3. For every goal mode that is not yet (full) in the current state:
       - Determine whether the mode supports calibration (supports mode) or is already
         equipped for soil analysis (equipped_for_soil_analysis mode). Both give a
         possible final action that yields (full mode): calibrate (if supports) or
         communicate_soil_data (if equipped_for_soil_analysis).
       - Compute an optimistic yet domain-informed estimate for the minimal number of
         actions needed for the chosen final action, taking into account:
         * whether there already exists an equipped_for_rock_analysis (waypoint, lander)
           that is usable (i.e., the lander can_traverse to the rover associated with
           the mode). If such a pair exists, often only the final action is needed (cost 1).
         * whether a visible waypoint exists (visible) and whether some lander can
           currently traverse rover1 (this is important because take_image requires
           can_traverse <lander> rover1). If so, an equipped_for_rock_analysis can often
           be created with a single take_image and then the final action executed.
         * if mappings must be changed (sample_soil can change which rover a lander can
           traverse), account for one extra action per needed mapping change (sample_soil).
         * obtaining visible_from (required for communicate_soil_data) is more demanding
           than obtaining visible: visible_from is typically produced by a
           communicate_image_data action which has additional preconditions, so we
           conservatively charge additional cost when visible_from is missing.
       - If both calibrate and communicate_soil_data are possible (supports and
         equipped_for_soil_analysis both hold), take the cheaper estimate.
       - If neither calibration nor communication path is available/recognizable from
         static facts, assign a conservative (but finite) penalty for that mode.
    4. Sum the per-mode estimates and return that integer as the heuristic value.

    Notes about numeric choices
    ---------------------------
    The numeric costs used by this heuristic are small integers (1..5) representing
    short action sequences. They are tuned to reflect typical small-step recipes in the
    domain:
    - 1: final action already applicable (calibrate or communicate_soil_data).
    - 2..3: one or two supporting actions needed (e.g., take_image, optional sample_soil)
           before the final action.
    - 4..5: more complex cases requiring both visibility creation and mapping changes.
    These values are intentionally heuristic (not exact) but usually informative because
    they correlate with the number of small actions that must be executed per mode.
    """

    def __init__(self, task):
        # Goals (set of fact strings)
        self.goals = task.goals
        # Static facts (frozenset of strings)
        self.static = set(task.static)

        # Precompute static-derived indices that are helpful across heuristic calls.
        self.supports_modes = set()
        self.equipped_for_soil_modes = set()
        self.static_mode_to_rover = {}  # mode -> rover (from static facts)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "supports" and len(parts) >= 2:
                self.supports_modes.add(parts[1])
            elif parts[0] == "equipped_for_soil_analysis" and len(parts) >= 2:
                self.equipped_for_soil_modes.add(parts[1])
            elif parts[0] == "at" and len(parts) >= 3:
                # (at <mode> <rover>)
                self.static_mode_to_rover[parts[1]] = parts[2]

    def __call__(self, node):
        state = set(node.state)
        # If already a goal state return 0
        if self.goals <= state:
            return 0

        # Combined facts (treat static facts as always present for applicability checks)
        combined = state | self.static

        # Index dynamic facts
        visible_waypoints = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "visible"}
        visible_from_waypoints = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "visible_from"}
        equipped_for_rock = {(get_parts(f)[1], get_parts(f)[2]) for f in combined if get_parts(f)[0] == "equipped_for_rock_analysis"}
        can_traverse_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in combined if get_parts(f)[0] == "can_traverse"}
        # gather modes that are already full in current state
        full_modes_in_state = {get_parts(f)[1] for f in state if get_parts(f)[0] == "full"}

        # Build current at-mode -> rover mapping (prefer state over static if both present)
        mode_to_rover = dict(self.static_mode_to_rover)  # start with static
        for f in combined:
            parts = get_parts(f)
            if parts[0] == "at" and len(parts) >= 3:
                # Overwrite with combined facts (state or static)
                mode_to_rover[parts[1]] = parts[2]

        # Helper queries
        def has_equipped_for_rock_for_rover(rover):
            """Return True if there exists (equipped_for_rock_analysis wp lander) and (can_traverse lander rover)."""
            for (wp, lander) in equipped_for_rock:
                if (lander, rover) in can_traverse_pairs:
                    return True
            return False

        def has_any_equipped_for_rock():
            return len(equipped_for_rock) > 0

        def has_can_traverse_to_rover1():
            # rover1 is a domain constant for take_image precondition
            return any(r == "rover1" for (_, r) in can_traverse_pairs)

        def has_can_traverse_to_target(target_rover):
            return any(r == target_rover for (_, r) in can_traverse_pairs)

        # Gather goal modes
        goal_modes = [get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "full"]

        total_cost = 0
        for mode in goal_modes:
            # Skip modes already full
            if mode in full_modes_in_state:
                continue

            # Identify which rover is associated with this mode (if known)
            target_rover = mode_to_rover.get(mode, None)

            # Determine if mode supports calibration or is equipped for soil analysis (static or combined)
            supports = (mode in self.supports_modes) or any(get_parts(f)[0] == "supports" and get_parts(f)[1] == mode for f in combined)
            equipped_soil = (mode in self.equipped_for_soil_modes) or any(get_parts(f)[0] == "equipped_for_soil_analysis" and get_parts(f)[1] == mode for f in combined)

            estimates = []

            # 1) Consider calibrate path if mode supports calibration.
            if supports:
                # If target_rover unknown, be conservative but finite
                if target_rover is None:
                    # Assign a moderate penalty because 'at mode rover' is required for calibrate and missing.
                    estimates.append(4)
                else:
                    if has_equipped_for_rock_for_rover(target_rover):
                        # Already have (equipped_for_rock_analysis wp L) and can_traverse L target_rover => 1 action (calibrate)
                        estimates.append(1)
                    elif has_any_equipped_for_rock() and has_can_traverse_to_target(target_rover):
                        # There is an equipped_for_rock but its lander may not be mapped to the required rover;
                        # sample_soil can change mapping (1) then calibrate (1) => 2
                        estimates.append(2)
                    elif visible_waypoints and has_can_traverse_to_rover1():
                        # We can take_image (1) to create an equipped_for_rock for a lander that can traverse rover1.
                        # If there is already a lander that can traverse target_rover, assume (optimistically) we can use the same lander
                        # else sample_soil is needed to change mapping once.
                        if has_can_traverse_to_target(target_rover):
                            estimates.append(2)  # take_image + calibrate
                        else:
                            estimates.append(3)  # take_image + sample_soil + calibrate
                    else:
                        # Lacking easy visible/lander mappings, assume more work is needed (but finite)
                        estimates.append(4)

            # 2) Consider communicate_soil_data path if the mode is equipped_for_soil_analysis.
            if equipped_soil:
                # This action requires:
                # - equipped_for_rock_analysis wp L with can_traverse L target_rover
                # - visible_from wp
                if target_rover is None:
                    estimates.append(5)
                else:
                    eq_cost = 0
                    vis_from_cost = 0

                    # cost to get an equipped_for_rock_analysis that is usable for target rover
                    if has_equipped_for_rock_for_rover(target_rover):
                        eq_cost = 0
                    elif has_any_equipped_for_rock() and has_can_traverse_to_target(target_rover):
                        # change mapping then use it
                        eq_cost = 1  # sample_soil
                    elif visible_waypoints and has_can_traverse_to_rover1():
                        # take_image to produce equipped_for_rock on a lander tied to rover1
                        # possibly sample_soil to remap to target_rover
                        if has_can_traverse_to_target(target_rover):
                            eq_cost = 1  # take_image then direct communicate: 1 (take_image) plus final action counted later
                        else:
                            eq_cost = 2  # take_image + sample_soil
                    else:
                        eq_cost = 3  # conservative fallback to create required equipment

                    # cost to get visible_from for some waypoint
                    if visible_from_waypoints:
                        vis_from_cost = 0
                    elif visible_waypoints:
                        # visible exists but visible_from does not; producing visible_from is more expensive (communicate_image_data)
                        vis_from_cost = 2
                    else:
                        # Neither visible nor visible_from => need to create visibility first (sample_rock or communicate_image_data)
                        vis_from_cost = 3

                    # final communicate_soil_data action costs 1
                    estimates.append(1 + eq_cost + vis_from_cost)

            # If neither path is applicable according to static information, assign a finite conservative cost
            if not estimates:
                # Mode cannot be completed by obvious static paths; choose conservative but finite penalty
                estimates.append(5)

            # Take the most optimistic estimate among considered paths
            mode_cost = min(estimates)
            # Ensure non-zero and finite
            if mode_cost < 0:
                mode_cost = 0
            total_cost += int(mode_cost)

        return total_cost