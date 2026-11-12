from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the whitespace-separated parts of a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `rover-2` (secret_domain) domain.
    The heuristic estimates the number of actions required to reach a state
    in which all goal facts "(full <mode>)" hold. It reasons about the two
    ways a mode can become `full` in the domain:
      - by applying `calibrate` (requires the mode to `support` and an
        `equipped_for_rock_analysis` fact for some (waypoint, lander) that
        can traverse the rover at which the mode is `at`), or
      - by applying `communicate_soil_data` (requires the mode to be
        `equipped_for_soil_analysis`, an `equipped_for_rock_analysis` for
        some (waypoint, lander) that can traverse the rover at which the
        mode is `at`, and `visible_from` for that waypoint).
    The heuristic tries to estimate the minimal number of actions to obtain
    the missing `equipped_for_rock_analysis` / `visible_from` / traversal
    alignment facts needed to make each mode `full`. It is intentionally
    not admissible but aims to be informative and cheap to compute.

    Assumptions
    -----------
    - Mode `full` facts are the only goals we care about.
    - `at` facts for modes (which indicate which rover a mode is at) are
      typically static or present in the union of static and current state.
    - If a lander can traverse some rover (can_traverse L R) in the state
      we assume sample_soil can switch that lander to another rover in one
      action (sample_soil toggles traversal). This is an approximation to
      keep the heuristic simple and finite.
    - If neither `supports` nor `equipped_for_soil_analysis` exists for a
      mode the heuristic returns a relatively large finite cost (we assume
      such modes are rare or impossible in practice).
    - The heuristic uses only facts present in the union of the current
      state and static facts; it does not attempt deep combinatorial
      reasoning about sequences of many intermediate actions.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - goal modes (the names of modes that must be `full`),
    - static facts about which modes `support` or are `equipped_for_soil_analysis`,
    - static or initial `at` facts tying modes to rovers (mode -> rover),
    - the set of static facts (kept for use in the estimate).
    These are stored in easy-to-query structures to make per-state cost
    computation cheap.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each mode M that is not yet `(full M)` in the current state:
      1. If M is already full in the state, contribution = 0.
      2. Determine the rover R where M is `at` (from `at` facts).
      3. Two primary paths to make M `full`:
         - Calibrate path (preferred if `supports M` exists):
           * Preconditions: `(supports M)`, `(at M R)`,
             an `(equipped_for_rock_analysis W L)`, and `(can_traverse L R)`.
           * If an aligned `(equipped_for_rock_analysis W L)` exists
             (same L that can traverse R), cost = 1 (calibrate).
           * Otherwise we try to produce an aligned `equipped_for_rock_analysis`
             by:
               - using `take_image` (needs a `visible W` and a lander L
                 that can traverse `rover1`) which costs 1 action, and
               - possibly using `sample_soil` once to switch the lander's
                 traversal to R (cost 1) if `(can_traverse L R)` is missing.
             The optimistic cost estimate for this path is therefore 2 or 3.
         - Communicate path (used if `equipped_for_soil_analysis M` exists):
           * Preconditions: `(equipped_for_soil_analysis M)`, `(at M R)`,
             an `(equipped_for_rock_analysis W L)`, `(visible_from W)`,
             and `(can_traverse L R)`.
           * If both an aligned `(equipped_for_rock_analysis W L)` and
             `(visible_from W)` are present, cost = 1 (communicate_soil_data).
           * Otherwise we add small penalties for producing the missing
             pieces (`take_image` to get `equipped_for_rock_analysis`,
             `sample_soil` to switch traversal, and a small heuristic
             penalty if `visible_from` is absent). This yields optimistic
             costs in the 2..4 range.
      4. If neither `supports` nor `equipped_for_soil_analysis` exists for M
         we return a conservative finite penalty (e.g., 10) for that mode.
      5. Sum the per-mode contributions to form the total heuristic value.

    The heuristic is cheap because it only inspects facts directly present
    in the state/static sets and performs a small combinatorial check per
    goal mode. It is not guaranteed to be admissible but provides an
    informative guide for greedy best-first search in this domain.
    """

    def __init__(self, task):
        # Goal full facts: store the mode names that must be full.
        self.goals = task.goals  # set of goal fact strings like "(full mode1)"
        self.static = set(task.static) if task.static is not None else set()

        # Extract static knowledge useful for quick checks:
        self.supports = {
            get_parts(fact)[1]
            for fact in self.static
            if get_parts(fact)[0] == "supports"
        }
        self.equipped_for_soil = {
            get_parts(fact)[1]
            for fact in self.static
            if get_parts(fact)[0] == "equipped_for_soil_analysis"
        }

        # Pre-extract static can_traverse pairs (lander, rover) if any.
        self.static_can_traverse = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in self.static
            if get_parts(fact)[0] == "can_traverse"
        }

        # Pre-extract any static at(mode, rover) facts (modes often have an 'at' that is static).
        self.static_at_mode = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if get_parts(fact)[0] == "at"
        }

    def __call__(self, node):
        state = set(node.state)  # frozenset -> mutable set for membership checks

        # If all goals already satisfied return 0
        if self.goals <= state:
            return 0

        # Prepare convenient views over current facts (merge static and state where appropriate)
        all_can_traverse = set(self.static_can_traverse)
        all_can_traverse.update({
            (get_parts(f)[1], get_parts(f)[2])
            for f in state
            if get_parts(f)[0] == "can_traverse"
        })

        equipped_for_rock = {
            (get_parts(f)[1], get_parts(f)[2])
            for f in state
            if get_parts(f)[0] == "equipped_for_rock_analysis"
        }

        visible = {get_parts(f)[1] for f in state if get_parts(f)[0] == "visible"}
        visible_from = {get_parts(f)[1] for f in state if get_parts(f)[0] == "visible_from"}

        # Modes that are equipped_for_soil may be in static or in current state.
        equipped_for_soil_current = set(self.equipped_for_soil)
        equipped_for_soil_current.update({
            get_parts(f)[1] for f in state if get_parts(f)[0] == "equipped_for_soil_analysis"
        })

        supports_current = set(self.supports)
        supports_current.update({
            get_parts(f)[1] for f in state if get_parts(f)[0] == "supports"
        })

        # Mode -> rover mapping (static or in current state)
        at_mode = dict(self.static_at_mode)
        for f in state:
            parts = get_parts(f)
            if parts[0] == "at":
                # Only keep entries where the first argument is a mode (domain typing does this)
                # but we cannot check types here; we simply store the mapping if it looks like `at <mode> <rover>`
                at_mode[parts[1]] = parts[2]

        total_cost = 0

        # Helper to test if some equipped_for_rock exists aligned with mode_rover
        def has_aligned_equip(mode_rover):
            for (wp, l) in equipped_for_rock:
                if (l, mode_rover) in all_can_traverse:
                    return True
            return False

        # Helper to check if there exists an equipped_for_rock for which lander has some can_traverse
        def has_any_equip_with_some_traverse():
            for (wp, l) in equipped_for_rock:
                # any traversal for this lander?
                if any(pair[0] == l for pair in all_can_traverse):
                    return True
            return False

        # Helper to check if we can produce an equipped_for_rock by taking an image
        # i.e., there exists a visible waypoint and a lander that can traverse rover1
        def can_produce_equip_via_take_image():
            # take_image requires: (visible W) and (can_traverse L rover1)
            for l, r in all_can_traverse:
                if r == "rover1" and visible:
                    return True
            return False

        # For each goal mode compute a small optimistic cost
        for goal_fact in self.goals:
            # goal_fact is like "(full mode1)"
            parts = get_parts(goal_fact)
            if parts[0] != "full":
                # Skip non-full goals (shouldn't happen in this domain examples)
                continue
            mode = parts[1]
            # If already full in current state -> 0 contribution
            if f"(full {mode})" in state:
                continue

            # Base pessimistic/default costs
            INF_COST = 10

            mode_rover = at_mode.get(mode, None)  # rover where this mode sits, if known
            # If mode has support or equipped_for_soil (either static or current) we can try two paths
            has_support = mode in supports_current
            has_equipped_soil = mode in equipped_for_soil_current

            best_cost = INF_COST

            # Try calibrate path if supports exists
            if has_support:
                # calibrate itself costs 1
                if mode_rover and has_aligned_equip(mode_rover):
                    best_cost = min(best_cost, 1)  # calibrate directly
                else:
                    # If there is an equipped_for_rock but misaligned we can sample_soil once to switch traversal
                    if has_any_equip_with_some_traverse() and mode_rover:
                        # sample_soil to make the lander traverse the required rover + calibrate
                        best_cost = min(best_cost, 2)  # sample_soil (1) + calibrate (1)
                    # Try to produce equipped_for_rock via take_image if possible
                    if can_produce_equip_via_take_image() and mode_rover:
                        # take_image (1) + maybe sample_soil (0/1) + calibrate (1)
                        # check if there exists a lander L that can traverse both rover1 (for take_image) and mode_rover
                        found_landers = {l for (l, r) in all_can_traverse if r == "rover1"}
                        aligned_exists = any((l, mode_rover) in all_can_traverse for l in found_landers)
                        if aligned_exists:
                            best_cost = min(best_cost, 1 + 1)  # take_image + calibrate
                        else:
                            # need take_image + sample_soil + calibrate
                            best_cost = min(best_cost, 1 + 1 + 1)

            # Try communicate_soil_data path if equipped_for_soil exists
            if has_equipped_soil:
                # communicate itself costs 1, but requires visible_from and aligned equipped_for_rock
                if mode_rover and has_aligned_equip(mode_rover) and visible_from:
                    # We also must ensure that visible_from corresponds to the same waypoint as equipped_for_rock,
                    # but to keep the check cheap we allow this optimistic shortcut if both exist.
                    best_cost = min(best_cost, 1)
                else:
                    # If aligned equip exists but visible_from missing, assume we can produce visible_from with a small penalty
                    if mode_rover and has_aligned_equip(mode_rover):
                        best_cost = min(best_cost, 2)  # produce visible_from (optimistic) + communicate
                    # If no aligned equip but we can make one via take_image:
                    if can_produce_equip_via_take_image() and mode_rover:
                        # take_image(1) + maybe sample_soil(0/1) + produce visible_from(1) + communicate(1)
                        found_landers = {l for (l, r) in all_can_traverse if r == "rover1"}
                        aligned_exists = any((l, mode_rover) in all_can_traverse for l in found_landers)
                        if aligned_exists:
                            best_cost = min(best_cost, 1 + 1 + 1)  # take_image + produce_visible_from + communicate
                        else:
                            best_cost = min(best_cost, 1 + 1 + 1 + 1)  # +sample_soil

            # If neither support nor equipped_for_soil, optimistic fallback finite cost
            if not has_support and not has_equipped_soil:
                best_cost = min(best_cost, INF_COST)

            # Add best_cost contribution for this mode
            total_cost += best_cost

        return int(total_cost)