from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()


def match_fact_with_pattern(fact_parts, pattern):
    """
    Check whether a fact (as list of parts) matches a pattern (tuple/list).
    pattern may contain '*' wildcards.
    """
    if len(fact_parts) != len(pattern):
        return False
    return all(fnmatch(fp, p) for fp, p in zip(fact_parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `rover-2` (secret_domain) domain.
    The domain's goals are typically of the form (full modeX). This heuristic
    estimates the number of actions required to make each required mode `full`.
    It reasons about the two main ways a mode can become full:
      - communicate_soil_data (requires the mode to be equipped_for_soil_analysis,
        an equipped_for_rock_analysis instance for some waypoint/lander, the mode
        to be at a rover, visible_from the waypoint, and a can_traverse relation
        between the chosen lander and the same rover), or
      - calibrate (requires the mode to be supported, the mode to be at some rover,
        an equipped_for_rock_analysis instance for some waypoint/lander, and a
        can_traverse relation between that lander and the rover).
    The heuristic is deliberately permissive (not admissible) and focuses on
    quickly discriminating promising states in greedy best-first search by
    counting the number of missing precondition facts (or approximate steps)
    per mode and then adding the final action to make the mode full.

    Assumptions
    -----------
    - The planner represents facts as strings of the form "(predicate arg1 arg2 ...)".
    - Some preconditions in the domain are existential (e.g., equipped_for_rock_analysis
      may hold for some waypoint/lander); the heuristic tries to exploit any existing
      matching fact in state or static info.
    - If an existential fact (e.g., an equipped_for_rock_analysis pair) does not exist,
      the heuristic assumes it can be produced with one extra action (e.g., take_image
      or other domain actions) and counts that as a single missing unit. This keeps the
      heuristic efficient while still penalizing missing supporting facts.
    - Static facts (task.static) are taken as always available and are consulted
      together with the current state when checking preconditions.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (modes that must be `full`).
    - Store static facts for quick membership queries.
    - Precompute and keep lists of static facts of interest (e.g., supports, equipped_for_soil_analysis,
      and any at facts about modes), so per-call cost is small.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each mode M required to be full (i.e., (full M) is in the goals but not present in the state):
      1. If (full M) is already true in the current state -> contribute 0.
      2. Otherwise estimate the minimal number of missing precondition "facts or actions"
         to enable either of the two mode-fulfilling operators (communicate_soil_data or calibrate).
         For each operator:
         a. Check whether the simple preconditions are present (e.g., (supports M), (equipped_for_soil_analysis M),
            or (at M R) for some rover R). If (at M R) exists pick each such R as a candidate rover.
            If no such (at M R) exists, count "at" as missing (1).
         b. For preconditions that are existential (e.g., equipped_for_rock_analysis ?wp ?lander),
            search for any existing fact in state/static that matches; if one exists, compute whether the
            additional connected preconditions (e.g., (visible_from ?wp) and (can_traverse ?lander R))
            are satisfied for that particular pairing. If any pairing yields all satisfied, that existential
            requirement is considered satisfied. Otherwise count missing pieces (each missing predicate
            contributes +1). If no such existential fact exists at all, the heuristic assumes it can be
            produced with roughly one extra action and accounts for that with +1.
         c. Sum missing pieces for this operator. If everything is present, the operator is applicable and
            requires exactly 1 action to produce (full M). Otherwise the estimated cost is (#missing pieces) + 1.
      3. Take the minimum estimated cost across the two operators (communicate_soil_data and calibrate). Add
         that to the total heuristic sum across goals.
    - The heuristic returns the sum of all per-mode estimates. If all goals are satisfied the heuristic returns 0.

    Notes on efficiency and behavior
    --------------------------------
    - The heuristic only performs simple set membership checks and small scans over relevant predicate lists.
    - It provides a finite estimate for any state (because missing facts are counted as finite positive integers).
    - It is not admissible (it may underestimate or overestimate), but is tuned to prefer states that already
      have critical supporting facts (e.g., equipped_for_rock_analysis, visible_from, can_traverse, supports).
    """

    def __init__(self, task):
        # Goals: set of goal fact strings (e.g., "(full mode1)")
        self.goals = set(task.goals)
        # Static facts are facts that never change; keep as set for fast membership tests.
        self.static = set(task.static)

        # Precompute static indices for quick lookups:
        # Map predicate -> list of argument-tuples for both static and later we will check state as well.
        self.static_index = {}
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = tuple(parts[1:])
            self.static_index.setdefault(pred, []).append(args)

    def facts_with_predicate(self, state, pred):
        """
        Return a list of argument-tuples for facts having predicate `pred`
        in the union of `state` and `static`.
        """
        results = []
        # static
        if pred in self.static_index:
            results.extend(self.static_index[pred])
        # state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == pred:
                results.append(tuple(parts[1:]))
        return results

    def fact_exists(self, state, pred, *args_pattern):
        """
        Check if any fact in state or static matches (pred args...) where
        args_pattern can contain '*' wildcards. Returns True/False.
        """
        # Check static
        if pred in self.static_index:
            for args in self.static_index[pred]:
                if match_fact_with_pattern((pred, *args), (pred, *args_pattern)):
                    return True
        # Check state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] != pred:
                continue
            if match_fact_with_pattern(parts, (pred, *args_pattern)):
                return True
        return False

    def __call__(self, node):
        """
        Compute the heuristic estimate given a search node.

        Returns an integer estimate of the number of actions required to reach a state
        satisfying all goal facts from the node's state.
        """
        state = node.state

        # If all goals already reached -> heuristic = 0 (requirement 2 of checklist).
        if self.goals <= state:
            return 0

        total_estimated = 0

        # Extract list of mode-goals to satisfy (we assume goals are "full" facts).
        mode_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "full":
                mode_goals.append(parts[1])

        # Helper: find all rovers where a given mode is presently `at` (static or state).
        def rovers_for_mode(mode):
            rovers = []
            # search both static and state for (at mode rover)
            for args in self.facts_with_predicate(state, "at"):
                if len(args) >= 2 and args[0] == mode:
                    rovers.append(args[1])
            return rovers

        # Helper: all equipped_for_rock_analysis facts in state/static as pairs (waypoint, lander)
        equipped_rock = self.facts_with_predicate(state, "equipped_for_rock_analysis")
        # Also gather visible_from and visible facts for quick checks
        visible_from_set = {args[0] for args in self.facts_with_predicate(state, "visible_from")}
        visible_set = {args[0] for args in self.facts_with_predicate(state, "visible")}
        # can_traverse pairs
        can_traverse_pairs = set(self.facts_with_predicate(state, "can_traverse"))
        # equipped_for_soil_analysis modes (static or state)
        equipped_soil_modes = {args[0] for args in self.facts_with_predicate(state, "equipped_for_soil_analysis")}
        # supports modes (from static or state)
        supports_modes = {args[0] for args in self.facts_with_predicate(state, "supports")}

        # For each mode that must be full but isn't, estimate a cost
        for mode in mode_goals:
            goal_fact = f"(full {mode})"
            if goal_fact in state:
                continue  # already satisfied

            # Candidates for minimal cost via different operators
            estimates = []

            # Candidate rovers where this mode is at
            rovers = rovers_for_mode(mode)
            if not rovers:
                # No at-fact available; treat as missing (+1) but still try to compute with placeholder None
                rovers = [None]

            # 1) Estimate via communicate_soil_data
            # communicate_soil_data preconditions:
            # - (equipped_for_soil_analysis ?v2)  --> here ?v2 is the mode
            # - (equipped_for_rock_analysis ?v3 ?v4)  --> existential: any (wp,lander)
            # - (at ?v2 ?v5)  --> mode is at some rover (we iterate rovers)
            # - (visible_from ?v3)  --> same waypoint as equipped_for_rock_analysis
            # - (can_traverse ?v4 ?v5)  --> lander must be able to traverse the rover chosen
            for rover in rovers:
                missing = 0
                # equipped_for_soil_analysis ?
                if mode not in equipped_soil_modes and not self.fact_exists(state, "equipped_for_soil_analysis", mode):
                    missing += 1
                # at mode rover
                if rover is None:
                    missing += 1
                # existential equipped_for_rock_analysis + visible_from + can_traverse
                # Try to find an existing (wp, lander) that minimizes missing pieces
                best_pair_missing = None
                if equipped_rock:
                    for (wp, lander) in equipped_rock:
                        pair_missing = 0
                        # visible_from wp?
                        if wp not in visible_from_set and not self.fact_exists(state, "visible_from", wp):
                            pair_missing += 1
                        # can_traverse lander rover?
                        if rover is None:
                            # rover missing counted above; here count can_traverse as missing because we don't know rover
                            pair_missing += 1
                        else:
                            if (lander, rover) not in can_traverse_pairs and not self.fact_exists(state, "can_traverse", lander, rover):
                                pair_missing += 1
                        if best_pair_missing is None or pair_missing < best_pair_missing:
                            best_pair_missing = pair_missing
                else:
                    # No equipped_for_rock_analysis exists: assume creating it costs roughly 1 action (e.g., take_image or sample_rock)
                    # That creation might also require can_traverse or visible facts; to keep heuristic cheap we add 1.
                    best_pair_missing = 1
                    # But if visible_from is required and we have visible_from for some wp, we can reduce one
                    if visible_from_set:
                        best_pair_missing = 1  # still at least 1 to create an equipped_for_rock_analysis pair
                missing += best_pair_missing if best_pair_missing is not None else 1

                # Total estimated steps: missing to enable + 1 to execute communicate_soil_data itself
                estimates.append(missing + 1)

            # 2) Estimate via calibrate
            # calibrate preconditions:
            # - (supports ?v2)  -> mode must be supported
            # - (at ?v2 ?v5)  -> mode is at some rover
            # - (equipped_for_rock_analysis ?v3 ?v4)  -> any (wp,lander)
            # - (can_traverse ?v4 ?v5)  -> lander must traverse same rover
            for rover in rovers:
                missing = 0
                # supports?
                if mode not in supports_modes and not self.fact_exists(state, "supports", mode):
                    missing += 1
                # at mode rover
                if rover is None:
                    missing += 1
                # existential equipped_for_rock_analysis + can_traverse
                best_pair_missing = None
                if equipped_rock:
                    for (wp, lander) in equipped_rock:
                        pair_missing = 0
                        if rover is None:
                            pair_missing += 1
                        else:
                            if (lander, rover) not in can_traverse_pairs and not self.fact_exists(state, "can_traverse", lander, rover):
                                pair_missing += 1
                        if best_pair_missing is None or pair_missing < best_pair_missing:
                            best_pair_missing = pair_missing
                else:
                    # If no equipped_for_rock_analysis present, assume one action needed to create it
                    best_pair_missing = 1
                missing += best_pair_missing if best_pair_missing is not None else 1

                # One action to execute calibrate itself
                estimates.append(missing + 1)

            # If no estimate was found (should not happen), add a conservative value
            if not estimates:
                estimates.append(3)

            # Choose the minimal estimate for this mode
            mode_cost = min(estimates)
            total_estimated += mode_cost

        # Ensure heuristic is a non-negative integer
        try:
            total_estimated = int(max(0, total_estimated))
        except Exception:
            total_estimated = 0

        return total_estimated