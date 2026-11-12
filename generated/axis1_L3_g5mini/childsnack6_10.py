from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string by stripping surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Return True if fact (string) matches the given pattern elements (wildcards '*' allowed).
    Example: match('(at mode1 rover2)', 'at', '*', 'rover2') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `rover-2` (secret_domain) domain.
    The heuristic estimates the number of actions needed to satisfy all goal
    facts of the form (full modeX). The domain's primary way of achieving
    `full(mode)` facts is via either the `calibrate` action (when a mode is
    supported) or the `communicate_soil_data` action (when a mode is equipped
    for soil analysis). Both of these actions require an `equipped_for_rock_analysis`
    fact produced by `take_image`, which in turn requires a `visible` fact that
    can be produced by `sample_rock` or `communicate_image_data`. This heuristic
    is deliberately approximate: it counts a small number of required actions
    along the shortest plausible chain to create `full(mode)` and sums these
    estimates for all unsatisfied mode goals.

    Assumptions
    -----------
    - Goals are of the form (full <mode>); other goal predicates are not
      considered in this heuristic implementation.
    - Static facts (task.static) are always available and should be considered
      alongside the current state.
    - We do not attempt to model precisely the identity constraints of rovers,
      landers and which lander must traverse which rover; instead we check for
      existence of the required kinds of facts and use small fixed costs to
      represent short chains of actions that typically produce those facts.
    - The heuristic is not admissible — it may underestimate or overestimate —
      but is designed to be cheap to compute and to reflect typical operator
      chains in the domain.

    Heuristic Initialization
    ------------------------
    - Save the set of goal facts (task.goals).
    - Save static facts (task.static) to consider them available at evaluation.
    - Nothing else is precomputed because most facts relevant to the heuristic
      are dynamic (appear in the current state). Static facts are used at
      evaluation time by merging them with the current state.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each goal of the form (full MODE) that is not yet true in the current
    state (note: "true" means present in node.state), we estimate the minimal
    number of actions needed to achieve that particular `full` fact using
    two main strategies and taking the cheaper one:

    1. Quick checks (zero/one action):
       - If (full MODE) is already in the current state -> cost 0 for this goal.
       - If there already exists an `equipped_for_rock_analysis (wp lander)`
         fact and the other crude preconditions appear available (e.g., there is
         an (at MODE rover) fact and some (can_traverse lander rover) facts),
         then either `calibrate` or `communicate_soil_data` can directly produce
         `(full MODE)` in 1 action (choose the applicable one). We check the
         presence of supports/equipped_for_soil_analysis to decide which action
         is applicable.

    2. Typical chain (3 actions):
       - If the quick 1-action case is not present, we check whether the common
         preparatory facts are present that allow the short chain
         sample_rock -> take_image -> (calibrate | communicate_soil_data).
         Concretely, we look for:
           - At least one camera with (have_image ...)
           - At least one empty objective
           - At least one waypoint with (have_soil_analysis ...)
         If such facts exist and the mode has a plausible final action (either
         supports(MODE) or equipped_for_soil_analysis(MODE) is available),
         we estimate 3 actions (sample_rock, take_image, final action).

    3. Special chain for `visible_from` requirement:
       - `communicate_soil_data` requires `visible_from waypoint`. If
         `visible_from` is not present but `communicated_soil_data` and
         `communicated_rock_data` exist for some camera/objective plus the
         usual `have_image`, `empty`, `have_soil_analysis` triple, then a
         short chain communicate_image_data -> take_image -> communicate_soil_data
         is likely and we again estimate 3 actions.

    4. Fallback:
       - If none of the above patterns can be matched we return a moderate
         finite fallback cost (5) for that mode. This keeps the heuristic finite
         for solvable states while reflecting that we lack evidence for a
         short chain.

    The final heuristic value is the sum of per-mode estimates. The heuristic
    returns 0 only when the planner's node.state already contains all goals.

    """

    def __init__(self, task):
        # The explicit goal facts to be achieved.
        self.goals = set(task.goals)
        # Static facts that are always available.
        self.static = set(task.static)

    def __call__(self, node):
        # If the node already satisfies all goals (according to the planner's
        # state representation), return 0. We respect the planner's view:
        # Task.goal_reached checks goals <= node.state, so the heuristic must
        # return 0 only in that case.
        if self.goals <= node.state:
            return 0

        # Merge dynamic state and static facts for lookups used by the heuristic.
        combined = set(node.state) | self.static

        # Extract quick-access sets/maps from combined facts.
        have_image_cameras = {get_parts(f)[1] for f in combined if match(f, "have_image", "*")}
        empty_objectives = {get_parts(f)[1] for f in combined if match(f, "empty", "*")}
        have_soil_waypoints = {get_parts(f)[1] for f in combined if match(f, "have_soil_analysis", "*")}
        visible_waypoints = {get_parts(f)[1] for f in combined if match(f, "visible", "*")}
        visible_from_waypoints = {get_parts(f)[1] for f in combined if match(f, "visible_from", "*")}
        equipped_rock = {(get_parts(f)[1], get_parts(f)[2]) for f in combined if match(f, "equipped_for_rock_analysis", "*", "*")}
        equipped_soil_modes = {get_parts(f)[1] for f in combined if match(f, "equipped_for_soil_analysis", "*")}
        supports_modes = {get_parts(f)[1] for f in combined if match(f, "supports", "*")}
        can_traverse_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in combined if match(f, "can_traverse", "*", "*")}
        at_mode_map = {}  # mode -> set of rovers where it is located
        for f in combined:
            if match(f, "at", "*", "*"):
                obj, loc = get_parts(f)[1], get_parts(f)[2]
                # only keep modes for which "at" is relevant (we'll check goals)
                at_mode_map.setdefault(obj, set()).add(loc)
        # communicated_* facts (used to decide if communicate_image_data is possible)
        communicated_soil_cameras = {get_parts(f)[1] for f in combined if match(f, "communicated_soil_data", "*")}
        communicated_rock_objectives = {get_parts(f)[1] for f in combined if match(f, "communicated_rock_data", "*")}

        total_cost = 0

        # Helper boolean checks for presence of basic resources needed to create
        # a visible/equipped_for_rock_analysis chain.
        basic_sample_possible = bool(have_image_cameras and empty_objectives and have_soil_waypoints)

        for goal in self.goals:
            parts = get_parts(goal)
            # expect goals like "(full modeX)"
            if parts[0] != "full" or len(parts) < 2:
                # If the goal is not of the kind this heuristic models, give a
                # conservative small cost (1) if not yet true; this keeps the
                # heuristic finite and modest for unknown goal types.
                if goal in node.state:
                    continue
                total_cost += 1
                continue

            mode = parts[1]
            # If already true in the node.state (planner's state), cost 0 for this mode.
            if goal in node.state:
                continue

            # Per-mode estimate: try the cheaper of the two canonical strategies.
            best_cost_for_mode = float("inf")

            # 1) Calibrate strategy: requires supports(mode), at(mode, rover),
            #    equipped_for_rock_analysis(wp, lander) and can_traverse(lander, rover)
            if mode in supports_modes:
                # find rovers where mode is located (must exist to apply calibrate).
                rovers_for_mode = at_mode_map.get(mode, set())
                if rovers_for_mode:
                    # If an equipped_for_rock_analysis already exists that can be
                    # used together with a can_traverse to reach the rover, cost=1.
                    for (wp, lander) in equipped_rock:
                        for rover in rovers_for_mode:
                            if (lander, rover) in can_traverse_pairs:
                                best_cost_for_mode = min(best_cost_for_mode, 1)
                    # If we can create equipped_for_rock_analysis cheaply:
                    if best_cost_for_mode > 1:
                        # If we have resources to sample_rock and then take_image,
                        # assume sample_rock + take_image + calibrate = 3 actions
                        # (sample_rock, take_image, calibrate).
                        # We require that there is at least one lander that can
                        # plausibly be moved to the rover or is already can_traverse.
                        # To keep it simple, prefer existence of any can_traverse pair
                        # linking some lander to some rover (not necessarily the same),
                        # as sample_soil can change such links.
                        if basic_sample_possible and can_traverse_pairs:
                            best_cost_for_mode = min(best_cost_for_mode, 3)
                        else:
                            # Without evidence for a short chain, assign a finite fallback.
                            best_cost_for_mode = min(best_cost_for_mode, 5)
                else:
                    # There's no (at mode rover) fact visible; cannot apply calibrate now.
                    # Assign a finite fallback cost.
                    best_cost_for_mode = min(best_cost_for_mode, 5)

            # 2) communicate_soil_data strategy: requires equipped_for_soil_analysis(mode),
            #    equipped_for_rock_analysis(wp, lander), visible_from(wp), at(mode, rover), can_traverse(lander, rover)
            if mode in equipped_soil_modes:
                rovers_for_mode = at_mode_map.get(mode, set())
                if rovers_for_mode:
                    # Fast case: we already have an equipped_rock and visible_from that
                    # align with a can_traverse -> 1 action.
                    for (wp, lander) in equipped_rock:
                        if wp in visible_from_waypoints:
                            for rover in rovers_for_mode:
                                if (lander, rover) in can_traverse_pairs:
                                    best_cost_for_mode = min(best_cost_for_mode, 1)
                    # If not directly available, check if we can create visible_from
                    # via communicate_image_data and then proceed:
                    # communicate_image_data (requires have_image, empty, communicated_*,
                    # have_soil_analysis) -> creates visible_from, then take_image -> equipped_rock,
                    # then communicate_soil_data -> full. That's 3 actions if prerequisites exist.
                    can_make_visible_from = bool(have_image_cameras and empty_objectives and have_soil_waypoints and communicated_soil_cameras and communicated_rock_objectives)
                    if can_make_visible_from and can_traverse_pairs:
                        best_cost_for_mode = min(best_cost_for_mode, 3)
                    else:
                        # If we cannot produce visible_from cheaply but we can produce
                        # equipped_for_rock_analysis via sample_rock/take_image, we still
                        # might be blocked because communicate_soil_data requires visible_from.
                        # Give a fallback cost to keep heuristic finite.
                        best_cost_for_mode = min(best_cost_for_mode, 5)
                else:
                    best_cost_for_mode = min(best_cost_for_mode, 5)

            # If neither strategy produced a finite estimate (should not happen because
            # of fallback assignments), ensure we have a finite cost.
            if best_cost_for_mode == float("inf"):
                best_cost_for_mode = 5

            total_cost += int(best_cost_for_mode)

        return int(total_cost)