from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens inside a PDDL fact string, e.g. "(at mode1 rover2)" -> ["at","mode1","rover2"]."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches a pattern with wildcards (uses fnmatch)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret 'rover-2' domain that estimates the
    number of actions required to achieve all goals of the form (full ?mode).

    The heuristic is not admissible (it is intended to guide greedy best-first search),
    but is designed to be informative by focusing on the domain-specific requirements
    to make a mode `full`, which -- according to the domain operators -- can be achieved
    either by the `communicate_soil_data` action or by the `calibrate` action.
    Both of these actions require a rock-analysis-capable sensor (an `equipped_for_rock_analysis`
    fact or a short sequence of actions to obtain it) and certain mode-specific or lander-specific
    preconditions. The heuristic inspects which preconditions are already satisfied and
    estimates minimal extra actions to satisfy the missing ones.

    Assumptions
    -----------
    - Goals are primarily of the form (full ?mode). For other goal predicates the heuristic
      will still return an estimate but is mainly targeted to (full ...) goals.
    - Static facts (task.static) such as `supports`, `equipped_for_soil_analysis`, and `at`
      for modes are not changed by normal execution in many instances; they are treated as
      useful static information.
    - The heuristic relaxes certain object-equality constraints (it reasons existentially
      about waypoints/landers/rovers). Concretely, when an operator requires some
      (equipped_for_rock_analysis ?wp ?lander) and (can_traverse ?lander ?rover),
      the heuristic will treat any matching pair in the state/static facts as sufficient
      even if the exact object identities are not checked for cross-dependencies. This
      relaxation makes the heuristic fast and keeps it finite even when a full grounding
      or exact matching would be expensive or impossible in the current representation.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
    - The set of goal predicates (task.goals).
    - Useful static information:
      - supports_modes: modes that have (supports mode) in static facts.
      - equipped_soil_modes: modes that have (equipped_for_soil_analysis mode) in static facts.
      - mode_at: a map of mode -> rover for (at mode rover) facts found in static facts.
      - communicated_soil_data_cameras: cameras with (communicated_soil_data camera).
      - communicated_rock_data_objectives: objectives with (communicated_rock_data objective).
    These static structures are used to cheaply check some preconditions required by the
    `calibrate` and `communicate_soil_data` actions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic works as follows:

    1. Quick goal check:
       - If all goals are satisfied in the given state, return 0.

    2. Parse the current state to build fast membership tests for the predicates that
       matter in this domain: sets for `full`, `equipped_for_rock_analysis`,
       `visible`, `visible_from`, `have_image`, `empty`, `have_soil_analysis`,
       `can_traverse`, `equipped_for_soil_analysis` (state-level), `at` for modes, etc.

    3. For each goal of the form (full ?mode) that is not yet satisfied, estimate the
       minimum number of actions to make that particular mode `full`. There are two
       domain actions that add (full ?mode): `communicate_soil_data` and `calibrate`.
       The heuristic computes a small relaxed cost estimate for both options and takes
       the smaller one.

       - Estimating the cost to reach preconditions shared by both actions:
         * If an (equipped_for_rock_analysis ?wp ?lander) already exists for some wp/lander,
           it costs 0 extra actions for that part.
         * If not, but there is a `visible` waypoint and at least one `can_traverse` fact,
           then a single `take_image` can (in the relaxed view) produce an
           `equipped_for_rock_analysis` fact -> cost 1.
         * Otherwise, if the agent has a camera image (have_image) and there is a waypoint
           with `have_soil_analysis` and an `empty` objective (so sample_rock could be applied
           in the relaxed model) we assume it costs 2 actions (one to produce visibility,
           one to `take_image`) -> cost 2.
         * Otherwise we assign a small fallback penalty (3) representing a longer sequence.

       - Additional precondition costs:
         * `supports mode` is taken from static facts. If missing, we add a larger penalty
           (5) because enabling it is generally not cheap or may be impossible in many instances.
         * `equipped_for_soil_analysis mode` (required by `communicate_soil_data`) is also
           read from static/state facts; if missing we add a larger penalty (5) because there
           is no cheap action in the domain to produce it.
         * `visible_from waypoint` (for communicate_soil_data) is checked; if absent a penalty
           (2) is charged since producing visible_from may require extra actions in the relaxed
           path.
         * `at mode rover` is typically static; if missing we add a moderate penalty (3).
         * `can_traverse lander rover` is required; if there exists any can_traverse in state/static
           we treat it as satisfied (cost 0), else a small penalty (2) is added.

       - The cost for a mode is: 1 (the final action that will produce `full`) plus the
         estimated cost to satisfy all missing preconditions (these sub-costs are computed
         as described above). The minimum of the two alternative paths (`calibrate` vs
         `communicate_soil_data`) is used as the estimate for that mode.

    4. The final heuristic value is the sum of the per-mode estimates. This is a relaxed
       additive heuristic (it ignores helpful interactions between different modes), but
       it is quick to compute and captures the domain-specific structure (that producing
       an `equipped_for_rock_analysis` fact and having `supports` or `equipped_for_soil_analysis`
       are the central ingredients to make a mode `full`).

    Implementation notes
    --------------------
    - The heuristic uses small integer penalties for missing preconditions. Penalties are
      chosen to keep values finite and to reflect relative difficulty: missing static predicates
      that cannot easily be produced get higher costs.
    - The heuristic always returns a finite integer, even if it may be optimistic or
      over-optimistic relative to the true minimal cost.
    """

    def __init__(self, task):
        # Goals (set of fact strings)
        self.goals = task.goals
        # Static facts (frozenset of strings)
        self.static = task.static

        # Extract static information into useful structures
        self.supports_modes = {get_parts(f)[1] for f in self.static if match(f, "supports", "*")}
        self.equipped_soil_static = {
            get_parts(f)[1] for f in self.static if match(f, "equipped_for_soil_analysis", "*")
        }
        # at mode rover in static (mode->rover); there could be multiple but we keep a set mapping
        self.mode_at_static = {}
        for f in self.static:
            if match(f, "at", "*", "*"):
                pred, a, b = get_parts(f)
                # only care about modes with at facts; store as set for each mode
                self.mode_at_static.setdefault(a, set()).add(b)

        # Useful static flags for cameras/objectives/rock/soil communication (may speed checks)
        self.communicated_soil_data = {get_parts(f)[1] for f in self.static if match(f, "communicated_soil_data", "*")}
        self.communicated_rock_data = {get_parts(f)[1] for f in self.static if match(f, "communicated_rock_data", "*")}

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals satisfied, return 0
        if self.goals <= state:
            return 0

        # Parse state facts into useful sets
        full_modes = {get_parts(f)[1] for f in state if match(f, "full", "*")}
        equipped_rock = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "equipped_for_rock_analysis", "*", "*")}
        visible_waypoints = {get_parts(f)[1] for f in state if match(f, "visible", "*")}
        visible_from = {get_parts(f)[1] for f in state if match(f, "visible_from", "*")}
        have_image = {get_parts(f)[1] for f in state if match(f, "have_image", "*")}
        empty_objectives = {get_parts(f)[1] for f in state if match(f, "empty", "*")}
        have_soil_analysis = {get_parts(f)[1] for f in state if match(f, "have_soil_analysis", "*")}
        equipped_soil_state = {get_parts(f)[1] for f in state if match(f, "equipped_for_soil_analysis", "*")}
        can_traverse_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "can_traverse", "*", "*")}
        # at facts for modes in state
        mode_at_state = {}
        for f in state:
            if match(f, "at", "*", "*"):
                p, a, b = get_parts(f)
                mode_at_state.setdefault(a, set()).add(b)

        # Helper to test if any can_traverse exists (we relax specific identity)
        any_can_traverse = len(can_traverse_pairs) > 0 or len({f for f in self.static if match(f, "can_traverse", "*", "*")}) > 0

        # Helper to estimate cost to obtain equipped_for_rock_analysis if missing
        def estimate_equipped_rock_cost():
            # 0 if any equipped_for_rock_analysis exists in state
            if equipped_rock:
                return 0
            # 1 if there is a visible waypoint and some can_traverse exists (we can take_image)
            if visible_waypoints and any_can_traverse:
                return 1
            # 2 if we can produce visible by sample_rock (have_image, empty objective, have_soil_analysis) then take_image
            if have_image and empty_objectives and have_soil_analysis:
                return 2
            # 3 if we at least have images (so we might do a longer sequence)
            if have_image:
                return 3
            # fallback cost
            return 4

        total_cost = 0

        # For each goal that is (full mode)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "full":
                # for non-full goals: simple 1 if not present, else 0 (conservative)
                if goal not in state:
                    total_cost += 1
                continue

            mode = parts[1]
            if goal in state:
                continue  # already satisfied

            # Consider two alternative operators: calibrate OR communicate_soil_data
            # We compute relaxed costs for each and take the minimum.

            # Common sub-cost: cost to obtain equipped_for_rock_analysis
            cost_equipped_rock = estimate_equipped_rock_cost()

            # --------- calibrate path ----------
            # calibrate preconditions:
            # (supports ?mode)  -- usually static
            # (at ?mode ?rover) -- static or state
            # (equipped_for_rock_analysis ?wp ?lander)
            # (can_traverse ?lander ?rover)
            calibrate_cost = 1  # the calibrate action itself

            # supports
            if mode not in self.supports_modes:
                # missing supports is relatively expensive to make available; we penalize heavily
                calibrate_cost += 5
            # at mode rover
            if (mode not in mode_at_state) and (mode not in self.mode_at_static):
                calibrate_cost += 3
            # equipped_for_rock_analysis
            calibrate_cost += cost_equipped_rock
            # can_traverse -- if we already saw a can_traverse anywhere, assume 0 cost else small penalty
            if not any_can_traverse:
                calibrate_cost += 2

            # --------- communicate_soil_data path ----------
            # communicate_soil_data preconditions:
            # (equipped_for_soil_analysis ?mode)
            # (equipped_for_rock_analysis ?wp ?lander)
            # (at ?mode ?rover)
            # (visible_from ?wp)
            # (can_traverse ?lander ?rover)
            comm_cost = 1  # the communicate_soil_data action itself

            # equipped_for_soil_analysis
            if (mode not in equipped_soil_state) and (mode not in self.equipped_soil_static):
                # No easy action to produce this in the domain -> large penalty
                comm_cost += 5
            # at mode rover
            if (mode not in mode_at_state) and (mode not in self.mode_at_static):
                comm_cost += 3
            # equipped_for_rock_analysis
            comm_cost += cost_equipped_rock
            # visible_from
            if not visible_from:
                # if there is a visible waypoint we assume we can convert it to visible_from with some steps (penalty 2)
                if visible_waypoints:
                    comm_cost += 2
                else:
                    comm_cost += 3
            # can_traverse
            if not any_can_traverse:
                comm_cost += 2

            mode_cost = min(calibrate_cost, comm_cost)
            total_cost += mode_cost

        # Ensure heuristic is finite
        if total_cost is None:
            return 1000
        return int(total_cost)