from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact (string) matches a pattern of tokens (wildcards '*' allowed).
    Example: match("(at mode1 rover2)", "at", "*", "rover2") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `rover-2` (secret_domain) domain. The
    heuristic estimates the number of actions required to make all goal
    `full`(mode) facts true. It is intentionally not admissible: it uses
    compact domain knowledge and optimistic assumptions about action parameter
    re-use to produce informative estimates while remaining cheap to compute.

    Assumptions
    -----------
    - Goals in this domain are of the form (full modeX). The heuristic only
      considers achieving these `full` facts.
    - Modes that can become `full` are made `full` either by the `calibrate`
      action (requires (supports mode)) or by `communicate_soil_data`
      (requires (equipped_for_soil_analysis mode)). Both actions require a
      matching `equipped_for_rock_analysis` fact for some (waypoint lander)
      pair and the appropriate `at` and `can_traverse` facts.
    - The heuristic treats available facts about (visible), (equipped_for_rock_analysis),
      (can_traverse) and static facts (supports, equipped_for_soil_analysis,
      at ...) as the primary enablers and abstracts away many low-level
      preconditions for some ancillary actions (e.g., changing which rover a
      lander can traverse to). This yields a compact but informative estimate.
    - If the required direct supporting facts are missing, we count a small
      additional cost for producing them (e.g., taking an image to obtain
      equipped_for_rock_analysis). We do not try to fully reconstruct long
      parameter-dependent chains (we instead use small, optimistic fixed costs).

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - The set of goal facts (task.goals).
    - Static information that does not change during planning:
      * supports_modes: set of mode names that have (supports mode) in static.
      * equipped_soil_modes: set of modes with (equipped_for_soil_analysis mode).
      * mode_at_rover: mapping mode -> rover from static (at mode rover).
    This static information is used to quickly decide whether the fast
    `calibrate` path (uses supports) is available.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state), the heuristic proceeds as follows:

    1. If the state's facts already satisfy all goal facts, return 0.

    2. Parse the current state and collect the following dynamic facts:
       - full_modes: modes that are already (full mode).
       - equipped_rock_pairs: set of (waypoint, lander) that are
         (equipped_for_rock_analysis waypoint lander).
       - visible_waypoints: waypoints that are (visible waypoint).
       - visible_from_waypoints: waypoints that are (visible_from waypoint).
       - can_traverse_pairs: set of (lander, rover) that are
         (can_traverse lander rover).

    3. For each goal mode that is not yet full, compute a small estimated
       cost to make it full using the most promising (cheapest) path:
       - Preferred (cheap) path: if (supports mode) is available (from static)
         and the mode's `at` rover is known, and there exists a pair
         (waypoint, lander) in equipped_rock_pairs with (can_traverse lander rover)
         in the state, then we can apply `calibrate` immediately -> cost 1.
       - Secondary path (also cheap): if (equipped_for_soil_analysis mode) is
         available (static) and we already have (equipped_for_rock_analysis waypoint lander)
         with (can_traverse lander rover), and (visible_from waypoint) exists,
         then `communicate_soil_data` can be applied -> cost 1.
       - If the required equipped_for_rock_analysis fact is missing but there
         exists a (visible waypoint) and some lander with some can_traverse
         entry (i.e., the lander is usable somewhere), we optimistically
         assume we can produce equipped_for_rock_analysis by taking an image
         (take_image) possibly combined with a small number of supporting
         moves of the `can_traverse` relation. In that case we assign cost 2
         (one action to obtain equipped_for_rock_analysis, then one to make
         the mode full via calibrate/communicate).
       - If only weaker enabling facts are present (e.g., some have_image,
         empty objective, have_soil_analysis, etc.) but no direct visible or
         equipped facts, we assign a slightly larger optimistic cost (3).
       - If a mode lacks both supports and equipped_for_soil_analysis
         (i.e., there is no known way to make it full with available domain
         operators), we conservatively assign a high finite penalty (10).
       The numbers above (1,2,3,10) are tuned to provide informative
       differentiation while keeping the computation cheap.

    4. The final heuristic value is the sum of the per-mode estimates. This
       additive assumption approximates the number of operator applications
       required when multiple modes must be made full. The heuristic returns
       an integer >= 0.

    Notes
    -----
    - The heuristic is deliberately optimistic in several places (for
      efficiency and guidance) and therefore is not guaranteed admissible.
    - It always returns a finite value for the kind of problems expected by
      this domain encoding (we use a bounded penalty when no clear path is
      visible).
    """

    def __init__(self, task):
        # Goals as a frozenset of strings (facts)
        self.goals = task.goals

        # Static facts
        static_facts = task.static if hasattr(task, "static") else frozenset()

        # Modes that have (supports mode)
        self.supports_modes = {get_parts(fact)[1] for fact in static_facts if match(fact, "supports", "*")}

        # Modes that have (equipped_for_soil_analysis mode)
        self.equipped_soil_modes = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")}

        # Map mode -> rover from static (at mode rover)
        self.mode_at_rover = {}
        for fact in static_facts:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                mode = parts[1]
                rover = parts[2]
                self.mode_at_rover[mode] = rover

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is zero (check list item 2)
        if self.goals <= state:
            return 0

        # Extract dynamic facts from the state
        full_modes = {get_parts(f)[1] for f in state if match(f, "full", "*")}
        equipped_rock_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "equipped_for_rock_analysis", "*", "*")}
        visible_waypoints = {get_parts(f)[1] for f in state if match(f, "visible", "*")}
        visible_from_waypoints = {get_parts(f)[1] for f in state if match(f, "visible_from", "*")}
        can_traverse_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "can_traverse", "*", "*")}

        # Also consider dynamic at(mode rover) facts (might be in static or state)
        mode_at_rover_dyn = dict(self.mode_at_rover)  # start with static mapping
        for f in state:
            if match(f, "at", "*", "*"):
                parts = get_parts(f)
                mode_at_rover_dyn[parts[1]] = parts[2]

        total_cost = 0

        # Determine list of modes that must be made full from goal facts
        goal_modes = [get_parts(g)[1] for g in self.goals if match(g, "full", "*")]

        for mode in goal_modes:
            if mode in full_modes:
                continue  # already satisfied

            # Default large-but-finite penalty if we cannot find any path
            cost = 10

            # Which rover is this mode at? If unknown, remain with penalty
            rover = mode_at_rover_dyn.get(mode, None)

            # Fast path: calibrate (requires supports(mode)) if we have an equipped rock pair that can reach that rover
            if mode in self.supports_modes and rover is not None:
                # check if any (waypoint, lander) is equipped and lander can_traverse to the mode's rover
                can_immediate_calibrate = any((lander, rover) in can_traverse_pairs for (_wp, lander) in equipped_rock_pairs)
                if can_immediate_calibrate:
                    cost = 1
                    total_cost += cost
                    continue
                # If there's no equipped pair but there exists at least one visible waypoint or a lander that can traverse somewhere,
                # we optimistically assume we can get to equipped_for_rock_analysis with one take_image and then calibrate.
                has_visible_and_lander = len(visible_waypoints) > 0 and any(True for (_l, r) in can_traverse_pairs)
                if has_visible_and_lander:
                    cost = 2  # take_image (produce equipped) + calibrate
                    total_cost += cost
                    continue

            # Second path: communicate_soil_data if mode has equipped_for_soil_analysis
            if mode in self.equipped_soil_modes and rover is not None:
                # Need an equipped_for_rock_analysis that also has a lander able to traverse to rover and the waypoint visible_from
                can_immediate_communicate = False
                for (wp, lander) in equipped_rock_pairs:
                    if (lander, rover) in can_traverse_pairs and wp in visible_from_waypoints:
                        can_immediate_communicate = True
                        break
                if can_immediate_communicate:
                    cost = 1
                    total_cost += cost
                    continue

                # If we have a visible waypoint and some lander with can_traverse somewhere, assume:
                #  - take_image to get equipped_for_rock_analysis
                #  - (optimistically) visible_from is available or cheap to get (we assign one more action)
                #  - then communicate_soil_data
                if len(visible_waypoints) > 0 and any(True for (_l, r) in can_traverse_pairs):
                    cost = 2  # optimistic: take_image + communicate
                    total_cost += cost
                    continue

                # If we have some supporting pre-facts like have_image/empty/have_soil_analysis
                # (these are not enumerated precisely here) assign a slightly larger optimistic cost
                # because communicate_image_data or sample_rock could produce visible/visible_from.
                has_image = any(match(f, "have_image", "*") for f in state)
                has_empty_objective = any(match(f, "empty", "*") for f in state)
                has_soil_analysis_wp = any(match(f, "have_soil_analysis", "*") for f in state)
                if has_image and has_empty_objective and has_soil_analysis_wp:
                    cost = 3
                    total_cost += cost
                    continue

            # Fallback: if we still haven't set a better cost, keep the penalty
            total_cost += cost

        return total_cost