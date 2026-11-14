from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, e.g. '(at mode1 rover2)' -> ['at','mode1','rover2']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the rover-2 (secret_domain) domain.

    The primary goals in instances of this domain are typically predicates of
    the form (full <mode>). A predicate (full mode) can be achieved either by
    the action `calibrate` (if the mode "supports" calibration) or by
    `communicate_soil_data` (when the mode is equipped for soil analysis).
    Both of these actions require that some lander is "equipped_for_rock_analysis"
    for some waypoint and that that lander can traverse to the rover where the
    mode is located. Creating or moving such an equipped lander typically
    requires `take_image` (to create equipped_for_rock_analysis from a visible
    waypoint) and `sample_soil` (to change the can_traverse link between a
    lander and rovers).

    This heuristic therefore estimates, for each mode that is not yet `full`,
    the minimal number of high-level actions (take_image, sample_soil,
    calibrate / communicate_soil_data) required to make that mode full, using
    the readily-available facts in the current state and static information.
    The heuristic sums these per-mode minima to produce the returned estimate.

    Assumptions
    -----------
    - The important actions for reaching goals are:
      - take_image : creates (equipped_for_rock_analysis waypoint lander) if a
        waypoint is visible and the lander can_traverse rover1.
      - sample_soil : can change which rover a lander can_traverse in one action.
      - calibrate / communicate_soil_data : produce (full mode) if preconditions
        are met (main difference: calibrate requires supports(mode)).
    - We assume that any visible waypoint or any equipped_for_rock_analysis
      fact present in the state can be used (we do not model specific camera/
      objective matching or interference between modes beyond resource
      contention).
    - We do not attempt to be admissible; we aim for a fast, informative,
      cheaply-computed estimate that guides greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - The set of goal facts (task.goals).
    - Static facts relevant to planning decisions:
      - Which modes `supports`.
      - Which modes are initially `equipped_for_soil_analysis`.
      - For which modes we know their `at` rover (mode -> rover).
    These structures are used when computing per-mode costs quickly.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each goal (full mode):
    1. If the goal fact already holds in the state, cost contribution = 0.
    2. Determine the rover to which the mode is attached (from static `at`
       facts). If no such static `at` is known, assume a conservative fallback
       cost (finite).
    3. Inspect the current state for:
       - equipped_for_rock_analysis facts (pairs (waypoint, lander))
       - visible waypoints and visible_from waypoints
       - can_traverse facts (pairs (lander, rover))
       - whether equipped_for_soil_analysis(mode) is present
    4. Compute a small set of plausible ways to get (full mode) and take the
       minimum cost among them. The principal plans considered:
       A) Calibrate route (if supports(mode)):
          - If some equipped_for_rock_analysis exists on a lander already
            having can_traverse to the mode's rover -> cost = 1 (calibrate).
          - If equipped exists on lander but not connected -> cost = 2
            (sample_soil to move lander to the target rover + calibrate).
          - If no equipped exists but a visible waypoint and a lander that can
            traverse rover1 exist -> use take_image (+1) to produce equipped,
            possibly sample_soil to move lander to the mode's rover (+1),
            then calibrate (+1). (Cost 2..3 depending on whether the target
            rover is rover1.)
          - If no lander currently can_traverse rover1 but some lander has a
            can_traverse link to a rover, allow one sample_soil to move that
            lander to rover1 before take_image; this increases the cost
            accordingly (kept finite).
       B) Communicate-soil route (if equipped_for_soil_analysis(mode) exists):
          - Very similar to the calibrate route but the final action is
            communicate_soil_data (cost 1) and it additionally needs
            visible_from for some waypoint (we prefer visible_from in the
            state if present; otherwise we assume it can be produced by other
            domain actions at additional cost — simplified here by a small
            additive cost when missing).
    5. If none of the reasonable construction routes appear applicable with
       the available facts, return a conservative (but finite) higher cost to
       signal the planner that more work is required.
    6. Sum up the per-mode minimal estimates and return the sum.

    Note: The heuristic purposefully abstracts away specific camera/objective
    matching, availability of particular waypoints for individual modes, and
    interactions between different modes competing for the same lander. This
    abstraction keeps the heuristic fast while remaining informative for the
    search.
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Extract static information useful for heuristic decisions.
        self.supports = set()  # modes that support calibration
        self.equipped_for_soil_static = set()  # modes equipped for soil analysis
        self.mode_at = {}  # mode -> rover (from static 'at' facts if present)

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == "supports" and args:
                self.supports.add(args[0])
            elif pred == "equipped_for_soil_analysis" and args:
                self.equipped_for_soil_static.add(args[0])
            elif pred == "at" and len(args) >= 2:
                # could be (at modeX roverY)
                self.mode_at[args[0]] = args[1]

    def __call__(self, node):
        state = node.state

        # 0 for goal states (check requirement from checklist)
        if self.goals <= state:
            return 0

        # Collect dynamic facts from the current state (plus static can_traverse if present there)
        can_traverse = set()  # pairs (lander, rover)
        equipped_for_rock = set()  # pairs (waypoint, lander)
        visible = set()  # waypoints that are visible
        visible_from = set()  # waypoints visible_from
        equipped_for_soil_state = set()  # modes that are equipped for soil analysis

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == "can_traverse" and len(args) >= 2:
                can_traverse.add((args[0], args[1]))
            elif pred == "equipped_for_rock_analysis" and len(args) >= 2:
                equipped_for_rock.add((args[0], args[1]))
            elif pred == "visible" and args:
                visible.add(args[0])
            elif pred == "visible_from" and args:
                visible_from.add(args[0])
            elif pred == "equipped_for_soil_analysis" and args:
                equipped_for_soil_state.add(args[0])

        # also consider modes that were equipped_for_soil statically
        equipped_for_soil_state |= self.equipped_for_soil_static

        # Utility tests
        def any_equipped_for_rover(target_rover):
            """Is there an equipped_for_rock_analysis on a lander that can_traverse target_rover?"""
            for wp, lander in equipped_for_rock:
                if (lander, target_rover) in can_traverse:
                    return True
            return False

        def any_equipped_any_lander():
            return len(equipped_for_rock) > 0

        def some_lander_with_can_traverse_to(rover_name):
            for lander, r in can_traverse:
                if r == rover_name:
                    return True
            return False

        def some_lander_exists():
            return len(can_traverse) > 0

        # Sum minimal costs over all modes that must become full
        total_cost = 0

        # Goals in task.goals might include other predicates; only consider (full mode) goals.
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "full":
                # Not a mode-full goal; ignore (or one could model other goal types).
                continue
            mode = parts[1]

            # If this mode is already full in current state, skip.
            if ("full " + mode) in "".join([]):  # placeholder to keep style; not used
                pass
            if goal in state:
                continue

            # Determine rover for this mode (prefer static mapping)
            target_rover = self.mode_at.get(mode, None)

            # If no knowledge of which rover the mode is on, we still produce a
            # conservative finite estimate (assume 3..5 actions).
            if target_rover is None:
                # conservative guess: need to produce an equipped lander and calibrate/communicate
                total_cost += 4
                continue

            # Evaluate cheapest way to get (full mode)
            best = float("inf")

            # Option A: use calibrate if supports(mode)
            if mode in self.supports:
                # Case A1: already have an equipped lander connected to the target rover
                if any_equipped_for_rover(target_rover):
                    best = min(best, 1)  # calibrate
                # Case A2: have an equipped lander but not connected -> move then calibrate
                elif any_equipped_any_lander():
                    best = min(best, 2)  # sample_soil + calibrate
                else:
                    # Need to create an equipped_for_rock_analysis via take_image
                    # Check whether we have a visible waypoint and a lander that can_traverse rover1
                    has_visible = len(visible) > 0
                    has_lander_rover1 = some_lander_with_can_traverse_to("rover1")
                    if has_visible and has_lander_rover1:
                        # If the lander that can traverse rover1 is already the target rover
                        if target_rover == "rover1":
                            # take_image + calibrate
                            best = min(best, 2)
                        else:
                            # take_image (on lander connected to rover1) + sample_soil to move to target + calibrate
                            best = min(best, 3)
                    elif has_visible and some_lander_exists():
                        # Move some lander to rover1 (sample_soil) + take_image + maybe move to target + calibrate
                        # Costs between 3 and 4; pick 4 conservative
                        best = min(best, 4)
                    else:
                        # No visible waypoint and no lander links known — conservative fallback
                        best = min(best, 5)

            # Option B: use communicate_soil_data if mode is equipped for soil analysis
            if mode in equipped_for_soil_state:
                # communicate_soil_data requires:
                # - equipped_for_soil_analysis(mode)
                # - some equipped_for_rock_analysis on lander that can connect to target rover
                # - visible_from waypoint
                # If visible_from absent, we assume small extra cost to produce it (conservative)
                visible_from_present = len(visible_from) > 0

                if any_equipped_for_rover(target_rover) and visible_from_present:
                    best = min(best, 1)  # communicate_soil_data
                elif any_equipped_any_lander() and visible_from_present:
                    best = min(best, 2)  # move lander + communicate
                elif any_equipped_any_lander() and not visible_from_present:
                    # need extra action(s) to make visible_from (conservatively 1)
                    best = min(best, 3)
                else:
                    # No equipped_for_rock present; try to create it first similarly to calibrate case
                    has_visible = len(visible) > 0
                    has_lander_rover1 = some_lander_with_can_traverse_to("rover1")
                    if has_visible and has_lander_rover1:
                        extra = 0 if visible_from_present else 1
                        if target_rover == "rover1":
                            best = min(best, 2 + extra)  # take_image + communicate (+visible_from)
                        else:
                            best = min(best, 3 + extra)  # take_image + move + communicate (+visible_from)
                    elif has_visible and some_lander_exists():
                        best = min(best, 4)
                    else:
                        best = min(best, 5)

            # If neither supports nor equipped_for_soil present, still provide a finite
            # but larger estimate: assume we must both equip the mode for soil and
            # create/move an equipped lander and then communicate.
            if mode not in self.supports and mode not in equipped_for_soil_state:
                # Conservative fallback value representing multiple required steps
                fallback = 5
                best = min(best, fallback)

            # Safety: if best is still infinity (should not happen), use finite fallback
            if best == float("inf"):
                best = 6

            total_cost += best

        # Ensure finite result
        if total_cost == float("inf"):
            total_cost = 1000

        return int(total_cost)