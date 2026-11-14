from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the list of tokens inside a PDDL fact string '(pred a b ...)'."""
    return fact[1:-1].split()

def match(fact, *pattern):
    """Check whether fact (string) matches the wildcard pattern (like fnmatch)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain dependent heuristic for the `rover-2` (secret_domain) domain.
    The goals in this domain are to make certain `mode` objects `full`.
    Two main ways exist to make a mode full:
      - calibrate (requires supports(mode), an equipped_for_rock_analysis fact
        and a suitable can_traverse mapping), or
      - communicate_soil_data (requires equipped_for_soil_analysis(mode),
        an equipped_for_rock_analysis fact, visible_from on a waypoint, and a
        suitable can_traverse mapping).
    The heuristic estimates, for every goal mode that is not yet `full`, a
    small number of actions required to achieve `full` by reasoning about:
      - whether the mode is already full,
      - whether static facts (supports / equipped_for_soil_analysis / at)
        allow calibrate or communicate_soil_data,
      - whether there are already equipped_for_rock_analysis facts or visible
        or visible_from facts in the state,
      - whether can_traverse mappings exist and whether they can be changed
        via one sample_soil action.
    The final heuristic value is the sum of the per-mode minimum estimates.

    Assumptions
    -----------
    - We assume the planner's actions behave as in the domain file:
      * take_image produces equipped_for_rock_analysis from a visible waypoint
        if the chosen lander can be traversed by rover1 (it requires
        (can_traverse <lander> rover1)).
      * sample_soil can transfer the can_traverse mapping of a lander from one
        rover to another in a single action (precondition: current mapping).
      * calibrate and communicate_soil_data produce full mode given their
        respective preconditions.
    - The heuristic is not required to be admissible; it aims to provide a
      cheap, informative estimate to guide greedy best-first search.
    - We conservatively approximate some sub-goal creation costs (e.g. making
      visible_from) with small integer costs if the necessary enabling facts
      (like have_image, empty objective, have_soil_analysis) appear present.
    - If exact preconditions for a path are not present, we still return a
      small finite estimate (we favor having finite estimates over proving
      impossibility), which keeps the heuristic informative while fast.

    Heuristic Initialization
    ------------------------
    In the constructor we extract from the task's static facts:
      - which modes are supported (supports_mode set),
      - which modes are equipped_for_soil_analysis (equipped_soil_modes set),
      - mappings from mode -> rover via (at mode rover) static facts,
      - any communicated_soil_data cameras and communicated_rock_data objectives
        from static (useful for allowing communicate_image_data approximations).
    We also keep the goal list (modes that must be full).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node / state:
    1. Parse the state's facts to find:
       - which modes are already full,
       - all equipped_for_rock_analysis pairs (waypoint, lander),
       - visible waypoints,
       - visible_from waypoints,
       - can_traverse pairs (lander, rover),
       - have_image cameras,
       - empty objectives,
       - have_soil_analysis waypoints.
    2. For each goal mode not yet full compute an estimated minimal cost by
       considering two high-level paths:
       - calibrate path (requires supports(mode) and at(mode, rover)):
           * If an equipped_for_rock_analysis for some lander L already exists
             and (L, rover_of_mode) is in can_traverse -> calibrate costs 1.
           * If equipped_for_rock_analysis exists for L but L cannot be
             traversed by the required rover, one sample_soil can transfer the
             can_traverse mapping -> cost 2 (sample_soil + calibrate).
           * If no equipped_for_rock_analysis exists but a visible waypoint is
             available and there is some lander L with (L, rover1) in
             can_traverse (so take_image is applicable), then:
               - take_image (1) creates equipped_for_rock_analysis(L),
               - possibly 1 sample_soil to fix can_traverse,
               - +1 calibrate.
             So typical costs for this route are 2..3.
           * If visible is not present but we can produce visible cheaply
             (we detect have_image, empty objective and have_soil_analysis on
             some waypoint), we add +1 for creating visible (sample_rock or
             communicate_image_data), then the costs above.
       - communicate_soil_data path (requires equipped_for_soil_analysis(mode),
         at(mode, rover), visible_from waypoint and equipped_for_rock_analysis
         for the same waypoint & traversability):
           * If visible_from and suitably-equipped equipped_for_rock_analysis
             and can_traverse exist -> cost 1.
           * Otherwise we approximate the cost to produce visible_from or to
             produce equipped_for_rock_analysis analogously to the calibrate
             path and sum costs accordingly.
    3. Pick the minimal estimate across considered paths and add this per-mode
       estimate to the total heuristic value.
    4. If a mode cannot be supported by either path according to static facts
       (for instance neither supports(mode) nor equipped_for_soil_analysis(mode)
       is true), the heuristic assigns a small finite fallback cost so the
       heuristic remains finite and search can still proceed (the planner may
       discover impossibility via actual expansion).

    Notes on soundness of the heuristic:
    - The heuristic returns 0 iff all goal `full` facts are present in the
      state (checked directly).
    - The heuristic always returns a finite integer (we use small bounded
      estimates per mode) for solvable states and also for states that look
      incomplete with respect to static facts. This is acceptable because the
      heuristic is not required to be admissible, but it must be efficiently
      computable and informative.
    """

    def __init__(self, task):
        # Goals: expect 'full <mode>' facts in task.goals
        self.goals = task.goals
        # Extract useful static information
        static = task.static

        # Modes that support calibrate
        self.supports_modes = {
            get_parts(fact)[1]
            for fact in static
            if match(fact, "supports", "*")
        }

        # Modes that have equipped_for_soil_analysis (enables communicate_soil_data)
        self.equipped_soil_modes = {
            get_parts(fact)[1]
            for fact in static
            if match(fact, "equipped_for_soil_analysis", "*")
        }

        # Map mode -> rover from static (at mode rover)
        self.mode_at_rover = {}
        for fact in static:
            if match(fact, "at", "*", "*"):
                a, b = get_parts(fact)[1], get_parts(fact)[2]
                # If the first arg is a mode (type info not explicit here), we
                # store it; in the domain instances `at modeX roverY` appears.
                self.mode_at_rover[a] = b

        # Static knowledge of communicated_soil_data cameras and communicated_rock_data objectives
        self.static_communicated_soil = {
            get_parts(fact)[1] for fact in static if match(fact, "communicated_soil_data", "*")
        }
        self.static_communicated_rock = {
            get_parts(fact)[1] for fact in static if match(fact, "communicated_rock_data", "*")
        }

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse state facts into useful sets
        full_modes = {get_parts(f)[1] for f in state if match(f, "full", "*")}
        equipped_rock = set()  # set of (waypoint, lander)
        visible = {get_parts(f)[1] for f in state if match(f, "visible", "*")}
        visible_from = {get_parts(f)[1] for f in state if match(f, "visible_from", "*")}
        can_traverse = set()  # set of (lander, rover)
        have_image = {get_parts(f)[1] for f in state if match(f, "have_image", "*")}
        empty_objectives = {get_parts(f)[1] for f in state if match(f, "empty", "*")}
        have_soil_analysis = {get_parts(f)[1] for f in state if match(f, "have_soil_analysis", "*")}

        for f in state:
            if match(f, "equipped_for_rock_analysis", "*", "*"):
                w, l = get_parts(f)[1], get_parts(f)[2]
                equipped_rock.add((w, l))
            if match(f, "can_traverse", "*", "*"):
                l, r = get_parts(f)[1], get_parts(f)[2]
                can_traverse.add((l, r))

        total_cost = 0

        # Helper to check if any can_traverse exists for a lander
        def any_can_traverse_for_lander(lander):
            return any(l == lander for (l, _) in can_traverse)

        # For each required goal mode compute an estimated minimal cost
        for goal in self.goals:
            if not match(goal, "full", "*"):
                # skip non-full goals (shouldn't happen in given domain)
                continue
            mode = get_parts(goal)[1]
            if mode in full_modes:
                continue  # already satisfied

            best = float("inf")

            # Helper: get rover for mode (if static); if unknown we still allow paths
            rover_for_mode = self.mode_at_rover.get(mode, None)

            # --- Path 1: calibrate (requires supports(mode)) ---
            if mode in self.supports_modes and rover_for_mode is not None:
                # Consider any lander candidate. We try:
                # - existing equipped_for_rock_analysis landers
                # - landers that are usable by rover1 (so take_image can be applied)
                candidate_landers = {l for (_, l) in equipped_rock} | {l for (l, r) in can_traverse if r == "rover1"}

                # If there are no candidate landers, conservatively allow a small finite cost
                if not candidate_landers:
                    # fallback: assume we can create what's needed in 3 actions
                    best = min(best, 3)
                else:
                    for L in candidate_landers:
                        # base cost of calibrate
                        cost = 1
                        # Do we already have equipped_for_rock_analysis for some waypoint with L?
                        have_equip_with_L = any(l == L for (_, l) in equipped_rock)
                        if not have_equip_with_L:
                            # Need to create equipped_for_rock_analysis:
                            # Prefer to use visible waypoint + take_image if possible
                            if visible and ("rover1" in {r for (l, r) in can_traverse if l == L}):
                                # take_image consumes visible and creates equipped_for_rock_analysis
                                cost += 1  # take_image
                            elif have_image and empty_objectives and have_soil_analysis and ("rover1" in {r for (l, r) in can_traverse if l == L}):
                                # We can create visible using sample_rock or communicate_image_data, then take_image
                                cost += 2  # sample_rock/comm_image + take_image
                            else:
                                # fallback: assume creating equip costs 2
                                cost += 2

                        # Now we need (L, rover_for_mode) in can_traverse or to create it
                        if (L, rover_for_mode) in can_traverse:
                            transfer_cost = 0
                        elif any_can_traverse_for_lander(L):
                            # change mapping via one sample_soil action
                            transfer_cost = 1
                        else:
                            # If no mapping exists for L, but we can create one from rover1 possibly
                            # (take_image needs (L, rover1) for creating equip). If we would create that mapping,
                            # we count an extra action. Be conservative and add 1.
                            transfer_cost = 1
                        cost += transfer_cost
                        best = min(best, cost)

            # --- Path 2: communicate_soil_data (requires equipped_for_soil_analysis(mode)) ---
            if mode in self.equipped_soil_modes and rover_for_mode is not None:
                # communicate_soil_data needs: equipped_for_soil_analysis(mode) [static],
                # equipped_for_rock_analysis(waypoint, lander) , at(mode rover) [static],
                # visible_from waypoint, can_traverse(lander, rover)
                # We try to find any waypoint W and lander L that can satisfy these.
                candidate_waypoints = set()
                # any waypoint that is visible_from already
                candidate_waypoints |= visible_from
                # any waypoint that is visible (we can turn visible -> visible_from with an extra action in some cases)
                candidate_waypoints |= visible
                # any waypoint that has have_soil_analysis might help in producing visible/visible_from
                candidate_waypoints |= have_soil_analysis

                if not candidate_waypoints:
                    # fallback small estimate
                    best = min(best, 3)
                else:
                    for W in candidate_waypoints:
                        # check if we already have equipped_for_rock_analysis for W
                        equip_pairs_for_W = [ (w,l) for (w,l) in equipped_rock if w == W ]
                        # We'll pick some lander L (if existing equip) or any lander candidate
                        lander_candidates = {l for (_, l) in equip_pairs_for_W}
                        # also consider landers that can be used by take_image (those with can_traverse to rover1)
                        lander_candidates |= {l for (l, r) in can_traverse if r == "rover1"}

                        if not lander_candidates:
                            # fallback estimate
                            best = min(best, 3)
                        else:
                            for L in lander_candidates:
                                cost = 1  # base communicate_soil_data action
                                # Need visible_from W:
                                if W in visible_from:
                                    vf_cost = 0
                                else:
                                    # If visible exists we may need one action to create visible_from
                                    # (approximate via communicate_image_data if conditions hold), otherwise 1
                                    if W in visible:
                                        vf_cost = 1
                                    elif (have_image and empty_objectives and W in have_soil_analysis
                                          and self.static_communicated_soil and self.static_communicated_rock):
                                        # we can approximate creating visible_from via communicate_image_data
                                        vf_cost = 1
                                    else:
                                        vf_cost = 1
                                cost += vf_cost

                                # Need equipped_for_rock_analysis(W, L)
                                if any(w == W and l == L for (w, l) in equipped_rock):
                                    equip_cost = 0
                                else:
                                    # Create equip via take_image (needs visible and (L, rover1) can_traverse)
                                    if W in visible and ("rover1" in {r for (l, r) in can_traverse if l == L}):
                                        equip_cost = 1  # take_image
                                    elif have_image and empty_objectives and W in have_soil_analysis and ("rover1" in {r for (l, r) in can_traverse if l == L}):
                                        equip_cost = 2  # sample_rock/comm_image + take_image
                                    else:
                                        equip_cost = 2
                                cost += equip_cost

                                # Need (L, rover_for_mode) in can_traverse or create via sample_soil
                                if (L, rover_for_mode) in can_traverse:
                                    transfer_cost = 0
                                elif any_can_traverse_for_lander(L):
                                    transfer_cost = 1
                                else:
                                    transfer_cost = 1
                                cost += transfer_cost

                                best = min(best, cost)

            # If neither path gave a finite estimate (shouldn't happen due to fallbacks),
            # use a small finite fallback cost to keep heuristic finite.
            if best == float("inf"):
                best = 4

            total_cost += int(best)

        # Ensure heuristic returns 0 for goal states (double-check)
        if self.goals <= state:
            return 0

        return int(total_cost)