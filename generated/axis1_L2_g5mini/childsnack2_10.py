from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove outer parentheses and split a PDDL fact string into parts.
    Example: '(qrohx31m tktj8rhk s154x56h)' -> ['qrohx31m', 'tktj8rhk', 's154x56h']
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (aixg4uih). The domain's goals
    are atoms of the form (qxafq0yb <z65nyce9-object>).
    The heuristic estimates how many actions are required to make each goal-fact true.
    It reasons about the two operator templates that can produce qxafq0yb:
      - l6w9j7tw  (requires: dym1mop6, xx5ljjai, zmomsdb3, waexgd4o, qrohx31m)
      - eav722gv  (requires: qrohx31m, dym1mop6, mcoxi8qw, waexgd4o)
    The heuristic tries to cheaply estimate the minimum actions needed to fulfill
    missing preconditions (e.g., producing dym1mop6 via nzh0a058 which itself may require
    omcsgizs, and omcsgizs may be produced by ppnkvo80 or aegb5ot3). It uses available
    static facts and the current state's facts to compute inexpensive approximations.

    Assumptions
    -----------
    - Static facts provided in task.static are true in every state (e.g., many waexgd4o
      and mcoxi8qw entries).
    - ss17onrb can change a qrohx31m mapping for some s1clodkz object in one action,
      provided that qrohx31m entry for that s1clodkz exists in the current state.
      The heuristic accounts for this by allowing a cost of 1 to set qrohx31m if needed.
    - If a specific precondition cannot be produced in one action but appears producible
      by a short sequence, the heuristic approximates costs conservatively with small
      integer values (1-3). This keeps computation fast.
    - The heuristic may overcount prerequisites that multiple goals could share; this
      is acceptable because the heuristic is not required to be admissible. We favor
      cheap, informative estimates that guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores useful static relations from the task:
    - waex_map: mapping from z65nyce9 object -> set of u59rqa6u (imdah) that are statically
      associated via (waexgd4o <z> <u>).
    - static_mco: set of z65nyce9 objects that have static (mcoxi8qw <z>).
    - static_zmoms: set of z65nyce9 objects that have static (zmomsdb3 <z>).
    - It also stores the goal facts for quick membership checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal atom (qxafq0yb R) missing in the current state:
      1. If the goal atom is already true, cost 0 for that goal.
      2. Otherwise attempt to estimate minimum cost using two alternative producing actions:
         - Route E (eav722gv): produces qxafq0yb(R) if:
             a) some imdah exists with waexgd4o(R, imdah),
             b) there exists axwhj with qrohx31m(axwhj, imdah),
             c) there exists dym1mop6(C, axwhj),
             d) mcoxi8qw(R) holds.
           Estimate missing pieces:
             - qrohx31m(axwhj, imdah): cost 0 if present; else cost 1 if we can change an existing
               qrohx31m mapping via ss17onrb (we check whether there is any qrohx31m fact in state
               and charge 1 to retarget it).
             - dym1mop6(C, ax): cost 0 if present; else 1 if omcsgizs(C) present and qrohx31m(ax, s154x56h)
               present (then nzh0a058 creates dym1mop6); if qrohx31m(ax, s154x56h) missing but a qrohx31m
               for some ax exists, cost 2 (ss17onrb + nzh0a058). If omcsgizs is also absent but
               producible by ppnkvo80/aegb5ot3 in one action, cost increases appropriately (1-2).
             - mcoxi8qw(R): cost 0 if present in state or static, otherwise treat route as expensive.
           Total route cost = estimated_precondition_cost + 1 (for eav722gv action).
         - Route L (l6w9j7tw): produces qxafq0yb(R) if:
             a) waexgd4o(R, imdah) and qrohx31m(axwhj, imdah),
             b) dym1mop6(C, axwhj),
             c) xx5ljjai(C) and zmomsdb3(R).
           Estimate costs for missing pieces similarly to Route E, but include cost to obtain xx5ljjai(C)
           (ppnkvo80 may produce xx5ljjai and omcsgizs together).
           Total route cost = estimated_precondition_cost + 1 (for l6w9j7tw action).
      3. If neither route is realistically available (e.g., no waexgd4o mapping exists at all), assign
         a finite large penalty (INF) to reflect difficulty.
      4. The heuristic cost for the problem state is the sum of the minimal estimated costs for each
         missing goal atom. (This is admissible-violating in general but informative and cheap to compute.)
    """

    def __init__(self, task):
        # store goals and static facts for quick access
        self.goals = task.goals
        # static facts (usually never changed by actions)
        static = task.static

        # Map waexgd4o relations (z65nyce9 -> set of u59rqa6u)
        self.waex_map = {}
        self.static_mco = set()
        self.static_zmoms = set()
        # Also record which s1clodkz have some static qrohx31m mapping to any u (useful to allow ss17onrb)
        self.static_qroh_any = set()

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o":
                z, u = parts[1], parts[2]
                self.waex_map.setdefault(z, set()).add(u)
            elif parts[0] == "mcoxi8qw":
                self.static_mco.add(parts[1])
            elif parts[0] == "zmomsdb3":
                self.static_zmoms.add(parts[1])
            elif parts[0] == "qrohx31m":
                # qrohx31m <axwhj> <imdah>
                self.static_qroh_any.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick goal-check: return 0 when all goals are satisfied.
        if self.goals <= state:
            return 0

        # Build indices for fast membership tests in current state.
        # We allow the state facts to override/augment static facts.
        qroh_map = {}           # axwhj -> set(imdah)
        dym_set = set()         # (c3xfw, axwhj)
        xx_set = set()          # c3xfw with xx5ljjai
        omc_set = set()         # c3xfw with omcsgizs
        mco_set = set()         # r with mcoxi8qw in current state
        zmoms_set = set()       # r with zmomsdb3 in current state
        waex_current = {}       # r -> set(imdah) from state facts in addition to static
        any_qroh_in_state = set()  # s1clodkz that have any qrohx31m mapping in state

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "qrohx31m":
                ax, u = parts[1], parts[2]
                qroh_map.setdefault(ax, set()).add(u)
                any_qroh_in_state.add(ax)
            elif pred == "dym1mop6":
                c3, ax = parts[1], parts[2]
                dym_set.add((c3, ax))
            elif pred == "xx5ljjai":
                xx_set.add(parts[1])
            elif pred == "omcsgizs":
                omc_set.add(parts[1])
            elif pred == "mcoxi8qw":
                mco_set.add(parts[1])
            elif pred == "zmomsdb3":
                zmoms_set.add(parts[1])
            elif pred == "waexgd4o":
                r, imd = parts[1], parts[2]
                waex_current.setdefault(r, set()).add(imd)

        # Combine static and current waex maps
        combined_waex = {}
        for r, s in self.waex_map.items():
            combined_waex.setdefault(r, set()).update(s)
        for r, s in waex_current.items():
            combined_waex.setdefault(r, set()).update(s)

        # combined qroh map: include static qrohx31m entries from static (if any)
        # Note: static qroh entries were recorded only as existence of some ax in constructor.
        # We prefer to check actual qroh entries in state; for the ability to retarget via ss17onrb,
        # it's enough to know whether any qroh exists in the state at all.
        any_qroh_exists = len(any_qroh_in_state) > 0 or len(self.static_qroh_any) > 0

        # Helper to quickly test if qxafq0yb r in state
        def goal_satisfied(r):
            return ("(qxafq0yb " + r + ")") in state

        # Cost constants and INF
        INF = 100  # finite large penalty

        # Precompute sets for quick membership
        # Note: also include static mco and zmoms knowledge
        mco_combined = mco_set | self.static_mco
        zmoms_combined = zmoms_set | self.static_zmoms

        total = 0

        # For each goal atom, compute an estimated minimal cost
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "qxafq0yb":
                # This domain's goals are expected to be of this form, but be robust.
                if goal in state:
                    continue
                else:
                    # assign a small-cost fallback
                    total += 1
                    continue

            r = parts[1]
            if goal in state:
                continue  # already satisfied

            # If no waex mapping exists at all for r (neither static nor state), it's hard to achieve
            if r not in combined_waex:
                total += INF
                continue

            # For each candidate imdah associated with r, compute two route costs and take min
            best_for_r = INF
            for imdah in combined_waex[r]:
                # Determine cost to have qrohx31m(ax, imdah) for some ax
                # If there exists any ax with that mapping in the state -> cost 0
                qroh_present_for_imd = False
                for ax, imds in qroh_map.items():
                    if imdah in imds:
                        qroh_present_for_imd = True
                        break
                if qroh_present_for_imd:
                    cost_qroh = 0
                else:
                    # We can try to retarget an existing qroh via ss17onrb if any qroh exists
                    if any_qroh_exists:
                        cost_qroh = 1
                    else:
                        # No qroh facts at all in state and none statically known -> expensive
                        cost_qroh = 3

                # Route E (eav722gv) requirements:
                # - qrohx31m(ax, imdah) -> cost_qroh
                # - dym1mop6(c3, ax)
                # - mcoxi8qw(r) -> must be present in combined static/state
                if r in mco_combined:
                    cost_mco = 0
                else:
                    cost_mco = INF  # route E effectively blocked without mcoxi8qw

                # Now estimate cost to obtain a matching dym1mop6 for some (c3, ax)
                # If any dym1mop6 exists that uses an ax that can be used (i.e., ax with qroh possibility),
                # then cost 0. Otherwise estimate cost to create dym1mop6 via nzh0a058:
                # nzh0a058 requires omcsgizs(c3) and qrohx31m(ax, s154x56h).
                cost_dym_best = INF
                # Check if any existing dym1mop6 can be used without modifications
                for (c3, ax) in dym_set:
                    # this dym1mop6 uses ax; we need qrohx31m(ax, imdah) to also hold for eav or l6
                    # but qrohx31m(ax, imdah) can be created with cost_qroh (possibly by retargeting)
                    cost_dym_best = 0
                    break

                if cost_dym_best != 0:
                    # No existing dym1mop6 present; try to see if nzh0a058 can be applied cheaply:
                    # nzh0a058 preconditions: omcsgizs(c3) and qrohx31m(ax, s154x56h)
                    # Check if there is any omcsgizs in state (cost 0) or producible in 1 step (ppnkvo80/aegb5ot3)
                    omc_present = len(omc_set) > 0
                    omc_producible_in_1 = False
                    # ppnkvo80 preconds: vrips567 ?, usj0xp5t ?, g7wohkyk ?, sl9kfffg ?, okcl3dr0 c3
                    # We do not have fine-grained static for those in constructor, but we can check current state:
                    # If any c3 exists with okcl3dr0 and there exists at least one vrips567 and one usj0xp5t and g7wohkyk and sl9kfffg
                    # then ppnkvo80 is applicable for some combo.
                    have_vrips = any(get_parts(f)[0] == "vrips567" for f in state)
                    have_usj0xp5t = any(get_parts(f)[0] == "usj0xp5t" for f in state)
                    have_g7woh = any(get_parts(f)[0] == "g7wohkyk" for f in state)
                    have_sl9 = any(get_parts(f)[0] == "sl9kfffg" for f in state)
                    have_okcl = any(get_parts(f)[0] == "okcl3dr0" for f in state)
                    if omc_present:
                        omc_producible_in_1 = False
                    else:
                        if have_vrips and have_usj0xp5t and have_g7woh and have_sl9 and have_okcl:
                            omc_producible_in_1 = True
                        else:
                            # aegb5ot3 requires g7wohkyk, sl9kfffg and okcl3dr0 -> similar availability check:
                            if have_g7woh and have_sl9 and have_okcl:
                                omc_producible_in_1 = True

                    # Check if qrohx31m(ax, s154x56h) exists already; if not it can be obtained via ss17onrb (cost 1)
                    qroh_has_s154 = False
                    for ax, imds in qroh_map.items():
                        if "s154x56h" in imds:
                            qroh_has_s154 = True
                            break

                    if omc_present and qroh_has_s154:
                        cost_dym_best = 1  # one nzh0a058
                    elif omc_present and not qroh_has_s154:
                        # need one ss17onrb + nzh0a058
                        if any_qroh_exists:
                            cost_dym_best = 2
                        else:
                            cost_dym_best = 3
                    elif omc_producible_in_1 and qroh_has_s154:
                        # ppnkvo80/aegb5ot3 -> produces omcsgizs in 1, then nzh0a058 -> total 2
                        cost_dym_best = 2
                    elif omc_producible_in_1:
                        # need ppnkvo80/aegb5ot3 + (maybe) ss17onrb + nzh0a058 => estimate 3
                        cost_dym_best = 3
                    else:
                        # fallback conservative estimate
                        cost_dym_best = 3

                # Route E cost estimate
                route_e_cost = cost_qroh + cost_dym_best + cost_mco
                if route_e_cost < INF:
                    route_e_cost += 1  # eav722gv action itself

                # Route L (l6w9j7tw) requirements:
                # - same qrohx31m(ax, imdah) (cost_qroh)
                # - dym1mop6(c3, ax) (cost_dym_best)
                # - xx5ljjai(c3) present for same c3 as dym or producible
                # - zmomsdb3(r) must be present (static or state)
                if r in zmoms_combined:
                    cost_zmoms = 0
                else:
                    cost_zmoms = INF  # l6 impossible without zmomsdb3 for r

                # Estimate cost to obtain xx5ljjai for some c3. If any xx exists -> 0
                if len(xx_set) > 0:
                    cost_xx = 0
                else:
                    # ppnkvo80 produces xx5ljjai and omcsgizs if its preconditions satisfied in state
                    if have_vrips and have_usj0xp5t and have_g7woh and have_sl9 and have_okcl:
                        cost_xx = 1
                    else:
                        cost_xx = 2  # conservative estimate: can be produced but may need extra actions

                route_l_cost = cost_qroh + cost_dym_best + cost_xx + cost_zmoms
                if route_l_cost < INF:
                    route_l_cost += 1  # l6w9j7tw action itself

                best_route = min(route_e_cost, route_l_cost)
                if best_route < best_for_r:
                    best_for_r = best_route

            # Accumulate best estimate for this goal (bounded)
            # Ensure finite by capping INF
            total += min(best_for_r, INF)

        # Ensure heuristic is 0 iff all goals satisfied (we handled earlier),
        # and return finite numeric estimate.
        return int(total)