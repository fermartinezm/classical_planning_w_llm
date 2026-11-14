from heuristics.heuristic_base import Heuristic
from itertools import product


def get_parts(fact):
    """Return the tokens of a fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (aixg4uih) domain.
    The domain's goals require achieving facts of the form (qxafq0yb ?z)
    for certain objects ?z of type z65nyce9. The actions that produce qxafq0yb
    require a chain of supporting facts: a dym1mop6(c3,ax) pairing, an appropriate
    qrohx31m(ax,imdah) mapping, and either mcoxi8qw(z) (for eav722gv) or
    zmomsdb3(z) together with xx5ljjai(c3) (for l6w9j7tw). dym1mop6 facts can be
    produced by nzh0a058, which requires omcsgizs(c3) and a specific
    qrohx31m(ax s154x56h). omcsgizs can in turn be produced by aegb5ot3 or
    ppnkvo80 if supporting predicates are present. The heuristic estimates
    the number of actions required to establish for every goal object z a
    supporting chain that enables one of the goal-achieving operators.

    Assumptions
    -----------
    - We assume facts in task.static are always available (static) and
      should be considered along with the dynamic facts in the current state.
    - We treat each action cost as equal (cost 1). The heuristic is not
      admissible; it is optimistic but sensitive to currently available
      supporting facts so it tends to prefer states closer to enabling
      goal-producing actions.
    - When a supporting fact is not present but appears to be producible
      in a single step from existing static/dynamic predicates (e.g.,
      omcsgizs from okcl3dr0 together with some g7wohkyk and sl9kfffg),
      we estimate a cost of 1 for producing it. If producing the fact
      looks harder we assign a slightly larger small constant (2 or 3).
    - The special constant s154x56h (the u59rqa6u constant in the domain)
      is used by the operator nzh0a058 and is assumed present as a token
      in facts (this matches the domain definition).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goals: the set of goal fact strings (as provided by the task).
    - self.static: the static facts supplied by the task.
    - self.waex_map: a mapping from each z65nyce9 object z to the set of
      u59rqa6u objects imdah such that (waexgd4o z imdah) holds (from static).
      (waexgd4o is static in examples and encodes which u59 values pair with z).
    - We do not lock in other facts as static beyond task.static; during
      evaluation we always consider the union of current state and static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal (qxafq0yb z) that is not already true in the state we compute
    a small optimistic estimate of required actions as follows.

    1. Quick success check:
       - If (qxafq0yb z) is already in the (state ∪ static) facts, cost 0.

    2. Identify candidate imdah values:
       - From waex_map[z] (static waexgd4o facts) obtain the candidate u59
         values that can be used with this z in the goal-producing actions.
       - If none exist, we mark this goal as expensive (fallback cost).

    3. Two alternative goal-producing operators:
       - eav722gv: requires
           * dym1mop6(c3, ax)
           * qrohx31m(ax, imdah) for some imdah ∈ waex_map[z]
           * mcoxi8qw(z)
         If all these hold at once, eav722gv directly gives qxafq0yb(z) in 1 step.
       - l6w9j7tw: requires
           * dym1mop6(c3, ax)
           * xx5ljjai(c3)
           * zmomsdb3(r7) where r7 is the z (i.e., zmomsdb3(z))
           * qrohx31m(ax, imdah) and waexgd4o(z,imdah)
         This alternative is considered if mcoxi8qw(z) is missing.

    4. Producing dym1mop6(c3, ax):
       - If already present, cost 0.
       - Otherwise, nzh0a058 can create dym1mop6(c3,ax) if omcsgizs(c3)
         and qrohx31m(ax s154x56h) hold. So we estimate:
           cost_make_omc = 0 if omcsgizs(c3) present,
                           1 if okcl3dr0(c3) present and supporting predicates
                             (g7wohkyk and sl9kfffg) exist somewhere in the
                             (state ∪ static) facts (we can then apply aegb5ot3),
                           2 if okcl3dr0(c3) present but support missing,
                           3 otherwise.
           cost_qroh_to_s154 = 0 if qrohx31m(ax, s154x56h) present,
                               1 if some qrohx31m(ax, other) exists (one ss17onrb
                                 can change it),
                               2 otherwise (we are pessimistic but finite).
           cost_dym = cost_make_omc + cost_qroh_to_s154 + 1 (for nzh0a058).

       - Note: changing qrohx31m(ax, X) to a desired value costs 1 using
         ss17onrb if a qrohx31m for ax exists.

    5. Completing the chain and applying the final operator:
       - For eav path: after obtaining dym1mop6 and qrohx31m(ax,imdah) and
         ensuring mcoxi8qw(z), apply eav722gv (1 action).
       - For l6 path: after obtaining dym1mop6, xx5ljjai(c3), zmomsdb3(z),
         qrohx31m(ax,imdah), apply l6w9j7tw (1 action).
       - We conservatively add the cost for creating xx5ljjai(c3) as:
           0 if already present, 1 if okcl3dr0(c3) present (ppnkvo80 or aegb5ot3),
           2 otherwise.
         zmomsdb3(z) is treated as 0 if present, else assigned 2 (slightly
         pessimistic but finite).

    6. Aggregation:
       - For each goal z we compute the minimal optimistic cost across
         feasible candidate triples (c3, ax, imdah) for both eav and l6
         family of solutions. If no candidate appears feasible we return
         a moderate fallback cost for that goal.
       - The heuristic value is the sum of the per-goal estimated costs.
       - If the entire state already satisfies all goals, return 0.

    Implementation notes
    --------------------
    - Costs are small integers; every operator application is counted as 1.
    - The heuristic is intentionally optimistic (underestimates), but it is
      domain-aware and prefers states where the immediate goal-producing
      operator(s) are close to being applicable. It is efficiently computed
      because it examines only facts present in the (state ∪ static) sets
      and enumerates a limited set of candidate objects observed in these facts.
    """

    def __init__(self, task):
        # goals are strings like "(qxafq0yb scvulwhv)"
        self.goals = task.goals
        # static facts are given as a set of strings by the planner
        self.static = task.static

        # waex_map: z65nyce9 -> set of u59rqa6u values (from waexgd4o facts).
        self.waex_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "waexgd4o" and len(parts) >= 3:
                z_obj = parts[1]
                imd = parts[2]
                self.waex_map.setdefault(z_obj, set()).add(imd)

        # Keep the special u59 constant token name used by nzh0a058 in domain.
        # It's present in domain constants; we assume it's "s154x56h" (as in domain).
        # If it does not appear in static facts, we still use the token since
        # that is required by operators that refer to it.
        self.s154 = "s154x56h"

    def __call__(self, node):
        # If goals already satisfied, return 0.
        state = set(node.state)
        combined = state | set(self.static)

        # Quick goal check: if all goal facts are in combined return 0.
        if self.goals <= combined:
            return 0

        # Collect present predicate sets for quick membership tests.
        def facts_with_pred(pred):
            return {get_parts(f)[1:] for f in combined if get_parts(f)[0] == pred}

        # Helper to check membership of a fact (predicate + args) in combined.
        def has_fact(pred, *args):
            fact = "(" + " ".join((pred,) + args) + ")"
            return fact in combined

        # Build convenient lookup structures from combined facts.
        # qroh_map: ax (s1clodkz) -> set of imdah (u59) such that qrohx31m(ax,imdah)
        qroh_map = {}
        for f in combined:
            parts = get_parts(f)
            if parts[0] == "qrohx31m" and len(parts) >= 3:
                ax = parts[1]
                imd = parts[2]
                qroh_map.setdefault(ax, set()).add(imd)

        # dym_set: set of (c3, ax) pairs present as dym1mop6
        dym_set = {(p[1], p[2]) for f in combined for p in [get_parts(f)] if p[0] == "dym1mop6" and len(p) >= 3}

        # omcsg set, xx5 set, okcl3 set, mcoxi set, zmoms set, g7wohkyk set, sl9kfffg set
        omcsg_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "omcsgizs" and len(get_parts(f)) >= 2}
        xx5_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "xx5ljjai" and len(get_parts(f)) >= 2}
        okcl3_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "okcl3dr0" and len(get_parts(f)) >= 2}
        mcoxi_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "mcoxi8qw" and len(get_parts(f)) >= 2}
        zmoms_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "zmomsdb3" and len(get_parts(f)) >= 2}
        g7_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "g7wohkyk" and len(get_parts(f)) >= 2}
        sl9_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "sl9kfffg" and len(get_parts(f)) >= 2}
        vrips_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "vrips567" and len(get_parts(f)) >= 2}
        usj_set = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "usj0xp5t" and len(get_parts(f)) >= 2}

        # Candidate c3 objects (ggld6jmz) we know about (from okcl3, xx5, omcsg, dym_set)
        candidate_c3 = set()
        candidate_c3 |= okcl3_set | xx5_set | omcsg_set
        candidate_c3 |= {c for (c, a) in dym_set}

        # Candidate ax objects known (from qroh_map and dym_set)
        candidate_ax = set(qroh_map.keys()) | {a for (c, a) in dym_set}

        # Bound for fallback when no feasible chain is found for a goal.
        FALLBACK_PER_GOAL = 10

        total_cost = 0

        # For each goal qxafq0yb z, compute per-goal estimated minimal cost.
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] != "qxafq0yb" or len(parts) < 2:
                # If the goal is not of expected form, skip with small cost
                continue
            z = parts[1]
            # If already satisfied (in combined), skip
            if has_fact("qxafq0yb", z):
                continue

            # Candidate imdah values that pair with z via waexgd4o (static mapping)
            imd_candidates = set()
            if z in self.waex_map:
                imd_candidates |= self.waex_map[z]
            # Also check combined for waexgd4o (just in case it's not static)
            for f in combined:
                p = get_parts(f)
                if p[0] == "waexgd4o" and p[1] == z and len(p) >= 3:
                    imd_candidates.add(p[2])

            if not imd_candidates:
                # No waex mapping known; give fallback cost for this goal.
                total_cost += FALLBACK_PER_GOAL
                continue

            best_for_goal = float("inf")

            # QUICK checks: is there a direct 1-action way via eav722gv?
            # For eav: need (dym1mop6 c3 ax) + (qrohx31m ax imd) + mcoxi8qw(z) + waexgd4o(z,imd)
            for (c3, ax) in dym_set:
                for imd in imd_candidates:
                    if has_fact("qrohx31m", ax, imd) and (z in mcoxi_set):
                        # eav is applicable immediately
                        best_for_goal = min(best_for_goal, 1)
            # QUICK check for l6w9j7tw: need dym1mop6 + xx5ljjai(c3) + zmomsdb3(z) + qrohx31m(ax,imd) + waexgd4o(z,imd)
            for (c3, ax) in dym_set:
                for imd in imd_candidates:
                    if has_fact("qrohx31m", ax, imd) and (c3 in xx5_set) and (z in zmoms_set):
                        best_for_goal = min(best_for_goal, 1)

            # If immediate 1-step solution not found, explore candidate (c3, ax, imd) combinations
            # We enumerate plausible c3 from candidate_c3 and plausible ax from candidate_ax.
            if not candidate_c3:
                candidate_c3 = set(okcl3_set) or set()
            if not candidate_ax:
                candidate_ax = set(qroh_map.keys()) or set()

            # To limit work, if both sets empty, try a small default search: combine observed okcl3 and qroh keys.
            if not candidate_c3 and not candidate_ax:
                # no knowledge; assign fallback
                total_cost += FALLBACK_PER_GOAL
                continue

            # Helper functions to estimate small optimistic costs.

            def cost_qroh_to(ax, target_imd):
                """
                Cost to ensure qrohx31m(ax, target_imd) holds.
                0 if already present in combined,
                1 if ax has other qroh mapping in combined (one ss17onrb to change),
                2 otherwise.
                """
                if has_fact("qrohx31m", ax, target_imd):
                    return 0
                if ax in qroh_map and qroh_map[ax]:
                    return 1
                # If no qroh for this ax, maybe one can be created but that is more expensive.
                return 2

            def cost_make_omc(c3):
                """
                Cost to make omcsgizs(c3).
                0 if already present,
                1 if okcl3dr0(c3) present and there exists supporting g7wohkyk and sl9kfffg
                  (one aegb5ot3 or ppnkvo80 can produce it),
                2 if okcl3dr0(c3) present but support not present,
                3 otherwise.
                """
                if c3 in omcsg_set:
                    return 0
                if c3 in okcl3_set:
                    # If there exists at least one g7 and one sl9 anywhere, we can use aegb5ot3/ppnkvo80.
                    if g7_set and sl9_set:
                        return 1
                    # If vrips/usj present as well it may allow ppnkvo80
                    if (vrips_set and usj_set and g7_set and sl9_set) or (g7_set and sl9_set):
                        return 1
                    return 2
                return 3

            def cost_make_xx5(c3):
                """
                Cost to make xx5ljjai(c3).
                0 if already present,
                1 if okcl3dr0(c3) present and additional supporting facts exist (optimistic),
                2 otherwise.
                """
                if c3 in xx5_set:
                    return 0
                if c3 in okcl3_set:
                    # optimistic: can call ppnkvo80 (requires also vrips567 and usj0xp5t and g7/sl9)
                    if (vrips_set and usj_set and g7_set and sl9_set):
                        return 1
                    # still optimistic, assume 1 if okcl3 present
                    return 1
                return 2

            # Now enumerate candidate triples (limit enumeration size for efficiency)
            # Use observed c3 and ax; also consider pairing c3 from okcl3_set with ax from qroh_map.
            c3_candidates = list(candidate_c3)[:40] + list(okcl3_set - candidate_c3)[:40]
            ax_candidates = list(candidate_ax)[:40] + list(qroh_map.keys() - candidate_ax)[:40]

            # include dym_set pairs preferentially
            for (c3, ax) in list(dym_set)[:40]:
                if c3 not in c3_candidates:
                    c3_candidates.append(c3)
                if ax not in ax_candidates:
                    ax_candidates.append(ax)

            # Evaluate costs for plausible triples
            for c3, ax, imd in product(c3_candidates, ax_candidates, imd_candidates):
                # Estimate cost to obtain dym1mop6(c3,ax)
                if (c3, ax) in dym_set:
                    cost_dym = 0
                else:
                    # To create dym via nzh0a058: need omcsgizs(c3) and qrohx31m(ax, s154)
                    cm = cost_make_omc(c3)
                    cq = cost_qroh_to(ax, self.s154)
                    # 1 for the nzh0a058 action
                    cost_dym = cm + cq + 1

                # Estimate cost to ensure qrohx31m(ax,imd) for target imd (may require ss17onrb)
                cost_q_to_imd = cost_qroh_to(ax, imd)

                # Two paths to get qxafq0yb(z):

                # Path EAV (eav722gv): requires mcoxi8qw(z) as well
                if z in mcoxi_set:
                    # if mcoxi present then eav once dym and qroh(ax,imd) present => 1 action
                    cost_eav = cost_dym + cost_q_to_imd + 1  # +1 for eav action
                    best_for_goal = min(best_for_goal, cost_eav)
                else:
                    # mcoxi missing, prefer l6 path or account for producing mcoxi (rare).
                    # We will consider l6 path next.
                    pass

                # Path L6 (l6w9j7tw): requires xx5ljjai(c3) and zmomsdb3(z)
                # Also requires waexgd4o(z,imd) (we ensured imd in imd_candidates so that's satisfied)
                cost_xx = cost_make_xx5(c3)
                cost_zm = 0 if z in zmoms_set else 2  # assume moderate cost to obtain zmomsdb3 if missing
                cost_l6 = cost_dym + cost_x_to_imd if False else None  # placeholder to avoid NameError

                # compute l6 cost explicitly:
                cost_l6 = cost_dym + cost_xx + cost_zm + cost_q_to_imd + 1  # +1 for l6w9j7tw
                best_for_goal = min(best_for_goal, cost_l6)

                # Additionally consider the possibility to produce mcoxi8qw(z) cheaply and then use eav:
                if z not in mcoxi_set:
                    # Estimate small optimistic cost to produce mcoxi8qw: often in examples mcoxi8qw is static,
                    # but if missing, assume cost 2 to produce it.
                    cost_make_mcoxi = 2
                    cost_eav_via_make = cost_dym + cost_q_to_imd + cost_make_mcoxi + 1
                    best_for_goal = min(best_for_goal, cost_eav_via_make)

            # If no candidate produced finite best, fallback
            if best_for_goal == float("inf"):
                best_for_goal = FALLBACK_PER_GOAL

            # best_for_goal should be >= 1 (we already handled 0-case). Accumulate.
            # But be robust: if best_for_goal < 1 (unlikely), floor to 1.
            if best_for_goal < 1:
                best_for_goal = 1

            total_cost += int(best_for_goal)

        # Ensure finite heuristic: if somehow inf, return sum of fallbacks
        if total_cost == float("inf"):
            return FALLBACK_PER_GOAL * len(self.goals)
        return int(total_cost)