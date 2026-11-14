from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (aixg4uih) domain.
    The heuristic estimates the number of actions required to achieve all
    goal facts of the form (qxafq0yb <r>) that are missing in the current
    state. It reasons about the typical causal chain that produces a qxafq0yb
    fact in this domain and counts the missing supporting facts and the
    minimal number of actions needed to create them in a relaxed (optimistic)
    manner.

    Assumptions
    -----------
    - Goals are primarily qxafq0yb facts (as in the domain examples).
    - Static facts (task.static) such as waexgd4o, mcoxi8qw, zmomsdb3 are
      reliable mappings and used to select relevant parameter bindings.
    - The heuristic is not required to be admissible; it is optimistic but
      accounts for likely required operator applications.
    - When multiple alternative ways exist to achieve a subgoal, the
      heuristic chooses the cheapest (optimistic) option.
    - If a prerequisite fact is not currently present but there exists an
      operator that can generate it in principle (based on available other
      facts), the heuristic assigns a small finite cost to create it.
      If no plausible producer exists at all (based on static and current
      facts), a larger finite penalty is used, but the heuristic keeps values
      finite for solvable states.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - Goal facts (task.goals)
    - Static mappings:
      * waex_map: map from r -> set of imdah values such that (waexgd4o r imdah)
      * static_mco: set of r for which (mcoxi8qw r) is static
      * static_zm: set of r for which (zmomsdb3 r) is static
    The constructor does not assume which facts are in the initial state;
    it only extracts the static facts needed repeatedly during evaluation.

    Step-By-Step Thinking for Computing Heuristic (per call)
    --------------------------------------------------------
    1. Quick goal check:
       - If all goals are already satisfied in the node.state, return 0.

    2. Index current facts:
       - For the current state, build predicate-based indices for quick lookups:
         * which qrohx31m facts exist (mapping ax -> set(imdah))
         * which dym1mop6 facts exist (set of (c, ax) pairs)
         * which other useful unary facts exist (omcsgizs, okcl3dr0, xx5ljjai,
           g7wohkyk, sl9kfffg, vrips567, usj0xp5t, mcoxi8qw, zmomsdb3)
       - Note: waexgd4o is treated as static (from initialization), but if it
         also appears in the current state it will be considered equivalent.

    3. For each goal object r that does not yet have (qxafq0yb r) in the
       current state, estimate the minimal cost to obtain qxafq0yb r by
       considering the primary two action schemas that produce it:
         - eav722gv (requires: qrohx31m(ax,imdah), dym1mop6(c,ax), mcoxi8qw(r),
           waexgd4o(r,imdah))
         - l6w9j7tw (requires: qrohx31m(ax,imdah), dym1mop6(c,ax), xx5ljjai(c),
           zmomsdb3(r), waexgd4o(r,imdah))
       For each waexgd4o binding (r -> imdah) we compute an optimistic
       minimal cost for both routes and pick the minimum.

    4. Cost components (optimistic estimates):
       - final action cost: 1 (applying eav722gv or l6w9j7tw to create qxafq0yb)
       - qrohx31m cost:
         * 0 if some (qrohx31m ax imdah) is already present
         * 1 if there is at least one qrohx31m for some ax but with different imdah
           (we can use ss17onrb to change it in one action)
         * large (treated as skip) if no qrohx31m exists at all for any ax
       - dym1mop6 cost for chosen ax:
         * 0 if some (dym1mop6 c ax) already exists
         * 1 if (omcsgizs c) exists for some c (we can use nzh0a058 to create it
           if the required qrohx31m for s154x56h is available or can be set)
         * 2 if some okcl3dr0 c exists (we optimistically assume creating omcsgizs
           from okcl3dr0 costs a single action such as aegb5ot3/ppnkvo80, then
           plus nzh0a058)
         * otherwise a small penalty (3) assuming deeper creation steps may be
           required
       - mcoxi8qw / zmomsdb3 requirement:
         * 0 if the respective fact is already present (either in state or
           static facts)
         * 1 if missing (optimistic single action/penalty)
       - xx5ljjai requirement (for l6 route):
         * 0 if (xx5ljjai c) exists for the c chosen
         * 1 if no xx5ljjai but some okcl3dr0 exists (we can produce xx5ljjai
           via ppnkvo80 given okcl3dr0 and other preconditions)
         * 2 otherwise (optimistic penalty)

    5. Combine optimistic components:
       - For each r, we evaluate the minimal optimistic cost across all
         imdah bindings and across both eav and l6 routes. If nothing is
         plausible for any imdah (e.g., no waexgd4o bindings), we assign a
         modest large finite penalty.
       - The heuristic value returned for the state is the sum of the minimal
         costs for all individual unsatisfied qxafq0yb goals. Summation is a
         reasonable approximation assuming independent subgoals (this domain
         often allows independent provisioning per r).

    Notes on design choices
    -----------------------
    - The heuristic intentionally uses optimistic small integer costs for
      producing intermediate facts (0/1/2/3). This keeps computation cheap
      and makes the heuristic informative while remaining finite for
      solvable states.
    - The heuristic focuses on the causal chain relevant to creating qxafq0yb
      facts and ignores many low-level details; this is acceptable because
      the heuristic will be used with greedy best-first search where
      admissibility is not required.
    """

    def __init__(self, task):
        # Goals: a set of fact strings like "(qxafq0yb scvulwhv)"
        self.goals = task.goals
        # Extract static facts into convenient structures
        static = task.static if task.static is not None else frozenset()

        # Map r -> set(imdah) for waexgd4o facts (static mapping linking r to imdah)
        self.waex_map = defaultdict(set)
        # Static mcoxi8qw and zmomsdb3 if present
        self.static_mco = set()
        self.static_zm = set()
        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "waexgd4o" and len(parts) >= 3:
                r = parts[1]
                imdah = parts[2]
                self.waex_map[r].add(imdah)
            elif pred == "mcoxi8qw" and len(parts) >= 2:
                self.static_mco.add(parts[1])
            elif pred == "zmomsdb3" and len(parts) >= 2:
                self.static_zm.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Fast goal check: heuristic is 0 for goal states
        if self.goals <= state:
            return 0

        # Build indices for current state facts
        pred_to_args = defaultdict(list)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            pred_to_args[pred].append(tuple(args))

        # Helper sets from current state (union with static where appropriate)
        qxafq0_present = {args[0] for args in pred_to_args.get("qxafq0yb", [])}
        qroh_map = defaultdict(set)   # ax -> set(imdah)
        for ax, imdah in pred_to_args.get("qrohx31m", []):
            qroh_map[ax].add(imdah)

        dym_pairs = set(pred_to_args.get("dym1mop6", []))  # set of (c, ax)
        omcsgizs_set = {args[0] for args in pred_to_args.get("omcsgizs", [])}
        okcl3dr0_set = {args[0] for args in pred_to_args.get("okcl3dr0", [])}
        xx5_set = {args[0] for args in pred_to_args.get("xx5ljjai", [])}
        g7_set = {args[0] for args in pred_to_args.get("g7wohkyk", [])}
        sl9_set = {args[0] for args in pred_to_args.get("sl9kfffg", [])}
        vrips_set = {args[0] for args in pred_to_args.get("vrips567", [])}
        usj_set = {args[0] for args in pred_to_args.get("usj0xp5t", [])}
        state_mco = {args[0] for args in pred_to_args.get("mcoxi8qw", [])}
        state_zm = {args[0] for args in pred_to_args.get("zmomsdb3", [])}
        # Combine static and state for mco and zm presence checks
        mco_all = self.static_mco | state_mco
        zm_all = self.static_zm | state_zm

        total_cost = 0

        # Extract list of goal objects r (assume goals are unary qxafq0yb facts)
        goal_rs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "qxafq0yb" and len(parts) >= 2:
                goal_rs.append(parts[1])

        # For each goal r that is not yet achieved estimate cost
        for r in goal_rs:
            if r in qxafq0_present:
                continue  # already satisfied

            best_cost_for_r = 999  # large initial

            # Get waexgd4o imdah candidates. Use static waex map; also consider waex in state.
            imdah_candidates = set(self.waex_map.get(r, set()))
            for wa in pred_to_args.get("waexgd4o", []):
                if wa[0] == r:
                    imdah_candidates.add(wa[1])
            # If no waex mapping, the direct operators that create qxafq0yb cannot be applied.
            if not imdah_candidates:
                # No mapping from r to imdah known: assign a small penalty but finite.
                total_cost += 4
                continue

            # Evaluate each waex (imdah) option
            for imdah in imdah_candidates:
                # --------- Evaluate eav722gv route ----------
                # eav requires: qrohx31m(ax,imdah), dym1mop6(c,ax), mcoxi8qw(r), waexgd4o(r,imdah)
                # cost components:
                final_cost = 1  # applying eav/l6 to produce qxafq0yb

                # mco requirement
                mco_penalty = 0 if r in mco_all else 1

                # qroh availability
                axs_with_imdah = [ax for ax, imds in qroh_map.items() if imdah in imds]
                qroh_cost_if_using_ax = {}
                # If we have direct ax with correct imdah, qroh_cost = 0.
                for ax in axs_with_imdah:
                    qroh_cost_if_using_ax[ax] = 0
                # If none directly available but some qroh exists for some ax, we can change it with ss17onrb (cost 1)
                if not axs_with_imdah and len(qroh_map) > 0:
                    for ax in qroh_map.keys():
                        qroh_cost_if_using_ax[ax] = 1

                # If there are no ax candidates at all (no qroh facts in the state), we cannot realistically produce qroh cheaply.
                if not qroh_cost_if_using_ax:
                    # mark eav route as implausible by setting a high cost
                    eav_best = 999
                else:
                    # For each candidate ax pick optimistic dym cost and combine
                    eav_best = 999
                    for ax, qroh_cost in qroh_cost_if_using_ax.items():
                        # dym cost for this ax:
                        # 0 if some (dym1mop6 c ax) exists
                        dym_existing = [c for (c_, ax_) in dym_pairs if ax_ == ax for c in [c_] if ax_ == ax]
                        if dym_existing:
                            dym_cost = 0
                        elif omcsgizs_set:
                            # optimistic: create dym via nzh0a058 if some omcsgizs exists
                            dym_cost = 1
                        elif okcl3dr0_set:
                            # optimistic two-step: create omcsgizs from okcl3dr0 (aegb5ot3/ppnkvo80), then nzh0a058
                            dym_cost = 2
                        else:
                            # deeper construction assumed
                            dym_cost = 3

                        cost = final_cost + mco_penalty + qroh_cost + dym_cost
                        if cost < eav_best:
                            eav_best = cost

                # --------- Evaluate l6w9j7tw route ----------
                # l6 requires: qrohx31m(ax,imdah), dym1mop6(c,ax), xx5ljjai(c), zmomsdb3(r), waexgd4o(r,imdah)
                zm_penalty = 0 if r in zm_all else 1

                # xx5 cost: optimistic based on whether any xx5ljjai exists or okcl3dr0 exists
                def xx5_cost_for_c(c):
                    if c in xx5_set:
                        return 0
                    if okcl3dr0_set:
                        return 1
                    return 2

                # As with eav, consider ax candidates (same qroh options)
                if not qroh_cost_if_using_ax:
                    l6_best = 999
                else:
                    l6_best = 999
                    for ax, qroh_cost in qroh_cost_if_using_ax.items():
                        # If there is an existing dym pair (c,ax), pick the best c w.r.t. xx5 presence.
                        dym_cs = [c for (c_, ax_) in dym_pairs if ax_ == ax for c in [c_]]
                        if dym_cs:
                            # pick the c that gives minimal xx5 cost
                            best_xx5_for_existing = min(xx5_cost_for_c(c) for c in dym_cs)
                            dym_cost = 0
                            xx5_cost = best_xx5_for_existing
                        else:
                            # No dym exists for this ax: we must create it.
                            if omcsgizs_set:
                                # create dym via nzh0a058 for some c (optimistic)
                                dym_cost = 1
                                # choose c to minimize xx5 cost: if some c in okcl3dr0_set exists choose it, else pick any and cost 2
                                possible_cs = list(okcl3dr0_set) if okcl3dr0_set else (list(okcl3dr0_set) + ["<any>"])
                                # if okcl3dr0_set nonempty pick cheapest among them
                                if okcl3dr0_set:
                                    xx5_cost = min(xx5_cost_for_c(c) for c in okcl3dr0_set)
                                else:
                                    # no okcl3dr0: assume xx5 may be created with penalty
                                    xx5_cost = 2
                            elif okcl3dr0_set:
                                # optimistic two-step: create omcsgizs then dym
                                dym_cost = 2
                                xx5_cost = min(xx5_cost_for_c(c) for c in okcl3dr0_set) if okcl3dr0_set else 2
                            else:
                                dym_cost = 3
                                xx5_cost = 2
                        cost = final_cost + zm_penalty + qroh_cost + dym_cost + xx5_cost
                        if cost < l6_best:
                            l6_best = cost

                best_for_imdah = min(eav_best, l6_best)
                if best_for_imdah < best_cost_for_r:
                    best_cost_for_r = best_for_imdah

            # If after evaluating all imdah candidates we still have a very large cost,
            # assign a bounded finite penalty rather than infinity.
            if best_cost_for_r >= 999:
                best_cost_for_r = 6  # finite fallback penalty for tricky/unexpected cases

            total_cost += best_cost_for_r

        # Ensure finite return
        return int(total_cost)