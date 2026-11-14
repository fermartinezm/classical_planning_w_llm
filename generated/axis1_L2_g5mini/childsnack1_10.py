from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()

def match(fact, *pattern):
    """
    Match a fact against a pattern with '*' wildcards.
    Example: match("(at a b)", "at", "*", "b") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (aixg4uih) domain.
    The planning problems in this domain require making certain z65nyce9
    objects true for predicate (qxafq0yb <z>) (these are the goals). Two
    different actions (eav722gv and l6w9j7tw) can add qxafq0yb for a z65nyce9
    object, but both actions share several prerequisites:
      - a waexgd4o relation linking the z65nyce9 object to a u59rqa6u token,
      - a qrohx31m mapping from a s1clodkz to that same u59rqa6u token,
      - a dym1mop6 fact linking a ggld6jmz object to that s1clodkz.
    The domain also contains operators that produce the dym1mop6 facts (nzh0a058),
    operators that produce omcsgizs / xx5ljjai (aegb5ot3 / ppnkvo80), and an
    operator (ss17onrb) that can change the qrohx31m mapping for a given s1clodkz.

    The heuristic estimates the number of actions required to satisfy all
    qxafq0yb goal facts by reasoning about which prerequisites are already
    present in the current state and by estimating the (small) number of
    actions needed to produce missing prerequisites. It is not admissible:
    it prefers optimistic, short repair sequences that often match typical
    problem encodings in this domain.

    Assumptions
    -----------
    - The goal facts are only of the form (qxafq0yb <z>) (as in the provided
      instances).
    - waexgd4o facts linking z65nyce9 objects to u59rqa6u tokens are either
      present in the state or in the static facts. These mappings determine
      which qrohx31m mappings are relevant.
    - qrohx31m facts for s1clodkz objects that appear in the instance can
      be changed by applying ss17onrb (cost ~1 action).
    - omcsgizs and xx5ljjai can often be produced by single actions (aegb5ot3
      or ppnkvo80) when okcl3dr0 and the required resource facts are present.
      When those resource facts are absent the heuristic uses a small extra
      penalty (optimistic) rather than marking the goal unreachable.
    - If a required predicate does not appear anywhere and no domain operator
      can produce it in a reasonable short sequence (according to the simple
      checks), the heuristic will assign a relatively large but finite cost.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts and stores:
    - the list of goal facts (task.goals),
    - the static facts (task.static) for quick checks (notably waexgd4o and
      other static linkages).
    No expensive precomputation is performed.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal (qxafq0yb z):
    1. If the goal fact is already true in the state, cost contribution = 0.
    2. Otherwise, identify the set of u59rqa6u tokens that are linked to z
       via waexgd4o (considering both static facts and state facts). If no
       waexgd4o mapping is present, the goal is considered difficult: assign
       a moderate penalty.
    3. For each such token (imd), look for any s1clodkz `ax` for which the
       state contains a qrohx31m mapping to the same token. If no such mapping
       exists but there exists a qrohx31m for the same `ax` to a different
       token, ss17onrb can change it (cost +1). If qrohx31m for that `ax`
       is completely missing, we treat the case as harder (larger penalty).
    4. For a chosen (c: ggld6jmz, ax: s1clodkz, imd: u59rqa6u) triple there
       are two natural ways to achieve (qxafq0yb z):
       - eav722gv: requires dym1mop6(c,ax), waexgd4o(z,imd), qrohx31m(ax,imd),
         and mcoxi8qw(z). If these hold, eav722gv can be applied directly
         (cost 1). If dym1mop6 is missing it can be produced by nzh0a058,
         but nzh0a058 requires omcsgizs(c) and qrohx31m(ax,s154x56h); creating
         omcsgizs(c) is often possible with aegb5ot3/ppnkvo80 when okcl3dr0(c)
         and resource facts exist.
       - l6w9j7tw: requires dym1mop6(c,ax), xx5ljjai(c), zmomsdb3(z),
         waexgd4o(z,imd), and qrohx31m(ax,imd). If xx5ljjai or dym1mop6 is
         missing they can often be produced with a small number of actions.
    5. The heuristic for a single goal is the minimum optimistic cost across
       feasible (c,ax,imd) choices and across the two action patterns above.
       The heuristic sums these per-goal estimates to produce a global value.
    6. The heuristic always returns 0 on a state that already satisfies all
       qxafq0yb goals. If prerequisites appear unreachable by the simple
       checks above, the heuristic returns a large finite cost instead of
       infinity, ensuring the value is finite for solvable states.

    Notes on cost accounting
    ------------------------
    - Basic action costs: any single operator application is counted as 1.
    - ss17onrb (change qrohx31m mapping) is counted as 1 when needed.
    - nzh0a058 (create dym1mop6 from omcsgizs and qrohx31m(... s154x56h)) is 1.
    - aegb5ot3/ppnkvo80 (create omcsgizs/xx5ljjai) are counted as 1 when their
      simpler preconditions are present; otherwise the heuristic adds a small
      extra penalty to reflect additional required actions.
    """

    def __init__(self, task):
        """Extract goals and static facts for quick checks."""
        self.goals = set(task.goals)
        self.static = set(task.static) if task.static is not None else set()

        # Pre-extract waexgd4o relations from static facts for a fast hint.
        # Map z65nyce9 -> set of u59rqa6u tokens.
        self.static_waex = {}
        for fact in self.static:
            if match(fact, "waexgd4o", "*", "*"):
                z, u = get_parts(fact)[1], get_parts(fact)[2]
                self.static_waex.setdefault(z, set()).add(u)

        # Pre-extract any static mcoxi8qw or zmomsdb3 facts as hints.
        self.static_mcoxi = {get_parts(f)[1] for f in self.static if match(f, "mcoxi8qw", "*")}
        self.static_zmoms = {get_parts(f)[1] for f in self.static if match(f, "zmomsdb3", "*")}

    def __call__(self, node):
        state = set(node.state)  # frozenset -> set for convenience

        # Quick goal-check: if all goals already hold, give 0.
        if self.goals <= state:
            return 0

        # Build fast indices from the current state.
        waex = dict(self.static_waex)  # copy static waex, then add state waex
        for fact in state:
            if match(fact, "waexgd4o", "*", "*"):
                z, u = get_parts(fact)[1], get_parts(fact)[2]
                waex.setdefault(z, set()).add(u)

        # qrohx31m: map s1clodkz -> set of u59 tokens currently mapped
        qrohx = {}
        for fact in state:
            if match(fact, "qrohx31m", "*", "*"):
                ax, u = get_parts(fact)[1], get_parts(fact)[2]
                qrohx.setdefault(ax, set()).add(u)

        # dym1mop6 pairs (c, ax)
        dym = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "dym1mop6", "*", "*")}

        # helper predicate presence sets
        xx_set = {get_parts(f)[1] for f in state if match(f, "xx5ljjai", "*")}
        omcsg_set = {get_parts(f)[1] for f in state if match(f, "omcsgizs", "*")}
        okcl3_set = {get_parts(f)[1] for f in state if match(f, "okcl3dr0", "*")}
        g7_set = {get_parts(f)[1] for f in state if match(f, "g7wohkyk", "*")}
        sl9_set = {get_parts(f)[1] for f in state if match(f, "sl9kfffg", "*")}
        vrips_set = {get_parts(f)[1] for f in state if match(f, "vrips567", "*")}
        usj_set = {get_parts(f)[1] for f in state if match(f, "usj0xp5t", "*")}
        mcoxi_set = {get_parts(f)[1] for f in state if match(f, "mcoxi8qw", "*")}
        zmoms_set = {get_parts(f)[1] for f in state if match(f, "zmomsdb3", "*")}.union(self.static_zmoms)

        # For convenience: list of all s1clodkz that have any qrohx mapping
        qrohx_axs = set(qrohx.keys())

        BIG = 100  # large finite penalty used when something looks hard/unavailable
        total_cost = 0

        # Extract goal objects that are qxafq0yb facts
        goal_zs = []
        for g in self.goals:
            if match(g, "qxafq0yb", "*"):
                goal_zs.append(get_parts(g)[1])
        # If there are other kinds of goals, ignore them (domain-specific heuristic).

        for z in goal_zs:
            if f"(qxafq0yb {z})" in state:
                continue  # already satisfied

            # Gather candidate imd tokens linking z -> imd (from state and static)
            imds = waex.get(z, set())
            if not imds:
                # If no waex mapping to token is available, mark this goal as
                # relatively expensive but finite.
                total_cost += 5
                continue

            best_for_z = BIG

            # For each token and each candidate s1clodkz (ax), compute optimistic costs
            for imd in imds:
                # Find all ax that either already map to imd, or that map to any token
                # (so they could be changed via ss17onrb to imd)
                candidate_axs = set()
                for ax, us in qrohx.items():
                    candidate_axs.add(ax)
                # If there are no qrohx facts at all, creating them is nontrivial;
                # we treat it as expensive.
                if not candidate_axs:
                    # cannot easily get required qrohx mapping -> moderate penalty
                    continue

                for ax in candidate_axs:
                    # cost to ensure qrohx31m(ax, imd)
                    if imd in qrohx.get(ax, set()):
                        cost_qroh = 0
                    elif ax in qrohx:  # can change existing mapping via ss17onrb
                        cost_qroh = 1  # ss17onrb
                    else:
                        # no qrohx info for this ax - pessimistic
                        cost_qroh = 2

                    # Consider multiple candidate ggld6jmz objects c.
                    # We try objects that currently appear in predicates referencing ggld6jmz:
                    # use okcl3dr0, xx5ljjai, omcsgizs, or any dym pairing we already have.
                    candidate_cs = set(okcl3_set) | set(xx_set) | {c for (c, _) in dym} | set(omcsg_set)
                    if not candidate_cs:
                        # no candidate c known; treat as expensive
                        candidate_cs = {None}

                    for c in candidate_cs:
                        # Cost to obtain dym1mop6(c,ax)
                        if c is not None and (c, ax) in dym:
                            cost_dym = 0
                        else:
                            # To create dym we need omcsgizs(c) and qrohx31m(ax, s154x56h),
                            # then nzh0a058.
                            # Check omcsgizs presence
                            if c is None:
                                cost_dym = BIG
                            else:
                                if c in omcsg_set:
                                    # omcsgizs present: ensure qrohx is s154x56h or change it
                                    if f"(qrohx31m {ax} s154x56h)" in state:
                                        cost_dym = 1  # nzh0a058
                                    elif ax in qrohx:
                                        cost_dym = 2  # ss17onrb + nzh0a058
                                    else:
                                        cost_dym = 3  # create qrohx somehow is expensive
                                else:
                                    # omcsgizs not present: try to create it with aegb5ot3 or ppnkvo80
                                    # optimistic check: okcl3dr0(c) present and at least one of g7 & sl9 etc.
                                    if c in okcl3_set:
                                        # Check quick availability of resource facts that those ops often require.
                                        # If they exist, assume creating omcsgizs costs 1; otherwise 2.
                                        if g7_set and sl9_set:
                                            cost_create_omc = 1
                                        else:
                                            cost_create_omc = 2
                                        # After creating omcsgizs, we still need nzh0a058 (and possibly ss17onrb)
                                        if f"(qrohx31m {ax} s154x56h)" in state:
                                            cost_dym = cost_create_omc + 1
                                        elif ax in qrohx:
                                            cost_dym = cost_create_omc + 2
                                        else:
                                            cost_dym = cost_create_omc + 3
                                    else:
                                        # okcl3dr0 missing -> more expensive to produce omcsgizs
                                        cost_dym = 4

                        # Now evaluate two alternative final actions:
                        # Option E: use eav722gv (requires mcoxi8qw(z), dym1mop6(c,ax), qrohx31m(ax,imd), waexgd4o(z,imd))
                        cost_e = BIG
                        if z in mcoxi_set or z in self.static_mcoxi:
                            # feasible in principle
                            cost_e = cost_qroh + cost_dym + 1  # +1 for eav722gv itself
                        # Option L: use l6w9j7tw (requires xx5ljjai(c), dym1mop6(c,ax), zmomsdb3(z), qrohx31m(ax,imd), waex)
                        cost_l = BIG
                        if z in zmoms_set:
                            # check if xx exists
                            if c is not None and c in xx_set:
                                cost_l = cost_qroh + cost_dym + 1  # l6w9j7tw
                            else:
                                # optimistic creation of xx5ljjai via ppnkvo80 if okcl3dr0 and resource facts available
                                if c is None:
                                    cost_l = BIG
                                else:
                                    if c in okcl3_set:
                                        # check detailed resources: vrips567, usj0xp5t, g7wohkyk, sl9kfffg
                                        resources_ok = bool(vrips_set) and bool(usj_set) and bool(g7_set) and bool(sl9_set)
                                        cost_create_xx = 1 if resources_ok else 2
                                        cost_l = cost_qroh + cost_dym + cost_create_xx + 1
                                    else:
                                        cost_l = cost_qroh + cost_dym + 3  # pessimistic

                        # Pick best of the two options for this (c,ax,imd)
                        best_for_z = min(best_for_z, cost_e, cost_l)

            if best_for_z == BIG:
                # If nothing looked feasible, assign a moderate finite penalty.
                total_cost += 8
            else:
                total_cost += best_for_z

        # Ensure heuristic is integer and finite.
        return int(total_cost)