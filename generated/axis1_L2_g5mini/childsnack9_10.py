from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return list of parts of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to achieve all goals
    of the secret_domain (the qxafq0yb facts for certain z65nyce9 objects).
    It reasons in a lightweight, domain-dependent way about the main causal
    requirements to produce qxafq0yb for an object:
      - the final action that achieves qxafq0yb (l6w9j7tw or eav722gv),
      - the dym1mop6 fact linking a ggld6jmz object to a s1clodkz value,
      - the qrohx31m fact linking that s1clodkz to a u59rqa6u value that matches
        the z65nyce9 object's waexgd4o mapping,
      - and one of the alternative support facts (mcoxi8qw on the z65nyce9 or
        the combination zmomsdb3 on the z65nyce9 together with xx5ljjai on the
        ggld6jmz).
    The heuristic is intentionally not admissible — it gives a pragmatic cost
    estimate (small integer weights) for missing prerequisites to guide greedy
    best-first search and reduce node expansions.

    Assumptions
    -----------
    - The planner's goal facts are qxafq0yb <z> (one or more z65nyce9 objects).
    - waexgd4o facts mapping z65nyce9 -> u59rqa6u are considered "bindings" and
      are extracted from the static information in the task (these rarely change).
    - Producing qxafq0yb requires (in the relaxed sense) that:
        * some s1clodkz value (ax) is associated with the same u59rqa6u (imdah)
          as the target z via qrohx31m(ax,imdah) and waexgd4o(z,imdah),
        * some ggld6jmz object c has dym1mop6(c,ax),
        * and either mcoxi8qw(z) holds or (zmomsdb3(z) and xx5ljjai(c)).
    - If a required fact is missing, we estimate a small cost to create it:
      e.g., moving an existing qrohx31m to a different imdah via ss17onrb costs 1,
      creating dym1mop6 is estimated as 1 if omcsgizs exists or a few actions
      otherwise. These are heuristic approximations, not exact minimal counts.

    Heuristic Initialization
    ------------------------
    In the constructor we extract useful static information:
      - self.goals : the set of goal facts (task.goals)
      - self.waex_map : mapping from each z65nyce9 object to the set of u59rqa6u
        values it is associated with via waexgd4o facts found in task.static.
    The constructor also stores nothing else heavy; dynamic facts are read per state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic proceeds as follows for every goal atom
    (qxafq0yb z) that is not yet true in the state:

    1. Quick check: If the goal qxafq0yb(z) is already present, it contributes 0.

    2. Identify candidate imdah values for z via waex_map (static). If none exist
       in the static mapping we assume the mapping is unknown and penalize this
       goal with a moderate fixed cost (unbound mapping).

    3. Collect current facts from the state relevant to the causal chain:
       - all qrohx31m(ax,imdah) facts (map ax -> set(imdah))
       - all dym1mop6(c,ax) facts
       - mcoxi8qw(z), zmomsdb3(z), xx5ljjai(c), omcsgizs(c), okcl3dr0(c),
         g7wohkyk(...), sl9kfffg(...), vrips567(...), usj0xp5t(...)

       These facts determine which prerequisites are already satisfied.

    4. For each candidate imdah and for each plausible (c, ax) pair compute a
       small estimated cost composed of:
         - 1 for the final action that produces qxafq0yb,
         - cost_qroh to obtain qrohx31m(ax,imdah): 0 if present, 1 if ax exists
           with a different imdah (we can change it with ss17onrb), 2 otherwise,
         - cost_dym to obtain dym1mop6(c,ax): 0 if present, 1 if omcsgizs(c)
           and qrohx31m(ax,s154x56h) exist (so nzh0a058 can create it), 3 if
           okcl3dr0(c) exists (we estimate a few ops to make omcsgizs and then
           dym1mop6), 4 otherwise,
         - cost_support to satisfy either mcoxi8qw(z) or (zmomsdb3(z) together
           with xx5ljjai(c)). If mcoxi8qw(z) present cost 0; if zmomsdb3(z) and
           xx5ljjai(c) present cost 0; otherwise small positive costs (1-3)
           reflecting the additional actions needed to obtain those support facts.

       The total candidate cost = 1 + cost_qroh + cost_dym + cost_support.
       We take the minimum candidate cost across choices of imdah and (c,ax).

    5. Sum the per-goal minimal costs to get the heuristic estimate for the
       entire state. This is a (possibly inadmissible) estimate of the number
       of actions required to achieve all qxafq0yb goals.

    Implementation notes
    --------------------
    - The heuristic is intentionally coarse-grained (small integer penalties).
      The aim is to capture the main causal bottlenecks for producing qxafq0yb,
      not to compute an exact minimal action count. This keeps the heuristic
      cheap to compute and effective for guiding greedy best-first search.
    - The heuristic returns 0 if all goals are already satisfied.
    """

    def __init__(self, task):
        # Store goals (a set of goal fact strings)
        self.goals = task.goals

        # Build waex_map from static facts: z65nyce9 -> set of imdah (u59rqa6u)
        self.waex_map = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) >= 3:
                z = parts[1]
                imdah = parts[2]
                self.waex_map.setdefault(z, set()).add(imdah)

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If all goals already satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # Helper collectors from current state
        qroh_map = {}       # ax -> set(imdah)
        dym_set = set()     # set of (c, ax)
        mcoxi_set = set()
        zmoms_set = set()
        xx5_set = set()
        omcsg_set = set()
        okcl_set = set()
        g7_set = set()
        sl9_set = set()
        vrips_set = set()
        usj_set = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "qrohx31m" and len(parts) >= 3:
                ax, imdah = parts[1], parts[2]
                qroh_map.setdefault(ax, set()).add(imdah)
            elif pred == "dym1mop6" and len(parts) >= 3:
                c, ax = parts[1], parts[2]
                dym_set.add((c, ax))
            elif pred == "mcoxi8qw":
                mcoxi_set.add(parts[1])
            elif pred == "zmomsdb3":
                zmoms_set.add(parts[1])
            elif pred == "xx5ljjai":
                xx5_set.add(parts[1])
            elif pred == "omcsgizs":
                omcsg_set.add(parts[1])
            elif pred == "okcl3dr0":
                okcl_set.add(parts[1])
            elif pred == "g7wohkyk":
                g7_set.add(parts[1])
            elif pred == "sl9kfffg":
                sl9_set.add(parts[1])
            elif pred == "vrips567":
                vrips_set.add(parts[1])
            elif pred == "usj0xp5t":
                usj_set.add(parts[1])

        # Collect candidate axes (ax) and gg objects (c) known in the state for iteration.
        candidate_axes = set(qroh_map.keys()) | {ax for (_, ax) in dym_set}
        candidate_cs = {c for (c, _) in dym_set} | set(okcl_set) | set(omcsg_set) | set(xx5_set)

        # Fallback: if no candidates discovered, create small synthetic sets to avoid empty loops
        if not candidate_axes:
            candidate_axes = {"_ax_placeholder"}
        if not candidate_cs:
            candidate_cs = {"_c_placeholder"}

        total_cost = 0

        # For each goal qxafq0yb z that is not in the current state
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "qxafq0yb":
                # We only reason about qxafq0yb targets in this domain-dependent heuristic.
                # Non-matching goals are ignored by this heuristic (they would be handled
                # elsewhere); treat as zero-cost here.
                continue
            z = parts[1]
            if f"(qxafq0yb {z})" in state:
                continue  # already satisfied

            # If static waex mapping absent, penalize
            imdah_candidates = self.waex_map.get(z)
            if not imdah_candidates:
                # No waex mapping known from static info -> assume it's costly to bind.
                # Assign a moderate cost and continue.
                total_cost += 6
                continue

            best_for_z = float("inf")

            # Evaluate candidate combinations (imdah, c, ax)
            for imdah in imdah_candidates:
                for c in candidate_cs:
                    for ax in candidate_axes:
                        # 1) Final action cost (either l6w9j7tw or eav722gv) assumed 1
                        cost = 1

                        # cost to get qrohx31m(ax,imdah)
                        if ax in qroh_map and imdah in qroh_map[ax]:
                            cost_qroh = 0
                        elif ax in qroh_map:
                            # ax exists but with different imdah -> ss17onrb can change imdah
                            cost_qroh = 1
                        else:
                            # ax unknown in qrohx31m facts -> assume more expensive to produce
                            cost_qroh = 2
                        cost += cost_qroh

                        # cost to get dym1mop6(c,ax)
                        if (c, ax) in dym_set:
                            cost_dym = 0
                        else:
                            # If omcsgizs(c) and qrohx31m(ax,s154x56h) -> nzh0a058 can create it
                            if c in omcsg_set and ax in qroh_map and "s154x56h" in qroh_map.get(ax, set()):
                                cost_dym = 1
                            elif c in okcl_set:
                                # okcl3dr0 present: estimating a few actions to produce omcsgizs and then dym1mop6
                                cost_dym = 3
                            else:
                                # expensive fallback
                                cost_dym = 4
                        cost += cost_dym

                        # cost to get support for final action:
                        # Either mcoxi8qw(z) present OR (zmomsdb3(z) and xx5ljjai(c))
                        if z in mcoxi_set:
                            cost_support = 0
                        elif z in zmoms_set and c in xx5_set:
                            cost_support = 0
                        elif z in zmoms_set:
                            # need to create xx5ljjai for some c: estimate 2
                            cost_support = 2
                        else:
                            # If okcl3dr0(c) and g7wohkyk and sl9kfffg exist somewhere, small cost
                            if c in okcl_set and (len(g7_set) > 0) and (len(sl9_set) > 0):
                                cost_support = 2
                            else:
                                # fallback moderate cost
                                cost_support = 3
                        cost += cost_support

                        # If waex mapping exists but qroh requires ss17onrb from some special s154x56h
                        # we already accounted for moving qroh above.

                        if cost < best_for_z:
                            best_for_z = cost

            # Safety: if nothing improved, give a finite fallback cost
            if best_for_z == float("inf"):
                best_for_z = 8

            total_cost += best_for_z

        # Ensure finite integer result
        try:
            total_cost = int(total_cost)
        except Exception:
            total_cost = 9999

        return total_cost