from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """
    Remove surrounding parentheses and split a PDDL fact string into its parts.
    Example: "(qrohx31m tktj8rhk s154x56h)" -> ["qrohx31m", "tktj8rhk", "s154x56h"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored for the secret_domain (aixg4uih) domain.
    The primary goals in this domain are facts of the form (qxafq0yb <z65nyce9>).
    Achieving such a goal normally requires a chain of supporting facts, typically:
      - a waexgd4o linking the z65nyce9 object to some u59rqa6u (imdah),
      - a qrohx31m fact pairing a s1clodkz (axwhj) with that u59rqa6u,
      - a dym1mop6 fact that ties a ggld6jmz object (c3xfw) to that axwhj,
      - and finally an action (either l6w9j7tw or eav722gv) that produces qxafq0yb.
    There are multiple ways to obtain intermediate facts (e.g., omcsgizs -> nzh0a058 -> dym1mop6,
    or using ppnkvo80/aegb5ot3 to create omcsgizs/xx5ljjai). This heuristic makes a compact,
    efficiently computable estimate of the number of actions required to create the missing
    qxafq0yb facts by reasoning about the existence (or ease of creation) of those required
    supporting predicates.

    Assumptions
    -----------
    - The heuristic is intended for greedy best-first search guidance and is not required to be admissible.
    - It treats facts that are currently true in the state (and static facts) as free.
    - For missing intermediate facts, it estimates costs using cheapest available operator patterns:
        * ss17onrb can "retarget" an existing qrohx31m for an axwhj to a different u59rqa6u in 1 action.
        * nzh0a058 creates dym1mop6 if omcsgizs exists for the relevant ggld6jmz and qrohx31m(axwhj, s154x56h) holds.
        * aegb5ot3 or ppnkvo80 can create omcsgizs if the required supporting facts (okcl3dr0, g7wohkyk, sl9kfffg and possibly vrips567/usj0xp5t) exist; ppnkvo80 also creates xx5ljjai.
    - If crucial static links (e.g., waexgd4o mapping for a goal z65nyce9) are missing, the heuristic returns a large estimate
      (treat as hard / unlikely to be solvable from that branch) but still finite.

    Heuristic Initialization
    ------------------------
    The constructor extracts and indexes:
      - The set of goal facts (task.goals).
      - Static facts (task.static) and all facts that may be produced/used as prerequisites.
    Indexed structures:
      - waex_map: z65nyce9 -> set of u59rqa6u (imdah) (waexgd4o facts)
      - qroh_map: axwhj (s1clodkz) -> set of imdah (u59rqa6u) (qrohx31m facts)
      - dym_pairs: set of (c3xfw, axwhj) (dym1mop6 facts)
      - omc_set: set of ggld6jmz with omcsgizs
      - xx_set: set of ggld6jmz with xx5ljjai
      - mco_set, zmoms_set: sets of z65nyce9 with mcoxi8qw or zmomsdb3
      - precondition-availability booleans for creating omcsgizs or xx5ljjai:
          * aegb5ot3_possible: true if there is at least one okcl3dr0, one g7wohkyk and one sl9kfffg
          * ppnkvo80_possible: true if the extended set needed by ppnkvo80 (vrips567, usj0xp5t, g7wohkyk, sl9kfffg, okcl3dr0) is present
    These structures include both static and (initial) dynamic facts as they are received through `task.static` and later
    state observations; during evaluation we use the given node.state to see current dynamic facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic follows these steps:

    1. Return 0 if the entire goal set is already satisfied by the state (goal test).

    2. Index the facts currently true in the state into the same structures as in initialization
       (so we can reason about which supporting facts already exist).

    3. For each missing goal qxafq0yb(z):
       - If qxafq0yb(z) already in the state: cost contribution 0.
       - Otherwise consider the two main action patterns that can produce qxafq0yb:
         (a) eav722gv: requires mcoxi8qw(z), waexgd4o(z, imdah), dym1mop6(c3xfw, axwhj),
                       and qrohx31m(axwhj, imdah).
         (b) l6w9j7tw: requires zmomsdb3(z), waexgd4o(z, imdah), dym1mop6(c3xfw, axwhj),
                       qrohx31m(axwhj, imdah) and xx5ljjai(c3xfw).
       - For each waexgd4o(imdah) candidate for z (the waex mapping is static-like, if none exist we mark this goal as hard -> large cost):
         * Evaluate minimal cost over choices of axwhj (we iterate over known axwhj keys that appear in qroh or dym facts).
         * For a chosen axwhj:
            - qroh cost: 0 if qroh(axwhj, imdah) already holds; else 1 if some qroh for that axwhj exists and can be retargeted by ss17onrb; otherwise infeasible.
            - dym cost: 0 if a dym1mop6 linking some c3xfw to that axwhj exists already.
                       Else, one can create dym1mop6 using nzh0a058 which requires omcsgizs(c3xfw) and qroh(axwhj, s154x56h):
                       * cost to make qroh(axwhj, s154x56h) is 0 if already present, else 1 (ss17onrb).
                       * cost to have omcsgizs is 0 if present already; else 1 if we can create it via aegb5ot3 or ppnkvo80 (checked via the presence of their required supporting facts).
                       * plus 1 for the nzh0a058 action itself.
            - xx cost (for l6 path): 0 if xx5ljjai is present for the chosen c3xfw, else 1 if ppnkvo80 is possible (since ppnkvo80 produces xx5ljjai), otherwise infeasible.
         * Combine these costs plus 1 for the final action (eav722gv or l6w9j7tw) and choose minimal over axwhj and imdah.
       - The cost contribution for the goal is the minimal feasible estimate across all considered patterns (eav/l6) and waex mappings.

    4. The heuristic value returned is the sum of the individual estimated costs for all missing qxafq0yb goals.
       This gives a role-based, domain-informed estimate that remains efficiently computable (index lookups and
       small nested iterations over relevant argument sets), biases the search toward states that already satisfy
       many prerequisites, and prefers shorter operator chains (e.g., using existing dym1mop6 or omcsgizs where available).

    Notes on robustness
    -------------------
    - If a necessary supporting fact appears impossible to obtain under available operators (according to the
      limited checks performed), the heuristic assigns a large finite penalty (100) as a pragmatic indicator of
      difficulty but remains finite to keep search algorithms stable.
    - The heuristic ignores delete effects in its cost accounting (it is not admissible) and treats creation costs
      as small integers representing operator counts (1 per action in the chain).
    """

    def __init__(self, task):
        # Goals and static facts (static facts are always present)
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build static indices for quick checks (these will be combined with state facts at call time)
        self.static_waex = {}  # z65nyce9 -> set(imdah)
        self.static_qroh = {}  # axwhj -> set(imdah)
        self.static_dym = set()  # (c3xfw, axwhj)
        self.static_omc = set()  # set of c3xfw
        self.static_xx = set()  # set of c3xfw
        self.static_mco = set()  # z65nyce9
        self.static_zmoms = set()  # z65nyce9
        self.static_okcl3dr0 = set()
        self.static_sl9kfffg = set()
        self.static_g7wohkyk = set()
        self.static_vrips567 = set()
        self.static_usj0xp5t = set()

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "waexgd4o":
                r, imdah = parts[1], parts[2]
                self.static_waex.setdefault(r, set()).add(imdah)
            elif pred == "qrohx31m":
                ax, imdah = parts[1], parts[2]
                self.static_qroh.setdefault(ax, set()).add(imdah)
            elif pred == "dym1mop6":
                c3, ax = parts[1], parts[2]
                self.static_dym.add((c3, ax))
            elif pred == "omcsgizs":
                self.static_omc.add(parts[1])
            elif pred == "xx5ljjai":
                self.static_xx.add(parts[1])
            elif pred == "mcoxi8qw":
                self.static_mco.add(parts[1])
            elif pred == "zmomsdb3":
                self.static_zmoms.add(parts[1])
            elif pred == "okcl3dr0":
                self.static_okcl3dr0.add(parts[1])
            elif pred == "sl9kfffg":
                self.static_sl9kfffg.add(parts[1])
            elif pred == "g7wohkyk":
                self.static_g7wohkyk.add(parts[1])
            elif pred == "vrips567":
                self.static_vrips567.add(parts[1])
            elif pred == "usj0xp5t":
                self.static_usj0xp5t.add(parts[1])

    def __call__(self, node):
        # Quick goal-check: heuristic 0 if goals already satisfied
        state = set(node.state)
        if self.goals <= state:
            return 0

        # Merge static + current state facts into indices for current reasoning
        waex = {}  # z65nyce9 -> set(imdah)
        qroh = {}  # axwhj -> set(imdah)
        dym = set()  # (c3xfw, axwhj)
        omc = set()
        xx = set()
        mco = set()
        zmoms = set()
        okcl3dr0 = set()
        sl9kfffg = set()
        g7wohkyk = set()
        vrips567 = set()
        usj0xp5t = set()

        # initialize with static information
        for k, v in self.static_waex.items():
            waex.setdefault(k, set()).update(v)
        for k, v in self.static_qroh.items():
            qroh.setdefault(k, set()).update(v)
        dym |= self.static_dym
        omc |= self.static_omc
        xx |= self.static_xx
        mco |= self.static_mco
        zmoms |= self.static_zmoms
        okcl3dr0 |= self.static_okcl3dr0
        sl9kfffg |= self.static_sl9kfffg
        g7wohkyk |= self.static_g7wohkyk
        vrips567 |= self.static_vrips567
        usj0xp5t |= self.static_usj0xp5t

        # incorporate current state facts
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "waexgd4o":
                r, imdah = parts[1], parts[2]
                waex.setdefault(r, set()).add(imdah)
            elif pred == "qrohx31m":
                ax, imdah = parts[1], parts[2]
                qroh.setdefault(ax, set()).add(imdah)
            elif pred == "dym1mop6":
                c3, ax = parts[1], parts[2]
                dym.add((c3, ax))
            elif pred == "omcsgizs":
                omc.add(parts[1])
            elif pred == "xx5ljjai":
                xx.add(parts[1])
            elif pred == "mcoxi8qw":
                mco.add(parts[1])
            elif pred == "zmomsdb3":
                zmoms.add(parts[1])
            elif pred == "okcl3dr0":
                okcl3dr0.add(parts[1])
            elif pred == "sl9kfffg":
                sl9kfffg.add(parts[1])
            elif pred == "g7wohkyk":
                g7wohkyk.add(parts[1])
            elif pred == "vrips567":
                vrips567.add(parts[1])
            elif pred == "usj0xp5t":
                usj0xp5t.add(parts[1])

        # Convenience sets and capability checks
        any_qroh_exists = len(qroh) > 0
        any_omc_exists = len(omc) > 0
        any_xx_exists = len(xx) > 0

        # Can we create omcsgizs cheaply via aegb5ot3? (needs okcl3dr0, g7wohkyk, sl9kfffg present somewhere)
        aegb5ot3_possible = bool(okcl3dr0 and g7wohkyk and sl9kfffg)
        # ppnkvo80 requires additional vrips567 and usj0xp5t
        ppnkvo80_possible = bool(okcl3dr0 and g7wohkyk and sl9kfffg and vrips567 and usj0xp5t)

        # Helper to compute best estimated cost to get qroh(ax, imdah) (0 if present, 1 if can retarget from other imdah for same ax)
        def cost_for_qroh(ax, desired_imdah):
            if ax in qroh and desired_imdah in qroh[ax]:
                return 0
            if ax in qroh and len(qroh[ax]) > 0:
                # we can use ss17onrb to retarget one existing mapping for this ax to desired_imdah
                return 1
            # no qroh for this ax at all -> infeasible to directly get qroh for this ax
            return math.inf

        # Helper to compute cost to get dym1mop6 for ax; returns (cost, possible_c3_candidates_set)
        # cost of 0 if some (c3, ax) already exists; otherwise estimate cost to create one (nzh0a058 path)
        def cost_for_dym(ax):
            # If dym exists for this ax in the state/static
            existing_c3 = {c3 for (c3, a) in dym if a == ax}
            if existing_c3:
                return 0, existing_c3  # cost 0, we know which c3 works
            # Otherwise we need to create via nzh0a058: requires omcsgizs(c3) and qroh(ax, s154x56h)
            # Choose any c3 for which omcsgizs exists or could be created.
            possible_c3 = set()
            # c3 candidates where omc exists currently:
            if any_omc_exists:
                possible_c3 |= set(omc)
            # even if omc doesn't exist, we might be able to create it using aegb5ot3 or ppnkvo80
            can_create_omc = aegb5ot3_possible or ppnkvo80_possible

            # Check if qroh(ax, s154x56h) can be obtained
            qroh_s154_cost = 0
            if ax in qroh and "s154x56h" in qroh[ax]:
                qroh_s154_cost = 0
            elif ax in qroh and len(qroh[ax]) > 0:
                qroh_s154_cost = 1  # use ss17onrb to retarget to s154x56h
            else:
                qroh_s154_cost = math.inf

            if (possible_c3 or can_create_omc) and qroh_s154_cost < math.inf:
                # cost: possibly 1 to create omc (if not present) + cost to set qroh to s154x56h + 1 for nzh0a058
                cost_make_omc = 0 if any_omc_exists else (1 if can_create_omc else math.inf)
                if cost_make_omc == math.inf:
                    return math.inf, set()
                total_cost = cost_make_omc + qroh_s154_cost + 1
                # if possible_c3 empty but can_create_omc True, we assume it can be created for some c3 (abstract)
                if not possible_c3 and can_create_omc:
                    # create placeholder c3 label - we don't need concrete name for heuristic; return generic marker
                    possible_c3 = {"<created_c3>"}
                return total_cost, possible_c3
            # otherwise infeasible
            return math.inf, set()

        # Main loop: sum costs for each missing qxafq0yb goal
        total_cost = 0
        LARGE = 100  # finite but large penalty for seemingly infeasible subproblems

        # Extract the goals as strings like "(qxafq0yb scvulwhv)"
        # Work with the predicate name and argument
        goal_z_objects = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "qxafq0yb" and len(parts) >= 2:
                goal_z_objects.append(parts[1])
            else:
                # if there are other goal types (not expected), we give trivial estimate 1 if missing
                if g not in state:
                    total_cost += 1

        for z in goal_z_objects:
            if f"(qxafq0yb {z})" in state:
                continue  # already satisfied

            # waex candidates for this z
            imdah_candidates = waex.get(z, set())
            if not imdah_candidates:
                # impossible or very hard: no waex mapping available for this z under current/static facts
                total_cost += LARGE
                continue

            best_cost_for_z = math.inf

            # Try each imdah (u59rqa6u) that links the z object to some imdah
            for imdah in imdah_candidates:
                # For this imdah, we must get qroh(ax, imdah) and dym(c3, ax)
                # We'll iterate possible ax values that are present in qroh keys (those are the only ax we can retarget)
                candidate_ax_set = set(qroh.keys()) | {a for (_, a) in dym}
                if not candidate_ax_set:
                    # If there is no ax at all in the state/static, then we probably cannot create needed qroh/dym
                    continue

                # Pre-check: waex exists (it does by candidate selection)
                cost_waex = 0

                # For this z, there are two main final actions: eav722gv (needs mcoxi8qw(z)) or l6w9j7tw (needs zmomsdb3(z))
                # Evaluate both paths and take min

                # Evaluate eav path only if mcoxi8qw present or potentially present via static (we cannot create mcoxi8qw via operators)
                eav_possible = z in mco
                cost_mco = 0 if z in mco else math.inf

                # Evaluate l6 path only if zmomsdb3 present
                l6_possible = z in zmoms
                cost_zmoms = 0 if z in zmoms else math.inf

                # If neither path feasible, skip this imdah
                if not (eav_possible or l6_possible):
                    continue

                # For each candidate ax, compute cost components
                for ax in candidate_ax_set:
                    # cost to get qroh(ax, imdah)
                    cq = cost_for_qroh(ax, imdah)
                    if cq == math.inf:
                        continue  # cannot produce qroh for this ax and imdah

                    # cost to obtain dym for this ax (and get candidate c3 set)
                    cd, c3_candidates = cost_for_dym(ax)
                    if cd == math.inf:
                        continue  # cannot obtain dym for this ax

                    # Considering the eav path:
                    if eav_possible:
                        # eav requires qroh(ax, imdah), dym(c3,ax) (some c3), mcoxi8qw(z), waexgd4o(z,imdah)
                        # cost is cq + cd + 1 (final eav action) + cost_waex + cost_mco
                        path_cost = cq + cd + 1 + cost_waex + cost_mco
                        if path_cost < best_cost_for_z:
                            best_cost_for_z = path_cost

                    # Considering the l6 path:
                    if l6_possible:
                        # l6 requires qroh(ax, imdah), dym(c3,ax), xx5ljjai(c3), waexgd4o(z,imdah), zmomsdb3(z)
                        # We need an actual c3 candidate for which xx might already exist or be creatable.
                        # If cd == 0 we have existing_c3 set; otherwise c3_candidates may be created or abstract.
                        # If any existing/possible c3 has xx present -> cost_xx = 0; else if ppnkvo80_possible -> 1; else INF
                        # Note: ppnkvo80 also creates omcsgizs and xx5ljjai, so if we needed to create omc in cd we might combine operations.
                        # For simplicity use the following conservative accounting:
                        # cost_xx = 0 if any c3 in c3_candidates intersects xx; else 1 if ppnkvo80_possible else INF
                        if c3_candidates:
                            if any(c in xx for c in c3_candidates):
                                cost_xx = 0
                            elif ppnkvo80_possible:
                                cost_xx = 1
                            else:
                                cost_xx = math.inf
                        else:
                            # No specific c3 candidate (should not often happen), but allow creation via ppnkvo80 if possible
                            cost_xx = 0 if any_xx_exists else (1 if ppnkvo80_possible else math.inf)

                        if cost_xx == math.inf:
                            continue

                        path_cost = cq + cd + cost_xx + 1 + cost_waex + cost_zmoms
                        if path_cost < best_cost_for_z:
                            best_cost_for_z = path_cost

            # If after considering everything the best cost is still infinite, treat as large penalty
            if best_cost_for_z == math.inf:
                total_cost += LARGE
            else:
                # Round up to integer (costs are integral by construction)
                total_cost += int(best_cost_for_z)

        # Finally, ensure heuristic is finite; if nothing was estimable return a large finite number
        if total_cost >= math.inf:
            return LARGE
        return total_cost