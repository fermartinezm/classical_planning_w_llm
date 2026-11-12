from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (aixg4uih) domain.
    The domain's goals are facts of the form (qxafq0yb ?r - z65nyce9). Producing
    such a fact requires several conditions (mapping facts qrohx31m, connector facts
    dym1mop6, and certain *static* connectors like waexgd4o). This heuristic
    estimates the number of actions required to satisfy all missing qxafq0yb goals
    from a given state by reasoning about which prerequisite facts are already
    present and which are missing, and assigning small, informative costs to
    create those prerequisites.

    Assumptions
    -----------
    - The planner represents facts as strings like '(predicate arg1 arg2)'.
    - Static facts (task.static) include waexgd4o entries that associate each
      z65nyce9 object with a particular u59rqa6u value that is needed by the
      goal-producing actions.
    - For producing a single qxafq0yb ?r, one of the two actions (l6w9j7tw or
      eav722gv) will be used. Both require:
        * a waexgd4o (?r ?imdah),
        * a qrohx31m (?ax ?imdah) for the same ?imdah value,
        * a dym1mop6 (?c ?ax) referencing the same ?ax.
      l6w9j7tw additionally prefers xx5ljjai (?c) and zmomsdb3 (?r),
      while eav722gv requires mcoxi8qw (?r).
    - Creating dym1mop6 requires nzh0a058 which in turn requires omcsgizs
      and that some qrohx31m maps the chosen ?ax to s154x56h at the time of
      creation. Because of this, in practice creating the required pair
      (qrohx31m + dym1mop6) may need one or two extra mapping-change actions
      (ss17onrb). This heuristic approximates these needs with small integer
      costs rather than performing full backward search.

    Heuristic Initialization
    ------------------------
    The constructor extracts helpful static information from task.static and
    stores them for fast lookup:
      - waexgd_map: maps each z65nyce9 object -> set of u59rqa6u values (from waexgd4o)
      - has_zmoms: set of z65nyce9 objects that appear in zmomsdb3
      - has_mcoxi: set of z65nyce9 objects that appear in mcoxi8qw

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal object r (i.e., each goal fact (qxafq0yb r)) not yet satisfied in
    the current state:
      1. If the goal fact is already in the state, cost contribution = 0.
      2. Otherwise, attempt to estimate the minimal number of actions needed to
         establish the preconditions for either eav722gv or l6w9j7tw and then
         apply that action (which we count as one additional action).
      3. The heuristic identifies the waexgd4o mapping(s) imdah for r (from static).
         If no waexgd4o mapping is known, we increase the cost (such goals are
         unlikely to be achievable without additional mappings).
      4. It then inspects the current state for:
         - existing qrohx31m facts (mappings of s1clodkz objects to u59rqa6u),
         - existing dym1mop6 facts (links from ggld6jmz to s1clodkz),
         - presence of omcsgizs or xx5ljjai, zmomsdb3, mcoxi8qw.
      5. Using these facts, the heuristic assigns small integer costs for the
         following missing pieces:
         - missing_dym: approximate cost to obtain an appropriate dym1mop6
           (0 if already present with a suitable ax, 1 if some supporting
           facts like omcsgizs exist, 2 otherwise).
         - missing_qroh: approximate cost to get a qrohx31m mapping to the
           desired imdah (0 if already present, 1 if some qrohx31m exists and
           can be set by a single ss17onrb, 2 otherwise).
         - missing_support: approximate cost to obtain either mcoxi8qw r or
           the pair (xx5ljjai c and zmomsdb3 r) (0 if one of these is already
           present, 1 if some parts exist, 2 otherwise).
      6. The estimated cost for this goal is:
            1 (the final action eav/l6w) + missing_dym + missing_qroh + missing_support
         This sum keeps the heuristic finite while reflecting how many distinct
         types of preconditions are absent.
      7. The total heuristic value is the sum over all unsatisfied qxafq0yb goals.
         This makes the heuristic optimistic in one sense (it assumes parallelism
         between satisfying different goals only to the extent reflected in
         shared facts) but domain-aware enough to guide greedy best-first search.

    Note
    ----
    This heuristic is intentionally not admissible: it trades optimality for
    speed and discrimination. Its goal is to provide informative, cheaply
    computable estimates that reduce node expansions in greedy search for
    problems in this domain.
    """

    def __init__(self, task):
        # store goals (set of goal fact strings)
        self.goals = task.goals
        # static facts: extract useful sets/maps
        static = task.static

        # Map each z65nyce9 object to the set of u59rqa6u values it is associated with via waexgd4o
        waexgd_map = {}
        has_zmoms = set()
        has_mcoxi = set()

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) >= 3:
                r = parts[1]
                val = parts[2]
                waexgd_map.setdefault(r, set()).add(val)
            elif parts[0] == "zmomsdb3" and len(parts) >= 2:
                has_zmoms.add(parts[1])
            elif parts[0] == "mcoxi8qw" and len(parts) >= 2:
                has_mcoxi.add(parts[1])

        self.waexgd_map = waexgd_map
        self.has_zmoms = has_zmoms
        self.has_mcoxi = has_mcoxi

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Quick check: if all goals satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Index useful facts in the current state for fast lookup.
        # 1) qrohx31m: mapping ax -> set of values
        qroh_map = {}
        # 2) dym1mop6: set of (c, ax) pairs
        dym_pairs = set()
        # 3) omcsgizs set, xx5ljjai set, zmomsdb3 present facts (state-level), mcoxi8qw present in state
        omcsgizs_set = set()
        xx5_set = set()
        zmoms_in_state = set()
        mcoxi_in_state = set()
        # also collect any qrohx31m values equal to s154x56h for cheaper creation of dym1mop6
        qroh_s154_axs = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "qrohx31m" and len(parts) >= 3:
                ax = parts[1]
                val = parts[2]
                qroh_map.setdefault(ax, set()).add(val)
                if val == "s154x56h":
                    qroh_s154_axs.add(ax)
            elif pred == "dym1mop6" and len(parts) >= 3:
                dym_pairs.add((parts[1], parts[2]))  # (c, ax)
            elif pred == "omcsgizs" and len(parts) >= 2:
                omcsgizs_set.add(parts[1])
            elif pred == "xx5ljjai" and len(parts) >= 2:
                xx5_set.add(parts[1])
            elif pred == "zmomsdb3" and len(parts) >= 2:
                zmoms_in_state.add(parts[1])
            elif pred == "mcoxi8qw" and len(parts) >= 2:
                mcoxi_in_state.add(parts[1])

        # helper checks
        any_qroh_exists = len(qroh_map) > 0
        any_dym_exists = len(dym_pairs) > 0
        any_omc_exists = len(omcsgizs_set) > 0
        any_xx5_exists = len(xx5_set) > 0

        total_cost = 0

        # For each goal qxafq0yb r, estimate cost if not already present
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "qxafq0yb":
                # only handle qxafq0yb goals here; others count as 0 (conservative)
                continue
            r = parts[1]
            goal_fact = f"(qxafq0yb {r})"
            if goal_fact in state:
                continue  # already satisfied

            # Base impossible/penalty if no waexgd mapping is known for r
            if r not in self.waexgd_map:
                # unknown mapping -> pessimistic small penalty but finite
                total_cost += 6
                continue

            # pick one representative imdah value for r (prefer any)
            imdah_candidates = self.waexgd_map[r]
            # we will use a simple strategy: try to see if any current qroh maps to a candidate imdah
            mapping_cost = 2  # default worst small cost to obtain qrohx31m ax imdah
            matching_ax_for_imdah = None
            for ax, vals in qroh_map.items():
                for imdah in imdah_candidates:
                    if imdah in vals:
                        mapping_cost = 0
                        matching_ax_for_imdah = ax
                        break
                if mapping_cost == 0:
                    break
            if mapping_cost != 0:
                # if some qroh exists at all, we can set one to the desired imdah with one ss17onrb
                if any_qroh_exists:
                    mapping_cost = 1
                else:
                    mapping_cost = 2  # no qroh facts at all; need to create/change twice (approx)

            # Estimate cost to obtain dym1mop6 for an ax that will match qrohx31m to imdah.
            # If there already exists dym1mop6 whose ax already maps to imdah, then 0.
            dym_cost = 3  # default conservative
            # Check for direct usable dym: find any (c, ax) where ax maps to one of imdah candidates
            found_direct_dym = False
            for (c, ax) in dym_pairs:
                if ax in qroh_map:
                    if any(imdah in qroh_map[ax] for imdah in imdah_candidates):
                        found_direct_dym = True
                        break
            if found_direct_dym:
                dym_cost = 0
            else:
                # If any dym exists at all, assume we can adapt one with moderate effort (e.g., remap qroh once)
                if any_dym_exists:
                    # but nzh0a058 creates dym1mop6 from omcsgizs + qrohx31m ax s154x56h
                    # if omcsgizs exists, creating a dym is cheap (1); else we need to create omcsgizs then dym (2)
                    if any_omc_exists:
                        dym_cost = 1
                    else:
                        dym_cost = 2
                else:
                    # no dym exists: need to create omcsgizs (maybe) and dym: cost 2 (approx)
                    dym_cost = 2 if any_omc_exists else 3

                # also account for the fact that nzh0a058 requires qrohx31m ax s154x56h at creation time.
                # If there is already a qrohx31m ax with s154x56h, we don't need an extra mapping; otherwise
                # we will have to perform one additional mapping (ss17onrb) to set some qroh to s154x56h.
                if len(qroh_s154_axs) == 0:
                    # need an extra mapping operation before nzh0a058 (but that mapping might overlap with
                    # the mapping we need to set to imdah later; to keep heuristic admissible we count it)
                    dym_cost += 1

            # Estimate cost for support facts (either mcoxi8qw r OR xx5ljjai c & zmomsdb3 r)
            support_cost = 2  # default
            if r in mcoxi_in_state or r in self.has_mcoxi:
                support_cost = 0
            else:
                # consider possibility of using l6w9j7tw which needs zmomsdb3 r and some xx5ljjai c
                if (r in zmoms_in_state) or (r in self.has_zmoms):
                    if any_xx5_exists:
                        support_cost = 0
                    else:
                        # If no xx5ljjai exists at present, roughly one action (ppnkvo80) can often create xx5ljjai
                        support_cost = 1
                else:
                    # neither mcoxi nor zmoms present, pick moderate cost to obtain mcoxi or zmoms+xx5
                    support_cost = 2

            # final action (apply eav722gv or l6w9j7tw)
            final_action_cost = 1

            # Sum the components. This is an approximation that tries to count distinct missing
            # prerequisites rather than individual low-level details.
            est_cost_for_r = final_action_cost + mapping_cost + dym_cost + support_cost

            total_cost += est_cost_for_r

        # ensure heuristic is finite and non-negative
        if total_cost < 0:
            total_cost = 0
        return int(total_cost)