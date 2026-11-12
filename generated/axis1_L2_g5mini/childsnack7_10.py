from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if fact (a string like "(pred a b)") matches the provided pattern
    (pred, arg1, arg2...). Wildcards '*' are supported (using fnmatch).
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'secret_domain' (aixg4uih).
    The domain's goals are facts of the form (qxafq0yb ?r) for z65nyce9 objects.
    This heuristic estimates the number of actions still required to make each
    goal fact true. It inspects which supporting predicates are already true
    and provides a compact, efficiently computable estimate of the remaining
    steps. The heuristic is not required to be admissible.

    Assumptions
    -----------
    - A single action (eav722gv or l6w9j7tw) will ultimately establish each
      missing (qxafq0yb ?r). Both actions require that:
        * there is a link (waexgd4o ?r ?imdah) that ties the goal object to
          some u59rqa6u identifier, and
        * there exists some chain producing (dym1mop6 ?c3xfw ?axwhj)
          and a matching (qrohx31m ?axwhj ?imdah).
    - dym1mop6 is produced by nzh0a058, which requires (omcsgizs ?c3xfw)
      and (qrohx31m ?axwhj s154x56h) (i.e., axwhj must at some point be
      mapped to the special constant s154x56h).
    - omcsgizs facts are produced from okcl3dr0 facts (via aegb5ot3/ppnkvo80).
    - We approximate costs for creating intermediate facts with small integer
      estimates. These are tuned to reflect required operator counts while
      staying cheap to compute.
    - If static facts or the current state already contain supporting facts
      (e.g., mcoxi8qw, omcsgizs, dym1mop6, xx5ljjai, qrohx31m), those reduce
      the estimated cost accordingly.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Extract static waexgd4o mappings (z65nyce9 -> u59rqa6u) from task.static.
      These static mappings are helpful to determine which qrohx31m value is
      required for a given goal-object.
    - Remember the static facts set itself for quick checks if needed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given node/state:
    1. If all goals are already satisfied in the state, return 0.
    2. For each goal (qxafq0yb ?r) that is not yet true:
       a) Determine the u59rqa6u identifier (call it imdah) required for r by
          checking waexgd4o facts in the current state first, then static facts.
       b) Check whether an immediate action can produce qxafq0yb ?r with cost 1:
          - If there exists in the current state a dym1mop6 linking some
            ?c3xfw to an ?axwhj whose qrohx31m already equals imdah, then
            eav722gv (or l6w9j7tw if zmomsdb3 is required) can be applied.
       c) Otherwise estimate the cost to create a suitable (dym1mop6 ?c ?ax)
          and to ensure (qrohx31m ?ax ?imdah) at the time of the final action:
          - Creating dym1mop6 usually requires:
             * producing omcsgizs for some ?c (if absent). If omcsgizs exists:
               omc_cost = 0. Else if okcl3dr0 exists for some candidate ?c:
               omc_cost = 1 (we assume one operator can create omcsgizs),
               otherwise we set omc_cost = 2 (an additional step).
             * nzh0a058 itself (cost 1) to produce dym1mop6 once omcsgizs and
               qroh(?ax, s154x56h) are present.
          - Producing the required qrohx31m mapping:
             * nzh0a058 requires some axwhj to be mapped to s154x56h at the
               time it's called. If a qrohx31m with s154x56h exists already,
               cost is 0 for this part; otherwise setting some axwhj to
               s154x56h takes an estimated cost 1 (if some qrohx31m exists)
               or 2 (if none exist).
             * After producing dym1mop6, the final action (eav722gv/l6w9j7tw)
               requires qrohx31m ?axwhj to equal imdah. If no axwhj currently
               has that mapping, we count 1 extra cost to perform a mapping
               change via ss17onrb (if possible) or 2 if even a qrohx31m
               doesn't exist in the state.
          - For the l6w9j7tw route (when zmomsdb3 is present on the goal
            object instead of mcoxi8qw), also ensure xx5ljjai ?c3xfw exists:
            if xx5ljjai is absent but okcl3dr0 exists, assume 1 step, else 2.
       d) Sum the small integer sub-costs above and add 1 for the final action
          that produces (qxafq0yb ?r).
    3. The heuristic value is the sum of the per-goal estimates. The heuristic
       is intentionally optimistic about parallelism across goals; it assumes
       independent achievement of different qxafq0yb facts.

    Notes
    -----
    - The heuristic trades strict admissibility for informative, fast-to-compute
      estimates that reduce node expansions in greedy best-first search.
    - All approximations are finite integers; unreachable goals will simply
      produce larger finite estimates instead of infinity.
    """

    def __init__(self, task):
        """Extract goals and useful static mappings from the task."""
        self.goals = task.goals
        self.static = set(task.static)

        # Map from z65nyce9 object -> u59rqa6u identifier via waexgd4o
        self.waex_map_static = {}
        for fact in self.static:
            if match(fact, "waexgd4o", "*", "*"):
                _, obj, ident = get_parts(fact)
                self.waex_map_static[obj] = ident

        # Remember whether the special constant s154x56h appears in any static qrohx31m
        self.has_static_s154 = any(match(fact, "qrohx31m", "*", "s154x56h") for fact in self.static)

    def __call__(self, node):
        """Compute a compact estimated number of actions to achieve all goals."""
        state = set(node.state)  # frozenset -> set for faster membership tests

        # If the goal set is subset of state, we're done.
        if self.goals <= state:
            return 0

        # Combine state and static when looking for waex mappings (prefer current state).
        waex_map = dict(self.waex_map_static)
        for fact in state:
            if match(fact, "waexgd4o", "*", "*"):
                _, obj, ident = get_parts(fact)
                waex_map[obj] = ident

        # Helper queries on the state
        def has(pred, *args):
            return any(match(f, pred, *args) for f in state)

        def find_facts(pred):
            return [f for f in state if match(f, pred, "*", "*") or match(f, pred, "*")]

        # Precompute quick lookups
        dym_facts = [get_parts(f) for f in state if match(f, "dym1mop6", "*", "*")]
        # dym_facts: list of [predicate, c3xfw, axwhj]
        # Build mapping sets for quick checks
        dym_pairs = {(parts[1], parts[2]) for parts in dym_facts}

        omc_exists = any(match(f, "omcsgizs", "*") for f in state)
        okcl_exists = any(match(f, "okcl3dr0", "*") for f in state)
        xx_exists = {get_parts(f)[1] for f in state if match(f, "xx5ljjai", "*")}
        omc_c3s = {get_parts(f)[1] for f in state if match(f, "omcsgizs", "*")}
        okcl_c3s = {get_parts(f)[1] for f in state if match(f, "okcl3dr0", "*")}
        qroh_facts = [get_parts(f) for f in state if match(f, "qrohx31m", "*", "*")]
        # qroh_facts: [predicate, axwhj, ident]
        qroh_map = {parts[1]: parts[2] for parts in qroh_facts}
        any_qroh_exists = len(qroh_facts) > 0
        any_qroh_s154 = any(parts[2] == "s154x56h" for parts in qroh_facts)

        total_cost = 0

        # For each goal (qxafq0yb <r>), estimate cost if missing
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            if parts[0] != "qxafq0yb":
                # unexpected goal form: count 1 as fallback
                total_cost += 1
                continue

            r = parts[1]

            # Determine required identifier (imdah) for this r from waexgd4o
            imdah = waex_map.get(r, None)

            # Quick check: can we immediately apply an action to make qxafq0yb?
            # eav722gv requires: qrohx31m ?axwhj ?imdah, dym1mop6 ?c ?ax, mcoxi8qw ?r, waexgd4o ?r ?imdah
            immediate_possible = False
            if imdah is not None and (match(f"(mcoxi8qw {r})") or has("mcoxi8qw", r)):
                # find any dym pair where axwhj has qroh == imdah
                for (c, ax) in dym_pairs:
                    if qroh_map.get(ax) == imdah:
                        immediate_possible = True
                        break
            # Alternative l6w9j7tw: needs dym1mop6 (c,ax), xx5ljjai c, zmomsdb3 r, waexgd4o r imdah, qrohx31m ax imdah
            if not immediate_possible and imdah is not None and has("zmomsdb3", r):
                for (c, ax) in dym_pairs:
                    if (c in xx_exists) and qroh_map.get(ax) == imdah:
                        immediate_possible = True
                        break

            if immediate_possible:
                # One final action suffices
                total_cost += 1
                continue

            # Otherwise estimate cost for either eav722gv path (if mcoxi8qw exists) or
            # l6w9j7tw path (if zmomsdb3 exists). Choose the cheaper estimated route.

            # Helper to estimate cost to create a dym1mop6 + matching qroh mapping
            def estimate_dym_and_qroh_cost(target_imdah):
                """
                Estimate cost to obtain a pair (dym1mop6 ?c ?ax) together with
                qrohx31m ?ax == target_imdah at the time of final application.
                This accounts for:
                  - producing omcsgizs (if needed)
                  - setting some qrohx31m ?ax to s154x56h (for nzh0a058)
                  - invoking nzh0a058 (cost 1)
                  - changing qrohx31m ?ax to target_imdah (if needed)
                """
                # If we already have a dym pair with ax that can be set to target_imdah,
                # then only mapping change may be necessary.
                existing_ax_with_dym = {ax for (c, ax) in dym_pairs}
                # If some existing dym ax already has required mapping, cost is small.
                if any(qroh_map.get(ax) == target_imdah for ax in existing_ax_with_dym):
                    # only final action needed elsewhere; but to create dym was zero cost
                    return 0

                # Need to create dym1mop6 via nzh0a058:
                # nzh0a058 costs 1, but requires omcsgizs for some c and qroh(ax, s154x56h).
                # Estimate omc cost:
                if omc_exists:
                    omc_cost = 0
                elif okcl_exists:
                    # We assume one operator (aegb5ot3/ppnkvo80) can produce omcsgizs from okcl3dr0
                    omc_cost = 1
                else:
                    omc_cost = 2

                # cost to get some ax to s154x56h (required by nzh0a058):
                if any_qroh_s154 or self.has_static_s154:
                    qroh_s154_cost = 0
                else:
                    qroh_s154_cost = 1 if any_qroh_exists else 2

                # cost to get some ax to target_imdah for final application:
                # if any qroh already equals target_imdah, zero; else 1 or 2
                if any(match(f, "qrohx31m", "*", target_imdah) for f in state) or \
                   any(get_parts(f)[2] == target_imdah for f in self.static if match(f, "qrohx31m", "*", "*")):
                    qroh_to_imd_cost = 0
                else:
                    qroh_to_imd_cost = 1 if any_qroh_exists else 2

                # We will need one nzh0a058 action
                nzh_cost = 1

                # It's possible that the same ss17onrb call could both create s154 and later change mapping,
                # but to be conservative, sum both small costs.
                return omc_cost + qroh_s154_cost + nzh_cost + qroh_to_imd_cost

            # Estimate for eav path (prefers mcoxi8qw r)
            eav_cost = float("inf")
            if imdah is not None:
                # Check if waexgd4o exists (it may be in static or state) - if waex missing,
                # eav/l6 can't be applied; we still estimate but add penalty.
                waex_exists = (any(match(f, "waexgd4o", r, imdah) for f in state) or (r in self.waex_map_static and self.waex_map_static[r] == imdah))
                if not waex_exists:
                    # penalize because required mapping is missing
                    waex_penalty = 1
                else:
                    waex_penalty = 0

                # If mcoxi8qw r already exists (static or state), prefer eav (no extra cost for mcoxi)
                mco_exists = any(match(f, "mcoxi8qw", r) for f in state) or any(match(f, "mcoxi8qw", r) for f in self.static)
                if mco_exists:
                    # final action eav722gv cost 1 plus cost to create dym and qroh
                    eav_cost = 1 + waex_penalty + estimate_dym_and_qroh_cost(imdah)
                else:
                    # If mcoxi8qw absent, eav can't be used directly; leave cost high
                    eav_cost = 1000

            # Estimate for l6w9j7tw path (prefers zmomsdb3 r)
            l6_cost = float("inf")
            if imdah is not None:
                zmoms = any(match(f, "zmomsdb3", r) for f in state) or any(match(f, "zmomsdb3", r) for f in self.static)
                if zmoms:
                    # l6 requires also xx5ljjai ?c3xfw for the c used in dym
                    # Check if any c in dym_pairs has xx5ljjai already
                    if any(c in xx_exists for (c, ax) in dym_pairs):
                        xx_penalty = 0
                    else:
                        # if okcl3dr0 exists (we can create xx via ppnkvo80/aegb5ot3), smaller cost
                        xx_penalty = 0 if any(c in xx_exists for (c, ax) in dym_pairs) else (1 if okcl_exists else 2)
                    l6_cost = 1 + xx_penalty + estimate_dym_and_qroh_cost(imdah)

            # If neither route is viable (imdah missing), impose a fallback estimate
            if eav_cost >= 1000 and l6_cost >= 1000:
                # impose a generic cost for this goal: try to create waex mapping (1)
                # plus a couple of intermediate ops
                fallback = 4
                total_cost += fallback
            else:
                total_cost += min(eav_cost, l6_cost)

        # Ensure finite integer
        return int(total_cost)