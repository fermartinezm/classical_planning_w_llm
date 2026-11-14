from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """Return tokens of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (aixg4uih) domain.
    The domain's goals are facts of the form (qxafq0yb <z>), i.e. certain z65nyce9 objects
    must be produced. Each such goal can be produced by one of two operators:
      - eav722gv: consumes a dym1mop6 fact (for some ggld6jmz and s1clodkz), requires that
        the s1clodkz is mapped (qrohx31m) to the same u59rqa6u that the target z65nyce9
        is associated with via (waexgd4o <z> <u>), and requires (mcoxi8qw <z>) for the
        eav path.
      - l6w9j7tw: similar to eav722gv but requires (zmomsdb3 <z>) and additionally
        requires xx5ljjai for the ggld6jmz.
    Supporting facts for producing dym1mop6 exist (omcsgizs) and can be created
    by other operators (ppnkvo80 / aegb5ot3). Also qrohx31m mappings can be altered by
    ss17onrb. This heuristic estimates the number of actions needed to achieve all
    (qxafq0yb <z>) goals from the given state using inexpensive, conservative
    reachability approximations and small fixed costs for producing missing supporting facts.

    Assumptions
    -----------
    - Goals are only facts of the form (qxafq0yb <z>).
    - The mapping facts (waexgd4o <z> <imdah>) (usually static) indicate which u59rqa6u
      value a target z needs to be associated with by qrohx31m.
    - qrohx31m facts in the current state indicate which s1clodkz objects already have
      the correct u59rqa6u; ss17onrb can change a mapping in one action.
    - dym1mop6 facts are required (for some ggld6jmz and s1clodkz) and can be created
      by nzh0a058 when omcsgizs for the ggld6jmz exists (nzh0a058 requires that the
      s1clodkz currently maps to s154x56h). omcsgizs can be created by one of the
      preparation operators (ppnkvo80 or aegb5ot3) in one action in many situations.
    - We make optimistic but computationally cheap estimates: if some supporting
      fact exists anywhere in the state we assume it can be re-used for a goal.
      If it is missing we count a small number of actions to create it
      (1 or 2), depending on the fact.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - The set of goal facts (self.goals).
      - Static mappings (from task.static) for waexgd4o facts (maps z65nyce9 -> u59rqa6u).
      - Static markers for mcoxi8qw and zmomsdb3 when present in static facts.
    These static data structures are used as a first lookup to determine which
    production path (eav722gv vs l6w9j7tw) is applicable more cheaply for each goal.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state we compute an estimated cost as follows:

    1. If all goals are already satisfied in `state`, return 0.

    2. Parse the state's facts into efficient lookup structures:
       - qroh_map: maps each s1clodkz to its current u59rqa6u (from qrohx31m facts).
       - dym_pairs: set of (ggld6jmz, s1clodkz) pairs present via dym1mop6 facts.
       - omcsgizs_set: ggld6jmz that currently have omcsgizs.
       - xx_set: ggld6jmz that currently have xx5ljjai.
       - mcoxi_state and zmoms_state sets from the state (in addition to static sets).
       - waex_map: prefer static mapping; if missing statically, attempt to read from state.

    3. For each missing goal (qxafq0yb <z> not in state) estimate the minimal cost
       to produce it by considering the two known production operators (eav722gv and l6w9j7tw).
       For the chosen operator we aggregate small sub-costs:
         - cost_final = 1 (to apply the final operator that produces qxafq0yb).
         - cost_dym: an estimate to ensure a suitable dym1mop6 exists:
            * If a dym1mop6 already exists for any pair -> cost_dym = 0.
            * Else if any omcsgizs exists -> cost_dym = 1 (we assume nzh0a058 can create dym1mop6).
            * Else -> cost_dym = 2 (one action to create omcsgizs, one action to create dym1mop6).
         - cost_xx (only for l6w9j7tw): if xx5ljjai is required and not present for any ggld6jmz -> +1.
           (ppnkvo80 can in practice create both xx5ljjai and omcsgizs; we keep a single small cost)
         - cost_qroh: if there is any s1clodkz already mapped to the desired u59rqa6u
           (waexgd4o mapping for the goal), we assume no cost; otherwise we add 1 to change a mapping
           using ss17onrb.
       The heuristic takes the minimum estimated cost of the two production options
       (when both are applicable: i.e., mcoxi8qw path and zmomsdb3 path) and sums these
       per-goal estimates.

    The numbers used (0, 1, or 2 for subgoals) are chosen to be cheap to compute
    while capturing the rough number of actions required to create missing helper facts.
    This heuristic is not admissible but is efficient and tends to reflect the true
    planning effort: it accounts for whether dym1mop6, omcsgizs, xx5ljjai and correct qrohx31m
    mappings are already present or must be created.
    """

    def __init__(self, task):
        # Goals (a set of fact strings)
        self.goals = set(task.goals)

        # Extract static waexgd4o mapping (z65nyce9 -> u59rqa6u) from static facts
        self.waex_static = {}
        self.mcoxi_static = set()
        self.zmoms_static = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) >= 3:
                # (waexgd4o <z> <u>)
                self.waex_static[parts[1]] = parts[2]
            elif parts[0] == "mcoxi8qw" and len(parts) >= 2:
                self.mcoxi_static.add(parts[1])
            elif parts[0] == "zmomsdb3" and len(parts) >= 2:
                self.zmoms_static.add(parts[1])

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse state
        qroh_map = {}  # axwhj -> imdah
        dym_pairs = set()  # (c3xfw, axwhj)
        omcsgizs_set = set()
        xx_set = set()
        mcoxi_state = set()
        zmoms_state = set()
        waex_state = {}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "qrohx31m" and len(parts) >= 3:
                qroh_map[parts[1]] = parts[2]
            elif pred == "dym1mop6" and len(parts) >= 3:
                dym_pairs.add((parts[1], parts[2]))
            elif pred == "omcsgizs" and len(parts) >= 2:
                omcsgizs_set.add(parts[1])
            elif pred == "xx5ljjai" and len(parts) >= 2:
                xx_set.add(parts[1])
            elif pred == "mcoxi8qw" and len(parts) >= 2:
                mcoxi_state.add(parts[1])
            elif pred == "zmomsdb3" and len(parts) >= 2:
                zmoms_state.add(parts[1])
            elif pred == "waexgd4o" and len(parts) >= 3:
                waex_state[parts[1]] = parts[2]

        # Helper presence checks
        any_dym_exists = len(dym_pairs) > 0
        any_omc_exists = len(omcsgizs_set) > 0
        any_xx_exists = len(xx_set) > 0

        total_cost = 0

        # Consider each qxafq0yb goal
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "qxafq0yb" or len(parts) < 2:
                # Not a qxafq0yb goal; skip (defensive)
                continue
            target = parts[1]
            goal_fact = f"(qxafq0yb {target})"
            if goal_fact in state:
                continue  # already satisfied

            # Determine required imdah for this target (prefer static mapping)
            imdah = self.waex_static.get(target, waex_state.get(target, None))

            # Determine which production paths are supported (eav vs l6)
            # eav path requires mcoxi8qw(target)
            eav_possible = (target in self.mcoxi_static) or (target in mcoxi_state)
            # l6 path requires zmomsdb3(target)
            l6_possible = (target in self.zmoms_static) or (target in zmoms_state)

            # If neither static nor state indicates which path to use, be optimistic and
            # allow both (heuristic will pick smaller estimate).
            if not eav_possible and not l6_possible:
                # allow both (choose whichever is cheaper later)
                eav_possible = True
                l6_possible = True

            best_for_target = float("inf")

            # A small routine that estimates cost when using either final operator (eav or l6)
            def estimate_cost(use_l6: bool):
                # base cost to apply final operator
                cost = 1

                # cost to ensure a dym1mop6 exists
                if any_dym_exists:
                    cost_dym = 0
                elif any_omc_exists:
                    # one action (nzh0a058) to create dym1mop6 from omcsgizs
                    cost_dym = 1
                else:
                    # optimistic: one action to create omcsgizs (ppnkvo80/aegb5ot3), one action to create dym1mop6
                    cost_dym = 2
                cost += cost_dym

                # if using l6, final op requires xx5ljjai for the ggld6jmz
                if use_l6:
                    if any_xx_exists:
                        cost_xx = 0
                    else:
                        # one action to create xx5ljjai (ppnkvo80). Note: ppnkvo80 often creates omcsgizs
                        # too; our cost model is optimistic but captures the extra work.
                        cost_xx = 1
                    cost += cost_xx

                # cost to ensure some s1clodkz is mapped to the desired imdah
                # If mapping unknown (imdah is None) be pessimistic and add one.
                if imdah is None:
                    cost_qroh = 1
                else:
                    # If any s1clodkz already maps to the required imdah, we assume we can use it:
                    if imdah in qroh_map.values():
                        cost_qroh = 0
                    else:
                        # Otherwise assume we need one ss17onrb to change a mapping
                        cost_qroh = 1
                cost += cost_qroh

                return cost

            if eav_possible:
                best_for_target = min(best_for_target, estimate_cost(use_l6=False))
            if l6_possible:
                best_for_target = min(best_for_target, estimate_cost(use_l6=True))

            # As a fallback (should not remain infinite), use a moderate cost.
            if best_for_target == float("inf"):
                best_for_target = 5

            total_cost += best_for_target

        # Heuristic is finite and 0 exactly for goal states
        return int(total_cost)