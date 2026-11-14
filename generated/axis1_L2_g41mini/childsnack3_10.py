from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by removing surrounding parentheses and splitting.
    E.g., '(qxafq0yb rap6j3pu)' -> ['qxafq0yb', 'rap6j3pu']
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (aixg4uih).

    Summary
    -------
    The heuristic estimates the number of actions required to achieve all goal facts
    of the form (qxafq0yb ?r7dv9). These goal facts indicate completion status on
    z65nyce9-typed objects.

    The domain manipulates predicates mainly around ggld6jmz, z65nyce9, s1clodkz, and u59rqa6u typed objects.
    Key goal facts are (qxafq0yb ?r7dv9).

    Assumptions
    -----------
    - Each goal fact (qxafq0yb ?r7dv9) is achieved by removing certain "dym1mop6" facts and
      establishing (qxafq0yb ?r7dv9).
    - The foremost way to produce (qxafq0yb ?r7dv9) is by executing one of:
        * l6w9j7tw or eav722gv actions, which require:
          - Appropriate (dym1mop6 ?c3xfw ?axwhj) facts,
          - (xx5ljjai ?c3xfw) or specific ggld6jmz states,
          - and access to suitable (waexgd4o ?r7dv9 ?imdah) and (qrohx31m ?axwhj ?imdah) pairs.
    - qrohx31m facts can be moved/changed via ss17onrb action.
    - Other static facts like (mcoxi8qw ?r7dv9), (zmomsdb3 ?r7dv9) signal if related to goal or operation state.
    - The constant s154x56h is a special u59rqa6u object often paired with qrohx31m facts.
    - The heuristic focuses on counting the minimal number of steps needed to:
      1. Transfer appropriate qrohx31m mappings to correct u59rqa6u objects.
      2. Ensure (dym1mop6 ?c3xfw ?axwhj) are prepared (via nzh0a058).
      3. Fulfill all goal qxafq0yb facts.

    Heuristic Initialization
    ------------------------
    - Extract all goal (qxafq0yb ?r7dv9) objects to track which z65nyce9 objects must have this fact.
    - Precompute:
      * Static mappings (waexgd4o) linking z65nyce9 to u59rqa6u.
      * The constant s154x56h u59rqa6u for convenience.
      * The set of s1clodkz objects involved (extracted from dym1mop6 facts in static).
    - Store the sets of:
      * ggld6jmz objects that are currently (xx5ljjai) or (okcl3dr0) (to estimate fixes),
      * and other relevant unary predicates that hint at requirements for actions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state:

    1. Identify which goal (qxafq0yb ?r7dv9) are still missing.
       The heuristic is 0 if all goal facts hold.

    2. For each missing goal (qxafq0yb ?r7dv9):
       - It requires the presence of (waexgd4o ?r7dv9 ?imdah) and an aligned (qrohx31m ?axwhj ?imdah).
         We check if qrohx31m facts need to be "moved" closer to correct u59rqa6u objects.
         The number of moves needed corresponds roughly to counts of mismatched qrohx31m facts,
         which can be fixed via ss17onrb actions.

       - For each ggld6jmz related via (dym1mop6 ?c3xfw ?axwhj) with given ?axwhj:
         * These predicates enable the goal qxafq0yb via l6w9j7tw or eav722gv.
         * If dym1mop6 facts do not hold where needed, nzh0a058 actions are required to produce them.
         * (xx5ljjai ?c3xfw) and (okcl3dr0 ?c3xfw) indicate readiness/availability.
         * If related predicates are missing to enable l6w9j7tw or eav722gv, additional fix actions like aegb5ot3 or ppnkvo80
           might be required. We approximate those costs.

    3. The heuristic sums:
       - The number of missing goals.
       - Estimated minimal number of qrohx31m moves (ss17onrb action) needed to align qrohx31m predicates.
       - Estimated preparations needed to produce (dym1mop6) facts via nzh0a058.
       - Estimated fixes for ggld6jmz facts to trigger l6w9j7tw/eav722gv (like enabling xx5ljjai, okcl3dr0, omcsgizs).
       - Estimation of actions to move the constant qrohx31m facts if needed.

    This heuristic tries to balance complexity and computational efficiency,
    providing guidance but not guaranteed admissibility.


    """

    def __init__(self, task):
        # Store goal predicates (only those qxafq0yb facts w/ z65nyce9 objects).
        self.goals = set()
        for fact in task.goals:
            parts = get_parts(fact)
            if parts[0] == "qxafq0yb":
                self.goals.add(parts[1])

        # Store static facts and parse useful static relations.
        self.static_waexgd4o = {}  # map z65nyce9 -> set of u59rqa6u
        self.static_qrohx31m = {}  # map s1clodkz -> set of u59rqa6u (usually constant s154x56h)
        self.static_mcoxi8qw = set()  # z65nyce9 with mcoxi8qw
        self.static_zmomsdb3 = set()  # z65nyce9 with zmomsdb3

        # We know the constant s154x56h always exists:
        self.const_s154x56h = "s154x56h"

        # Store static waexgd4o facts for quick access.
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) == 3:
                zobj, uobj = parts[1], parts[2]
                if zobj not in self.static_waexgd4o:
                    self.static_waexgd4o[zobj] = set()
                self.static_waexgd4o[zobj].add(uobj)
            elif parts[0] == "qrohx31m" and len(parts) == 3:
                s1obj, uobj = parts[1], parts[2]
                if s1obj not in self.static_qrohx31m:
                    self.static_qrohx31m[s1obj] = set()
                self.static_qrohx31m[s1obj].add(uobj)
            elif parts[0] == "mcoxi8qw" and len(parts) == 2:
                self.static_mcoxi8qw.add(parts[1])
            elif parts[0] == "zmomsdb3" and len(parts) == 2:
                self.static_zmomsdb3.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick check if all goals satisfied:
        goals_fulfilled = all(
            f"(qxafq0yb {zobj})" in state for zobj in self.goals
        )
        if goals_fulfilled:
            return 0  # Goal state reached

        # Extract relevant current facts by predicate for efficient access:
        facts_by_pred = {}
        for fact in state:
            pred = fact[1 : fact.find(" ")] if " " in fact else fact[1:-1]
            if pred not in facts_by_pred:
                facts_by_pred[pred] = set()
            facts_by_pred[pred].add(fact)

        def extract_objs(pred_name):
            """Return set of tuples of objects for a predicate in current state."""
            if pred_name not in facts_by_pred:
                return set()
            objs = set()
            for f in facts_by_pred[pred_name]:
                parts = get_parts(f)
                objs.add(tuple(parts[1:]))
            return objs

        # 1) Find missing goals (z65nyce9 obj where qxafq0yb is missing)
        missing_goals = [zobj for zobj in self.goals if f"(qxafq0yb {zobj})" not in state]

        # 2) Count current qrohx31m facts -> s1clodkz objects with u59rqa6u links
        qrohx31m_facts = extract_objs("qrohx31m")  # tuples (s1clodkz_obj, u59rqa6u_obj)

        # 3) Count current dym1mop6 facts: (ggld6jmz, s1clodkz)
        dym1mop6_facts = extract_objs("dym1mop6")

        # 4) Which ggld6jmz objects have (xx5ljjai) and (okcl3dr0) and (omcsgizs)?
        xx5ljjai = {objs[0] for objs in extract_objs("xx5ljjai")}
        okcl3dr0 = {objs[0] for objs in extract_objs("okcl3dr0")}
        omcsgizs = {objs[0] for objs in extract_objs("omcsgizs")}
        sl9kfffg = {objs[0] for objs in extract_objs("sl9kfffg")}
        mcoxi8qw = {objs[0] for objs in extract_objs("mcoxi8qw")}
        waexgd4o = {(objs[0], objs[1]) for objs in extract_objs("waexgd4o")}
        qrohx31m = {(objs[0], objs[1]) for objs in qrohx31m_facts}
        dym1mop6 = {(objs[0], objs[1]) for objs in dym1mop6_facts}

        # 5) We note for goals (qxafq0yb ?r7dv9):
        #    They require:
        #    - (waexgd4o ?r7dv9 ?imdah)
        #    - (qrohx31m ?axwhj ?imdah)
        #    - (dym1mop6 ?c3xfw ?axwhj)
        #    - (xx5ljjai ?c3xfw) or (mcoxi8qw ?r7dv9) for some c3xfw
        #    Actions l6w9j7tw or eav722gv connect these to achieve goal (qxafq0yb).

        # Heuristic components initialization:
        cost = 0

        # Heuristic step A: Cost to fix missing goal qxafq0yb ?r7dv9.
        # Each missing goal requires at least one action l6w9j7tw or eav722gv
        cost += len(missing_goals)

        # Heuristic step B: For each missing goal, estimate moves to align qrohx31m mappings:
        # Count how many needed movements of qrohx31m are needed to get correct (qrohx31m ?axwhj ?imdah) pairs.
        # Note that qrohx31m can be changed by ss17onrb (one action moves mapping from one u59rqa6u to another).
        # For each needed qrohx31m mapping to correct u59rqa6u for goal's waexgd4o, count 1 if missing.

        ss17onrb_moves = 0
        for zobj in missing_goals:
            # waexgd4o ?r7dv9 ?imdah static or dynamic must hold:
            waex_u59s = {u for (z, u) in waexgd4o if z == zobj}
            if not waex_u59s and zobj in self.static_waexgd4o:
                waex_u59s = self.static_waexgd4o.get(zobj, set())
            if not waex_u59s:
                # No known waexgd4o for this zobj, assume 1 fix needed
                ss17onrb_moves += 1
                continue
            # For each u59rqa6u in waex_u59s verify if qrohx31m with s1clodkz and that u59rqa6u exists in state.
            valid_qrohx31m_found = False
            for u in waex_u59s:
                for (s1obj, u_curr) in qrohx31m:
                    # Check if u_curr matches u or different (means a movement needed)
                    if u_curr == u:
                        valid_qrohx31m_found = True
                        break
                if valid_qrohx31m_found:
                    break
            if not valid_qrohx31m_found:
                # Need at least one ss17onrb move to fix this qrohx31m mapping
                ss17onrb_moves += 1

        cost += ss17onrb_moves

        # Heuristic step C: Estimate fixing missing dym1mop6 facts required for goal achievement.
        # If (dym1mop6 ?c3xfw ?axwhj) not present but needed, one nzh0a058 action needed.
        # Count missing dym1mop6 facts related to current qrohx31m mappings connected to waexgd4o.

        # Identify required s1clodkz from qrohx31m in current state:
        required_s1clodkz = set(s1 for (s1, u) in qrohx31m)

        missing_dym1mop6 = 0
        # For each dym1mop6 expected according to qrohx31m we check presence:
        for (ggld9, s1) in dym1mop6:
            if s1 not in required_s1clodkz:
                continue  # Heuristic focuses on known s1clodkz used in qrohx31m

        # Instead, check which s1clodkz occur in qrohx31m and count how many do not have a dym1mop6 with any ggld6jmz.
        s1_clodkz_with_dym = {s1 for (_, s1) in dym1mop6}
        for s1 in required_s1clodkz:
            if s1 not in s1_clodkz_with_dym:
                missing_dym1mop6 += 1

        cost += missing_dym1mop6

        # Heuristic step D: Fixing ggld6jmz readiness for actions l6w9j7tw/eav722gv:
        # If a ggld6jmz is involved in dym1mop6 (dym1mop6 ?c3xfw ?axwhj), but not in (xx5ljjai) or (okcl3dr0),
        # then we estimate at least 1 action to fix that (e.g., aegb5ot3 or ppnkvo80).

        ggld6jmz_in_dym = {ggld for (ggld, s1) in dym1mop6}
        not_ready_ggld = set()
        for ggldobj in ggld6jmz_in_dym:
            if ggldobj not in xx5ljjai and ggldobj not in okcl3dr0:
                not_ready_ggld.add(ggldobj)
        cost += len(not_ready_ggld)

        # Heuristic step E: Consider the need to fix (omcsgizs) facts – if needed for actions like nzh0a058.
        # If ggld6jmz involved are not omcsgizs, count 1 fix per such object.
        omcsgizs_set = omcsgizs
        missing_omcsgizs = 0
        for ggldobj in ggld6jmz_in_dym:
            if ggldobj not in omcsgizs_set:
                missing_omcsgizs += 1
        cost += missing_omcsgizs

        # Final heuristic cost is sum of all approximated fix action counts.
        # The heuristic assumes each missing or misaligned component roughly costs one action.

        return cost
