from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (aixg4uih).

    Summary:
    ----------
    This heuristic estimates how many actions are approximately needed to achieve the goal conditions 
    by focusing primarily on the key goal predicate `(qxafq0yb ?r7dv9)` and the relevant preconditions 
    and operators that produce them. 
    The heuristic counts the number of goal-related "z65nyce9" objects that have not yet been marked 
    with the predicate `(qxafq0yb ?r7dv9)`. To estimate the effort, it tracks supporting predicates 
    and intermediary predicates needed to achieve `(qxafq0yb)` for these objects, along with 
    relevant resources (predicates and object bindings) needed, guided by domain operators.

    Assumptions:
    ----------
    - The main goal predicates are `(qxafq0yb <z65nyce9-object>)`.
    - To achieve `(qxafq0yb ?r7dv9)`, at least one of the actions `l6w9j7tw` or `eav722gv` must be applied.
    - Those actions require specific supporting facts about other object types and predicates.
    - Some predicates indicate resources or states that must be arranged for success (e.g., 
      `(dym1mop6 ?c3xfw ?axwhj)`, `(xx5ljjai ?c3xfw)`, `(omcsgizs ?c3xfw)`).
    - The action `ss17onrb` moves `(qrohx31m ?axwhj ?imdah)` between `u59rqa6u` objects and can be used to 
      prepare this predicate on desired objects.
    - The constant `s154x56h` (type `u59rqa6u`) appears special and is used in goal-related action preconditions.
    - Heuristic is not necessarily admissible, only informative to prune the search efficiently.

    Heuristic Initialization:
    ----------
    - The set of goal objects that must achieve `(qxafq0yb ?r7dv9)` is extracted by parsing the goal facts.
    - Static facts (from `task.static`) indexed by predicates `(waexgd4o ?r7dv9 ?imdah)` and `(zmomsdb3 ?r7dv9)` 
      and others are stored in dictionaries for quick lookup.
    - The unique constant `s154x56h` is stored for specialized checks.
    - The heuristic prepares mappings for relevant predicate instances per object to estimate needed steps.

    Step-By-Step Thinking for Computing Heuristic:
    ----------
    1. Extract from the current state all objects for which `(qxafq0yb ?r7dv9)` is true (achieved goals).
    2. Count how many goal objects (the ones required in goals) lack this predicate.
    3. For each missing goal object:
       - Check if it satisfies certain supporting predicates that enable actions to achieve `(qxafq0yb)`.
       - Check for presence of intermediary predicates like `(dym1mop6 ?c3xfw ?axwhj)`, `(xx5ljjai ?c3xfw)`, `(omcsgizs ?c3xfw)`.
       - Check if the correct `(qrohx31m ?axwhj ?imdah)` binding exists, especially for the special constant `s154x56h`.
       - Confirm required predicates `(waexgd4o ?r7dv9 ?imdah)` and `(zmomsdb3 ?r7dv9)` are in state or static.
    4. Estimate the minimum number of action steps to transition missing goal objects from the current state 
       to the achieved state `(qxafq0yb ?r7dv9)` based on which predicates are missing and which must be produced 
       or moved. For example:
       - Missing `(qxafq0yb)` means at least one relevant action must be applied.
       - Missing `(dym1mop6 ?c3xfw ?axwhj)` and `(xx5ljjai ?c3xfw)` means actions to produce `dym1mop6` or `xx5ljjai` are needed.
       - Changing `(qrohx31m ?axwhj ?imdah)` requires one or more `ss17onrb` actions.
       - Certain predicates correspond to resource changes carried out by actions like `aegb5ot3` or `ppnkvo80`.
    5. Accumulate these minimal action costs for all missing goal objects; this sum is the heuristic estimate.
    6. Return 0 if all goal objects already satisfy `(qxafq0yb)`.
    7. Ensure the heuristic is always finite and efficiently computable by bounding cost increments and using known static facts.

    Overall, this heuristic synthesizes the layered dependencies of predicates and actions to guess how far 
    the current state is from achieving the goal predicates `(qxafq0yb ?)`.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.s154x56h = "s154x56h"  # The unique u59rqa6u constant in domain (given in constants).

        # Extract goal objects for (qxafq0yb ?r7dv9)
        self.goal_z65nyce9 = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "qxafq0yb" and len(parts) == 2:
                self.goal_z65nyce9.add(parts[1])

        # Index static facts by predicate for quick lookup
        self.static_waexgd4o = set()
        self.static_zmomsdb3 = set()
        self.static_qrohx31m = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) == 3:
                self.static_waexgd4o.add((parts[1], parts[2]))  # (r7dv9, imdah)
            elif parts[0] == "zmomsdb3" and len(parts) == 2:
                self.static_zmomsdb3.add(parts[1])  # r7dv9
            elif parts[0] == "qrohx31m" and len(parts) == 3:
                self.static_qrohx31m.add((parts[1], parts[2]))  # (axwhj, imdah)

    def __call__(self, node):
        state = node.state

        # Quick helper sets for current state predicates
        qxafq0yb_achieved = {get_parts(fact)[1] for fact in state if fact.startswith("(qxafq0yb ")}
        zmomsdb3_current = {get_parts(fact)[1] for fact in state if fact.startswith("(zmomsdb3 ")}
        waexgd4o_current = set()
        for fact in state:
            if fact.startswith("(waexgd4o "):
                parts = get_parts(fact)
                waexgd4o_current.add((parts[1], parts[2]))
        qrohx31m_current = set()
        for fact in state:
            if fact.startswith("(qrohx31m "):
                parts = get_parts(fact)
                qrohx31m_current.add((parts[1], parts[2]))
        dym1mop6_current = set()
        for fact in state:
            if fact.startswith("(dym1mop6 "):
                parts = get_parts(fact)
                dym1mop6_current.add((parts[1], parts[2]))
        xx5ljjai_current = {get_parts(fact)[1] for fact in state if fact.startswith("(xx5ljjai ")}
        omcsgizs_current = {get_parts(fact)[1] for fact in state if fact.startswith("(omcsgizs ")}
        okcl3dr0_current = {get_parts(fact)[1] for fact in state if fact.startswith("(okcl3dr0 ")}
        g7wohkyk_current = {get_parts(fact)[1] for fact in state if fact.startswith("(g7wohkyk ")}
        sl9kfffg_current = {get_parts(fact)[1] for fact in state if fact.startswith("(sl9kfffg ")}
        mcoxi8qw_current = {get_parts(fact)[1] for fact in state if fact.startswith("(mcoxi8qw ")}
        waexgd4o_all = waexgd4o_current | self.static_waexgd4o
        zmomsdb3_all = zmomsdb3_current | self.static_zmomsdb3
        qrohx31m_all = qrohx31m_current | self.static_qrohx31m

        # The heuristic cost accumulator
        total_cost = 0

        # If all goal objects have (qxafq0yb), heuristic is 0 (goal reached)
        if self.goal_z65nyce9.issubset(qxafq0yb_achieved):
            return 0

        # For each goal object not yet achieved with (qxafq0yb)
        for r7dv9 in self.goal_z65nyce9:
            if r7dv9 in qxafq0yb_achieved:
                continue  # Already achieved

            # To achieve (qxafq0yb r7dv9), at least one of l6w9j7tw or eav722gv needed.
            # Analyze preconditions for those actions to estimate cost.

            # Preconditions for l6w9j7tw:
            # (dym1mop6 ?c3xfw ?axwhj), (xx5ljjai ?c3xfw), (zmomsdb3 ?r7dv9), (waexgd4o ?r7dv9 ?imdah), (qrohx31m ?axwhj ?imdah)
            # Effects includes (not (dym1mop6 ?c3xfw ?axwhj)) and (qxafq0yb ?r7dv9)

            # Preconditions for eav722gv:
            # (qrohx31m ?axwhj ?imdah), (dym1mop6 ?c3xfw ?axwhj), (mcoxi8qw ?r7dv9), (waexgd4o ?r7dv9 ?imdah)
            # Effects: (qxafq0yb ?r7dv9), (not (dym1mop6 ?c3xfw ?axwhj))

            # Check if zmomsdb3 or mcoxi8qw predicates are true for r7dv9 (they seem mutually exclusive):
            has_zmomsdb3 = r7dv9 in zmomsdb3_all
            has_mcoxi8qw = r7dv9 in mcoxi8qw_current

            # waexgd4o predicate: need at least one binding with r7dv9 and some imdah
            waexgd4o_bindings = [imdah for (r, imdah) in waexgd4o_all if r == r7dv9]

            # If no waexgd4o bindings, must perform ss17onrb to create needed (qrohx31m ?axwhj ?imdah)
            if not waexgd4o_bindings:
                # Forced to move (qrohx31m) bindings by ss17onrb actions (likely > 1 step)
                # Estimating a penalty cost for establishing waexgd4o predicate presence
                total_cost += 3
                continue  # Without waexgd4o, achieving goal is difficult; no need for further checks here.

            # For each relevant waexgd4o binding (r7dv9, imdah), check if (qrohx31m ?axwhj ?imdah) exists:
            # We try to find some axwhj such that (qrohx31m axwhj imdah) in state or static:
            possible_axwhj_for_imdah = [axwhj for (axwhj, imdah2) in qrohx31m_all if imdah2 in waexgd4o_bindings]
            if not possible_axwhj_for_imdah:
                total_cost += 2  # Need to create (qrohx31m) relation via ss17onrb
                continue

            # Check if dym1mop6 predicate exists for any c3xfw with those axwhj:
            dym1mop6_for_axwhj = [(c3xfw, ax) for (c3xfw, ax) in dym1mop6_current if ax in possible_axwhj_for_imdah]

            # Also, check if any c3xfw in those dym1mop6 has (xx5ljjai c3xfw):
            c3xfw_with_xx5ljjai = {c3 for c3, ax in dym1mop6_for_axwhj if c3 in xx5ljjai_current}

            # If none of the above, we need to produce dym1mop6 and xx5ljjai predicates with actions (ppnkvo80, aegb5ot3, nzh0a058)
            # Estimate higher cost if these are not available:
            if not dym1mop6_for_axwhj or len(c3xfw_with_xx5ljjai) == 0:
                total_cost += 5  # Steps needed to create these supporting predicates
                continue

            # Further checks for omcsgizs predicate on c3xfw may reduce uncertainty:
            c3xfw_with_omcsgizs = {c3 for c3, ax in dym1mop6_for_axwhj if c3 in omcsgizs_current}
            if len(c3xfw_with_omcsgizs) == 0:
                total_cost += 3  # Need to create omcsgizs via aegb5ot3 before l6w9j7tw action

            # Check if (mcoxi8qw r7dv9) is present, which allows eav722gv action path
            if not has_mcoxi8qw:
                total_cost += 1  # Likely need to achieve this predicate too

            # Minimal cost of applying the final l6w9j7tw or eav722gv action to turn goal predicate true
            total_cost += 1

        # Heuristic always returns finite number; for unreachable states cost may be higher but finite
        return total_cost
