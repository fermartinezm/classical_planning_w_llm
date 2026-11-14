from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by removing surrounding parentheses and splitting by spaces.
    Example: "(qxafq0yb rap6j3pu)" -> ["qxafq0yb", "rap6j3pu"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (aixg4uih domain).

    Summary:
    This heuristic estimates the minimal number of actions required to reach a goal state by mainly
    focusing on achieving the goal predicate (qxafq0yb ?z65nyce9) for all required objects.
    The domain involves toggling predicates related to objects of types z65nyce9 and ggld6jmz,
    combined with conditions involving s1clodkz and u59rqa6u objects.
    The heuristic counts how many goal qxafq0yb predicates are still missing and estimates
    the cost to achieve these using the relevant predicates and actions observed.

    Assumptions:
    - The goal requires a set of objects of type z65nyce9 to hold the predicate (qxafq0yb ?).
    - The main achievable predicates for goal qxafq0yb depend on removing a (dym1mop6 ?ggld6jmz ?s1clodkz)
      predicate via actions that require specific conditions on the ggld6jmz and z65nyce9 objects,
      also involving u59rqa6u objects with qrohx31m predicates.
    - The static facts relate to waexgd4o predicates linking z65nyce9 and u59rqa6u,
      and the goal relies on the appropriate configuration of these predicates.
    - We approximate costs by counting unsatisfied goal qxafq0yb predicates and estimating the 
      number of "dym1mop6" facts and "okcl3dr0" / "xx5ljjai" predicates necessary for achieving
      these goals.
    - The heuristic is not admissible but aims to be informative and easy to compute.

    Heuristic Initialization:
    - Extract the static mappings:
      - For each z65nyce9 object, record which u59rqa6u objects it is linked to via waexgd4o.
    - Extract the set of goal objects of type z65nyce9 that must have qxafq0yb true.
    - Extract all constants of type u59rqa6u for quick reference.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which goal objects of type z65nyce9 still do not have (qxafq0yb ?).
       Each missing qxafq0yb will require applying:
       - One of the actions l6w9j7tw or eav722gv that produce (qxafq0yb ?).
    2. For each missing goal z65nyce9 objects:
       - Check if the predicate (mcoxi8qw ?z65nyce9) is true. This is often a static condition but checked for consistency.
       - Find related u59rqa6u objects (from static waexgd4o) linked to this z65nyce9.
       - For each such u59rqa6u, find s1clodkz objects linked via qrohx31m.
       - Check in the state whether (dym1mop6 ?ggld6jmz ?s1clodkz) holds for some ggld6jmz and s1clodkz.
         Also check if (okcl3dr0 ?ggld6jmz) or (xx5ljjai ?ggld6jmz) hold, since different actions require these.
    3. Estimate cost components:
       - Removing a (dym1mop6 ?ggld6jmz ?s1clodkz) predicate costs at least one action (l6w9j7tw or eav722gv).
       - If (xx5ljjai) is needed, it requires ppnkvo80 action to make omcsgizs true, which is a prerequisite for l6w9j7tw.
       - If (omcsgizs) is missing, nzh0a058 can be applied once with s154x56h constant to get (dym1mop6).
       - Reassigning u59rqa6u on qrohx31m predicates (ss17onrb action) might also be needed if u59rqa6u differ.
    4. For simplicity and efficiency:
       - Count missing goals qxafq0yb (M).
       - Count number of dym1mop6 predicates currently true (D).
       - Count number of ggld6jmz objects with okcl3dr0 or xx5ljjai true.
       - Assume roughly 1-3 actions per missing qxafq0yb depending on readiness of supporting predicates.
    5. Return a heuristic value that weighs these factors, assigning 0 if all qxafq0yb goals are met.

    Note: The heuristic is designed heuristically using domain analysis and might overestimate or underestimate,
    but it is efficiently computable from available state and static predicates.
    """

    def __init__(self, task):
        # Extract goals of type (qxafq0yb ?z65nyce9) to know which z65nyce9 objects need qxafq0yb true
        self.goal_z65 = {
            get_parts(goal)[1]
            for goal in task.goals
            if get_parts(goal)[0] == "qxafq0yb"
        }

        # Extract static waexgd4o facts linking z65nyce9 -> u59rqa6u
        self.z65_to_u59rqa6u = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o":
                z_obj, u_obj = parts[1], parts[2]
                self.z65_to_u59rqa6u.setdefault(z_obj, set()).add(u_obj)

        # Extract all u59rqa6u constants (important: one constant named s154x56h always exists)
        self.u59rqa6u_consts = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o":
                self.u59rqa6u_consts.add(parts[2])
        # s154x56h constant is also a known constant of type u59rqa6u
        self.u59rqa6u_consts.add("s154x56h")

    def __call__(self, node):
        state = node.state

        # Quickly check if the goal is already reached
        # Heuristic must return 0 in that case
        # self.goal_z65 are all z65 objects that must have (qxafq0yb ?)
        qxafq0yb_true = {
            get_parts(fact)[1]
            for fact in state
            if fact.startswith("(qxafq0yb ")
        }
        missing_goal_objects = self.goal_z65 - qxafq0yb_true
        if not missing_goal_objects:
            return 0  # Goal reached

        # Count how many dym1mop6 predicates are currently true
        current_dym1mop6 = [get_parts(fact) for fact in state if fact.startswith("(dym1mop6 ")]
        dym1mop6_count = len(current_dym1mop6)

        # Count how many ggld6jmz have okcl3dr0 and xx5ljjai predicates true
        okcl3dr0_true = set()
        xx5ljjai_true = set()
        for fact in state:
            if fact.startswith("(okcl3dr0 "):
                okcl3dr0_true.add(get_parts(fact)[1])
            elif fact.startswith("(xx5ljjai "):
                xx5ljjai_true.add(get_parts(fact)[1])

        # Also count how many ggld6jmz have omcsgizs predicate true (needed for some actions)
        omcsgizs_true = set()
        for fact in state:
            if fact.startswith("(omcsgizs "):
                omcsgizs_true.add(get_parts(fact)[1])

        # Count how many (qrohx31m ?s1clodkz ?u59rqa6u) predicates are true,
        # since these link s1clodkz and u59rqa6u, used in several actions
        qrohx31m_true = set()
        for fact in state:
            if fact.startswith("(qrohx31m "):
                parts = get_parts(fact)
                qrohx31m_true.add((parts[1], parts[2]))

        # Heuristic computation:
        # For each missing qxafq0yb:
        #   - We need at least one action l6w9j7tw or eav722gv which requires dym1mop6 true on a ggld6jmz with s1clodkz
        #   - If no dym1mop6 currently, must run nzh0a058 once to create dym1mop6
        #   - If ggld6jmz lacks okcl3dr0 or xx5ljjai, may need ppnkvo80 to switch states
        #   - Possibly ss17onrb actions to reassign u59rqa6u in qrohx31m predicates

        # Simple scoring model:
        # base_cost = number of missing qxafq0yb, as each requires an action to achieve
        base_cost = len(missing_goal_objects)

        # bonus for lacking dym1mop6 predicates (means extra action to create them)
        dym1mop6_needed = 1 if dym1mop6_count == 0 else 0

        # For each missing qxafq0yb, check how "ready" it is:
        readiness_penalty = 0

        for z_obj in missing_goal_objects:
            # Check if mcoxi8qw is present for z_obj (should be true or static)
            mcoxi8qw_present = any(
                fact == f"(mcoxi8qw {z_obj})" for fact in state
            )  # Assume always given once static

            # Get linked u59rqa6u for z_obj
            u_objs = self.z65_to_u59rqa6u.get(z_obj, set())

            # For each u_obj, check for s1clodkz objects linked by qrohx31m
            # quick count how many s1clodkz have qrohx31m to u_obj
            s1clodkz_for_u = {
                s for (s, u) in qrohx31m_true if u in u_objs
            }

            # Check if at least one (dym1mop6 ?ggld6jmz ?s1clodkz) for this s1clodkz is present
            dym1mop6_ready = False
            for d_fact in current_dym1mop6:
                ggld, s1 = d_fact[1], d_fact[2]
                if s1 in s1clodkz_for_u:
                    # Also check if ggld has okcl3dr0 or xx5ljjai, since these enable different actions
                    if (ggld in okcl3dr0_true) or (ggld in xx5ljjai_true):
                        dym1mop6_ready = True
                        break

            if not dym1mop6_ready:
                # Missing a crucial dym1mop6 predicate for this goal object
                readiness_penalty += 2  # Penalize this lack

            # Penalize if no okcl3dr0 or xx5ljjai on any ggld6jmz to help reach the action
            if not okcl3dr0_true and not xx5ljjai_true:
                readiness_penalty += 1

            # Penalize if qrohx31m predicates for this z_obj's u59rqa6u are minimal (hard to reassign)
            if len(s1clodkz_for_u) == 0:
                readiness_penalty += 1

        # Total heuristic estimate:
        # base_cost + penalty for missing predicates + 1 if we must create dym1mop6 from scratch
        heuristic_value = base_cost + readiness_penalty + dym1mop6_needed

        # Return heuristic value, at least 1 (if not goal) to guide search
        return max(1, heuristic_value)
