from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (aixg4uih) domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state, based primarily on the achievement of goal predicates of the form
    (qxafq0yb <z65nyce9-object>). The goal requires one or more of these predicates 
    (on certain z65nyce9-typed objects) to be true. Actions affecting these predicates
    depend on intermediate conditions involving predicates linked to ggld6jmz (a kind 
    of component/agent) and s1clodkz (some key or token).

    Assumptions:
    - Goal facts are of the form (qxafq0yb <z65nyce9-object>).
    - Every goal qxafq0yb fact requires enabling (qxafq0yb) on a certain z65nyce9-object.
    - Achieving (qxafq0yb ?r7dv9) is done by actions that require disabling the predicate 
      (dym1mop6 ?c3xfw ?axwhj) for some ggld6jmz and s1clodkz, and predicates 
      (xx5ljjai ?c3xfw) and (zmomsdb3 ?r7dv9) + a waexgd4o predicate relating ?r7dv9 and a u59rqa6u object,
      and qrohx31m linking s1clodkz and a u59rqa6u object.

    - The u59rqa6u objects are transferable (see action ss17onrb),
      so the heuristic assumes these can be moved in a fixed small number of steps.
    - The ggld6jmz predicates (okcl3dr0, omcsgizs, xx5ljjai) represent modes or states of components
      that gate whether action l6w9j7tw or nzh0a058 can be applied.
    - Some actions toggle states of these ggld6jmz and n4cwbwmy, vwkdymi3 typed objects,
      that are needed for goal achievement.
    - The heuristic simplifies all dependencies and approximates the minimum required transformations
      to enable the goal predicates (qxafq0yb ...).

    Heuristic Initialization:
    - Extract from goal the set of z65nyce9-objects for which (qxafq0yb ?) must hold.
    - Extract static predicates for quick membership tests, particularly waexgd4o relations,
      and zmomsdb3 facts for relevant z65nyce9-objects.
    - Extract initial states of ggld6jmz predicates and mappings qrohx31m (s1clodkz to u59rqa6u).
    
    Step-By-Step Thinking for Computing Heuristic:
    1. Count how many goal (qxafq0yb ?) facts are not yet satisfied.
       Each missing (qxafq0yb ?r) must eventually be achieved by applying l6w9j7tw or eav722gv actions.

    2. For each missing goal object ?r:
       - Check if (zmomsdb3 ?r) and (waexgd4o ?r ?imdah) hold for some u59rqa6u ?imdah.
       - Check if there is a ggld6jmz ?c3xfw with (xx5ljjai ?c3xfw) or (okcl3dr0 ?c3xfw) or (omcsgizs ?c3xfw)
         and (dym1mop6 ?c3xfw ?axwhj) for a s1clodkz ?axwhj such that (qrohx31m ?axwhj ?imdah) holds.
       This means a complex chain is needed to satisfy the preconditions of the goal-achieving action.

    3. Estimate costs for enabling ggld6jmz components to the required states:
       - To have (xx5ljjai ?c3xfw) and (omcsgizs ?c3xfw), we may need actions ppnkvo80 and nzh0a058.
       - To transfer u59rqa6u objects over s1clodkz keys we have ss17onrb.
       - Each subgoal roughly costs 1-3 actions to enable toggles or transfers.

    4. Sum approximations assuming each missing goal predicate requires about 3-5 key actions:
       - One or two state switches for ggld6jmz objects.
       - One key transfer action if needed.
       - One final goal achievement action l6w9j7tw or eav722gv.

    5. The heuristic is zero only if all goal (qxafq0yb) facts hold.

    This heuristic is not guaranteed admissible, but aims to prioritize states closer 
    to enabling the goal predicates by counting missing goals and estimating minimal transformations.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting goal objects and static facts for quick lookup."""
        self.goals = task.goals  # Goal facts (strings).
        self.static = task.static

        # Extract the set of z65nyce9-objects for which the goal requires (qxafq0yb ?r).
        self.goal_objs = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "qxafq0yb" and len(parts) == 2:
                self.goal_objs.add(parts[1])

        # Map waexgd4o for fast lookup: (r7dv9, u59rqa6u)
        self.static_waexgd4o = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) == 3:
                self.static_waexgd4o.add((parts[1], parts[2]))

        # Static set of (zmomsdb3 ?r7dv9) for quick membership.
        self.static_zmomsdb3 = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "zmomsdb3" and len(parts) == 2:
                self.static_zmomsdb3.add(parts[1])

    def __call__(self, node):
        """
        Compute the heuristic estimate for the number of actions to reach the goal.

        The heuristic works by:
        - Counting how many goal (qxafq0yb) facts are missing.
        - For each missing goal object, checking whether critical predicates 
          required to achieve this goal are currently enabled.
        - Estimating minimal enabling cost for toggling states of ggld6jmz components,
          transferring u59rqa6u objects with s1clodkz keys, and enabling dym1mop6 predicates.
        """
        state = node.state

        # Quick membership sets for current state predicates by predicate name.
        pred_dict = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred not in pred_dict:
                pred_dict[pred] = []
            pred_dict[pred].append(parts[1:])

        def has_pred(pred_name, *args):
            """Check if state has a fact pred_name(arg1, arg2, ...), args must match exactly."""
            if pred_name not in pred_dict:
                return False
            for fact_args in pred_dict[pred_name]:
                if len(args) == len(fact_args) and all(a == b for a, b in zip(args, fact_args)):
                    return True
            return False

        # Count how many goal (qxafq0yb ?r) missing.
        missing_goals = []
        for obj in self.goal_objs:
            if not has_pred("qxafq0yb", obj):
                missing_goals.append(obj)
        if not missing_goals:
            # Goal reached
            return 0

        # For each missing goal object estimate minimal number of actions to enable it.
        total_estimated_cost = 0
        for r7dv9 in missing_goals:
            # Check (zmomsdb3 ?r7dv9) - must hold to use l6w9j7tw or eav722gv -- if not in static or state, consider cost high.
            is_zmomsdb3 = (r7dv9 in self.static_zmomsdb3) or has_pred("zmomsdb3", r7dv9)

            # Find all (waexgd4o ?r7dv9 ?imdah) (u59rqa6u objects) in static or currently true.
            waex_candidates = [imdah for (r, imdah) in self.static_waexgd4o if r == r7dv9]
            # Also check current state waexgd4o, possibly transient
            if "waexgd4o" in pred_dict:
                waex_candidates += [args[1] for args in pred_dict["waexgd4o"] if args[0] == r7dv9]
            waex_candidates = list(set(waex_candidates))
            if not waex_candidates:
                # If no waexgd4o for this object, likely can't progress - assign medium cost
                total_estimated_cost += 5
                continue

            # For each candidate, try to find supporting (qrohx31m ?axwhj ?imdah) and (dym1mop6 ?c3xfw ?axwhj)
            # plus (xx5ljjai ?c3xfw) or (okcl3dr0 ?c3xfw) or (omcsgizs ?c3xfw)
            # Use any available combination that requires minimal enabling cost.

            # First gather all ggld6jmz objects from predicates
            ggld6jmz_objs = set()
            for p in ("okcl3dr0", "xx5ljjai", "omcsgizs"):
                if p in pred_dict:
                    for args in pred_dict[p]:
                        ggld6jmz_objs.add(args[0])

            # Gather all known (dym1mop6 ?c3xfw ?axwhj)
            dym1mop6_facts = []
            if "dym1mop6" in pred_dict:
                dym1mop6_facts = pred_dict["dym1mop6"]

            # Gather all (qrohx31m ?axwhj ?imdah)
            qrohx31m_facts = []
            if "qrohx31m" in pred_dict:
                qrohx31m_facts = pred_dict["qrohx31m"]

            # We try to find a tuple (?c3xfw ?axwhj ?imdah) such that
            # (dym1mop6 ?c3xfw ?axwhj), (qrohx31m ?axwhj ?imdah) holds,
            # with ?c3xfw in ggld6jmz_objs, and ?imdah in waex_candidates

            found_good_combo = False
            for (c3xfw, axwhj) in dym1mop6_facts:
                if c3xfw not in ggld6jmz_objs:
                    continue
                for (ax, imdah) in qrohx31m_facts:
                    if ax == axwhj and imdah in waex_candidates:
                        # Check if c3xfw is in a suitable "enabled" state in state:
                        # (xx5ljjai ?c3xfw) or (okcl3dr0 ?c3xfw) or (omcsgizs ?c3xfw)
                        c3xfw_enabled = (
                            has_pred("xx5ljjai", c3xfw)
                            or has_pred("okcl3dr0", c3xfw)
                            or has_pred("omcsgizs", c3xfw)
                        )
                        if c3xfw_enabled and is_zmomsdb3:
                            found_good_combo = True
                            break
                if found_good_combo:
                    break

            # Cost estimation:
            # If good combo exists, estimated cost to achieve goal for r7dv9 is about 1 action (l6w9j7tw or eav722gv)
            # else, add higher cost to represent toggling ggld6jmz states, transferring keys, and enabling predicates.

            if found_good_combo:
                total_estimated_cost += 2  # 1 for goal action + some buffer for minimal enabling.
            else:
                # If no good combo, estimate costs for enabling states:
                # Assume toggling ggld6jmz states twice, a key transfer (ss17onrb), and final action.
                total_estimated_cost += 5

        return total_estimated_cost
