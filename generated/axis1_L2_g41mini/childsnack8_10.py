from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal condition(s) in the secret_domain.
    The primary goal literal(s) are of the form (qxafq0yb <z65nyce9_object>).
    Achieving (qxafq0yb <obj>) requires removing certain predicates (like dym1mop6) and enabling (qxafq0yb)
    on that object using the domain's actions. 

    Assumptions:
    - The goal is conjunction(s) of facts (qxafq0yb ?r7dv9).
    - To achieve (qxafq0yb ?r7dv9), the predicate (zmomsdb3 ?r7dv9) and (waexgd4o ?r7dv9 ?imdah) must hold for some ?imdah.
    - The action l6w9j7tw can achieve (qxafq0yb ?r7dv9) by deleting (dym1mop6 ?c3xfw ?axwhj) and requires (dym1mop6 ?c3xfw ?axwhj), (xx5ljjai ?c3xfw), (zmomsdb3 ?r7dv9), (waexgd4o ?r7dv9 ?imdah), and (qrohx31m ?axwhj ?imdah).
    - (dym1mop6) facts relate ggld6jmz and s1clodkz objects; these sometimes must be created via action nzh0a058 from (omcsgizs) and (qrohx31m ?axwhj s154x56h).
    - Some predicates toggle the state of objects: (okcl3dr0), (xx5ljjai), (omcsgizs).
    - The action ppnkvo80 can switch (okcl3dr0) to (xx5ljjai) and (omcsgizs).
    - The action aegb5ot3 can enable (omcsgizs) and disables other predicates.
    - Heuristic works by counting the needed (qxafq0yb) goal literals unmet,
      then estimates the required intermediate steps: enabling (xx5ljjai) and (omcsgizs),
      creating (dym1mop6), and moving the (qrohx31m) facts as needed.

    Heuristic Initialization:
    - Extract the goal facts of form (qxafq0yb ?obj).
    - Extract static mappings (waexgd4o predicate linking z65nyce9 to u59rqa6u).
    - Collect static facts such as (zmomsdb3 ?r7dv9).
    - Map and index relevant predicates for quick membership checking.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which goal (qxafq0yb ?r7dv9) facts are already true. Ignore those.
    2. For each goal fact not satisfied:
       a. Check if (zmomsdb3 ?r7dv9) holds; if not, must enable it first (estimate cost=1).
       b. Check if (waexgd4o ?r7dv9 ?imdah) exists for some ?imdah (from static facts).
       c. Check if for that ?imdah there is a (qrohx31m ?axwhj ?imdah) fact in current state,
          if not, one must apply transfer action ss17onrb (cost += 1).
       d. Check if there exists some ggld6jmz object ?c3xfw with (dym1mop6 ?c3xfw ?axwhj).
          If not, it can be created by applying nzh0a058 (cost += 1), requiring (omcsgizs ?c3xfw).
       e. To enable (omcsgizs ?c3xfw) may require applying aegb5ot3 or ppnkvo80 actions (cost estimate).
       f. Enable (xx5ljjai ?c3xfw) if not present to allow l6w9j7tw use.
       g. Finally, applying l6w9j7tw action achieves (qxafq0yb ?r7dv9) (cost += 1).
    3. Sum estimated actions needed for all missing goal facts.
    4. Return summed cost as heuristic estimate.

    This heuristic is not guaranteed to be admissible but aims to approximate the minimal steps
    balancing computation time and heuristic informativeness.
    """

    def __init__(self, task):
        self.goals = task.goals

        # Extract goal objects for (qxafq0yb ?r7dv9)
        self.goal_objs = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "qxafq0yb":
                self.goal_objs.add(parts[1])

        # Static mapping: waexgd4o links (z65nyce9 obj) to (u59rqa6u obj)
        self.waexgd4o = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o":
                zobj, uobj = parts[1], parts[2]
                self.waexgd4o[zobj] = uobj

        # Collect static facts (zmomsdb3), indicating readiness of z65nyce9 objects
        self.zmomsdb3_objs = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "zmomsdb3":
                self.zmomsdb3_objs.add(parts[1])

        # Store all ggld6jmz objects occurring in static facts (for enumeration)
        self.ggld6jmz_objs = set()
        for fact in task.static:
            parts = get_parts(fact)
            # We gather fixed ggld6jmz objects by their appearance in typical predicates
            if parts[0] in ["okcl3dr0", "xx5ljjai", "omcsgizs", "dym1mop6"]:
                self.ggld6jmz_objs.add(parts[1])

        # Store all s1clodkz objects from static facts (for enumeration)
        self.s1clodkz_objs = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "qrohx31m" and parts[1] not in self.s1clodkz_objs:
                self.s1clodkz_objs.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick membership checks
        state_facts = state
        qxafq0yb_true = {get_parts(f)[1] for f in state_facts if f.startswith("(qxafq0yb ")}

        # Heuristic is zero if all goals satisfied
        if self.goal_objs <= qxafq0yb_true:
            return 0

        total_cost = 0

        # Helper to check presence of predicate with given args in state
        def pred_in_state(pred, *args):
            search = "(" + " ".join((pred,) + args) + ")"
            return search in state_facts

        for goal_obj in self.goal_objs:
            if goal_obj in qxafq0yb_true:
                # Goal already met, no cost.
                continue

            # 1) Check if zmomsdb3 ?goal_obj holds; if not, estimate cost 1 to enable it
            if goal_obj not in self.zmomsdb3_objs and not pred_in_state("zmomsdb3", goal_obj):
                total_cost += 1  # Cost to enable zmomsdb3 (approximate)
            
            # 2) waexgd4o must exist for goal_obj to some ?imdah (from static)
            if goal_obj in self.waexgd4o:
                u_obj = self.waexgd4o[goal_obj]
            else:
                # If no waexgd4o mapping, it's likely unsolvable; assign a high cost
                total_cost += 5
                continue  # Skip to next goal_obj

            # 3) Need qrohx31m ?axwhj ?imdah in state for some ?axwhj in s1clodkz
            # Try to find any ?axwhj in state for qrohx31m ?axwhj u_obj
            qrohx_candidates = {get_parts(f)[1] for f in state_facts if f.startswith("(qrohx31m ") and get_parts(f)[2] == u_obj}
            if not qrohx_candidates:
                # Try to create or transfer via ss17onrb (heuristic cost 1)
                total_cost += 1
                # For heuristic, pick one candidate from static s1clodkz if known else 1
                qrohx_candidates = set()
                if self.s1clodkz_objs:
                    qrohx_candidates.add(next(iter(self.s1clodkz_objs)))
                else:
                    qrohx_candidates.add("axwhj_dummy")

            # Select one ?axwhj to reason about
            axwhj = next(iter(qrohx_candidates))

            # 4) Check for some ggld6jmz ?c3xfw with (dym1mop6 ?c3xfw ?axwhj) in state
            dym1mop6_facts = [f for f in state_facts if f.startswith("(dym1mop6 ")]
            dym1mop6_available = False
            related_c3xfw = None
            for f in dym1mop6_facts:
                parts = get_parts(f)
                if parts[2] == axwhj:
                    dym1mop6_available = True
                    related_c3xfw = parts[1]
                    break

            # If no such (dym1mop6) fact, estimate cost += 1 for creating it (nzh0a058)
            if not dym1mop6_available:
                # Need to enable (omcsgizs ?c3xfw) first; check if any ggld6jmz with omcsgizs
                omcsgizs_in_state = {get_parts(f)[1] for f in state_facts if f.startswith("(omcsgizs ")}
                if omcsgizs_in_state:
                    related_c3xfw = next(iter(omcsgizs_in_state))
                elif self.ggld6jmz_objs:
                    related_c3xfw = next(iter(self.ggld6jmz_objs))
                    # Cost to enable omcsgizs on related_c3xfw (aegb5ot3 or ppnkvo80): +1
                    total_cost += 1
                else:
                    # No ggld6jmz known, assign penalty
                    total_cost += 3
                total_cost += 1  # Create (dym1mop6) with nzh0a058

            # 5) Check if (xx5ljjai ?c3xfw) is in state; if not, +1 cost to enable (ppnkvo80)
            if related_c3xfw and not pred_in_state("xx5ljjai", related_c3xfw):
                total_cost += 1

            # 6) Finally, apply l6w9j7tw to achieve (qxafq0yb ?goal_obj) - cost 1
            total_cost += 1

        # Return heuristic estimate (finite and 0 only if goals met)
        return total_cost
