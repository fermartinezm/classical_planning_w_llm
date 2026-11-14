from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a fact string like '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the "spanner" / secret_domain domain.

    This heuristic estimates the number of actions required to achieve all
    goal facts of the form (tightened-2 <locatable> <location>).
    It is designed to be cheaply computable while capturing the main
    structural constraints that enable creation of a tightened-2 fact:
      - the existence of a nut that has a loose-2 fact at the desired location,
      - that nut currently being tightened at some locatable,
      - the relation between that locatable and the target locatable
        (one of loose / at / carrying / usable),
      - and the target locatable having the `link` flag (required by many
        actions that create tightened-2).
    The heuristic is not required to be admissible; it aims to reduce node
    expansions by favoring states that already satisfy many of those
    enabling conditions.

    Assumptions
    -----------
    - Goals are conjunctions of (tightened-2 <locatable> <location>) facts.
    - Static relation facts (loose, at, carrying, usable, at-2, ...) are
      available in task.static and represent structural adjacency used by
      actions.
    - Dynamic facts in the state include (tightened ...), (loose-2 ...),
      and (link ...). The presence of (link L) on a locatable L is usually
      required by the actions that produce tightened-2, so lacking that flag
      means at least one extra action is needed to move or create the link.
    - If multiple goals require the same enabling resources, we simply sum
      per-goal estimated costs (this is intentionally optimistic/pessimistic
      trade-off that helps greedy search without expensive global reasoning).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goals: the set of goal fact strings.
    - static relation sets built from task.static:
        - self.static_loose: set of (locA, locB) for (loose locA locB)
        - self.static_at: set of (locA, locB) for (at locA locB)
        - self.static_carrying: set of (locA, locB) for (carrying locA locB)
        - self.static_usable: set of (locA, locB) for (usable locA locB)
      These are used to quickly test whether a target locatable L is related
      to a locatable V where a nut is currently tightened (a common
      precondition pattern for actions that produce tightened-2).
    - No other heavy precomputation is performed; everything else is
      computed from the current state inside __call__.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal fact g = (tightened-2 L Loc):
    1. If g already holds in the current state, cost contribution is 0.
    2. Otherwise, identify the following from the current state:
       - tightened_pairs: all (nut, V) such that (tightened nut V) holds.
       - loose2_pairs: all (nut, Location) such that (loose-2 nut Location) holds.
       - link_set: all locatables X such that (link X) holds.
    3. We try to detect whether a single action can produce the goal:
       - Many domain actions that directly add (tightened-2 L Loc) require:
         * a nut N with (loose-2 N Loc)
         * that same nut N tightened at some locatable V: (tightened N V)
         * L being related to V via one of the static relations:
           loose(L,V) OR at(L,V) OR carrying(L,V) OR usable(L,V)
         * and (link L) present in the current state.
       - If all these conditions hold for some N and V, we estimate base_cost = 1.
    4. If there is some nut N with (loose-2 N Loc) and a tightened(N,V) but
       L is not directly related to that V (so an intermediate move is needed),
       we estimate base_cost = 2 (one move to position the tightened nut
       relative to L, then one action to produce tightened-2).
    5. If the location Loc has some loose-2 nut but no tightened(N,V) in the
       current state for that nut N, we estimate base_cost = 3 (we assume
       extra work needed to create or reposition tightened on some locatable).
    6. If there is no nut with (loose-2 N Loc) in the state at all, we estimate
       a larger base_cost = 4 (we must first move or create a loose-2 nut at Loc,
       then position tightened accordingly and finally produce tightened-2).
    7. Finally, many actions that create (tightened-2 L Loc) require (link L).
       If (link L) is not present in the current state we add +1 to the per-goal
       cost to account for the action(s) required to make L linked.
    8. The heuristic value is the sum of these per-goal estimates for all
       missing tightened-2 goal facts.
    This scheme is quick to compute, always finite for solvable states, and
    returns 0 exactly when all goals are already satisfied.

    Note: This heuristic intentionally uses simple local checks derived from
    common action preconditions in the domain (tightened, loose-2, link and
    the four static relations). It trades exactness for speed and is tailored
    to prefer states where nuts are already in useful positions and target
    locatables have the link flag set.
    """

    def __init__(self, task):
        # store goal facts (strings)
        self.goals = task.goals

        # store static facts into convenient relation sets
        static_facts = task.static

        self.static_loose = set()
        self.static_at = set()
        self.static_carrying = set()
        self.static_usable = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "loose" and len(parts) >= 3:
                self.static_loose.add((parts[1], parts[2]))
            elif pred == "at" and len(parts) >= 3:
                self.static_at.add((parts[1], parts[2]))
            elif pred == "carrying" and len(parts) >= 3:
                self.static_carrying.add((parts[1], parts[2]))
            elif pred == "usable" and len(parts) >= 3:
                self.static_usable.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Quick check: if goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse dynamic state facts relevant for heuristic
        tightened_pairs = set()  # (nut, locatable)
        loose2_pairs = set()  # (nut, location)
        link_set = set()  # locatables with link

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "tightened" and len(parts) >= 3:
                tightened_pairs.add((parts[1], parts[2]))
            elif pred == "loose-2" and len(parts) >= 3:
                loose2_pairs.add((parts[1], parts[2]))
            elif pred == "link" and len(parts) >= 2:
                link_set.add(parts[1])

        # Helper to test whether locatable L is related to V by one of the static relations
        def is_related(L, V):
            return (L, V) in self.static_loose or (L, V) in self.static_at or (
                L,
                V,
            ) in self.static_carrying or (L, V) in self.static_usable

        total_cost = 0

        # Consider only tightened-2 goals (other goal predicates are not expected
        # in this domain, but we handle generically by ignoring non-tightened-2)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "tightened-2" or len(parts) < 3:
                # non-target goal: if it's not in state we conservatively add 1
                if goal not in state:
                    total_cost += 1
                continue

            target_locatable = parts[1]
            target_location = parts[2]

            if goal in state:
                continue  # already achieved

            # Find nuts that have loose-2 at the desired target_location
            nuts_with_loose2_at_loc = {nut for (nut, loc) in loose2_pairs if loc == target_location}

            # Base cost estimation
            base_cost = None

            if nuts_with_loose2_at_loc:
                # Try to find a nut N that is already tightened at some locatable V
                # such that V is related to target_locatable (one action case).
                found_one_action = False
                found_any_tightened = False
                for N in nuts_with_loose2_at_loc:
                    for (nut, V) in tightened_pairs:
                        if nut != N:
                            continue
                        found_any_tightened = True
                        if is_related(target_locatable, V):
                            found_one_action = True
                            break
                    if found_one_action:
                        break

                if found_one_action:
                    base_cost = 1
                elif found_any_tightened:
                    # There is a tightened(N,V) at some V, but not directly related:
                    # assume it takes one move to position, and one action to create tightened-2.
                    base_cost = 2
                else:
                    # loose-2 exists at location but nothing tightened for that nut in state:
                    # assume more work needed to create tightened at some locatable first.
                    base_cost = 3
            else:
                # No nut has loose-2 at the required location: we must move/create one.
                # Conservative estimate: 4 actions.
                base_cost = 4

            # If link is required but not present, add extra cost for acquiring link
            # (many domain operators require (link target_locatable) as a precondition).
            link_penalty = 0 if target_locatable in link_set else 1

            total_cost += base_cost + link_penalty

        return total_cost