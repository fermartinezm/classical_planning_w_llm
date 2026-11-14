from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting by space."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern.
    Wildcards '*' are allowed in the pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (content-portion) domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state by focusing on the "no_gluten_bread-2" goals for (content-portion-2, tray)
    pairs as specified in the goal. Informally, the domain is about preparing sandwiches
    with gluten/no-gluten constraints, moving content portions and bread portions,
    handling trays and sandwiches, serving children, and maintaining waiting places.

    The heuristic primarily counts how many goal-required (content-portion-2, tray)
    pairs still need to satisfy the predicate (no_gluten_bread-2 content-portion-2 tray),
    and estimates the minimal number of actions required to achieve these conditions.
    It incorporates relevant static and state facts to estimate complexities such as
    waiting places that must be cleared and trays on which sandwiches must be prepared.

    Assumptions:
    - The goal predicate of interest is "no_gluten_bread-2" with (content-portion-2, tray).
    - Getting a pair to the goal requires a sequence of specific actions,
      roughly involving: preparing no_gluten_bread-2 at trays for content portions.
    - Waiting places introduce dependencies (a sandwich can only be prepared if 
      its place is waiting).
    - The heuristic need not be admissible but should guide search efficiently.
    - Certain intermediate predicates like "waiting", "waiting-2", and "served"
      impose necessary ordering.

    Heuristic Initialization:
    - Extract from static facts the served relations between bread-portions.
    - Extract the goals for (content-portion-2, tray) pairs requiring no_gluten_bread-2.
    - Extract sets of waiting places and waiting-2 facts.
    - Extract relevant static facts linking sandwiches, trays, bread-portions, and content portions.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all goal (content-portion-2, tray) pairs that require no_gluten_bread-2.
    2. For each such pair:
       a) Check if the pair already satisfies the goal predicate in the current state.
          If yes, it does not add to the heuristic cost.
       b) Otherwise, estimate how many actions are needed to achieve that predicate:
          - At least one "make_sandwich" or "make_sandwich_no_gluten" action.
          - Possibly a "move_tray" action to position the tray.
          - Possibly "put_on_tray" or other intermediate preparation actions.
          - Possibly a "serve_sandwich_no_gluten" or "serve_sandwich-2" action.
          - Further counting of waiting places that need servicing because sandwiches
            cannot be prepared until places are waiting.
          - Bake in that multiple trays may be handled in parallel but we conservatively
            count one action per necessary step per goal pair.
    3. Sum the costs for all unsatisfied goal pairs.
    4. Additionally, count the number of places currently waiting, if these places
       block sandwich preparation for any of the goal pairs.
    5. Return the sum as the heuristic estimate.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all goal (content-portion-2, tray) pairs from no_gluten_bread-2 predicates
        self.goal_content_tray_pairs = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "no_gluten_bread-2" and len(parts) == 3:
                self.goal_content_tray_pairs.add((parts[1], parts[2]))

        # Extract served relations from static facts: set of (bread1, bread2)
        self.served_pairs = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 3:
                self.served_pairs.add((parts[1], parts[2]))

        # Extract places that are waiting
        self.waiting_places = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waiting" and len(parts) == 2:
                self.waiting_places.add(parts[1])

        # Extract waiting-2 facts (content-portion-2, bread-portion)
        self.waiting_2_pairs = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waiting-2" and len(parts) == 3:
                self.waiting_2_pairs.add((parts[1], parts[2]))

        # Extract facts linking sandwiches, trays, and content portions for heuristic usefulness
        # We'll build a dictionary of sandwiches on trays for quick lookup in __call__
        self.sandwich_on_tray = {}  # sandwich -> set of trays
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                if sandwich not in self.sandwich_on_tray:
                    self.sandwich_on_tray[sandwich] = set()
                self.sandwich_on_tray[sandwich].add(tray)

        # Extract sandwiches and their gluten allergy-related facts (not_allergic_gluten-2)
        self.not_allergic_gluten_2 = {}  # sandwich -> set of content-portion-2
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "not_allergic_gluten-2" and len(parts) == 3:
                sandwich, content_2 = parts[1], parts[2]
                if sandwich not in self.not_allergic_gluten_2:
                    self.not_allergic_gluten_2[sandwich] = set()
                self.not_allergic_gluten_2[sandwich].add(content_2)

    def __call__(self, node):
        state = node.state

        # Helper functions
        def fact_true_in_state(predicate, *args):
            """Checks whether the predicate with given args is true in the current state."""
            patt = "({} {})".format(predicate, " ".join(args))
            # We must allow for predicates with multiple arguments, so reconstruct full fact
            # but the above is incomplete: we need parentheses and all arguments inside.
            # We'll just do a fast check against all state facts with matching predicate and args
            # Simplify: we check all facts starting with '(' + predicate + ' ' and match rest
            # but to avoid false matches, check all state facts exactly matching predicate and args
            want = "({} {})".format(predicate, " ".join(args))
            return want in state

        def contains_predicate(predicate):
            """Check if any fact in state starts with predicate."""
            for f in state:
                if f.startswith("(" + predicate + " "):
                    return True
            return False

        # Extract all current no_gluten_bread-2 pairs in the state
        current_no_gluten_bread_2_pairs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "no_gluten_bread-2" and len(parts) == 3:
                current_no_gluten_bread_2_pairs.add((parts[1], parts[2]))

        # Extract places currently waiting
        current_waiting_places = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "waiting" and len(parts) == 2:
                current_waiting_places.add(parts[1])

        # Extract places with waiting-2 (content-portion-2, bread-portion)
        current_waiting_2_pairs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "waiting-2" and len(parts) == 3:
                current_waiting_2_pairs.add((parts[1], parts[2]))

        # Extract sandwiches on trays in the current state:
        current_sandwich_on_tray = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                if sandwich not in current_sandwich_on_tray:
                    current_sandwich_on_tray[sandwich] = set()
                current_sandwich_on_tray[sandwich].add(tray)

        # Extract no_gluten_sandwich-2 in current state (goal relevant)
        current_no_gluten_sandwich_2 = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "no_gluten_sandwich-2" and len(parts) == 2:
                current_no_gluten_sandwich_2.add(parts[1])

        # Extract allergic_gluten sandwich facts in current state (might help penalties)
        current_allergic_gluten = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten" and len(parts) == 3:
                sandwich = parts[1]
                current_allergic_gluten.add(sandwich)

        # Extract 'served' relations from current state
        current_served_pairs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 3:
                current_served_pairs.add((parts[1], parts[2]))

        # Heuristic calculation begins here

        cost = 0

        # 1) Primary cost: number of goal (content-portion-2, tray) pairs still unsatisfied
        # Each unsatisfied pair, conservatively count 3 actions to achieve:
        # - make_sandwich or equivalent (1 action)
        # - possibly move_tray (1 action)
        # - serve sandwich (1 action)
        # We add a penalty of 1 for each waiting place still waiting,
        # and extra penalty for waiting-2 pairs blocking progress.

        for (content_2, tray) in self.goal_content_tray_pairs:
            if (content_2, tray) not in current_no_gluten_bread_2_pairs:
                cost += 3  # Composite cost estimate per unsatisfied pair

        # 2) Add the number of waiting places currently waiting which block sandwich prep
        # Because waiting places must be cleared before sandwiches can be made
        # Each waiting place roughly corresponds to one 'serve_sandwich_no_gluten' or wait
        cost += len(current_waiting_places)

        # 3) Add penalty for waiting-2 pairs in the current state
        # Waiting-2 presumably blocks actions at the bread-portion level,
        # count as 1 action penalty per blocking pair.
        cost += len(current_waiting_2_pairs)

        # 4) Add small penalty for allergic_gluten sandwiches that may require extra make_sandwich actions
        # We add 1 per allergic_gluten sandwich currently in the state.
        cost += len(current_allergic_gluten)

        # 5) Add small penalty if served pairs are missing that are required (from static served pairs)
        # Count how many of the static served pairs are not yet in current_served_pairs
        cost += len(self.served_pairs - current_served_pairs)

        # Ensure heuristic is 0 if state satisfies all goals
        # If all goal pairs are satisfied and no waiting or allergic situations, cost = 0
        goal_satisfied = all((content_2, tray) in current_no_gluten_bread_2_pairs
                             for (content_2, tray) in self.goal_content_tray_pairs)
        if goal_satisfied and len(current_waiting_places) == 0 and len(current_waiting_2_pairs) == 0 and len(current_allergic_gluten) == 0:
            cost = 0

        return cost
