from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(no_gluten_bread-2 content-portion2 tray3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain 'content-portion'.

    Summary:
    This heuristic estimates the number of remaining steps to reach the goal by focusing on the outstanding "no_gluten_bread-2" facts,
    which must hold for each (content-portion-2, tray) pair as prescribed in the goal. The heuristic considers the current state of
    these facts and estimates how many actions are required to achieve all of them.

    Assumptions:
    - The goal requires achieving certain "no_gluten_bread-2" predicates for combinations of content-portion-2 and tray.
    - To achieve a "no_gluten_bread-2" fact, the domain requires a sequence of actions making sandwiches, putting them on trays, 
      and serving them.
    - Actions depend on preparing sandwiches with or without gluten allergy considerations, moving trays, and serving sandwiches.
    - Achieving each missing "no_gluten_bread-2" fact roughly takes a bounded number of complex actions, typically at least one make or move action plus eventually a serve.
    - The heuristic need not be admissible but should minimize node expansion by closely estimating effort per missing goal.

    Heuristic Initialization:
    - Extract all goal (no_gluten_bread-2 ?contentportion2 ?tray) facts to know which pairs must be completed.
    - Extract static facts related to:
      * (served ?breadportion1 ?breadportion2): pairs of bread serving relations,
      * (waiting ?place) and (waiting-2 ?contentportion2 ?breadportion): to understand readiness/waiting states,
      * (at_kitchen_bread-2 ?contentportion ?contentportion2 ?tray): to identify which content-portion-2 and trays are involved,
      * (no_gluten_content ?contentportion), (no_gluten_bread ?contentportion),
      * (no_gluten_content-2 ?breadportion), etc. to estimate at least if gluten-free sandwiches are needed.
    - Index these facts as sets for quick membership tests.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal (no_gluten_bread-2 content-portion-2 tray) pair:
       - Check if current state contains that fact → if yes, it's done, cost is 0.
       - Else estimate cost to achieve it:
         a) Check if the associated content-portion-2 is currently waiting (via waiting-2) or on tray preparations.
         b) Check if the bread portions relevant to the tray and content-portion-2 are served and if "at_kitchen_bread-2" facts exist.
         c) If no "at_kitchen_bread-2" fact or bread portion served facts, imply sandwich must be made or tray moved, costing several actions.
         d) Consider if there is a gluten allergy (allergic_gluten predicate) influencing which sandwich making action variant to use.
         e) Count roughly 2-4 actions per incomplete sandwich/tray goal due to make, move, put_on_tray, serve actions.
    2. Sum over all missing goals to estimate total heuristic cost.
    3. Return 0 if all goals are achieved.
    
    This heuristic trades accuracy for computational efficiency by using presence checks and static information
    to approximate required actions to complete the sandwiches on trays while respecting gluten constraints,
    thus guiding the search effectively in this complex domain.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal and static information."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal no_gluten_bread-2 facts: (no_gluten_bread-2 ?contentportion2 ?tray)
        self.goal_pairs = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "no_gluten_bread-2":
                self.goal_pairs.add((parts[1], parts[2]))

        # Extract served pairs: (served breadportion1 breadportion2)
        self.served_pairs = set()
        for fact in self.static:
            if match(fact, "served", "*", "*"):
                parts = get_parts(fact)
                self.served_pairs.add((parts[1], parts[2]))

        # Extract waiting places
        self.waiting_places = set()
        for fact in self.static:
            if match(fact, "waiting", "*"):
                parts = get_parts(fact)
                self.waiting_places.add(parts[1])

        # Extract waiting-2 facts: (waiting-2 contentportion2 breadportion)
        self.waiting_2 = set()
        for fact in self.static:
            if match(fact, "waiting-2", "*", "*"):
                parts = get_parts(fact)
                self.waiting_2.add((parts[1], parts[2]))

        # Extract at_kitchen_bread-2: (at_kitchen_bread-2 contentportion contentportion2 tray)
        self.at_kitchen_bread_2 = set()
        for fact in self.static:
            if match(fact, "at_kitchen_bread-2", "*", "*", "*"):
                parts = get_parts(fact)
                self.at_kitchen_bread_2.add((parts[1], parts[2], parts[3]))

        # Extract no_gluten_content, no_gluten_bread, no_gluten_content-2 for quick checks
        self.no_gluten_content = set()
        for fact in self.static:
            if match(fact, "no_gluten_content", "*"):
                parts = get_parts(fact)
                self.no_gluten_content.add(parts[1])

        self.no_gluten_bread = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                parts = get_parts(fact)
                self.no_gluten_bread.add(parts[1])

        self.no_gluten_content_2 = set()
        for fact in self.static:
            if match(fact, "no_gluten_content-2", "*"):
                parts = get_parts(fact)
                self.no_gluten_content_2.add(parts[1])

    def __call__(self, node):
        """
        Estimate the cost to complete the goal from the current state.

        The heuristic counts, for each goal no_gluten_bread-2 fact missing from the current state,
        an approximate action cost based on readiness of bread portions, trays, content portions,
        allergy conditions, and waiting states.

        Returns 0 if goal is reached.
        """
        state = node.state

        # Quick check: if all goal facts are satisfied => heuristic = 0
        if self.goals <= state:
            return 0

        cost = 0

        # Helper sets for current state
        current_no_gluten_bread_2 = set()
        for fact in state:
            if match(fact, "no_gluten_bread-2", "*", "*"):
                parts = get_parts(fact)
                current_no_gluten_bread_2.add((parts[1], parts[2]))

        current_at_kitchen_bread_2 = set()
        for fact in state:
            if match(fact, "at_kitchen_bread-2", "*", "*", "*"):
                parts = get_parts(fact)
                current_at_kitchen_bread_2.add((parts[1], parts[2], parts[3]))

        allergic_gluten_pairs = set()
        for fact in state:
            if match(fact, "allergic_gluten", "*", "*"):
                parts = get_parts(fact)
                allergic_gluten_pairs.add((parts[1], parts[2]))

        waiting_places_now = set()
        for fact in state:
            if match(fact, "waiting", "*"):
                parts = get_parts(fact)
                waiting_places_now.add(parts[1])

        waiting_2_now = set()
        for fact in state:
            if match(fact, "waiting-2", "*", "*"):
                parts = get_parts(fact)
                waiting_2_now.add((parts[1], parts[2]))

        # served facts in current state
        served_pairs_now = set()
        for fact in state:
            if match(fact, "served", "*", "*"):
                parts = get_parts(fact)
                served_pairs_now.add((parts[1], parts[2]))

        # For each goal pair not yet satisfied estimate effort
        for (cp2_goal, tray_goal) in self.goal_pairs:
            if (cp2_goal, tray_goal) in current_no_gluten_bread_2:
                # Already achieved, no cost
                continue

            # Estimate cost to prepare sandwich on tray and achieve no_gluten_bread-2

            # Check if at_kitchen_bread-2 is available for this content-portion-2 and tray
            has_kitchen_bread_2 = any(
                at_cp == cp2_goal and at_tray == tray_goal for (_, at_cp, at_tray) in current_at_kitchen_bread_2
            )

            # If no_kitchen_bread_2 available, we probably need to make sandwich or move tray
            # Approximate baseline cost for missing no_gluten_bread-2:
            # - At least 2 actions: make_sandwich or move_tray + serve_sandwich_no_gluten or serve_sandwich-2
            # We add 3 to reflect possible extra prep and placing actions.

            est_cost = 3

            # Check if allergic_gluten impact: if allergic_gluten present for this content portion and some sandwich,
            # likely more complex make_sandwich (allergic) needed raising cost
            allergic = False
            for (sandwich, cp) in allergic_gluten_pairs:
                if cp == cp2_goal:
                    allergic = True
                    break
            if allergic:
                est_cost += 1  # Penalize allergy requiring special make actions

            # Check waiting states: sandwiches or trays not waiting cause extra steps
            # Penalize if waiting not in place or waiting-2 for the pair missing in current state
            # Use waiting predicates in current state for place and waiting-2 for (content-portion-2, bread-portion)

            # We do not have direct mapping to relate cp2_goal to bread portions here,
            # so we check broadly if waiting-2 facts exist for cp2_goal with any bread portion to estimate delays.
            waiting_penalty = 0
            waiting_any = any(cp2 == cp2_goal for (cp2, _) in waiting_2_now)
            if not waiting_any:
                # Possibly need to wait or prepare place/tray
                waiting_penalty += 1

            place_waiting = False
            for place in waiting_places_now:
                # If any place waiting, minimal effect
                place_waiting = True
                break
            if not place_waiting:
                waiting_penalty += 1

            est_cost += waiting_penalty

            # Check served pairs: many actions depend on served relation among bread portions,
            # missing served pairs indicate more make or serve actions needed.
            # We try to guess roughly how many "notexist" preconditions are active.

            # Count notexist facts for this content portion and bread portion pairs
            notexist_count = 0
            for fact in state:
                if match(fact, "notexist", cp2_goal, "*", "*"):
                    notexist_count += 1
                if match(fact, "notexist", "*", cp2_goal, "*"):
                    notexist_count += 1
            # Use notexist count to increase estimate moderately
            if notexist_count > 0:
                est_cost += 1

            cost += est_cost

        return cost
