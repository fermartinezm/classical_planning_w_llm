from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at_kitchen_bread content-portion1 bread-portion2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (content-portion) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal state, 
    which primarily depends on assembling and serving gluten-free sandwiches onto trays
    with correct bread and content portions, respecting gluten allergies.

    The goal predicates are typically of the form (no_gluten_bread-2 <content-portion-2> <tray>),
    meaning each tray must have gluten-free bread with correct content.

    Assumptions
    -----------
    - Each goal tray must have at least one unique gluten-free bread-content portion assembled
      properly (represented by facts like no_gluten_bread-2).
    - Serving sandwiches requires moving content-portions and bread-portions through the kitchen
      and placing them on trays.
    - Actions such as make_sandwich_no_gluten, put_on_tray, serve_sandwich_no_gluten and their
      variants are relevant to the task.
    - Each bread-portion and content-portion pair must be prepared and placed on the right tray.
    - The number of incomplete servings/trays primarily drives action counts.
    - Waiting places (places where sandwiches are waiting) indicate readiness states.
    - The static facts give served relations for bread-portion pairs indicating allowable substitutions.

    Heuristic Initialization
    ------------------------
    - Extract all goal conditions (such as no_gluten_bread-2 on specific trays).
    - Extract static served pairs to understand bread substitutions.
    - Extract sets of trays, bread-portions, content-portions, and sandwiches relevant to the problem.
    - Extract all waiting places and waiting-2 objects as indicators of pending tasks.
    - Index static data for quick membership queries.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify how many goal gluten-free breads on trays are still missing in the current state.
       This gives a minimal baseline on how many servings remain.
    2. For each missing gluten-free bread-2 fact (goal), check if it is "close" to completion:
       - Is there already a corresponding at_kitchen_bread-2 fact for that content-portion-2 and tray?
       - Is the content portion gluten-free and available in the kitchen?
       - Is a sandwich (allergic or not) already made or being made for that tray/content?
    3. Count incomplete sandwiches that require preparation or serving.
    4. Include waiting states as additional indicators of actions needed.
    5. Heuristically assign costs per incomplete serving:
       - Making a sandwich is estimated to cost a fixed number of steps (e.g., 3-5).
       - Serving actions also cost steps.
    6. Combine counts into a total heuristic estimate.
    7. Return 0 if all goals are satisfied (goal state).

    Notes:
    - The heuristic is not admissible but aims to be informative.
    - It avoids expensive full relaxed planning graph computations.
    - It focuses on tasks related to bread-portion-2 and trays as indicated by goals and static facts.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goals that require (no_gluten_bread-2 <content-portion-2> <tray>)
        self.goal_bread2_trays = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "no_gluten_bread-2" and len(parts) == 3:
                self.goal_bread2_trays.add((parts[1], parts[2]))

        # Extract all trays and content-portion-2 objects from static facts (from no_gluten_bread-2 and at_kitchen_bread-2)
        self.trays = set()
        self.content_portions_2 = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and (parts[0] == "no_gluten_bread-2" or parts[0] == "at_kitchen_bread-2"):
                if len(parts) >= 3:
                    self.content_portions_2.add(parts[1])
                    self.trays.add(parts[-1])

        # Extract served pairs: (served bread-portionX bread-portionY)
        # We use this to understand valid bread substitutions
        self.served_pairs = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) == 3:
                self.served_pairs.add((parts[1], parts[2]))

        # Extract waiting places (waiting ?place) and waiting-2 facts
        self.waiting_places = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "waiting" and len(parts) == 2:
                self.waiting_places.add(parts[1])

        self.waiting_2_pairs = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "waiting-2" and len(parts) == 3:
                self.waiting_2_pairs.add((parts[1], parts[2]))

        # Precompute static facts lookup sets for quick lookup
        self.static_no_gluten_sandwich_2 = set(
            f for f in self.static if match(f, "no_gluten_sandwich-2", "*")
        )
        self.static_no_gluten_bread_2 = set(
            f for f in self.static if match(f, "no_gluten_bread-2", "*", "*")
        )
        self.static_allergic_gluten = set(
            f for f in self.static if match(f, "allergic_gluten", "*", "*")
        )

    def __call__(self, node):
        state = node.state

        # If goal reached (all goal facts true), heuristic is zero
        if self.goal_bread2_trays <= {(p[0], p[1]) for fact in state if (fact.startswith("(no_gluten_bread-2 ")) for p in [tuple(get_parts(fact)[1:3])] }:
            return 0

        # Track how many goal gluten-free breads on trays are still missing
        bread2_trays_completed = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "no_gluten_bread-2" and len(parts) == 3:
                bread2_trays_completed.add((parts[1], parts[2]))
        goal_remaining = self.goal_bread2_trays - bread2_trays_completed

        # Heuristic cost estimate initialization
        estimated_cost = 0

        # --- Helper functions to check for clues in the state ---

        def has_at_kitchen_bread_2(content_portion_2, tray):
            """Check if (at_kitchen_bread-2 <content_portion_1> <content_portion_2> <tray>) exists for given content_portion_2 and tray"""
            for fact in state:
                parts = get_parts(fact)
                if (
                    len(parts) == 4
                    and parts[0] == "at_kitchen_bread-2"
                    and parts[2] == content_portion_2
                    and parts[3] == tray
                ):
                    return True
            return False

        def is_waiting(place):
            return any(match(f, "waiting", place) for f in state)

        def is_waiting_2(content_portion_2, bread_portion):
            return any(match(f, "waiting-2", content_portion_2, bread_portion) for f in state)

        def gluten_free_content(content_portion):
            return any(match(f, "no_gluten_content", content_portion) or match(f, "no_gluten_content-2", content_portion) for f in state)

        def gluten_free_bread(content_portion):
            return any(match(f, "no_gluten_bread", content_portion) for f in state)

        def gluten_free_sandwich(child, bread_portion):
            return any(match(f, "no_gluten_sandwich", child, bread_portion) for f in state)

        def allergic_gluten(sandwich, content_portion):
            return any(match(f, "allergic_gluten", sandwich, content_portion) for f in state)

        # Estimate the cost for each incomplete bread-2 on a tray in goals:
        for (content_portion_2_goal, tray_goal) in goal_remaining:
            # Initial rough cost per incomplete goal, adjusted by clues from the state
            cost_subestimate = 3

            # 1) Check if the bread-2 is present in the kitchen for this tray with some content portion
            if has_at_kitchen_bread_2(content_portion_2_goal, tray_goal):
                # Partly prepared, reduce cost estimate
                cost_subestimate -= 1

            # 2) Check if there are waiting-2 facts indicating preparation pending for this bread-portion
            # We look for waiting-2 facts with this content portion and any bread portion that might go with it
            # This suggests steps like make_sandwich or move_tray remain
            relevant_waiting2 = set()
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "waiting-2" and parts[1] == content_portion_2_goal:
                    relevant_waiting2.add((parts[1], parts[2]))
            if relevant_waiting2:
                cost_subestimate += 1  # Penalize waiting tasks

            # 3) Check if corresponding allergic_gluten sandwiches exist that may require special handling
            allergic_pending = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "allergic_gluten" and parts[-1] == content_portion_2_goal:
                    allergic_pending = True
                    break
            if allergic_pending:
                cost_subestimate += 2  # Allergic sandwiches may require more steps

            # 4) Check if tray currently has no_gluten_bread-2 fact: if missing, need to serve -> add cost
            if not any(
                match(f, "no_gluten_bread-2", content_portion_2_goal, tray_goal) for f in state
            ):
                cost_subestimate += 1

            # 5) Clamp cost_subestimate minimum to 1, maximum to 6
            cost_subestimate = max(1, min(cost_subestimate, 6))

            estimated_cost += cost_subestimate

        # Additional heuristics:

        # Count how many waiting places still wait; each corresponds to one serving action approximately
        waiting_count = sum(1 for wp in self.waiting_places if is_waiting(wp))
        estimated_cost += waiting_count

        # Count how many waiting-2 facts hold in state (potential additional steps)
        estimated_cost += sum(1 for w2 in self.waiting_2_pairs if is_waiting_2(*w2))

        return estimated_cost
