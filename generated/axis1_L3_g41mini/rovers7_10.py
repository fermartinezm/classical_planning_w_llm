from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    A domain-dependent heuristic for the secret_domain (content-portion) domain.

    Summary
    -------
    This heuristic estimates the number of remaining actions needed to prepare and serve all required gluten-free sandwiches
    on trays according to the goal conditions. It is designed to guide greedy best-first search efficiently, without
    guaranteeing admissibility.

    Assumptions
    -----------
    - The final goal typically involves facts like `(no_gluten_bread-2 content-portionX trayY)` or `(no_gluten_sandwich-2 bread-portionZ)`.
    - Preparing sandwiches involves multiple steps: preparing bread with correct gluten properties, managing content portions,
      placing sandwiches on trays, and serving them.
    - Gluten-free requirements imply that sandwiches and bread portions must be free from gluten according to predicates like
      `no_gluten_bread`, `no_gluten_content`, `no_gluten_sandwich`, and related +/-2 predicates.
    - The domain involves interaction between content-portions, bread-portions, places, trays, and sandwiches, with actions that
      move these items between states like at-kitchen, waiting, on tray, and served.
    - Static facts provide served relations between bread-portion pairs, gluten properties, and allergy information.
    - The heuristic ignores complex interactions of allergies and gluten at a fine granularity but pays attention to how many
      goal achievements remain and whether prerequisite conditions are achieved in the current state.

    Heuristic Initialization
    ------------------------
    - Extract all goal conditions relevant for sandwiches and bread served on trays.
    - Extract static facts about served bread-portion pairs.
    - Extract static gluten-related facts that affect which bread-portion/content-portion combinations are allowed.
    - Extract from static facts and goals, the list of trays involved in the final goal.
    - The heuristic uses simple counts of missing goal-related facts and approximates needed intermediate steps.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the goals related to gluten-free sandwiches on bread portions and gluten-free bread on trays.
    2. For each goal fact, check if it is currently satisfied in the state.
    3. Count how many such goal facts are unsatisfied.
    4. Also check for prerequisite facts that must hold to create such goal facts, e.g., waiting places,
       at_kitchen predicates, no_gluten predicates.
    5. For each unsatisfied goal fact, estimate a minimal cost of actions to achieve it:
       - If a bread-portion on a tray lacks the `no_gluten_bread-2` fact, estimate steps to prepare and move that bread onto the tray.
       - If a no_gluten_sandwich-2 fact is missing, consider steps to prepare sandwiches and serve them.
    6. Add small penalties for places that are waiting or waiting-2 fact missing since these are preconditions for action application.
    7. Approximate the heuristic as the sum of these minimal costs (mostly counted as 'number of missing goal facts' plus
       small overheads for preliminary steps).
    8. Return 0 if all goal facts are satisfied (goal reached).

    This heuristic attempts to be informative yet cheap to compute by focusing on the presence/absence of key goal facts and
    their immediate prerequisites, helping guide the search by pushing it toward achieving the critical bread/tray/sandwich states.
    """

    def __init__(self, task):
        """
        Extract and store useful static information and goal-related facts.

        Parameters:
        -----------
        task: The planning task instance containing goals, operators, static facts, initial state.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal gluten-free bread on trays, e.g. (no_gluten_bread-2 ?content-portion-2 ?tray)
        self.goal_bread_on_trays = set()
        # Extract goal gluten-free sandwiches, e.g. (no_gluten_sandwich-2 ?bread-portion)
        self.goal_sandwiches = set()

        # For quick membership checking of goal facts
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "no_gluten_bread-2" and len(parts) == 3:
                self.goal_bread_on_trays.add((parts[1], parts[2]))
            elif parts[0] == "no_gluten_sandwich-2" and len(parts) == 2:
                self.goal_sandwiches.add(parts[1])

        # Extract served relations from static facts: served(bread1 bread2)
        self.served_relations = set()
        for fact in self.static:
            if match(fact, "served", "*", "*"):
                parts = get_parts(fact)
                self.served_relations.add((parts[1], parts[2]))

        # Extract no_gluten content-2 bread facts from static for quick lookups
        self.no_gluten_bread_2 = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread-2", "*", "*"):
                parts = get_parts(fact)
                self.no_gluten_bread_2.add((parts[1], parts[2]))

        # Extract waiting places (waiting ?place)
        self.waiting_places = set()
        for fact in self.static:
            if match(fact, "waiting", "*"):
                parts = get_parts(fact)
                self.waiting_places.add(parts[1])

        # Extract waiting-2 facts (waiting-2 ?content-portion-2 ?bread-portion)
        self.waiting_2 = set()
        for fact in self.static:
            if match(fact, "waiting-2", "*", "*"):
                parts = get_parts(fact)
                self.waiting_2.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Quick helper to check fact presence in current state
        def fact_in_state(predicate, *args):
            fact = "({} {})".format(predicate, " ".join(args))
            return fact in state

        heuristic_cost = 0

        # 1) Estimate cost for all goal bread-on-tray that are not yet achieved:
        missing_bread_on_tray = 0
        for (content_portion_2, tray) in self.goal_bread_on_trays:
            # Check if this fact is satisfied in the current state
            if not fact_in_state("no_gluten_bread-2", content_portion_2, tray):
                missing_bread_on_tray += 1

                # We estimate roughly 2-3 actions needed for each bread on tray that is missing:
                # Including preparing bread, moving tray, and making gluten-free sandwich.
                heuristic_cost += 3

        # 2) Estimate cost for all goal gluten-free sandwiches not yet achieved
        missing_sandwiches = 0
        for bread_portion in self.goal_sandwiches:
            if not fact_in_state("no_gluten_sandwich-2", bread_portion):
                missing_sandwiches += 1

                # Each missing gluten-free sandwich likely requires some preparation steps.
                # Approximate as 2 actions per missing sandwich.
                heuristic_cost += 2

        # 3) Check if waiting places exist and are satisfied, add penalty if not
        # Any waiting ?place fact missing may delay sandwich making
        places_waiting = set()
        for fact in state:
            if match(fact, "waiting", "*"):
                parts = get_parts(fact)
                places_waiting.add(parts[1])

        for place in self.waiting_places:
            if place not in places_waiting:
                heuristic_cost += 1  # Penalty for missing waiting place state

        # Similarly, for waiting-2 facts that are preconditions for some actions, 
        # estimate small penalties for those missing from state.
        waiting2_in_state = set()
        for fact in state:
            if match(fact, "waiting-2", "*", "*"):
                parts = get_parts(fact)
                waiting2_in_state.add((parts[1], parts[2]))

        for w2fact in self.waiting_2:
            if w2fact not in waiting2_in_state:
                heuristic_cost += 1  # Penalize missing waiting-2 facts

        # 4) Handle 'served' relation prerequisites:
        # If any goal bread on tray or sandwich-2 requires served relations that are missing,
        # add small cost for them because they are needed for serving.
        # Find how many served relations needed by goals are missing in current state.

        # For missing bread on tray and sandwiches, check if served relations present
        # In practice, this is complex to infer exactly, so approximate as:
        missing_served = 0
        for (c1, c2) in self.served_relations:
            fact = f"(served {c1} {c2})"
            if fact not in state:
                missing_served += 1

        # Add a small weighted penalty for missing served relations
        heuristic_cost += missing_served // 5  # since these are many and not always critical

        # 5) If heuristic_cost is zero but goals are not fully satisfied, set minimal cost 1
        # because some final action is needed
        if heuristic_cost == 0 and not self.goals <= state:
            heuristic_cost = 1

        # 6) Return 0 if goal satisfied, else heuristic cost
        if self.goals <= state:
            return 0
        else:
            return heuristic_cost
