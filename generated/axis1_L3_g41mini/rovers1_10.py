from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact string, e.g. "(predicate obj1 obj2)".
    - `args`: Pattern with possible wildcards.
    Returns True if matches, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (content-portion domain).

    Summary:
    The heuristic estimates the number of remaining actions needed to
    complete all sandwich-serving goals involving gluten-free breads
    and sandwiches on trays for various places and children.
    It combines counts of remaining sandwiches/breads to be served,
    trays to be loaded, and places awaiting serving,
    leveraging both static and dynamic information from the current state.

    Assumptions:
    - The goal is to achieve conditions like `(no_gluten_bread-2 ?content-portion-2 ?tray)`
      and `(no_gluten_sandwich-2 ?bread-portion)` as specified by goal facts.
    - Relevant static facts and served relations are stable throughout planning.
    - Serving a sandwich or making a sandwich_no_gluten action basically
      progresses towards these goals by discharging waiting places or updating trays.
    - States encode facts about breads, contents, trays, sandwiches, and waiting places.
    - The heuristic need not be admissible, but should guide search efficiently by lower bounding
      the minimal remaining actions needed.

    Heuristic Initialization:
    - Extract goal facts indicating required `(no_gluten_bread-2 content-portion-2 tray)`
      and `(no_gluten_sandwich-2 bread-portion)`.
    - Extract static served relations `(served bread1 bread2)`.
    - Extract all trays, bread-portions, content-portions, and places from static and goal facts.
    - Prepare lookup structures for efficient counting and matching.

    Step-By-Step Thinking for Computing Heuristic:
    1. Count how many `(no_gluten_bread-2 content-portion-2 tray)` goals are not yet satisfied in the state.
       Each such missing fact typically corresponds to a required `make_sandwich` or `move_tray` action.
    2. Count how many `(no_gluten_sandwich-2 bread-portion)` goals are missing.
       Each corresponds roughly to a sandwich-making action at a bread portion.
    3. Identify how many places are currently `(waiting)` vs. already served.
       Each waiting place will require a sequence of actions to finish.
    4. Count how many sandwiches or breads are still "at_kitchen" but not completed.
       These typically require serving or manipulation actions.
    5. Use the served relation to decide if breads are already served or need additional serving steps.
    6. Combine the above counts using weighted sums reflecting minimal required actions:
       - Actions like `make_sandwich`, `serve_sandwich_no_gluten`, `put_on_tray` etc. usually cost 1 each,
         but the heuristic can weight counts to reflect minimum expected sequences.
    7. Return the sum of these weighted counts as an estimate of remaining action cost.
       This is guaranteed finite for solvable states and zero only at goal states.

    This heuristic is tailored for guiding greedy best-first search balancing accuracy and efficiency.
    """

    def __init__(self, task):
        """Extract relevant static information and goal requirements for heuristic estimates."""
        self.goals = task.goals
        self.static = task.static

        # Extract all goal no_gluten_bread-2 facts and no_gluten_sandwich-2 facts.
        self.goal_no_gluten_bread_2 = set(
            fact for fact in self.goals if fact.startswith("(no_gluten_bread-2 ")
        )
        self.goal_no_gluten_sandwich_2 = set(
            fact for fact in self.goals if fact.startswith("(no_gluten_sandwich-2 ")
        )

        # Extract served relations from static facts.
        self.served_pairs = set()
        for fact in self.static:
            if fact.startswith("(served "):
                parts = get_parts(fact)
                # served relations are between two bread-portions (served bread1 bread2)
                if len(parts) == 3:
                    self.served_pairs.add((parts[1], parts[2]))

        # Extract sets of all trays, content-portions-2, bread-portions from static + goals to help parsing.
        self.trays = set()
        self.content_portion_2 = set()
        self.bread_portions = set()
        for fact in self.static | self.goals:
            parts = get_parts(fact)
            # no_gluten_bread-2 ?content-portion-2 tray
            if match(fact, "no_gluten_bread-2", "*", "*") and len(parts) == 3:
                self.content_portion_2.add(parts[1])
                self.trays.add(parts[2])
            # no_gluten_sandwich-2 bread-portion
            if match(fact, "no_gluten_sandwich-2", "*") and len(parts) == 2:
                self.bread_portions.add(parts[1])
            # Additional bread portions appear in served relations.
            if fact.startswith("(served "):
                self.bread_portions.add(parts[1])
                self.bread_portions.add(parts[2])

        # Extract all places from static facts with waiting predicate to capture waiting places.
        self.places = set()
        for fact in self.static:
            if fact.startswith("(waiting ") and len(get_parts(fact)) == 2:
                self.places.add(get_parts(fact)[1])

    def __call__(self, node):
        """
        Estimate the required number of actions to reach a goal state.

        Uses current facts and static knowledge to count missing goal facts,
        remaining waiting places, and sandwiches/breads to be served,
        combining them into a weighted sum reflecting minimal effort.
        """
        state = node.state

        # Quick check: if goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Count missing no_gluten_bread-2 goal facts
        missing_bread_2 = len([
            fact for fact in self.goal_no_gluten_bread_2 if fact not in state
        ])

        # Count missing no_gluten_sandwich-2 goal facts
        missing_sandwich_2 = len([
            fact for fact in self.goal_no_gluten_sandwich_2 if fact not in state
        ])

        # Count how many places are currently waiting
        waiting_places = sum(1 for fact in state if match(fact, "waiting", "*"))

        # Count how many (at_kitchen_bread-2 content-portion content-portion-2 tray) facts are still present for trays in goal
        # These represent breads at kitchen that still might require serving or moving.
        breads_at_kitchen_2 = sum(
            1 for fact in state if fact.startswith("(at_kitchen_bread-2 ") and any(tray in fact for tray in self.trays)
        )

        # Count how many (at_kitchen_sandwich-2 sandwich content-portion) facts are present for sandwiches expected on trays
        sandwiches_at_kitchen_2 = sum(
            1 for fact in state if fact.startswith("(at_kitchen_sandwich-2 ") and any(content_2 in fact for content_2 in self.content_portion_2)
        )

        # Count how many sandwiches are allergic_gluten and still need processing
        allergic_sandwiches = sum(
            1 for fact in state if fact.startswith("(allergic_gluten ")
        )

        # Count how many 'waiting-2' facts are present => content-portions waiting with bread-portions combos
        waiting_2_facts = sum(
            1 for fact in state if fact.startswith("(waiting-2 ")
        )

        # Count no_gluten_bread facts (bread portions with no gluten)
        no_gluten_bread_facts = sum(
            1 for fact in state if fact.startswith("(no_gluten_bread ")
        )

        # Count no_gluten_content facts (content portions with no gluten)
        no_gluten_content_facts = sum(
            1 for fact in state if fact.startswith("(no_gluten_content ")
        )

        # Count at_kitchen_content facts: contents at kitchen may still need sandwich making.
        at_kitchen_content_facts = sum(
            1 for fact in state if fact.startswith("(at_kitchen_content ")
        )

        # Count waiting places (waiting ?place)
        waiting_places_exact = sum(
            1 for fact in state if match(fact, "waiting", "*") and get_parts(fact)[1] in self.places
        )

        # Weigh the counted values to produce heuristic estimate.
        # Observations:
        # - Each missing no_gluten_bread-2 fact requires approx 2 actions: move_tray + serve steps
        # - Each missing no_gluten_sandwich-2 fact requires approx 2 actions: make_sandwich + serve_sandwich_no_gluten
        # - Each waiting place requires at least 1 action to finish serving
        # - breads_at_kitchen_2 and sandwiches_at_kitchen_2 represent intermediate steps toward goals, weight 1 each
        # - allergic sandwiches require extra handling actions, weight 1 each
        # - waiting-2 facts and no_gluten_content or no_gluten_bread indicate intermediate readiness, weight 1 each

        heuristic_value = 0
        heuristic_value += 2 * missing_bread_2
        heuristic_value += 2 * missing_sandwich_2
        heuristic_value += 1 * waiting_places_exact
        heuristic_value += 1 * breads_at_kitchen_2
        heuristic_value += 1 * sandwiches_at_kitchen_2
        heuristic_value += 1 * allergic_sandwiches
        heuristic_value += 1 * waiting_2_facts
        heuristic_value += 1 * no_gluten_bread_facts
        heuristic_value += 1 * no_gluten_content_facts
        heuristic_value += 1 * at_kitchen_content_facts

        # Ensure heuristic is at least 1 for non-goal states to avoid zero but not goal state
        return max(heuristic_value, 1)
