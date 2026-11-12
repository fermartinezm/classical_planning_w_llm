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
    Domain-dependent heuristic for the secret_domain domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to complete sandwich preparation and serving tasks, focusing on achieving all goal conditions, which mostly involve placing no-gluten bread-portions on certain trays or sandwiches for children. It does so by analyzing the current state facts related to:
      - presence of no-gluten breads on trays,
      - sandwiches prepared (gluten-free and allergic-gluten cases),
      - waiting places and contents to be processed,
      - and the complex chaining of gluten allergies and sandwich assembly.

    Assumptions:
    ------------
    - The goal facts require `no_gluten_bread-2` predicates, i.e., no-gluten bread portions correctly placed on trays.
    - Actions revolve around preparing sandwiches with or without gluten allergens, moving trays, serving sandwiches, and awaiting places.
    - Static facts, especially `served` relations form a graph of bread portion "serving" sequences.
    - Actions require intermediate steps such as moving content-portion objects, placing sandwiches on trays, and satisfying 'waiting' conditions.
    - This heuristic is not guaranteed admissible but aims to be informative and efficiently computable.
    - The heuristic focuses on remaining goals and partial satisfying conditions rather than a deep search of all dependencies.

    Heuristic Initialization:
    ---------------------------
    - Extract all goals, especially those stating `(no_gluten_bread-2 ?content ?tray)`.
    - Extract static facts such as `(served ?bp1 ?bp2)`, used to understand bread serving relations.
    - Build reverse mappings from bread portions to which breads are served next.
    - Extract all bread-portion objects and trays relevant for goal checking.
    - Store any relevant static no-gluten or allergic-gluten facts for heuristics on sandwiches.
    - Store facts about waiting places to identify how many places await action.

    Step-By-Step Thinking for Computing Heuristic:
    ------------------------------------------------
    1. Identify how many goal `no_gluten_bread-2` facts are already satisfied in the current state.
       Each unsatisfied goal fact corresponds roughly to at least one needed action (serving or preparing sandwiches properly).

    2. Count the number of waiting places (`waiting` and `waiting-2`) still present, as each requires actions to clear by serving or preparing.

    3. Identify bread portions involved in goals or static `served` relations that are not yet at the target state and count them as needed future efforts.

    4. Evaluate presence or absence of gluten allergen conditions `(allergic_gluten ...)` and `(not_allergic_gluten ...)` predicates to estimate sandwich preparation complexity.

    5. Estimate minimal sandwich preparation and serving steps by combining:
       - Number of waiting places times a small constant weight,
       - Number of not-yet-served `no_gluten_bread-2` goals,
       - Penalty for sandwiches currently marked as allergic gluten needing preparation.

    6. Provide final heuristic value as a weighted sum which generally grows with the number of unsatisfied key goals and waiting places.

    7. Return heuristic 0 if all goals are satisfied (goal state).

    Notes:
    -------
    - The heuristic uses counts rather than exact action sequences for efficiency.
    - Does not attempt full planning graph or landmark-based estimates to remain fast.
    """

    def __init__(self, task):
        """
        Initialize heuristic, extracting relevant static and goal facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all no_gluten_bread-2 goal facts with their content-portion and tray objects.
        self.goal_no_gluten_bread_2 = set()
        for fact in self.goals:
            if match(fact, "no_gluten_bread-2", "*", "*"):
                parts = get_parts(fact)
                # parts format: ["no_gluten_bread-2", content-portion, tray]
                self.goal_no_gluten_bread_2.add((parts[1], parts[2]))

        # Extract served relations (bread-portion to bread-portion)
        self.served_map = {}
        for fact in self.static:
            if match(fact, "served", "*", "*"):
                parts = get_parts(fact)
                self.served_map.setdefault(parts[1], set()).add(parts[2])

        # Extract sets to quickly check gluten sandwich or bread properties
        self.no_gluten_sandwich_statics = set()
        for fact in self.static:
            if match(fact, "no_gluten_sandwich-2", "*"):
                p = get_parts(fact)
                self.no_gluten_sandwich_statics.add(p[1])

        self.waiting_places_static = set()
        for fact in self.static:
            if match(fact, "waiting", "*"):
                p = get_parts(fact)
                self.waiting_places_static.add(p[1])

    def __call__(self, node):
        """
        Compute the heuristic value estimating the steps needed to reach a goal state.
        """
        state = node.state

        # If already at goal, heuristic is 0
        if self.goals <= state:
            return 0

        # Count how many of the no_gluten_bread-2 goals are currently satisfied
        satisfied_no_gluten_bread_2 = 0
        for (content_portion, tray) in self.goal_no_gluten_bread_2:
            predicate = f"(no_gluten_bread-2 {content_portion} {tray})"
            if predicate in state:
                satisfied_no_gluten_bread_2 += 1

        unsatisfied_no_gluten_bread_2 = len(self.goal_no_gluten_bread_2) - satisfied_no_gluten_bread_2

        # Count waiting places currently present (waiting + waiting-2) as these indicate pending serving/preparing steps
        waiting_places = sum(1 for fact in state if match(fact, "waiting", "*"))
        waiting_2 = sum(1 for fact in state if match(fact, "waiting-2", "*", "*"))
        total_waiting = waiting_places + waiting_2

        # Count number of sandwiches marked as allergic_gluten that presumably require preparation steps
        allergic_gluten_sandwiches = sum(1 for fact in state if match(fact, "allergic_gluten", "*", "*"))

        # Count not_allergic_gluten facts that potentially facilitate preparation (lower cost)
        not_allergic_gluten_pairs = sum(1 for fact in state if match(fact, "not_allergic_gluten", "*", "*"))

        # Base heuristic estimation:
        # Each unsatisfied no_gluten_bread_2 goal likely requires 2-3 actions (prepare sandwich, put on tray).
        # Each waiting place corresponds roughly to 1-2 actions to clear.
        # Each allergic gluten sandwich requires additional attention, adding ~2 actions.
        # Having many not_allergic_gluten facts reduces the overhead somewhat.

        # Calculate weighted heuristic value
        h_value = 0

        # Unsatisfied bread on tray goals: weighted higher to push search towards completing them
        h_value += unsatisfied_no_gluten_bread_2 * 3

        # Waiting places require clearing: estimate 2 actions each on average
        h_value += total_waiting * 2

        # Allergic gluten sandwich penalties
        h_value += allergic_gluten_sandwiches * 2

        # Subtract minor adjustment for not_allergic_gluten pairs to reward partial progress
        h_value -= not_allergic_gluten_pairs

        # Ensure heuristic is never negative
        if h_value < 0:
            h_value = 0

        # Return estimated number of required actions to reach goal
        return h_value
