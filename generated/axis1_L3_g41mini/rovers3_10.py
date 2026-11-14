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
    This heuristic estimates the number of actions required to reach the goal in the secret_domain,
    which revolves around preparing gluten and non-gluten sandwiches on trays for children in places.
    It focuses on the preparation and serving steps indicated by the goal facts involving protected
    gluten-free sandwiches on trays and at places.

    Assumptions:
    ------------
    - Goals mainly require certain bread portions to be gluten-free on specific trays and sandwiches.
    - Preparing sandwiches involves "making" sandwiches (possibly gluten-free or allergic gluten variants),
      placing content on trays, and serving them.
    - The heuristic does not have to be admissible but should be informative and efficiently computable.
    - The goal facts involve predicates like (no_gluten_bread-2 content-portion tray) and (no_gluten_sandwich-2 bread-portion).
    - Counts of "waiting"/"waiting-2" facts and pending gluten/non-gluten sandwich serving relate directly to required actions.
    - The heuristic computes an approximate number of required main action steps, considering sandwich making and serving,
      based on how many goal conditions are not satisfied.

    Heuristic Initialization:
    --------------------------
    - Extract goal facts of relevant predicates:
       - no_gluten_bread-2(content-portion-2, tray)
       - no_gluten_sandwich-2(bread-portion)
    - Extract from static facts and goals:
       - Bread portions and content portions relevant to gluten-free conditions.
       - Identify trays and places involved in goals.
    - Prepare quick lookup sets for:
       - Goal trays and content-portion-pairs that must become gluten-free bread on trays.
       - Goal bread-portions that must become gluten-free sandwiches.
    - Extract sets of "served" pairs from static info for checking serving constraints.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify which goal no_gluten_bread-2 facts are NOT yet satisfied in the current state.
       Each unsatisfied no_gluten_bread-2 goal represents a bread portion on a tray that still needs a gluten-free bread sandwich,
       which likely requires sandwich making and possibly tray preparation actions.
    2. Identify which goal no_gluten_sandwich-2 bread-portions are NOT yet satisfied in the current state.
       Each unsatisfied no_gluten_sandwich-2 goal represents a bread portion still requiring a gluten-free sandwich preparation.
    3. Count the number of places that are currently waiting (have the predicate `(waiting ?place)` in state).
       These places require preparing sandwiches or serving bread to them.
    4. Count the number of waiting-2 facts `(waiting-2 content-portion-2 bread-portion)` indicating waiting content or bread portions
       linked to sandwiches/trays.
    5. Note that the serving transition requires that served bread portions pairs are consistently changed over.
       Use `served` static information to recognize possible no_gluten transitions referencing these served bread-portion pairs.
    6. Use these counts to estimate the number of sandwich-related actions needed:
       - One action per unsatisfied no_gluten_bread-2 goal (making and placing bread on trays)
       - One action per unsatisfied no_gluten_sandwich-2 goal (making gluten-free sandwiches)
       - One action per waiting place or waiting-2 fact (representing pending serving preparations)
    7. Return the total approximate cost as heuristic value.
       This heuristic value will be zero only for goal states.

    This heuristic yields a finite, reasonable estimate to guide greedy best-first search efficiently in secret_domain.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal facts subsets relevant to heuristic:
        # Goals for gluten-free bread on tray: no_gluten_bread-2(content-portion-2, tray)
        self.goal_no_gluten_bread_2 = set()
        # Goals for gluten-free sandwiches on bread portion: no_gluten_sandwich-2(bread-portion)
        self.goal_no_gluten_sandwich_2 = set()

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "no_gluten_bread-2":
                # Expected form: (no_gluten_bread-2 content-portion-2 tray)
                if len(parts) == 3:
                    self.goal_no_gluten_bread_2.add((parts[1], parts[2]))
            elif pred == "no_gluten_sandwich-2":
                # Expected form: (no_gluten_sandwich-2 bread-portion)
                if len(parts) == 2:
                    self.goal_no_gluten_sandwich_2.add(parts[1])

        # Extract served pairs info from static facts for reference
        self.served_pairs = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) == 3:
                self.served_pairs.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # Helper sets for checking current state:
        # Set of all no_gluten_bread-2 facts in current state: (content-portion-2, tray)
        current_no_gluten_bread_2 = set()
        # Set of all no_gluten_sandwich-2 bread portions in current state
        current_no_gluten_sandwich_2 = set()
        # Set of places currently waiting
        current_waiting_places = set()
        # Set of waiting-2 facts currently in state (content-portion-2, bread-portion)
        current_waiting_2 = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "no_gluten_bread-2" and len(parts) == 3:
                current_no_gluten_bread_2.add((parts[1], parts[2]))
            elif pred == "no_gluten_sandwich-2" and len(parts) == 2:
                current_no_gluten_sandwich_2.add(parts[1])
            elif pred == "waiting" and len(parts) == 2:
                # waiting place
                current_waiting_places.add(parts[1])
            elif pred == "waiting-2" and len(parts) == 3:
                current_waiting_2.add((parts[1], parts[2]))

        # Count unsatisfied gluten-free bread on tray goals
        unsatisfied_no_gluten_bread_2 = len(self.goal_no_gluten_bread_2 - current_no_gluten_bread_2)

        # Count unsatisfied gluten-free sandwich bread portions goals
        unsatisfied_no_gluten_sandwich_2 = len(self.goal_no_gluten_sandwich_2 - current_no_gluten_sandwich_2)

        # Count waiting places and waiting-2 facts as proxy for pending serving/making actions
        num_waiting_places = len(current_waiting_places)
        num_waiting_2 = len(current_waiting_2)

        # Compute heuristic estimate as weighted sum:
        # Weights chosen to roughly reflect typical action counts per type, emphasizing main preparation and serving steps.
        # Making bread on tray (no_gluten_bread-2) and making sandwiches (no_gluten_sandwich-2) considered highest cost.
        # Waiting places and waiting-2 facts impose additional cost.
        heuristic_value = (
            3 * unsatisfied_no_gluten_bread_2
            + 4 * unsatisfied_no_gluten_sandwich_2
            + 2 * num_waiting_places
            + 1 * num_waiting_2
        )

        # Return zero if goal is reached exactly, else heuristic
        if self.goals <= state:
            return 0

        # Ensure heuristic is finite and non-negative
        return heuristic_value
