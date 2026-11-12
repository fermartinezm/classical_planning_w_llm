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

    Summary:
    This heuristic estimates the number of actions needed to reach the goal 
    state in the secret_domain by analyzing how many "no_gluten_bread-2" 
    goals remain unsatisfied for each tray (goal fact: (no_gluten_bread-2 content-portion-2 tray)),
    and assessing the required preparation steps to produce these gluten-free items.

    The heuristic guides a greedy best-first search through estimating how many 
    sandwich-making and serving actions are still required, based on the current 
    state facts involving trays, bread portions, content portions, children, and waiting places.

    Assumptions:
    - The goal facts are of the form (no_gluten_bread-2 <content-portion-2> <tray>).
    - Each goal requires making a sandwhich with specific gluten properties.
    - Making sandwiches requires content-portions and bread-portions to be at the kitchen with appropriate gluten-free conditions.
    - Serving sandwiches involves waiting places changing states from waiting to served.
    - Multiple trays and sandwiches may be involved; the heuristic counts the number of remaining sandwiches to prepare and serve.
    - The heuristic need not be admissible but should efficiently estimate remaining necessary actions.

    Heuristic Initialization:
    - Extract from the goal facts all (no_gluten_bread-2 <content-portion-2> <tray>) entries representing sandwiches to prepare.
    - Extract static facts that identify relationships like (served bread-portionX bread-portionY), (waiting placeX), and gluten properties.
    - Record all trays involved in the goals for indexing.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify each goal sandwich (content-portion-2 and tray) from goal facts.
    2. Check if the goal fact (no_gluten_bread-2 cp2 tray) is present in the current state:
       - If yes, no action required for that goal sandwich.
       - If no, sandstone must still be prepared and served.
    3. Count total missing sandwiches = number of goal sandwiches without corresponding fact in state.
    4. For each missing sandwich, estimate the minimum number of main actions:
       - Preparing sandwich (likely involves at least 2-3 key actions: 
         make_sandwich, put_on_tray, serve_sandwich_no_gluten or serve_sandwich).
       - Consider waiting places: sandwiches require waiting places to be ready.
         Count how many waiting places are available and how many sandwiches depend on these.
    5. Use static facts (such as served relations) to estimate which bread portions must be rotated or served.
    6. Encourage usage of waiting places by counting unsatisfied (waiting place) facts.
    7. Calculate heuristic cost as:
       estimated_actions = 
          missing_sandwiches * average_actions_per_sandwich + 
          waiting_places_to_free * waiting_action_cost +
          bread rotation / serving estimated cost (if applicable)
    8. Return 0 if all goal facts satisfied.

    The heuristic attempts to balance complexity and computational efficiency by counting unsatisfied gluten-free sandwiches
    and estimating a fixed number of actions per sandwich preparation and service, adjusted by waiting place availability.

    """

    def __init__(self, task):
        """
        Initialization by extracting goal sandwiches and relevant static facts.
        """
        self.goals = task.goals

        # Extract all goal sandwiches as tuples (content-portion-2, tray)
        self.goal_sandwiches = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "no_gluten_bread-2" and len(parts) == 3:
                cp2, tray = parts[1], parts[2]
                self.goal_sandwiches.add((cp2, tray))

        # Extract served relations as set of tuples (bread1, bread2)
        self.served_pairs = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 3:
                bread1, bread2 = parts[1], parts[2]
                self.served_pairs.add((bread1, bread2))

        # Extract waiting places from static (if any) or rely on dynamic state to count waiting places

    def __call__(self, node):
        state = node.state

        # Quick check if goal already reached: all goal sandwiches present in state as no_gluten_bread-2 facts
        satisfied = 0
        for cp2, tray in self.goal_sandwiches:
            goal_fact = f"(no_gluten_bread-2 {cp2} {tray})"
            if goal_fact in state:
                satisfied += 1

        if satisfied == len(self.goal_sandwiches):
            return 0  # Goal reached

        missing_sandwiches = len(self.goal_sandwiches) - satisfied

        # Count how many waiting places are currently waiting (waiting ?place)
        waiting_places = [fact for fact in state if fact.startswith("(waiting ")]
        num_waiting = len(waiting_places)

        # Count also waiting-2 facts (waiting-2 content-portion-2 bread-portion)
        waiting_2_facts = [fact for fact in state if fact.startswith("(waiting-2 ")]
        num_waiting_2 = len(waiting_2_facts)

        # Estimate average actions per missing sandwich (heuristically):
        #  - At least 3 actions for making sandwich and putting on tray
        #  - At least 2 actions to serve sandwich and free waiting place
        #  - Some bread rotation or serving might be needed per bread portion

        # Very rough constants, experimentally chosen to guide search:
        ACTIONS_PER_SANDWICH = 5  
        WAITING_COST = 1  

        # Penalty for missing bread rotations or served pairs not reflected in state:
        # Count how many served facts from static are not true in state and relate to sandwiches needed? 
        # This is a complex check; we approximate by counting served facts related to goal sandwiches bread portions missing from state.
        
        # Extract all bread-portion facts currently at kitchen or related to bread-portions on trays
        # Approximate unserved breads (served pairs in static but no fact to confirm in state = missing)

        # Count served facts missing in state among those related to bread portions (heuristic)
        missing_served_relations = 0
        for bread1, bread2 in self.served_pairs:
            served_fact = f"(served {bread1} {bread2})"
            if served_fact not in state:
                missing_served_relations += 1

        # Combine estimated costs
        estimated_cost = (
            missing_sandwiches * ACTIONS_PER_SANDWICH
            + max(0, missing_sandwiches - num_waiting) * WAITING_COST  # penalty for not enough waiting places
            + missing_served_relations * 0.5  # small penalty for missing served relations to encourage bread rotations
        )

        # Ensure heuristic is at least 1 if not goal; avoids zero heuristic misleading search
        return max(1, int(estimated_cost))
