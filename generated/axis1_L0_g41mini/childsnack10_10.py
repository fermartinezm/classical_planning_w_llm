from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with possible wildcards.

    Args:
        fact (str): A PDDL fact string, e.g., "(at_kitchen_bread bread1)".
        args (list[str]): Pattern elements to match against the fact components.

    Returns:
        bool: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    Summary
    -------
    This heuristic estimates the minimum number of actions required to serve all children
    with sandwiches according to their gluten allergy requirements. It balances sandwich
    preparation, tray placement, and serving, giving an informative but efficiently
    computable estimate to guide greedy best-first search.

    Assumptions
    -----------
    - Serving a child requires one sandwich delivered on a tray at the child's waiting place.
    - Children with gluten allergies must receive gluten-free sandwiches; others can have regular sandwiches.
    - Bread and content portions are consumed when making a sandwich; sandwiches are distinct objects.
    - Sandwiches can be made gluten-free if both bread and content are gluten-free.
    - Sandwiches start in the kitchen and need to be put on trays also in the kitchen.
    - Trays can be moved between places, and sandwiches can be served from trays at the child's waiting place.
    - The goal is reached when all children have "(served ?c)".
    - Each action costs 1, and the heuristic counts approximate action steps needed to produce,
      place, move, and serve sandwiches to all unserved children.

    Heuristic Initialization
    ------------------------
    - From static facts, determine which children are allergic to gluten.
    - Initialize sets of children allergic and not allergic to gluten for easy lookup.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify children not yet served.
    2. Separate unserved children into two sets: allergic to gluten and not allergic.
    3. Count how many gluten-free sandwiches are available:
       - Sandwiches marked "no_gluten_sandwich" and currently "at_kitchen_sandwich".
    4. Count how many regular sandwiches are available likewise.
    5. Identify how many bread and content portions are available, tracking gluten-free bread/content separately.
    6. For sandwiches needed:
       - Allergic children require gluten-free sandwiches.
       - Others require regular sandwiches (gluten-free sandwiches may not count as regular).
    7. Estimate the number of sandwiches that still must be made for both groups based on available sandwiches.
    8. For sandwich making, estimate number of sandwiches that can be made from available bread and content portions:
       - Gluten-free sandwiches require gluten-free bread AND gluten-free content portions.
       - Regular sandwiches can be made from any bread and content (may include gluten-containing).
       - Sandwich making consumes one bread and one content portion each.
    9. Count how many trays are currently at kitchen.
    10. Count how many sandwiches are already on trays.
    11. Estimate sandwiches still needing to be placed on trays = sandwiches available/not yet on trays.
    12. Count the number of unique places where children are waiting, because trays might need to be moved there.
    13. Estimate total actions as the sum of:
        - Number of sandwiches to make (each requiring 1 action).
        - Number of sandwiches to be put on trays (each requiring 1 action).
        - Number of tray moves needed at minimum: at least one per distinct place with waiting children.
        - Number of serve actions = number of unserved children.
    14. The heuristic value is zero only if all children are served (goal state).
    15. The heuristic strives to minimize overestimates but may be non-admissible to guide search effectively.

    This heuristic runs in O(n) time relative to the problem size, extracting counts from facts and static data.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children's gluten allergies.

        Args:
            task (Task): The planning task with static facts and goal conditions.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract set of children allergic to gluten
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }
        # Extract set of children NOT allergic to gluten
        self.not_allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "not_allergic_gluten", "*")
        }

    def __call__(self, node):
        """
        Estimate the number of actions required to reach a goal state from the given node.

        Args:
            node: Planner search node containing current state.

        Returns:
            int: Estimated cost (number of actions) to achieve the goal from current state.
                 Zero iff goal is reached.
        """
        state = node.state

        # Quick goal check: if all served, heuristic is zero.
        served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }
        # Children that are still waiting (not served)
        waiting_children = {
            child
            for fact in state
            if match(fact, "waiting", "*", "*")
            for child in [get_parts(fact)[1]]
            if child not in served_children
        }

        if not waiting_children:
            return 0  # Goal reached

        # Partition waiting children by allergy
        allergic_waiting = waiting_children.intersection(self.allergic_children)
        not_allergic_waiting = waiting_children.intersection(self.not_allergic_children)

        # Count sandwiches available at kitchen
        # Gluten-free sandwiches at kitchen
        gluten_free_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }.intersection({
            get_parts(fact)[1]
            for fact in state
            if match(fact, "no_gluten_sandwich", "*")
        })

        # Regular sandwiches at kitchen (those sandwiches at kitchen not gluten-free)
        all_sandwiches_at_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }
        regular_sandwiches = all_sandwiches_at_kitchen - gluten_free_sandwiches

        # Bread and content portions available at kitchen
        breads_at_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_bread", "*")
        }
        contents_at_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_content", "*")
        }

        # Gluten-free bread portions at kitchen
        gluten_free_breads = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "no_gluten_bread", "*")
        }.intersection(breads_at_kitchen)

        # Gluten-free content portions at kitchen
        gluten_free_contents = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "no_gluten_content", "*")
        }.intersection(contents_at_kitchen)

        # Estimate how many gluten-free sandwiches can still be made
        gluten_free_breads_count = len(gluten_free_breads)
        gluten_free_contents_count = len(gluten_free_contents)
        gluten_free_sandwiches_possible = min(gluten_free_breads_count, gluten_free_contents_count)

        # Estimate how many regular sandwiches can still be made
        # Regular bread and content are those not gluten-free
        regular_breads = breads_at_kitchen - gluten_free_breads
        regular_contents = contents_at_kitchen - gluten_free_contents
        num_regular_breads = len(regular_breads) + (gluten_free_breads_count)  # gluten-free breads can also be used in regular sandwiches
        num_regular_contents = len(regular_contents) + (gluten_free_contents_count)  # gluten-free contents can be used in regular sandwiches
        # But gluten-free breads/contents counted already in gluten_free_sandwiches_possible,
        # to avoid double counting assume worst case: breads/contents can be used for gluten-free or regular sandwiches but not both.

        # To simplify: regular sandwiches can be made from all breads and contents excluding those reserved for gluten-free sandwiches.
        # So:
        breads_for_regular = len(breads_at_kitchen) - gluten_free_sandwiches_possible
        contents_for_regular = len(contents_at_kitchen) - gluten_free_sandwiches_possible
        regular_sandwiches_possible = max(0, min(breads_for_regular, contents_for_regular))

        # How many sandwiches are required but not yet available
        gluten_free_needed = max(0, len(allergic_waiting) - len(gluten_free_sandwiches))
        regular_needed = max(0, len(not_allergic_waiting) - len(regular_sandwiches))

        # Sandwiches to make cannot exceed what is possible from available ingredients
        gluten_free_to_make = min(gluten_free_needed, gluten_free_sandwiches_possible)
        regular_to_make = min(regular_needed, regular_sandwiches_possible)

        # Count sandwiches currently on trays
        sandwiches_on_trays = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ontray", "*", "*")
        }

        # Count trays currently at kitchen
        trays_at_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at", "*", "kitchen")
        }
        tray_count = len(trays_at_kitchen)

        # Count sandwiches at kitchen (not on tray)
        sandwiches_at_kitchen = all_sandwiches_at_kitchen - sandwiches_on_trays

        # Estimate sandwiches needing to be put on trays: sandwiches available - already on trays
        # Note: We need enough sandwiches on trays to serve children.
        total_sandwiches_required = len(waiting_children)
        sandwiches_available = len(all_sandwiches_at_kitchen)
        sandwiches_on_trays_count = len(sandwiches_on_trays)
        sandwiches_to_tray = max(0, total_sandwiches_required - sandwiches_on_trays_count)

        # The trays need to be moved to waiting places of children.
        waiting_places = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "waiting", "*", "*")
        }

        # If no trays at kitchen but trays are needed, at least one tray move implies tray movement cost.
        tray_moves_needed = max(0, len(waiting_places))

        # Serving each waiting child requires at least one serve action.
        serve_actions = len(waiting_children)

        # Actions estimates:
        # 1 action per sandwich to make
        # 1 action per sandwich to put on tray
        # 1 action per tray move (at least one per distinct waiting place)
        # 1 action per serving sandwich to child

        heuristic_estimate = (
            gluten_free_to_make + regular_to_make +
            sandwiches_to_tray +
            tray_moves_needed +
            serve_actions
        )

        # Heuristic must always be finite and at least 1 if not goal.
        # Lower bound at least number of unserved children.
        heuristic_estimate = max(heuristic_estimate, 1)

        return heuristic_estimate
