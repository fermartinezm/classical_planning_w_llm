from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(waiting child1 table1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by:
    1. Counting unserved children
    2. Accounting for sandwiches that need to be made (considering gluten allergies)
    3. Tracking sandwiches that need to be placed on trays
    4. Considering tray movements needed

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in any order as long as requirements are met
    - Trays can carry multiple sandwiches
    - Moving a tray between locations counts as one action

    # Heuristic Initialization
    - Extract information about allergic children from static facts
    - Identify waiting locations for each child
    - Note which bread and content portions are gluten-free

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic value)
    2. For each unserved child:
       - If allergic: check if gluten-free sandwich exists or can be made
       - If not allergic: check if any sandwich exists or can be made
    3. Count sandwiches that need to be made:
       - For gluten-free: need both gluten-free bread and content available
       - For regular: need any bread and content available
    4. Count sandwiches that need to be placed on trays
    5. Count tray movements needed to serve all children:
       - At least one movement per unique location where children are waiting
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }

        # Extract waiting children and their locations
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static if match(fact, "waiting", "*", "*")
        }

        # Extract gluten-free ingredients
        self.gluten_free_breads = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.gluten_free_contents = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        total_cost = 0

        # Count unserved children (main component of heuristic)
        unserved_children = [
            child for child in self.waiting_children
            if f"(served {child})" not in state
        ]
        total_cost += len(unserved_children)

        # Count sandwiches needed
        sandwiches_needed = 0
        gluten_free_needed = 0

        for child in unserved_children:
            if child in self.allergic_children:
                gluten_free_needed += 1
            else:
                sandwiches_needed += 1

        # Check available sandwiches
        available_sandwiches = sum(
            1 for fact in state if match(fact, "at_kitchen_sandwich", "*")
        )
        available_gluten_free = sum(
            1 for fact in state if match(fact, "no_gluten_sandwich", "*")
        )

        # Count sandwiches that need to be made
        make_sandwich_actions = max(0, sandwiches_needed - available_sandwiches)
        make_gluten_free_actions = max(0, gluten_free_needed - available_gluten_free)
        total_cost += make_sandwich_actions + make_gluten_free_actions

        # Count sandwiches that need to be put on trays
        sandwiches_on_trays = sum(
            1 for fact in state if match(fact, "ontray", "*", "*")
        )
        put_on_tray_actions = max(0, len(unserved_children) - sandwiches_on_trays
        total_cost += put_on_tray_actions

        # Count tray movements needed (at least one per unique waiting location)
        waiting_locations = {
            loc for child, loc in self.waiting_children.items()
            if child in unserved_children
        }
        current_tray_locations = {
            get_parts(fact)[2] for fact in state if match(fact, "at", "tray*", "*")
        }
        move_actions = len(waiting_locations - current_tray_locations)
        total_cost += move_actions

        return total_cost
