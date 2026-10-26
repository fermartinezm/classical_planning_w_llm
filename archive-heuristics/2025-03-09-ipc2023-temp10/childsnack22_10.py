from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnack22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the number of children waiting, the number of sandwiches that need to be made,
    and the number of trays available. It prioritizes making sandwiches for children with gluten allergies.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten.
    - Gluten-free sandwiches are prioritized for children with gluten allergies.
    - Trays are used to transport sandwiches to children.
    - The heuristic assumes that all ingredients are available in the kitchen.

    # Heuristic Initialization
    - Extract the list of children with gluten allergies.
    - Extract the list of children without gluten allergies.
    - Extract the list of waiting children.
    - Extract the list of trays.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served.
    2. Count the number of children with gluten allergies.
    3. Count the number of children without gluten allergies.
    4. Count the number of sandwiches already on trays.
    5. Count the number of gluten-free sandwiches already on trays.
    6. Estimate the number of gluten-free sandwiches that need to be made.
    7. Estimate the number of regular sandwiches that need to be made.
    8. Estimate the number of actions needed to make the sandwiches.
    9. Estimate the number of actions needed to put the sandwiches on trays.
    10. Estimate the number of actions needed to move the trays to the children.
    11. Estimate the number of actions needed to serve the sandwiches to the children.
    12. The heuristic value is the sum of all these estimates.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "waiting", "*", "*")
        }
        self.trays = {get_parts(fact)[1] for fact in static_facts if match(fact, "at", "*", "*")}

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        served_children = sum(1 for fact in state if match(fact, "served", "*"))
        waiting_children_count = len(self.waiting_children) - served_children

        if waiting_children_count <= 0:
            return 0  # Goal state reached

        gluten_free_sandwiches_on_tray = sum(
            1 for fact in state if match(fact, "no_gluten_sandwich", "*") and match(fact, "ontray", "*", "*")
        )
        sandwiches_on_tray = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        allergic_children_waiting = 0
        for child in self.allergic_children:
            if any(match(fact, "waiting", child, "*") for fact in state) and not any(match(fact, "served", child) for fact in state):
                allergic_children_waiting += 1

        not_allergic_children_waiting = waiting_children_count - allergic_children_waiting

        # Estimate the number of sandwiches needed
        sandwiches_to_make = waiting_children_count - sandwiches_on_tray

        # Prioritize gluten-free sandwiches for allergic children
        gluten_free_sandwiches_needed = max(0, allergic_children_waiting - gluten_free_sandwiches_on_tray)
        regular_sandwiches_needed = max(0, sandwiches_to_make - gluten_free_sandwiches_needed)

        # Estimate the number of actions
        make_sandwich_actions = max(0, gluten_free_sandwiches_needed + regular_sandwiches_needed)
        put_on_tray_actions = max(0, sandwiches_to_make)
        serve_sandwich_actions = waiting_children_count

        # Each tray needs to be moved to the correct place
        move_tray_actions = len(self.trays)

        heuristic_value = make_sandwich_actions + put_on_tray_actions + serve_sandwich_actions + move_tray_actions

        return heuristic_value
