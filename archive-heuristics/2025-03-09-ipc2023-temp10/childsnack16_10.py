from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnack16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten.
    - Allergic children must receive gluten-free sandwiches.
    - Trays must be moved to the correct location before serving.
    - Ingredients are available at the kitchen.

    # Heuristic Initialization
    - Extract information about children's allergies from the static facts.
    - Identify waiting children and their locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served.
    2. Count the number of allergic and non-allergic children.
    3. Count the number of gluten-free and regular sandwiches needed.
    4. Estimate the number of sandwiches that need to be made.
    5. Estimate the number of trays that need to be moved.
    6. Estimate the number of serve actions needed.
    7. Sum the estimated costs of making sandwiches, moving trays, and serving children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Information about children's allergies.
        - Information about waiting children and their locations.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the goal is reached
        if task.goal_reached(node.state):
            return 0

        # Count the number of served children
        served_children = sum(1 for fact in state if match(fact, "served", "*"))

        # Count the number of waiting children
        num_waiting_children = len(self.waiting_children_locations)

        # Calculate the number of children that still need to be served
        children_to_serve = num_waiting_children - served_children

        if children_to_serve <= 0:
            return 0

        # Count the number of gluten-free sandwiches needed
        gluten_free_sandwiches_needed = sum(
            1 for child in self.allergic_children if f"(served {child})" not in state
        )

        # Count the number of regular sandwiches needed
        regular_sandwiches_needed = sum(
            1 for child in self.not_allergic_children if f"(served {child})" not in state
        )

        # Count the number of sandwiches on trays
        sandwiches_on_trays = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        # Count the number of sandwiches at the kitchen
        sandwiches_at_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))

        # Estimate the number of sandwiches that need to be made
        sandwiches_to_make = max(0, gluten_free_sandwiches_needed + regular_sandwiches_needed - sandwiches_on_trays - sandwiches_at_kitchen)

        # Estimate the number of trays that need to be moved
        trays_to_move = 0
        for child, place in self.waiting_children_locations.items():
            tray_at_correct_place = False
            for fact in state:
                if match(fact, "at", "*", place):
                    tray_at_correct_place = True
                    break
            if not tray_at_correct_place and f"(served {child})" not in state:
                trays_to_move += 1

        # Estimate the number of serve actions needed
        serve_actions_needed = children_to_serve

        # Sum the estimated costs
        heuristic_value = sandwiches_to_make + trays_to_move + serve_actions_needed

        return heuristic_value
