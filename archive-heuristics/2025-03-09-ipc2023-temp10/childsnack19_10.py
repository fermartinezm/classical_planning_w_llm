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


class childsnack19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the need to make sandwiches, put them on trays, move trays to the children, and serve them.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made at the kitchen.
    - Trays can be moved between places.
    - The heuristic focuses on the number of unserved children and the availability of sandwiches.

    # Heuristic Initialization
    - Identify children, their gluten allergies, and their waiting places from the static facts.
    - Determine the available bread and content portions at the kitchen.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of unserved children.
    2. Count the number of available sandwiches (both at the kitchen and on trays).
    3. Estimate the number of sandwiches that need to be made.
    4. Estimate the number of sandwiches that need to be put on trays.
    5. Estimate the number of trays that need to be moved to the children's locations.
    6. Estimate the number of serve actions required.
    7. Sum up all the estimated action counts to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting information from the task."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state

        # Count unserved children
        unserved_children = 0
        for child in self.waiting_children:
            if f"(served {child})" not in state:
                unserved_children += 1

        if unserved_children == 0:
            return 0  # Goal state

        # Count available sandwiches
        kitchen_sandwiches = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        tray_sandwiches = sum(1 for fact in state if match(fact, "ontray", "*", "*"))
        total_sandwiches = kitchen_sandwiches + tray_sandwiches

        # Estimate the number of sandwiches to make
        sandwiches_to_make = max(0, unserved_children - total_sandwiches)

        # Estimate the number of sandwiches to put on trays
        sandwiches_to_tray = max(0, unserved_children - tray_sandwiches)

        # Estimate the number of tray moves
        tray_moves = 0
        for child, place in self.waiting_children.items():
            if f"(served {child})" not in state:
                tray_found = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich, tray = get_parts(fact)[1:3]
                        if f"(at {tray} {place})" in state:
                            tray_found = True
                            break
                if not tray_found:
                    tray_moves += 1

        # Estimate the number of serve actions
        serve_actions = unserved_children

        # Sum up the estimated action counts
        heuristic_value = sandwiches_to_make + sandwiches_to_tray + tray_moves + serve_actions

        return heuristic_value
