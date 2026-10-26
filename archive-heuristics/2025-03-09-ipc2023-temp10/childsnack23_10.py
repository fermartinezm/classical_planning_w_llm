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


class childsnack23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the need to make sandwiches, put them on trays, move trays to the children, and serve them.
    It prioritizes serving children with gluten allergies with no-gluten sandwiches.

    # Assumptions
    - Each child needs one sandwich.
    - There are enough bread and content portions to make all necessary sandwiches.
    - Trays can be moved to any place.

    # Heuristic Initialization
    - Extract information about children with gluten allergies.
    - Extract information about waiting children and their locations.
    - Identify available bread and content portions in the kitchen.
    - Identify available trays and their locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of unserved children.
    2. Identify the number of children with gluten allergies who are waiting.
    3. Identify the number of children without gluten allergies who are waiting.
    4. Estimate the number of no-gluten sandwiches needed (equal to the number of allergic children waiting).
    5. Estimate the number of regular sandwiches needed (equal to the number of non-allergic children waiting).
    6. For each sandwich type (no-gluten and regular), estimate the number of actions required:
        - Make sandwich: 1 action
        - Put on tray: 1 action
        - Move tray to child: 1 action
        - Serve sandwich: 1 action
    7. If there are not enough sandwiches on trays at the correct locations, estimate the number of "move_tray" actions needed.
    8. Sum up the estimated actions for all sandwiches to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Extract children with gluten allergies.
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }

        # Extract children without gluten allergies.
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }

        # Extract waiting children and their locations.
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts if match(fact, "waiting", "*", "*")
        }

        # Extract available bread portions in the kitchen.
        self.available_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_kitchen_bread", "*")
        }

        # Extract available content portions in the kitchen.
        self.available_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_kitchen_content", "*")
        }

        # Extract available trays and their locations.
        self.tray_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts if match(fact, "at", "*", "*")
        }

        # Extract no_gluten bread and content
        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to serve all waiting children."""
        state = node.state

        # Check if the goal is already reached.
        unserved_children = 0
        for child in self.waiting_children:
            if f"(served {child})" not in state:
                unserved_children += 1

        if unserved_children == 0:
            return 0

        # Count the number of waiting allergic and non-allergic children.
        waiting_allergic = 0
        waiting_non_allergic = 0
        for child in self.waiting_children:
            if f"(served {child})" not in state:
                if child in self.allergic_children:
                    waiting_allergic += 1
                else:
                    waiting_non_allergic += 1

        # Estimate the number of no-gluten and regular sandwiches needed.
        no_gluten_sandwiches_needed = waiting_allergic
        regular_sandwiches_needed = waiting_non_allergic

        # Estimate the number of actions required for each sandwich type.
        no_gluten_actions = 0
        regular_actions = 0

        if no_gluten_sandwiches_needed > 0:
            no_gluten_actions = (
                no_gluten_sandwiches_needed * 4
            )  # make, put_on_tray, move_tray, serve

        if regular_sandwiches_needed > 0:
            regular_actions = (
                regular_sandwiches_needed * 4
            )  # make, put_on_tray, move_tray, serve

        # Adjust for existing sandwiches.
        no_gluten_sandwiches_available = sum(
            1 for fact in state if match(fact, "no_gluten_sandwich", "*")
        )
        regular_sandwiches_available = sum(
            1 for fact in state if match(fact, "at_kitchen_sandwich", "*")
        )

        # Adjust the number of make_sandwich actions if sandwiches already exist
        if no_gluten_sandwiches_available >= no_gluten_sandwiches_needed:
            no_gluten_actions -= no_gluten_sandwiches_needed
        else:
            no_gluten_actions -= no_gluten_sandwiches_available

        if regular_sandwiches_available >= regular_sandwiches_needed:
            regular_actions -= regular_sandwiches_needed
        else:
            regular_actions -= regular_sandwiches_available

        # Adjust for sandwiches already on trays
        no_gluten_sandwiches_on_trays = sum(
            1 for fact in state if match(fact, "ontray", "*", "*") and match(fact, "no_gluten_sandwich", "*")
        )
        regular_sandwiches_on_trays = sum(
            1 for fact in state if match(fact, "ontray", "*", "*") and not match(fact, "no_gluten_sandwich", "*")
        )

        if no_gluten_sandwiches_on_trays >= no_gluten_sandwiches_needed:
            no_gluten_actions -= no_gluten_sandwiches_needed
        else:
            no_gluten_actions -= no_gluten_sandwiches_on_trays

        if regular_sandwiches_on_trays >= regular_sandwiches_needed:
            regular_actions -= regular_sandwiches_needed
        else:
            regular_actions -= regular_sandwiches_on_trays

        # Adjust for trays already at the correct locations
        trays_at_correct_locations = 0
        for child, place in self.waiting_children.items():
            for tray, location in self.tray_locations.items():
                if location == place:
                    trays_at_correct_locations += 1
                    break

        # Calculate the total estimated cost.
        total_cost = no_gluten_actions + regular_actions

        return total_cost
