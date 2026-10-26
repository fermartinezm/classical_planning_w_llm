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


class childsnack2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the actions of making sandwiches, putting them on trays, moving trays to the children, and serving the sandwiches.

    # Assumptions
    - Each child needs one sandwich.
    - Making a sandwich requires bread and content.
    - Trays can be moved between places.
    - Gluten-allergic children require gluten-free sandwiches.

    # Heuristic Initialization
    - Extract the children, their gluten allergies, and their waiting locations from the static facts.
    - Identify available bread and content portions from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of unserved children.
    2. For each unserved child, determine if they need a gluten-free sandwich.
    3. Estimate the number of sandwiches that need to be made.
       - If there are enough sandwiches already made, no additional make_sandwich actions are needed.
       - Otherwise, estimate the number of make_sandwich actions required.
    4. Estimate the number of put_on_tray actions needed.
    5. Estimate the number of move_tray actions needed to bring the tray to the waiting children.
    6. Estimate the number of serve_sandwich actions needed.
    7. Sum up the estimated number of actions for each step.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Children and their gluten allergies.
        - Waiting locations of children.
        - Available bread and content.
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

        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        for op in task.operators:
            if op.name == 'make_sandwich_no_gluten':
                for pre in op.preconditions:
                    if 'no_gluten_bread' in pre:
                        self.no_gluten_bread.add(get_parts(pre)[1])
                    if 'no_gluten_content' in pre:
                        self.no_gluten_content.add(get_parts(pre)[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = (self.allergic_children | self.not_allergic_children) - served_children
        num_unserved_children = len(unserved_children)

        if num_unserved_children == 0:
            return 0

        # Count the number of gluten-free sandwiches needed
        num_gluten_free_needed = len(self.allergic_children - served_children)

        # Count the number of regular sandwiches needed
        num_regular_needed = len(self.not_allergic_children - served_children)

        # Count available bread and content
        available_bread = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        available_content = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        available_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        available_no_gluten_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        sandwiches_on_tray = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}

        # Estimate make_sandwich actions
        num_make_sandwich_actions = 0
        num_make_sandwich_no_gluten_actions = 0

        if len(available_no_gluten_sandwiches) + len(sandwiches_on_tray) < num_gluten_free_needed:
            num_make_sandwich_no_gluten_actions = num_gluten_free_needed - (len(available_no_gluten_sandwiches) + len(sandwiches_on_tray))

        if len(available_sandwiches) + len(sandwiches_on_tray) < num_regular_needed:
            num_make_sandwich_actions = num_regular_needed - (len(available_sandwiches) + len(sandwiches_on_tray))

        # Estimate put_on_tray actions
        num_put_on_tray_actions = num_gluten_free_needed + num_regular_needed - len(sandwiches_on_tray)

        # Estimate move_tray actions
        trays_at_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at", "*", "kitchen")}
        num_move_tray_actions = 0
        for child in unserved_children:
            tray_at_child_location = False
            for fact in state:
                if match(fact, "at", "*", self.waiting_children_locations[child]):
                    tray_at_child_location = True
                    break
            if not tray_at_child_location:
                num_move_tray_actions += 1

        # Estimate serve_sandwich actions
        num_serve_sandwich_actions = num_unserved_children

        total_cost = (
            num_make_sandwich_actions
            + num_make_sandwich_no_gluten_actions
            + num_put_on_tray_actions
            + num_move_tray_actions
            + num_serve_sandwich_actions
        )

        return total_cost
