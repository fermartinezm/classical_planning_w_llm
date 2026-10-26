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


class childsnack14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the number of children waiting, the number of sandwiches that need to be made,
    the number of trays available, and the need to move trays to the correct locations.

    # Assumptions
    - Each child needs one sandwich.
    - Making a sandwich requires one bread and one content portion.
    - A tray can hold multiple sandwiches.
    - The heuristic assumes that the number of trays is sufficient.

    # Heuristic Initialization
    - Count the number of children waiting.
    - Identify the children who are allergic to gluten.
    - Store the information about available bread and content.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. Count the number of children who are allergic to gluten.
    3. Count the number of sandwiches that are already on trays.
    4. Count the number of gluten-free sandwiches that are already on trays.
    5. Estimate the number of sandwiches that need to be made.
       - For allergic children, estimate the number of gluten-free sandwiches to make.
       - For non-allergic children, estimate the number of regular sandwiches to make.
    6. Estimate the number of 'put_on_tray' actions needed.
    7. Estimate the number of 'serve_sandwich' actions needed.
    8. Estimate the number of 'move_tray' actions needed.
    9. Sum up all the estimated actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        self.static = task.static

        # Count the number of waiting children and identify allergic children.
        self.waiting_children = set()
        self.allergic_children = set()
        self.waiting_places = {}

        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                place = parts[2]
                self.waiting_children.add(child)
                self.waiting_places[child] = place
            elif match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])

        # Count available bread and content.
        self.available_bread = set()
        self.available_content = set()
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        for fact in self.static:
            if match(fact, "at_kitchen_bread", "*"):
                self.available_bread.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "*"):
                self.available_content.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_bread", "*"):
                self.no_gluten_bread.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.no_gluten_content.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached.
        if all(goal in state for goal in self.goals):
            return 0

        # Count the number of sandwiches on trays.
        sandwiches_on_tray = 0
        gluten_free_sandwiches_on_tray = 0
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwiches_on_tray += 1
                sandwich = get_parts(fact)[1]
                if any(match(f, "no_gluten_sandwich", sandwich) for f in state):
                    gluten_free_sandwiches_on_tray += 1

        # Count the number of served children.
        served_children = 0
        for fact in state:
            if match(fact, "served", "*"):
                served_children += 1

        # Calculate the number of sandwiches needed.
        children_to_serve = len(self.waiting_children) - served_children
        if children_to_serve <= 0:
            return 0

        # Estimate the number of gluten-free sandwiches to make.
        num_allergic_children_unserved = len(self.allergic_children.intersection(self.waiting_children)) - sum(1 for fact in state if match(fact, "served", "*") and get_parts(fact)[1] in self.allergic_children)
        gluten_free_sandwiches_needed = max(0, num_allergic_children_unserved - gluten_free_sandwiches_on_tray)

        # Estimate the number of regular sandwiches to make.
        num_non_allergic_children_unserved = len(self.waiting_children) - len(self.allergic_children.intersection(self.waiting_children)) - sum(1 for fact in state if match(fact, "served", "*") and get_parts(fact)[1] not in self.allergic_children)
        regular_sandwiches_needed = max(0, num_non_allergic_children_unserved - (sandwiches_on_tray - gluten_free_sandwiches_on_tray))

        # Estimate the number of make_sandwich actions.
        make_sandwich_actions = gluten_free_sandwiches_needed + regular_sandwiches_needed

        # Estimate the number of put_on_tray actions.
        put_on_tray_actions = gluten_free_sandwiches_needed + regular_sandwiches_needed

        # Estimate the number of serve_sandwich actions.
        serve_sandwich_actions = children_to_serve

        # Estimate the number of move_tray actions.
        move_tray_actions = 0
        trays_at_correct_place = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray = get_parts(fact)[1]
                place = get_parts(fact)[2]
                for child in self.waiting_children:
                    if self.waiting_places[child] == place:
                        trays_at_correct_place.add(tray)

        move_tray_actions = len(self.waiting_children) - len(trays_at_correct_place)

        # Total estimated cost.
        total_cost = make_sandwich_actions + put_on_tray_actions + serve_sandwich_actions + move_tray_actions

        return total_cost
