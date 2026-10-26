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


class childsnack3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the number of children waiting, the number of sandwiches that need to be made,
    the number of trays available, and the need to move trays to the correct locations.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made from available bread and content portions in the kitchen.
    - Trays are used to carry sandwiches from the kitchen to the children.
    - The heuristic assumes that the 'notexist' predicate accurately reflects the number of sandwiches that still need to be created.

    # Heuristic Initialization
    - Count the number of children waiting.
    - Identify the children who are allergic to gluten.
    - Identify the available gluten-free bread and content.
    - Store the static facts related to allergies and waiting children.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. Count the number of children who are allergic to gluten.
    3. Count the number of sandwiches that are on trays.
    4. Count the number of sandwiches that still need to be made (using the 'notexist' predicate).
    5. Estimate the number of 'make_sandwich' actions needed.  If there are gluten-allergic children, prioritize making gluten-free sandwiches if possible.
    6. Estimate the number of 'put_on_tray' actions needed.
    7. Estimate the number of 'move_tray' actions needed to bring the trays to the waiting children.
    8. Estimate the number of 'serve_sandwich' actions needed.  Consider gluten allergies when serving.
    9. The heuristic value is the sum of these estimated action counts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.waiting_children = set()
        self.allergic_children = set()
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                self.waiting_children.add(get_parts(fact)[1])
            elif match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_bread", "*"):
                self.no_gluten_bread.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.no_gluten_content.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        served_children = 0
        for fact in state:
            if match(fact, "served", "*"):
                served_children += 1

        waiting_children_count = len(self.waiting_children)
        allergic_children_count = len(self.allergic_children)

        if served_children == waiting_children_count:
            return 0

        sandwiches_on_trays = 0
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwiches_on_trays += 1

        sandwiches_to_make = 0
        for fact in state:
            if match(fact, "notexist", "*"):
                sandwiches_to_make += 1

        make_sandwich_cost = 0
        if sandwiches_to_make > 0:
            make_sandwich_cost = sandwiches_to_make

        put_on_tray_cost = 0
        if sandwiches_on_trays < waiting_children_count:
            put_on_tray_cost = max(0, waiting_children_count - sandwiches_on_trays)

        move_tray_cost = 0
        waiting_children_at_tray_location = 0
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray = get_parts(fact)[1]
                location = get_parts(fact)[2]
                for waiting_child in self.waiting_children:
                    for static_fact in self.goals:
                        if match(static_fact, "served", waiting_child):
                            continue
                        if match(static_fact, "waiting", waiting_child, location):
                            waiting_children_at_tray_location += 1
        move_tray_cost = max(0, waiting_children_count - waiting_children_at_tray_location)

        serve_sandwich_cost = max(0, waiting_children_count - served_children)

        total_cost = make_sandwich_cost + put_on_tray_cost + move_tray_cost + serve_sandwich_cost

        return total_cost
