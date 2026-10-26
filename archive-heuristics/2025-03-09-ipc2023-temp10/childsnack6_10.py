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


class childsnack6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten.
    - Allergic children must receive gluten-free sandwiches.
    - Trays can be moved between places.
    - The heuristic focuses on the core actions: making sandwiches, putting them on trays, and serving them.

    # Heuristic Initialization
    - Identify children with gluten allergies.
    - Identify the waiting children and their locations.
    - Determine the number of gluten-free and regular sandwiches needed.
    - Track the available bread and content portions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. Count the number of children with gluten allergies.
    3. Estimate the number of 'make_sandwich' actions needed based on the number of waiting children.
    4. Estimate the number of 'make_sandwich_no_gluten' actions needed based on the number of allergic children.
    5. Estimate the number of 'put_on_tray' actions needed based on the number of sandwiches to be put on a tray.
    6. Estimate the number of 'serve_sandwich' and 'serve_sandwich_no_gluten' actions needed based on the number of waiting children.
    7. Estimate the number of 'move_tray' actions needed based on the number of places where children are waiting.
    8. Sum up the estimated number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, "waiting", "*", "*")
        }
        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if task.goal_reached(state):
            return 0

        waiting_children = 0
        served_children = 0
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                waiting_children += 1
            if match(fact, "served", "*"):
                served_children += 1

        allergic_children_to_serve = 0
        for child in self.allergic_children:
            waiting_fact = f"(waiting {child} *)"
            if any(match(fact, "waiting", child, "*") for fact in state):
                allergic_children_to_serve += 1

        not_allergic_children_to_serve = 0
        for child in self.not_allergic_children:
            waiting_fact = f"(waiting {child} *)"
            if any(match(fact, "waiting", child, "*") for fact in state):
                not_allergic_children_to_serve += 1

        make_sandwich_no_gluten_actions = allergic_children_to_serve
        make_sandwich_actions = not_allergic_children_to_serve

        sandwiches_on_tray = 0
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwiches_on_tray += 1

        put_on_tray_actions = make_sandwich_no_gluten_actions + make_sandwich_actions - sandwiches_on_tray if (make_sandwich_no_gluten_actions + make_sandwich_actions - sandwiches_on_tray) > 0 else 0

        serve_sandwich_actions = not_allergic_children_to_serve
        serve_sandwich_no_gluten_actions = allergic_children_to_serve

        tray_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray, location = get_parts(fact)[1], get_parts(fact)[2]
                if tray.startswith("tray"):
                    tray_locations.add(location)

        waiting_locations = set(self.waiting_children_locations.values())
        move_tray_actions = len(waiting_locations.difference(tray_locations))

        heuristic_value = (
            make_sandwich_actions
            + make_sandwich_no_gluten_actions
            + put_on_tray_actions
            + serve_sandwich_actions
            + serve_sandwich_no_gluten_actions
            + move_tray_actions
        )

        return heuristic_value
