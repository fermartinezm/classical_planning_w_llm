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


class childsnack13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the need to make sandwiches, put them on trays, and move the trays to the children.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten, depending on the child's allergies.
    - Trays can be moved between places.
    - Bread and content portions are always available at the kitchen.

    # Heuristic Initialization
    - Identify children who are allergic to gluten.
    - Identify children who are not allergic to gluten.
    - Store the waiting children and their locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. For each waiting child:
       - Determine if the child is allergic to gluten.
       - Estimate the cost to serve the child:
         - If a gluten-free sandwich is needed:
           - Check if a gluten-free sandwich is already on a tray at the child's location. If so, the cost is 0.
           - Otherwise, estimate the cost as: make_sandwich + put_on_tray + move_tray + serve_sandwich
         - If a non-gluten-free sandwich is needed:
           - Check if a non-gluten-free sandwich is already on a tray at the child's location. If so, the cost is 0.
           - Otherwise, estimate the cost as: make_sandwich + put_on_tray + move_tray + serve_sandwich
    3. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children = {}
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                place = parts[2]
                self.waiting_children[child] = place

    def __call__(self, node):
        """Estimate the minimum cost to serve all waiting children."""
        state = node.state

        if node.state.issuperset(self.goals):
            return 0

        # Define the cost of each action.
        make_sandwich_cost = 1
        put_on_tray_cost = 1
        move_tray_cost = 1
        serve_sandwich_cost = 1

        total_cost = 0
        served_children = 0

        for fact in state:
            if match(fact, "served", "*"):
                served_children += 1

        waiting_children_count = len(self.waiting_children)

        if served_children == waiting_children_count:
            return 0

        for child, place in self.waiting_children.items():
            # Check if the child is already served
            served = False
            for fact in state:
                if match(fact, "served", child):
                    served = True
                    break
            if served:
                continue

            # Check if a suitable sandwich is already on a tray at the child's location
            sandwich_available = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    tray_at_location = False
                    for fact2 in state:
                        if match(fact2, "at", tray, place):
                            tray_at_location = True
                            break
                    if tray_at_location:
                        if child in self.allergic_children:
                            if match(fact, "no_gluten_sandwich", sandwich):
                                sandwich_available = True
                                break
                        else:
                            sandwich_available = True
                            break

            if sandwich_available:
                continue

            # Estimate the cost to serve the child
            total_cost += (
                make_sandwich_cost
                + put_on_tray_cost
                + move_tray_cost
                + serve_sandwich_cost
            )

        return total_cost
