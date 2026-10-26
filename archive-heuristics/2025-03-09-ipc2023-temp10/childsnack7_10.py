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


class childsnack7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and tray movements. It prioritizes making sandwiches, placing them on trays,
    moving trays to the correct locations, and serving the children.

    # Assumptions
    - There are enough bread and content portions to make sandwiches for all children.
    - Trays are initially in the kitchen.
    - The heuristic focuses on minimizing the number of actions, not necessarily the makespan.

    # Heuristic Initialization
    - Identify children with gluten allergies.
    - Identify the waiting children and their locations.
    - Identify the available trays and their locations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Count the number of unserved children. If all children are served, return 0.
    2.  Count the number of allergic and non-allergic children waiting.
    3.  Estimate the number of sandwiches needed.
    4.  For each unserved child, estimate the cost of serving them:
        a.  If the child is allergic, check if a gluten-free sandwich exists on a tray at the child's location.
            If not, estimate the cost of making a gluten-free sandwich, placing it on a tray, moving the tray to the child's location, and serving the sandwich.
        b.  If the child is not allergic, check if a sandwich exists on a tray at the child's location.
            If not, estimate the cost of making a sandwich, placing it on a tray, moving the tray to the child's location, and serving the sandwich.
    5.  The heuristic value is the sum of the estimated costs for all unserved children.
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
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if all goals are achieved
        if self.goal_reached(node):
            return 0

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = set(self.waiting_children.keys()) - served_children
        num_unserved_children = len(unserved_children)

        if num_unserved_children == 0:
            return 0

        heuristic_value = 0
        for child in unserved_children:
            place = self.waiting_children[child]
            if child in self.allergic_children:
                # Check if a gluten-free sandwich is available at the child's location
                gluten_free_sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if match(fact, "no_gluten_sandwich", sandwich):
                            for fact2 in state:
                                if match(fact2, "at", tray, place):
                                    gluten_free_sandwich_available = True
                                    break
                        if gluten_free_sandwich_available:
                            break

                if not gluten_free_sandwich_available:
                    # Estimate cost: make sandwich + put on tray + move tray + serve
                    heuristic_value += 4
            else:
                # Check if a sandwich is available at the child's location
                sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        for fact2 in state:
                            if match(fact2, "at", tray, place):
                                sandwich_available = True
                                break
                    if sandwich_available:
                        break

                if not sandwich_available:
                    # Estimate cost: make sandwich + put on tray + move tray + serve
                    heuristic_value += 4

        return heuristic_value
