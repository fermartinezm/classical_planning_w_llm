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


class childsnack21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the actions of making sandwiches, putting them on trays, moving trays to the children,
    and serving the sandwiches.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten, depending on the child's allergies.
    - Trays can be moved between places.
    - The heuristic assumes that all necessary ingredients (bread and content) are available.

    # Heuristic Initialization
    - Count the number of children waiting to be served.
    - Identify children with gluten allergies.
    - Store the available trays.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting and not yet served.
    2. For each unserved child:
       a. Check if the child has a gluten allergy.
       b. If the child is allergic to gluten, check if a gluten-free sandwich is available on a tray at the child's location. If not, estimate the cost of making a gluten-free sandwich, putting it on a tray, moving the tray to the child, and serving the sandwich.
       c. If the child is not allergic to gluten, check if any sandwich is available on a tray at the child's location. If not, estimate the cost of making a sandwich, putting it on a tray, moving the tray to the child, and serving the sandwich.
    3. Sum the estimated costs for all unserved children.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Count the number of children waiting to be served.
        self.waiting_children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.trays = set()

        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                self.waiting_children.add(get_parts(fact)[1])
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            if match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])

        for fact in static_facts:
            if match(fact, "at", "*", "kitchen"):
                tray = get_parts(fact)[1]
                tray_type = tray.replace("tray", "")
                if tray_type.isdigit():
                    self.trays.add(tray)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all waiting children."""
        state = node.state

        # Count the number of served children.
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        # Count the number of unserved children.
        unserved_children = self.waiting_children - served_children

        total_cost = 0
        for child in unserved_children:
            # Get the location of the child.
            child_location = None
            for fact in state:
                if match(fact, "waiting", child, "*"):
                    child_location = get_parts(fact)[2]
                    break

            # Check if a suitable sandwich is available on a tray at the child's location.
            sandwich_available = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    tray_location = None
                    for fact2 in state:
                        if match(fact2, "at", tray, "*"):
                            tray_location = get_parts(fact2)[2]
                            break
                    if tray_location == child_location:
                        if child in self.allergic_children:
                            if match(fact, "no_gluten_sandwich", sandwich):
                                sandwich_available = True
                                break
                        else:
                            sandwich_available = True
                            break

            # If a suitable sandwich is not available, estimate the cost of making one.
            if not sandwich_available:
                # Estimate the cost of making a sandwich, putting it on a tray, moving the tray, and serving.
                total_cost += 4  # make_sandwich + put_on_tray + move_tray + serve_sandwich

        return total_cost
