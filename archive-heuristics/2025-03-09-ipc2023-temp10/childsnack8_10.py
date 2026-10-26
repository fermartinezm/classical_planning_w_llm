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


class childsnack8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten.
    - Allergic children must receive gluten-free sandwiches.
    - Trays must be at the location of the child to serve the sandwich.

    # Heuristic Initialization
    - Identify children with gluten allergies.
    - Store the waiting places of each child.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. For each waiting child:
       - Determine if the child is allergic to gluten.
       - If allergic, check if a gluten-free sandwich is on a tray at the child's location. If not, estimate the cost to make one and put it on a tray and move the tray to the child.
       - If not allergic, check if any sandwich is on a tray at the child's location. If not, estimate the cost to make one and put it on a tray and move the tray to the child.
    3. The heuristic value is the sum of these individual costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting information about children's allergies
        and their waiting locations from the static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_places = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions required to serve all waiting children."""
        state = node.state

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_children = set(self.waiting_places.keys())
        unserved_children = waiting_children - served_children

        if not unserved_children:
            return 0  # All children are served

        cost = 0
        for child in unserved_children:
            child_place = self.waiting_places[child]
            sandwich_available = False

            # Check if a suitable sandwich is already on a tray at the child's location
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    tray_at_place = False
                    for fact2 in state:
                        if match(fact2, "at", tray, child_place):
                            tray_at_place = True
                            break
                    if not tray_at_place:
                        continue

                    if child in self.allergic_children:
                        if match(fact, "no_gluten_sandwich", sandwich):
                            sandwich_available = True
                            break
                    else:
                        sandwich_available = True
                        break

            if not sandwich_available:
                # Estimate cost to make a sandwich, put it on a tray, and move the tray
                cost += 4  # make_sandwich + put_on_tray + move_tray + serve_sandwich

        return cost
