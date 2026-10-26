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


class childsnack1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the actions of making sandwiches, putting them on trays, moving trays to the children, and serving the sandwiches.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten, depending on the child's allergies.
    - Trays can be moved between places.
    - The heuristic assumes that we always have enough bread and content to make the sandwiches.

    # Heuristic Initialization
    - Extract information about children's allergies, waiting children, available bread and content, and tray locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. For each waiting child, determine if they are allergic to gluten.
    3. If a child is allergic to gluten, check if a gluten-free sandwich is available on a tray at the child's location. If not, estimate the cost of making a gluten-free sandwich and putting it on a tray.
    4. If a child is not allergic to gluten, check if any sandwich is available on a tray at the child's location. If not, estimate the cost of making a sandwich and putting it on a tray.
    5. Estimate the cost of moving a tray to the child's location if necessary.
    6. Sum the costs for all children to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting information about children's allergies,
        waiting children, available bread and content, and tray locations from the static facts.
        """
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
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_children_not_served = [
            child for child in self.waiting_children if child not in served_children
        ]

        total_cost = 0
        for child in waiting_children_not_served:
            child_location = self.waiting_children[child]
            if child in self.allergic_children:
                # Check if there is a gluten-free sandwich on a tray at the child's location
                gluten_free_sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        tray_at_location = False
                        for fact2 in state:
                            if match(fact2, "at", tray, child_location):
                                tray_at_location = True
                                break
                        no_gluten = False
                        for fact3 in state:
                            if match(fact3, "no_gluten_sandwich", sandwich):
                                no_gluten = True
                                break
                        if tray_at_location and no_gluten:
                            gluten_free_sandwich_available = True
                            break

                if not gluten_free_sandwich_available:
                    # Estimate the cost of making a gluten-free sandwich and putting it on a tray
                    total_cost += 3  # make_sandwich_no_gluten + put_on_tray + move_tray
            else:
                # Check if there is any sandwich on a tray at the child's location
                sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        tray_at_location = False
                        for fact2 in state:
                            if match(fact2, "at", tray, child_location):
                                tray_at_location = True
                                break
                        if tray_at_location:
                            sandwich_available = True
                            break

                if not sandwich_available:
                    # Estimate the cost of making a sandwich and putting it on a tray
                    total_cost += 3  # make_sandwich + put_on_tray + move_tray

        # If the goal is reached, return 0
        if self.goal_reached(state):
            return 0

        return total_cost
