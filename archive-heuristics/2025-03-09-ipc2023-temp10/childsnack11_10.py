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


class childsnack11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Making a sandwich takes one action.
    - Putting a sandwich on a tray takes one action.
    - Serving a sandwich takes one action.
    - Moving a tray takes one action.
    - We can make any sandwich if the ingredients are available.
    - We can move trays between places.

    # Heuristic Initialization
    - Extract the list of children, their gluten allergies, and their waiting places.
    - Identify available bread and content portions, distinguishing between gluten-free and regular options.
    - Determine the location of trays.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting and not yet served.
    2. For each unserved child, determine if they are allergic to gluten.
    3. If a child is allergic to gluten, check if there is a gluten-free sandwich on a tray at their location.
       - If not, estimate the cost of making a gluten-free sandwich (if ingredients are available),
         putting it on a tray, moving the tray to the child's location (if necessary), and serving the sandwich.
    4. If a child is not allergic to gluten, check if there is any sandwich on a tray at their location.
       - If not, estimate the cost of making a sandwich (if ingredients are available),
         putting it on a tray, moving the tray to the child's location (if necessary), and serving the sandwich.
    5. Sum the estimated costs for all unserved children.
    6. If all children are served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting information about children, ingredients, and trays.
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
        """
        Estimate the number of actions needed to serve all waiting children.
        """
        state = node.state

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {
            child for child in self.waiting_children if child not in served_children
        }

        if not unserved_children:
            return 0

        total_cost = 0
        for child in unserved_children:
            child_location = self.waiting_children[child]
            is_allergic = child in self.allergic_children

            if is_allergic:
                # Check for gluten-free sandwich on tray at the child's location
                gluten_free_sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if ("(no_gluten_sandwich {})".format(sandwich) in state) and any(match(f, "at", tray, child_location) for f in state):
                            gluten_free_sandwich_available = True
                            break

                if not gluten_free_sandwich_available:
                    # Estimate cost of making, putting on tray, moving tray, and serving
                    make_sandwich_cost = 1 if (self.no_gluten_bread and self.no_gluten_content) else float('inf')
                    put_on_tray_cost = 1
                    move_tray_cost = 1 if not any(match(f, "at", "*", child_location) for f in state) else 0
                    serve_sandwich_cost = 1
                    total_cost += make_sandwich_cost + put_on_tray_cost + move_tray_cost + serve_sandwich_cost

            else:
                # Check for any sandwich on tray at the child's location
                sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if any(match(f, "at", tray, child_location) for f in state):
                            sandwich_available = True
                            break

                if not sandwich_available:
                    # Estimate cost of making, putting on tray, moving tray, and serving
                    make_sandwich_cost = 1
                    put_on_tray_cost = 1
                    move_tray_cost = 1 if not any(match(f, "at", "*", child_location) for f in state) else 0
                    serve_sandwich_cost = 1
                    total_cost += make_sandwich_cost + put_on_tray_cost + move_tray_cost + serve_sandwich_cost

        return total_cost
