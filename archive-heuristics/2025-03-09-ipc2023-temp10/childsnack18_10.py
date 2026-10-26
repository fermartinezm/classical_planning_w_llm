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


class childsnack18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches must be made before being put on a tray.
    - Trays can be moved to the location of the waiting children.
    - Gluten-free sandwiches must be made for children with gluten allergies.

    # Heuristic Initialization
    - Extract the children, their gluten allergies, and their waiting locations from the static facts.
    - Determine the available bread and content portions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting and not yet served.
    2. For each unserved child, determine if they are allergic to gluten.
    3. If a child is allergic to gluten, check if a gluten-free sandwich exists on a tray at their location.
       If not, estimate the cost to make a gluten-free sandwich, put it on a tray, and move the tray to the child.
    4. If a child is not allergic to gluten, check if any sandwich exists on a tray at their location.
       If not, estimate the cost to make a sandwich, put it on a tray, and move the tray to the child.
    5. The cost to make a sandwich includes:
       - Checking if a sandwich of the correct type already exists.
       - If not, making the sandwich (1 action).
       - Putting the sandwich on a tray (1 action).
    6. The cost to move a tray includes:
       - Moving the tray to the child's location (1 action).
    7. Serving the sandwich (1 action).
    8. Sum the costs for all unserved children to get the heuristic value.
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
        self.waiting_children = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in static_facts if match(fact, "waiting", "*", "*")
        }
        self.no_gluten_breads = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_contents = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the minimum cost to serve all waiting children."""
        state = node.state

        served_children = {
            get_parts(fact)[1] for fact in state if match(fact, "served", "*")
        }

        unserved_children = [
            (child, place) for child, place in self.waiting_children
            if child not in served_children
        ]

        total_cost = 0
        for child, place in unserved_children:
            if child in self.allergic_children:
                # Check if a gluten-free sandwich is available at the child's location
                gluten_free_sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if ("(no_gluten_sandwich {})".format(sandwich) in state) and ("(at {} {})".format(tray, place) in state):
                            gluten_free_sandwich_available = True
                            break

                if not gluten_free_sandwich_available:
                    # Estimate cost to make a gluten-free sandwich, put it on a tray, and move the tray
                    make_sandwich_cost = 1  # Make sandwich
                    put_on_tray_cost = 1  # Put on tray
                    move_tray_cost = 1  # Move tray to the child's location
                    total_cost += make_sandwich_cost + put_on_tray_cost + move_tray_cost

                serve_sandwich_cost = 1  # Serve the sandwich
                total_cost += serve_sandwich_cost

            elif child in self.not_allergic_children:
                # Check if any sandwich is available at the child's location
                sandwich_available = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if ("(at {} {})".format(tray, place) in state):
                            sandwich_available = True
                            break

                if not sandwich_available:
                    # Estimate cost to make a sandwich, put it on a tray, and move the tray
                    make_sandwich_cost = 1  # Make sandwich
                    put_on_tray_cost = 1  # Put on tray
                    move_tray_cost = 1  # Move tray to the child's location
                    total_cost += make_sandwich_cost + put_on_tray_cost + move_tray_cost

                serve_sandwich_cost = 1  # Serve the sandwich
                total_cost += serve_sandwich_cost

        # If all goals are reached, return 0
        if node.task.goal_reached(state):
            return 0

        return total_cost
