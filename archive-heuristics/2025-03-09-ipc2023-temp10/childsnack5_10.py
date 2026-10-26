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


class childsnack5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Making a sandwich requires one 'make_sandwich' action.
    - Putting a sandwich on a tray requires one 'put_on_tray' action.
    - Serving a sandwich to a child requires one 'serve_sandwich' action.
    - Moving a tray between places requires one 'move_tray' action.
    - The heuristic prioritizes serving children based on their gluten allergy.
    - We assume that we need to create a sandwich for each child.

    # Heuristic Initialization
    - Extract information about children's gluten allergies from the static facts.
    - Extract information about waiting children and their locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served.
    2. For each waiting child:
       - Determine if the child is allergic to gluten.
       - Estimate the number of actions required to serve the child:
         - If a gluten-free sandwich is needed:
           - 1 action to make a gluten-free sandwich.
           - 1 action to put the sandwich on a tray.
           - 1 action to move the tray to the child's location (if necessary).
           - 1 action to serve the sandwich.
         - If a regular sandwich is sufficient:
           - 1 action to make a regular sandwich.
           - 1 action to put the sandwich on a tray.
           - 1 action to move the tray to the child's location (if necessary).
           - 1 action to serve the sandwich.
    3. Sum the estimated number of actions for all children.
    4. If the goal is already reached, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting information about children's gluten allergies
        and the locations of waiting children from the static facts.
        """
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
        """Estimate the number of actions needed to serve all waiting children."""
        state = node.state

        # If the goal is already reached, return 0.
        if task.goal_reached(state):
            return 0

        served_children = {
            get_parts(fact)[1] for fact in state if match(fact, "served", "*")
        }

        unserved_children = [
            child for child in self.waiting_children if child not in served_children
        ]

        total_cost = 0
        for child in unserved_children:
            # Each child needs a sandwich made, put on a tray, and served.
            total_cost += 3  # make_sandwich, put_on_tray, serve_sandwich

            # Check if a move_tray action is needed
            child_location = self.waiting_children[child]
            tray_locations = {get_parts(fact)[1] for fact in state if match(fact, "at", "*", child_location)}
            if not tray_locations:
                # If the tray is not at the child's location, a move_tray action is needed.
                total_cost += 1  # move_tray

        return total_cost
