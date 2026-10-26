from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robby rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with snacks.
    It counts the minimum actions needed for each unserved, waiting child, considering making a sandwich,
    putting it on a tray, moving the tray to the child's location (if necessary), and serving the sandwich.

    # Assumptions
    - For each waiting child, a sandwich needs to be made, placed on a tray, and served.
    - Trays are initially at the kitchen.
    - We need to move the tray from the kitchen to the child's location if the child is not waiting at the kitchen.
    - One sandwich is sufficient for each child.

    # Heuristic Initialization
    - Extracts static information about children's allergies and waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is currently waiting and not yet served:
    1. Determine if the child is allergic to gluten.
    2. Estimate the cost of making a sandwich: 1 action (make_sandwich or make_sandwich_no_gluten).
    3. Estimate the cost of putting the sandwich on a tray: 1 action (put_on_tray).
    4. Determine the child's waiting place. If it is not the kitchen, estimate the cost of moving a tray to that place: 1 action (move_tray).
    5. Estimate the cost of serving the sandwich: 1 action (serve_sandwich or serve_sandwich_no_gluten).
    6. Sum up these costs for all unserved waiting children to get the total heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by pre-processing static facts to quickly access
        allergy information and waiting children.
        """
        self.goals = task.goals
        static_facts = task.static

        self.child_allergies = {}
        self.waiting_children_places = {}

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergies[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergies[child] = False
            elif match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                place = parts[2]
                self.waiting_children_places[child] = place

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the estimated number of actions to serve all waiting children.
        """
        state = node.state
        heuristic_value = 0

        waiting_children = []
        served_children = set()

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                waiting_children.append(get_parts(fact)[1])
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        for child in waiting_children:
            if child not in served_children:
                heuristic_value += 1  # Serve action
                heuristic_value += 1  # Put on tray action
                heuristic_value += 1  # Make sandwich action
                waiting_place = self.waiting_children_places.get(child)
                if waiting_place and waiting_place != 'kitchen':
                    heuristic_value += 1  # Move tray action

        return heuristic_value
