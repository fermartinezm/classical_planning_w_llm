from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ontray sandw1 tray1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with sandwiches,
    considering their gluten allergies and the need to make, tray, move, and serve sandwiches.

    # Assumptions
    - There are always enough bread and content portions at the kitchen to make sandwiches.
    - Trays are initially at the kitchen and can be moved to serve children at different places.
    - The goal is to serve all children who are initially in a 'waiting' state.

    # Heuristic Initialization
    - Identify children who are allergic and not allergic to gluten from the static facts.
    - Identify children who are waiting and their waiting places from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each child who is currently 'waiting' and not yet 'served':
    1. Check if there is a suitable sandwich already on a tray at the child's waiting place.
       - A suitable sandwich is 'no_gluten_sandwich' for allergic children and any sandwich for non-allergic children.
    2. If a suitable sandwich is found, estimate 1 action (serve_sandwich or serve_sandwich_no_gluten).
    3. If no suitable sandwich is found, estimate 4 actions:
       - 1 action to make a sandwich (make_sandwich or make_sandwich_no_gluten).
       - 1 action to put the sandwich on a tray (put_on_tray).
       - 1 action to move a tray to the child's waiting place (move_tray).
       - 1 action to serve the sandwich (serve_sandwich or serve_sandwich_no_gluten).
    4. Sum up the estimated actions for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about children's allergies and waiting status.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children_places = {}

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                self.waiting_children_places[child] = place

    def __call__(self, node):
        """
        Estimate the number of actions needed to serve all waiting children.
        """
        state = node.state
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        heuristic_value = 0
        for child in self.waiting_children_places:
            if child not in served_children:
                waiting_place = self.waiting_children_places[child]
                suitable_sandwich_found = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        for tray_location_fact in state:
                            if match(tray_location_fact, "at", tray, waiting_place):
                                if child in self.allergic_children:
                                    for sandwich_type_fact in state:
                                        if match(sandwich_type_fact, "no_gluten_sandwich", sandwich):
                                            suitable_sandwich_found = True
                                            break
                                    if suitable_sandwich_found:
                                        break # Found gluten-free for allergic child
                                else: # Not allergic child, any sandwich is ok
                                    suitable_sandwich_found = True
                                    break # Found any sandwich for not allergic child
                        if suitable_sandwich_found:
                            break # Found suitable sandwich for this child, no need to check further

                if suitable_sandwich_found:
                    heuristic_value += 1  # Estimate 1 action to serve
                else:
                    heuristic_value += 4  # Estimate 4 actions: make, tray, move_tray, serve

        return heuristic_value
