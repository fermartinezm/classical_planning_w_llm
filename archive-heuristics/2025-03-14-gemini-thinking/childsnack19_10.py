from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-kitchen_bread bread1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with appropriate snacks.
    It considers the need to make sandwiches (gluten-free or regular), put them on trays, move trays to the children's locations, and finally serve them.

    # Assumptions:
    - Sufficient bread and content portions are available in the kitchen to make all required sandwiches.
    - Trays are initially in the kitchen and can be moved to any place where a child is waiting.
    - Making a sandwich, putting it on a tray, moving a tray, and serving a sandwich each count as one action.

    # Heuristic Initialization
    - Extracts static information about children's gluten allergies and their waiting places from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is currently 'waiting' and not yet 'served':
    1. Determine if the child is allergic to gluten.
    2. Check if a suitable sandwich (gluten-free if allergic, regular otherwise) is already prepared and on a tray at the child's waiting place. If yes, only a 'serve_sandwich' action is needed (cost 1).
    3. If not at the right place, check if a suitable sandwich is on any tray. If yes, a 'move_tray' and 'serve_sandwich' action are needed (cost 2).
    4. If not on a tray, check if a suitable sandwich is already made and in the kitchen. If yes, a 'put_on_tray' and 'serve_sandwich' action are needed (cost 2).
    5. If no suitable sandwich is made, then actions 'make_sandwich' (or 'make_sandwich_no_gluten'), 'put_on_tray', and 'serve_sandwich' are needed (cost 3).
    6. Sum up the minimum costs for all unserved children to get the total heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Gluten allergy information for each child.
        - Waiting place for each child.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergy_info = {}
        self.waiting_places = {}

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergy_info[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergy_info[child] = False
            elif match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                place = parts[2]
                self.waiting_places[child] = place

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
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
            if child in served_children:
                continue

            is_allergic = self.allergy_info.get(child, False)
            waiting_place = self.waiting_places.get(child)
            if not waiting_place:
                continue # Should not happen in valid problems

            suitable_sandwich_served = False
            # Check if suitable sandwich is already on tray at the correct place
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    for tray_location_fact in state:
                        if match(tray_location_fact, "at", "*", waiting_place):
                            tray_obj = get_parts(tray_location_fact)[1]
                            if tray_obj == tray:
                                is_gluten_free_sandwich = "(no_gluten_sandwich {})".format(sandwich) in state
                                if is_allergic == is_gluten_free_sandwich:
                                    heuristic_value += 1 # Serve action
                                    suitable_sandwich_served = True
                                    break
                    if suitable_sandwich_served:
                        break
                if suitable_sandwich_served:
                    break
            if suitable_sandwich_served:
                continue


            suitable_sandwich_on_tray = False
            # Check if suitable sandwich is on any tray (but maybe at wrong place)
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    is_gluten_free_sandwich = "(no_gluten_sandwich {})".format(sandwich) in state
                    if is_allergic == is_gluten_free_sandwich:
                        heuristic_value += 2 # Move tray + Serve
                        suitable_sandwich_on_tray = True
                        break
                if suitable_sandwich_on_tray:
                    break
            if suitable_sandwich_on_tray:
                continue


            suitable_sandwich_kitchen = False
            # Check if suitable sandwich is in kitchen
            for fact in state:
                if match(fact, "at_kitchen_sandwich", "*"):
                    sandwich = get_parts(fact)[1]
                    is_gluten_free_sandwich = "(no_gluten_sandwich {})".format(sandwich) in state
                    if is_allergic == is_gluten_free_sandwich:
                        heuristic_value += 2 # Put on tray + Serve
                        suitable_sandwich_kitchen = True
                        break
                if suitable_sandwich_kitchen:
                    break
            if suitable_sandwich_kitchen:
                continue

            # Need to make a sandwich
            heuristic_value += 3 # Make sandwich + Put on tray + Serve

        return heuristic_value
