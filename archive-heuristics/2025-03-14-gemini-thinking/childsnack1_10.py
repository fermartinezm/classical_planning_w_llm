from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    It considers the actions of making sandwiches, putting them on trays, moving trays, and serving sandwiches,
    taking into account gluten allergies.

    # Assumptions:
    - Bread and content portions are always available at the kitchen to make sandwiches if needed.
    - Trays are initially at the kitchen and can be moved to any place.
    - The heuristic focuses on serving each waiting child individually and sums up the estimated costs.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is currently waiting and not yet served:
    1. Check if the child is already served. If yes, no further actions are needed for this child.
    2. Determine if the child is allergic to gluten.
    3. Check if a suitable sandwich (gluten-free for allergic children, regular otherwise) is already on a tray at the child's waiting place.
       If yes, only a 'serve_sandwich' action is needed.
    4. If not, check if a suitable sandwich is at the kitchen.
       If yes, actions 'put_on_tray' and 'serve_sandwich' are needed (and potentially 'move_tray' if the tray is not at the serving place, but we assume tray movement is always needed from kitchen to serving place).
    5. If no suitable sandwich is at the kitchen, assume we need to make a sandwich first.
       Then actions 'make_sandwich', 'put_on_tray', and 'serve_sandwich' are needed (and potentially 'move_tray').
    6. Sum up the estimated number of actions for each unserved child to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the childsnack heuristic."""
        super().__init__(task)

    def __call__(self, node):
        """Estimate the number of actions to serve all waiting children."""
        state = node.state
        goal_state = self.task.goal_reached(state)
        if goal_state:
            return 0

        heuristic_value = 0
        waiting_children = []
        child_allergies = {}
        child_places = {}

        for fact in self.task.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                child_allergies[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                child_allergies[child] = False
            elif match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                child_places[child] = place

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                if not any(match(goal_fact, "served", child) for goal_fact in state): # Check if child is served in current state
                    waiting_children.append(child)

        for child in waiting_children:
            if any(match(fact, "served", child) for fact in state):
                continue # Child already served in current state

            heuristic_value += 1 # For serve action

            place = child_places[child]
            is_allergic = child_allergies[child]
            sandwich_type = "no_gluten_sandwich" if is_allergic else "at_kitchen_sandwich"
            sandwich_type_check = "no_gluten_sandwich" if is_allergic else "at_kitchen_sandwich" # for checking ontray

            sandwich_on_tray_at_place = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    for fact2 in state:
                        if match(fact2, "at", tray, place):
                            sandwich_gluten_fact = f'({sandwich_type_check} {sandwich})'
                            if sandwich_type_check == "no_gluten_sandwich":
                                if sandwich_gluten_fact in state:
                                    sandwich_on_tray_at_place = True
                                    break
                            elif sandwich_type_check == "at_kitchen_sandwich": # Regular sandwich check is just existence of sandwich on tray, no specific predicate for regular sandwich on tray.
                                sandwich_on_tray_at_place = True
                                break
                    if sandwich_on_tray_at_place:
                        break
                if sandwich_on_tray_at_place:
                    break

            if sandwich_on_tray_at_place:
                continue # Sandwich is on tray at the correct place, only serve is needed

            heuristic_value += 1 # For put_on_tray action

            sandwich_at_kitchen = False
            for fact in state:
                if match(fact, "at_kitchen_sandwich", "*"):
                    sandwich = get_parts(fact)[1]
                    sandwich_gluten_fact = f'({sandwich_type} {sandwich})'
                    if sandwich_type == "no_gluten_sandwich":
                        if sandwich_gluten_fact in state:
                            sandwich_at_kitchen = True
                            break
                    elif sandwich_type == "at_kitchen_sandwich": # Regular sandwich check is just existence of sandwich at kitchen, no specific predicate for regular sandwich at kitchen.
                        sandwich_at_kitchen = True
                        break
                if sandwich_at_kitchen:
                    break

            if sandwich_at_kitchen:
                heuristic_value += 1 # For move_tray action (kitchen to place)
                continue # Sandwich is at kitchen, put on tray and serve are needed

            heuristic_value += 1 # For make_sandwich action
            heuristic_value += 1 # For move_tray action (kitchen to place)


        return heuristic_value
