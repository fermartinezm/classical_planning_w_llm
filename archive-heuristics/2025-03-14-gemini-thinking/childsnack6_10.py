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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with snacks.
    It considers the necessary steps: making a sandwich, putting it on a tray, moving the tray to the child's location, and serving the sandwich.

    # Assumptions
    - For each unserved child, we need to perform a 'serve' action.
    - To serve a child, a suitable sandwich must be on a tray at the child's location. If not, 'put_on_tray' and 'move_tray' actions are needed.
    - A suitable sandwich must exist (at the kitchen). If not, a 'make_sandwich' action is needed.
    - We assume that bread and content are always available at the kitchen to make sandwiches if needed.
    - We prioritize serving each child individually and sum up the estimated costs.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1. Initialize the heuristic value to 0.
    2. Identify all children who are currently 'waiting' at some place.
    3. For each waiting child:
        a. Check if the child is already 'served'. If yes, no further actions are needed for this child.
        b. If the child is not 'served', we need to estimate the actions to serve them:
            i. Increment the heuristic value by 1 (for the 'serve_sandwich' action).
            ii. Determine the child's allergy type (allergic_gluten or not_allergic_gluten) and the place they are waiting at.
            iii. Check if there is a suitable sandwich on a tray at the child's waiting place:
                - Iterate through the state to find facts matching '(ontray ?s ?t)' and '(at ?t ?p)' where ?p is the child's waiting place.
                - Check if the sandwich '?s' is suitable for the child's allergy (no_gluten_sandwich for allergic children, any sandwich for not allergic).
                - If no suitable sandwich is found on a tray at the child's location, increment the heuristic value by 2 (for 'put_on_tray' and 'move_tray' actions).
            iv. Check if a suitable sandwich exists at the kitchen:
                - Iterate through the state to find facts matching '(at_kitchen_sandwich ?s)'.
                - Check if the sandwich '?s' is suitable for the child's allergy.
                - If no suitable sandwich is found at the kitchen, increment the heuristic value by 1 (for 'make_sandwich' action).
    4. Return the accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the childsnack heuristic."""
        super().__init__(task)

    def __call__(self, node):
        """Estimate the number of actions to serve all waiting children."""
        state = node.state
        heuristic_value = 0

        waiting_children = {}
        allergy_info = {}
        for fact in self.task.static:
            if match(fact, 'waiting', '*', '*'):
                parts = get_parts(fact)
                child = parts[1]
                place = parts[2]
                waiting_children[child] = place
            if match(fact, 'allergic_gluten', '*'):
                allergy_info[get_parts(fact)[1]] = 'allergic_gluten'
            elif match(fact, 'not_allergic_gluten', '*'):
                allergy_info[get_parts(fact)[1]] = 'not_allergic_gluten'

        for child in waiting_children:
            if f'(served {child})' not in state:
                heuristic_value += 1 # for serve action

                child_place = waiting_children[child]
                child_allergy = allergy_info.get(child, 'not_allergic_gluten')
                sandwich_on_tray_at_place = False

                for fact in state:
                    if match(fact, 'ontray', '*', '*'):
                        parts = get_parts(fact)
                        sandwich = parts[1]
                        tray = parts[2]
                        for fact2 in state:
                            if match(fact2, 'at', '*', child_place):
                                parts2 = get_parts(fact2)
                                tray2 = parts2[1]
                                if tray == tray2:
                                    suitable_sandwich = False
                                    if child_allergy == 'allergic_gluten':
                                        if f'(no_gluten_sandwich {sandwich})' in state:
                                            suitable_sandwich = True
                                    else:
                                        suitable_sandwich = True
                                    if suitable_sandwich:
                                        sandwich_on_tray_at_place = True
                                        break
                        if sandwich_on_tray_at_place:
                            break
                    if sandwich_on_tray_at_place:
                        break

                if not sandwich_on_tray_at_place:
                    heuristic_value += 2 # for put_on_tray and move_tray

                suitable_sandwich_at_kitchen = False
                for fact in state:
                    if match(fact, 'at_kitchen_sandwich', '*'):
                        sandwich = get_parts(fact)[1]
                        suitable_sandwich = False
                        if child_allergy == 'allergic_gluten':
                            if f'(no_gluten_sandwich {sandwich})' in state:
                                suitable_sandwich = True
                        else:
                            suitable_sandwich = True
                        if suitable_sandwich:
                            suitable_sandwich_at_kitchen = True
                            break
                if not suitable_sandwich_at_kitchen:
                    heuristic_value += 1 # for make_sandwich

        return heuristic_value
