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
    This heuristic estimates the number of actions required to serve all
    children who are currently waiting but not yet served. It counts the
    estimated number of 'make', 'put_on_tray', 'move_tray', and 'serve'
    actions needed based on the current state and the children's requirements.

    # Heuristic Components:
    1.  Number of unserved children: Each needs a 'serve' action.
    2.  Number of distinct locations where unserved children are waiting
        but no tray is currently present: Each needs at least one 'move_tray'
        action to bring a tray there.
    3.  Number of sandwiches needed that are not yet on a tray: These need
        a 'put_on_tray' action.
    4.  Number of sandwiches needed that do not yet exist: These need a
        'make_sandwich' action. This count considers both the total number
        of sandwiches needed and the specific requirement for gluten-free
        sandwiches for allergic children.

    # Assumptions:
    - Each unserved child requires one sandwich.
    - A tray can hold multiple sandwiches (implicit in counting needed sandwiches
      separately from trays).
    - Ingredients for making sandwiches are sufficient as long as 'notexist'
      sandwich objects are available (relaxed).
    - Allergy and waiting status of children are static (based on domain definition).

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all children who are in the goal state (i.e., need to be served)
       but are not currently marked as 'served' in the state. Count them
       (`N_unserved`). If 0, return 0.
    2. For each unserved child, determine their waiting location and allergy status
       using the static facts.
    3. Count the number of distinct waiting locations that currently do not have
       any tray present (`N_locations_needing_tray`).
    4. Count the total number of sandwiches currently on trays (`N_ontray_any`).
    5. Count the total number of sandwiches currently in the kitchen
       (`N_kitchen_any`).
    6. Count the number of gluten-free sandwiches currently on trays
       (`N_ontray_gf`).
    7. Count the number of gluten-free sandwiches currently in the kitchen
       (`N_kitchen_gf`).
    8. Calculate the number of sandwiches that need to be put on a tray: This is
       the total number of sandwiches needed (`N_unserved`) minus those already
       on trays (`N_ontray_any`). Take `max(0, ...)` (`N_put_needed`).
    9. Calculate the number of sandwiches that need to be made:
       a. Total sandwiches needed is `N_unserved`. Total available (ontray or kitchen)
          is `N_ontray_any + N_kitchen_any`. Total needing to be made (any type)
          is `max(0, N_unserved - (N_ontray_any + N_kitchen_any))`.
       b. Gluten-free sandwiches needed is the count of unserved allergic children
          (`unserved_gf_needed`). Available GF (ontray or kitchen) is
          `N_ontray_gf + N_kitchen_gf`. GF needing to be made is
          `max(0, unserved_gf_needed - (N_ontray_gf + N_kitchen_gf))`.
       c. The total number of 'make' actions is the maximum of the total deficit
          and the GF deficit (`N_make_actions`).
    10. The heuristic value is the sum of `N_unserved + N_locations_needing_tray + N_put_needed + N_make_actions`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by pre-processing static information about
        children's waiting locations and allergy statuses.
        """
        super().__init__(task)
        # Pre-process children info from static facts
        self._waiting_info = {}
        self._allergy_info = {}
        # The domain defines waiting and allergy as static predicates.
        for fact in task.static:
             parts = get_parts(fact)
             if parts[0] == 'waiting' and len(parts) == 3:
                 self._waiting_info[parts[1]] = parts[2]
             elif parts[0] == 'allergic_gluten' and len(parts) == 2:
                 self._allergy_info[parts[1]] = 'allergic'
             elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                 self._allergy_info[parts[1]] = 'not_allergic'

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        goals = self.goals

        unserved_children = []
        waiting_locations_of_unserved = set()
        unserved_gf_needed = 0
        unserved_any_needed = 0

        # 1. Identify unserved children and their needs/locations
        for goal in goals:
            parts = get_parts(goal)
            if parts[0] == 'served' and len(parts) == 2:
                child = parts[1]
                if goal not in state:
                    unserved_children.append(child)
                    # Get waiting location from pre-processed info
                    location = self._waiting_info.get(child)
                    if location: # Child in goal should be a waiting child based on domain structure
                        waiting_locations_of_unserved.add(location)
                        # Get allergy info from pre-processed info
                        if self._allergy_info.get(child) == 'allergic':
                            unserved_gf_needed += 1
                        else:
                            unserved_any_needed += 1
                    # else: This case indicates a problem definition inconsistency
                    # where a child is in the goal but not marked as waiting.
                    # Assuming valid problems where goal children are waiting.


        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached

        # 2. Count locations needing trays
        N_locations_needing_tray = 0
        for loc in waiting_locations_of_unserved:
            tray_at_loc = False
            for fact in state:
                parts = get_parts(fact)
                # Check for (at ?t ?p) where ?p is the location
                if parts[0] == 'at' and len(parts) == 3 and parts[2] == loc:
                     # Assuming any object at a place is a tray in this context
                     # Could add type check if necessary, but domain structure implies this
                     tray_at_loc = True
                     break
            if not tray_at_loc:
                N_locations_needing_tray += 1

        # 3. Count available sandwiches and GF status
        N_ontray_any = 0
        N_kitchen_any = 0
        sandwiches_ontray = set()
        sandwiches_kitchen = set()
        gf_sandwiches_in_state = set() # All sandwiches marked as GF in the current state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray' and len(parts) == 3:
                sandwich = parts[1]
                N_ontray_any += 1
                sandwiches_ontray.add(sandwich)
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                sandwich = parts[1]
                N_kitchen_any += 1
                sandwiches_kitchen.add(sandwich)
            elif parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                sandwich = parts[1]
                gf_sandwiches_in_state.add(sandwich)

        # Count GF sandwiches among those ontray or at kitchen
        N_ontray_gf = sum(1 for s in sandwiches_ontray if s in gf_sandwiches_in_state)
        N_kitchen_gf = sum(1 for s in sandwiches_kitchen if s in gf_sandwiches_in_state)


        # 4. Calculate deficits and heuristic components
        N_needed_total = N_unserved # Each unserved child needs one sandwich

        # Sandwiches needing put_on_tray: Those needed that aren't already on a tray
        # This counts the number of sandwiches that must transition to 'ontray' state.
        N_put_needed = max(0, N_needed_total - N_ontray_any)

        # Sandwiches needing make: Those needed that aren't ontray or at kitchen
        # This counts the number of sandwich objects that must transition from 'notexist'.
        N_available_total = N_ontray_any + N_kitchen_any
        N_make_any_needed = max(0, N_needed_total - N_available_total)

        # GF sandwiches needing make: Those GF needed that aren't available GF ontray or kitchen
        N_available_gf = N_ontray_gf + N_kitchen_gf
        N_make_gf_needed = max(0, unserved_gf_needed - N_available_gf)

        # The number of 'make' actions is the number of sandwich objects that need to be created.
        # This is the total deficit, but must be at least the number of GF sandwiches that *must* be made.
        N_make_actions = max(N_make_any_needed, N_make_gf_needed)


        # Sum the components
        heuristic_value = (
            N_unserved # Cost for serve actions (one per child)
            + N_locations_needing_tray # Cost for move_tray actions (one per location needing a tray)
            + N_put_needed # Cost for put_on_tray actions (one per sandwich needing to get on a tray)
            + N_make_actions # Cost for make actions (one per sandwich needing to be created)
        )

        return heuristic_value

