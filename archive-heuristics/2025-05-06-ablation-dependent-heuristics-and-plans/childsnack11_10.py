# Ensure the parent directory 'heuristics' is in the Python path if running directly.
# In a typical planning framework, this would be handled by the runner script.
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    waiting children. It counts the number of unserved children, the number
    of sandwiches that need to be made and put on trays, and the number of
    tray movements required to bring trays to children's locations and the kitchen.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic
      children can accept any sandwich.
    - Available gluten-free sandwiches are prioritized for allergic children.
      Any surplus gluten-free sandwiches can satisfy non-allergic children.
    - Making a sandwich requires available bread, content, and a sandwich object
      (these resource limits are not strictly modeled beyond counting needed makes).
    - Putting a sandwich on a tray requires the sandwich to be in the kitchen
      and a tray to be in the kitchen.
    - Serving a sandwich requires the sandwich to be on a tray and the tray
      to be at the child's location.
    - The place constant 'kitchen' is the kitchen location.
    - Objects starting with 'tray' are tray objects.

    # Heuristic Initialization
    - Identify all children that are goals (need to be served).
    - Store static facts, particularly allergy information for children.
    - Identify all tray objects and the kitchen location.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify Unserved Children: Count the number of children in the goal
        who are not yet marked as 'served'. Categorize them into allergic
        and non-allergic groups based on static facts.
    2.  Count Available Sandwiches: Count sandwiches currently 'ontray' or
        'at_kitchen_sandwich', distinguishing between gluten-free and regular
        sandwiches based on the 'no_gluten_sandwich' predicate.
    3.  Calculate Needed Sandwiches (Ontray Stage): Determine how many
        gluten-free and regular sandwiches still need to reach the 'ontray'
        stage to satisfy the unserved children, prioritizing GF for allergic.
        Sum these to get the total number of sandwiches needing the 'put_on_tray' action.
    4.  Calculate Needed Sandwiches (Make Stage): Determine how many of the
        sandwiches needing the 'ontray' stage are not currently 'at_kitchen_sandwich'.
        These must be made. Sum these to get the total number of sandwiches
        needing the 'make_sandwich' action.
    5.  Calculate Tray Movement Costs:
        - Identify all places where unserved children are waiting.
        - Identify all places where trays are currently located.
        - Count places with waiting children that do not have a tray. Each such
          place needs a tray moved there (cost +1 per place).
        - If any sandwiches need to be put on a tray (calculated in step 3)
          and no tray is currently in the kitchen, a tray needs to be moved
          to the kitchen (cost +1).
    6.  Sum Costs: The total heuristic is the sum of:
        - The number of unserved children (cost for 'serve' actions).
        - The total number of sandwiches needing the 'make_sandwich' action.
        - The total number of sandwiches needing the 'put_on_tray' action.
        - The total cost for tray movements.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, static facts,
        tray objects, and the kitchen location.
        """
        self.goals = task.goals
        self.static = task.static
        self.kitchen = 'kitchen' # Assuming 'kitchen' is the constant name for the kitchen place

        # Identify all children that are goals (need to be served)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == 'served'}

        # Identify allergic children from static facts
        self.allergic_children = {get_parts(fact)[1] for fact in self.static if match(fact, 'allergic_gluten', '*')}

        # Identify tray objects - assuming objects starting with 'tray' are trays
        # A more robust way would parse the :objects section of the PDDL,
        # but this is not available in the current Task object structure.
        self.tray_objects = set()
        # Look for objects used as the first argument of 'at' predicate in initial state or goals
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) > 1 and parts[1].startswith('tray'):
                 self.tray_objects.add(parts[1])
        # Fallback: if no trays found this way, try parsing objects from initial state directly
        # This is still brittle as object types aren't available.
        if not self.tray_objects:
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts and len(parts) > 1 and parts[1].startswith('tray'):
                      self.tray_objects.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = {c for c in self.goal_children if '(served ' + c + ')' not in state}
        N_unserved = len(unserved_children)

        # If all children are served, the goal is reached
        if N_unserved == 0:
            return 0

        # Categorize unserved children by allergy
        unserved_gluten = {c for c in unserved_children if '(allergic_gluten ' + c + ')' in self.static}
        unserved_regular = unserved_children - unserved_gluten
        N_unserved_gluten = len(unserved_gluten)
        N_unserved_regular = len(unserved_regular)

        # 2. Count Available Sandwiches by type and location
        ontray_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'ontray', '*', '*')}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_sandwich', '*')}

        is_gluten_free = lambda s: '(no_gluten_sandwich ' + s + ')' in state

        Avail_ontray_gluten = len({s for s in ontray_sandwiches if is_gluten_free(s)})
        Avail_ontray_regular = len(ontray_sandwiches) - Avail_ontray_gluten
        # Avail_ontray_total = len(ontray_sandwiches) # Not strictly needed for calculation

        Avail_kitchen_gluten = len({s for s in kitchen_sandwiches if is_gluten_free(s)})
        Avail_kitchen_regular = len(kitchen_sandwiches) - Avail_kitchen_gluten
        # Avail_kitchen_total = len(kitchen_sandwiches) # Not strictly needed for calculation

        # 3. Calculate Needed Sandwiches (Ontray Stage)
        # Needed GF sandwiches that must reach ontray
        Needed_ontray_GF = max(0, N_unserved_gluten - Avail_ontray_gluten)
        # Surplus GF sandwiches ontray that can serve regular children
        Surplus_ontray_GF = max(0, Avail_ontray_gluten - N_unserved_gluten)
        # Needed Regular sandwiches that must reach ontray (cannot use available regular or surplus ontray GF)
        Needed_ontray_Regular = max(0, N_unserved_regular - Avail_ontray_regular - Surplus_ontray_GF)
        Needed_ontray_total = Needed_ontray_GF + Needed_ontray_Regular

        # Cost for put_on_tray actions: Each sandwich needing to reach ontray needs one put_on_tray action
        Cost_put = Needed_ontray_total

        # 4. Calculate Needed Sandwiches (Make Stage)
        # Needed GF sandwiches that must be made (not yet ontray or in kitchen)
        Needed_make_GF = max(0, Needed_ontray_GF - Avail_kitchen_gluten)
        # Surplus GF sandwiches in kitchen that can serve regular children
        Surplus_kitchen_GF = max(0, Avail_kitchen_gluten - Needed_ontray_GF)
        # Needed Regular sandwiches that must be made (cannot use available regular or surplus kitchen GF)
        Needed_make_Regular = max(0, Needed_ontray_Regular - Avail_kitchen_regular - Surplus_kitchen_GF)
        Needed_make_total = Needed_make_GF + Needed_make_Regular

        # Cost for make_sandwich actions
        Cost_make = Needed_make_total

        # 5. Calculate Tray Movement Costs
        # Find places where unserved children are waiting
        places_with_waiting_children = {get_parts(fact)[2] for fact in state if match(fact, 'waiting', '*', '*') and get_parts(fact)[1] in unserved_children}

        # Find places where trays are located
        places_with_trays = {get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*') and get_parts(fact)[1] in self.tray_objects}

        # Cost to move trays to places with waiting children that don't have a tray
        N_places_need_tray = len(places_with_waiting_children - places_with_trays)
        Cost_move_to_place = N_places_need_tray

        # Cost to move a tray to the kitchen if needed for put_on_tray and no tray is there
        Cost_move_to_kitchen = 0
        if Needed_ontray_total > 0 and self.kitchen not in places_with_trays:
             Cost_move_to_kitchen = 1

        # 6. Sum Costs
        # Cost for serve actions: Each unserved child needs one serve action
        Cost_serve = N_unserved

        total_cost = Cost_serve + Cost_make + Cost_put + Cost_move_to_place + Cost_move_to_kitchen

        return total_cost
