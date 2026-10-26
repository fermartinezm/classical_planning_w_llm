# The code should be placed within a file structure that allows importing Heuristic
# e.g., in a file like `heuristics/childsnack_heuristic.py`

# from heuristics.heuristic_base import Heuristic # This line is commented out for standalone testing but required in the planner env

# Dummy Heuristic base class for standalone testing if needed
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

from fnmatch import fnmatch
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    parts = fact[1:-1].split()
    return parts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    fact_parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(fact_parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children.
    It sums up the estimated cost for each unserved child based on the current
    state of the required sandwich and tray relative to the child's location.

    # Assumptions
    - Each unserved child requires a suitable sandwich (regular or gluten-free)
      to be delivered on a tray to their waiting place.
    - The heuristic estimates the steps needed for each child independently,
      potentially overcounting shared actions like moving a tray that serves
      multiple children or making a sandwich needed by multiple children.
    - The estimated steps for a single child, from needing a sandwich made
      to being served, are: make (1) + put_on_tray (1) + move_tray (1) + serve (1) = 4.
      The heuristic checks which of these steps are already completed for the
      required sandwich/tray for that child.
    - Assumes sufficient bread, content, and notexist sandwich objects are
      available when needed for making sandwiches, and sufficient trays are
      available at the kitchen when needed for putting sandwiches on trays.

    # Heuristic Initialization
    - Extracts static information about children: their allergy type (allergic_gluten or not_allergic_gluten)
      and their waiting place (waiting ?c ?p).
    - Identifies all children in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Identify all unserved children by checking which children do NOT have the `(served ?c)` predicate in the current state.
    3. For each unserved child:
        a. Determine the child's waiting place `p` and required sandwich allergy type (`regular` or `gluten`).
        b. Check the state to see if a suitable sandwich (matching the allergy type) is available and where it is:
           - Is there *any* suitable sandwich `s` that is `ontray t` AND the tray `t` is `at p`?
             - If yes, this child needs only the `serve` action. Add 1 to the total cost.
           - Else (no suitable sandwich is on a tray at the child's location):
             - Add 1 to the total cost (representing the `move_tray` action needed).
             - Is there *any* suitable sandwich `s` that is `ontray t` anywhere (regardless of tray location)?
               - If yes, this child needs the `move_tray` and `serve` actions. (We already added 1 for move, add 1 for serve). Total cost for this child stage = 2.
             - Else (no suitable sandwich is on a tray anywhere):
               - Add 1 to the total cost (representing the `put_on_tray` action needed).
               - Is there *any* suitable sandwich `s` that is `at_kitchen_sandwich`?
                 - If yes, this child needs `put_on_tray`, `move_tray`, and `serve` actions. (We already added 1 for move and 1 for put, add 1 for serve). Total cost for this child stage = 3.
               - Else (no suitable sandwich exists yet, neither on tray nor in kitchen):
                 - Add 1 to the total cost (representing the `make_sandwich` action needed).
                 - This child needs `make_sandwich`, `put_on_tray`, `move_tray`, and `serve` actions. (We already added 1 for move, 1 for put, 1 for make, add 1 for serve). Total cost for this child stage = 4.
    4. The total heuristic value is the sum of the costs estimated for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        super().__init__(task)

        self.all_children = set()
        self.child_allergy = {} # Map child object to 'gluten' or 'regular'
        self.child_place = {}   # Map child object to place object

        # Extract static facts about children from the initial state
        # Allergy and waiting place are static throughout the problem.
        initial_state_facts = task.initial_state

        for fact in initial_state_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'allergic_gluten' and len(parts) == 2:
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = 'gluten'
            elif predicate == 'not_allergic_gluten' and len(parts) == 2:
                child = parts[1]
                self.all_children.add(child)
                self.child_allergy[child] = 'regular'
            elif predicate == 'waiting' and len(parts) == 3:
                child, place = parts[1:]
                self.all_children.add(child)
                self.child_place[child] = place

        # Basic validation: Ensure we found info for all children mentioned in allergy/waiting facts
        # In a real scenario, you might get children objects from the task definition itself
        # if they are listed in the :objects section, but parsing facts is also common.
        # For this problem, assuming children are defined by allergy/waiting facts is sufficient.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all unserved children.
        """
        state = node.state
        total_cost = 0

        # Identify available sandwiches and their current state/location
        # We only care about sandwiches that have been made (not notexist)
        available_made_sandwiches_by_type = defaultdict(list) # {allergy_type: [(sandwich_obj, state_type, location/tray_obj)]}
        tray_location = {} # {tray_obj: place_obj}

        # Populate tray locations
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    t, p = parts[1:]
                    if t.startswith('tray'): # Ensure it's a tray object
                         tray_location[t] = p

        # Populate available made sandwiches and their state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                s = parts[1]
                # Determine allergy type - need to check for no_gluten_sandwich predicate
                allergy = 'gluten' if '(no_gluten_sandwich ' + s + ')' in state else 'regular'
                available_made_sandwiches_by_type[allergy].append((s, 'kitchen', None))
            elif predicate == "ontray" and len(parts) == 3:
                s, t = parts[1:]
                 # Determine allergy type
                allergy = 'gluten' if '(no_gluten_sandwich ' + s + ')' in state else 'regular'
                available_made_sandwiches_by_type[allergy].append((s, 'ontray', t))

        # Compute cost for each unserved child
        for child in self.all_children:
            # Check if child is served
            if '(served ' + child + ')' in state:
                continue # This child is already served, cost is 0

            # Child is unserved. Determine their need.
            child_place = self.child_place.get(child)
            child_allergy = self.child_allergy.get(child)

            if child_place is None or child_allergy is None:
                 # Should not happen in valid problems, but handle defensively
                 # print(f"Warning: Missing place or allergy for child {child}")
                 continue # Cannot estimate cost for this child

            needed_allergy_type = child_allergy
            target_place = child_place

            # Estimate steps needed for this child based on sandwich/tray state
            # This is an additive heuristic summing per-child estimates.

            # Start with the cost of the final 'serve' action
            cost_for_child = 1

            # Check if the precondition for serving is met by any available sandwich
            # Precondition: suitable sandwich on tray at child's place
            can_serve_now = False
            # Ensure the needed allergy type exists in the available sandwiches dictionary
            if needed_allergy_type in available_made_sandwiches_by_type:
                for s_info in available_made_sandwiches_by_type[needed_allergy_type]:
                    s_obj, s_state, tray_obj = s_info
                    if s_state == 'ontray' and tray_obj in tray_location and tray_location[tray_obj] == target_place:
                        can_serve_now = True
                        break # Found a suitable sandwich on a tray at the correct place

            if can_serve_now:
                # Child needs only the serve action (cost 1 already added)
                pass
            else:
                # Precondition for serving is NOT met. Need to get sandwich/tray to place.
                # Add cost for the 'move_tray' action
                cost_for_child += 1

                # Check if the precondition for moving is met by any available sandwich
                # Precondition: suitable sandwich on tray anywhere
                is_ontray_anywhere = False
                if needed_allergy_type in available_made_sandwiches_by_type:
                    for s_info in available_made_sandwiches_by_type[needed_allergy_type]:
                        s_obj, s_state, tray_obj = s_info
                        if s_state == 'ontray': # On any tray, anywhere
                            is_ontray_anywhere = True
                            break # Found a suitable sandwich on a tray

                if is_ontray_anywhere:
                    # Child needs move_tray and serve actions (cost 1+1=2 already added)
                    pass
                else:
                    # Precondition for moving is NOT met. Need to put sandwich on tray.
                    # Add cost for the 'put_on_tray' action
                    cost_for_child += 1

                    # Check if the precondition for putting is met by any available sandwich
                    # Precondition: suitable sandwich in kitchen
                    is_in_kitchen = False
                    if needed_allergy_type in available_made_sandwiches_by_type:
                        for s_info in available_made_sandwiches_by_type[needed_allergy_type]:
                            s_obj, s_state, tray_obj = s_info
                            if s_state == 'kitchen': # In the kitchen
                                is_in_kitchen = True
                                break # Found a suitable sandwich in the kitchen

                    if is_in_kitchen:
                        # Child needs put_on_tray, move_tray, and serve actions (cost 1+1+1=3 already added)
                        pass
                    else:
                        # Precondition for putting is NOT met. Need to make sandwich.
                        # Add cost for the 'make_sandwich' action
                        cost_for_child += 1
                        # Child needs make_sandwich, put_on_tray, move_tray, and serve actions (cost 1+1+1+1=4 already added)
                        pass

            total_cost += cost_for_child

        return total_cost
