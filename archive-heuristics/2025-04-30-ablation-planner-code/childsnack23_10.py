from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real environment, this would be provided by the planner framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Assuming task object has an 'objects' attribute listing all objects by type
            # e.g., task.objects = [('child1', 'child'), ('tray1', 'tray'), ...]
            self.objects = getattr(task, 'objects', []) # Use getattr for robustness

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Return empty list or handle appropriately if non-predicate facts are possible
         return []
    # Split by whitespace, ignoring empty strings from multiple spaces
    return fact_str[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts: # Handle non-predicate facts
        return False
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the final 'serve' action for each unserved child and adds the estimated
    actions needed to get a suitable sandwich on a tray to the child's location,
    including making the sandwich if necessary, putting it on a tray, and moving the tray.

    # Assumptions
    - All children targeted by the goal are initially in a 'waiting' state at some location.
    - Initial ingredients and 'notexist' sandwich objects are sufficient to make any required sandwiches.
    - Trays can be reused immediately after a sandwich is served from them.
    - The cost of moving a tray between any two places is 1 action.
    - The cost of making a sandwich, putting it on a tray, and serving is 1 action each.
    - A tray holds at most one sandwich at a time (inferred from predicate `ontray ?s ?t`).

    # Heuristic Initialization
    - Extracts static information about which children are allergic to gluten.
    - Identifies all children, sandwiches, trays, and places from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the heuristic value `h` to 0.
    2.  Identify all children that are not yet served based on the goal facts and the current state. Let this count be `N_unserved`.
    3.  Add `N_unserved` to `h`. This accounts for the final 'serve' action required for each unserved child.
    4.  Identify which of the unserved children *do not* currently have a suitable sandwich on a tray located at their waiting place. Let this set be `children_need_delivery` and its count `N_children_need_delivery`.
        - A sandwich is suitable for an allergic child only if it is gluten-free.
        - A sandwich is suitable for a non-allergic child if it is any sandwich.
        - A sandwich is considered 'delivered' if it is on a tray and that tray is at the child's waiting location.
    5.  Count the total number of sandwiches that currently exist (either in the kitchen or on any tray). These are sandwiches for which the `notexist` predicate is false. Let this be `N_sandwiches_available`.
    6.  Calculate the number of *new* sandwiches that need to be made to satisfy the children needing delivery, assuming existing available sandwiches are used first. This is `N_to_make = max(0, N_children_need_delivery - N_sandwiches_available)`.
    7.  Add `N_to_make` to `h`. This accounts for the 'make_sandwich' actions.
    8.  Each of the `N_children_need_delivery` requires a sandwich to be put on a tray. Add `N_children_need_delivery` to `h` (for 'put_on_tray' actions).
    9.  Each of the `N_children_need_delivery` requires the tray with the sandwich to be moved to their location. Add `N_children_need_delivery` to `h` (for 'move_tray' actions).
    10. The total heuristic value is the sum calculated in steps 3, 7, 8, and 9.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        # Extract static information about child allergies
        self.child_is_allergic = {}
        # Assuming static facts are strings like '(allergic_gluten child1)'
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.child_is_allergic[parts[1]] = True
            elif parts and parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                 self.child_is_allergic[parts[1]] = False # Store explicitly or assume False if not allergic_gluten

        # Identify all objects of relevant types from task.objects
        # task.objects is expected to be a list of (object_name, object_type) tuples
        self.all_children = {obj for obj, obj_type in self.objects if obj_type == 'child'}
        self.all_sandwiches = {obj for obj, obj_type in self.objects if obj_type == 'sandwich'}
        self.all_trays = {obj for obj, obj_type in self.objects if obj_type == 'tray'}
        self.all_places = {obj for obj, obj_type in self.objects if obj_type == 'place'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # 1. Identify unserved children
        # Goal facts are typically like '(served child1)'
        goal_served_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == 'served'}
        unserved_children = {c for c in goal_served_children if f'(served {c})' not in state}

        N_unserved = len(unserved_children)
        h += N_unserved # Cost for the final 'serve' action for each

        if N_unserved == 0:
            return 0 # Goal state

        # Get current state information efficiently
        state_facts = set(state) # Convert frozenset to set for faster lookups

        child_location = {}
        tray_location = {}
        sandwich_on_tray = {}
        sandwich_in_kitchen = set()
        sandwich_is_gf_state = set() # Sandwiches that currently have the no_gluten_sandwich predicate

        # Pre-process state facts into dictionaries/sets for quick access
        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                if child in self.all_children and place in self.all_places:
                    child_location[child] = place
            elif predicate == 'at' and len(parts) == 3 and parts[1] in self.all_trays:
                 tray, place = parts[1], parts[2]
                 if place in self.all_places:
                    tray_location[tray] = place
            elif predicate == 'ontray' and len(parts) == 3 and parts[1] in self.all_sandwiches and parts[2] in self.all_trays:
                 sandwich, tray = parts[1], parts[2]
                 sandwich_on_tray[sandwich] = tray
            elif predicate == 'at_kitchen_sandwich' and len(parts) == 2 and parts[1] in self.all_sandwiches:
                 sandwich = parts[1]
                 sandwich_in_kitchen.add(sandwich)
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2 and parts[1] in self.all_sandwiches:
                 sandwich = parts[1]
                 sandwich_is_gf_state.add(sandwich)

        # 2. Identify children needing delivery
        children_need_delivery = set()
        for c in unserved_children:
            p = child_location.get(c)
            if p is None:
                # A child in the goal should be in a waiting state initially.
                # If not found in state, something is unexpected, but assume they need delivery.
                # Or, if they are not waiting, they cannot be served by the current actions.
                # Let's assume valid states maintain the waiting predicate for unserved children.
                continue

            is_allergic = self.child_is_allergic.get(c, False)

            # Check if a suitable sandwich is already on a tray at location p
            suitable_sandwich_delivered = False
            for s in self.all_sandwiches:
                if s in sandwich_on_tray:
                    t = sandwich_on_tray[s]
                    if tray_location.get(t) == p:
                        # Sandwich s is on tray t, and tray t is at location p
                        is_gf_sandwich = s in sandwich_is_gf_state
                        if (is_allergic and is_gf_sandwich) or (not is_allergic):
                            # Found a suitable sandwich delivered to the location
                            suitable_sandwich_delivered = True
                            break # Found one for this child, move to next child

            if not suitable_sandwich_delivered:
                children_need_delivery.add(c)

        N_children_need_delivery = len(children_need_delivery)

        # 3. Count available sandwiches (that exist, regardless of location/tray)
        # Sandwiches that exist are those not marked as notexist
        N_sandwiches_available = len({s for s in self.all_sandwiches if f'(notexist {s})' not in state_facts})

        # 4. Calculate make actions needed
        # We need N_children_need_delivery sandwiches in total for these children.
        # We have N_sandwiches_available already made.
        # We need to make the difference.
        N_to_make = max(0, N_children_need_delivery - N_sandwiches_available)
        h += N_to_make # Cost for 'make_sandwich' actions

        # 5. Add put_on_tray and move_tray actions for needed deliveries
        # Each child needing delivery requires a sandwich to be put on a tray and the tray moved.
        h += N_children_need_delivery # Cost for 'put_on_tray' actions
        h += N_children_need_delivery # Cost for 'move_tray' actions

        return h
