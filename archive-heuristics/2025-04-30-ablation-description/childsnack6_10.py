from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # print(f"Warning: Unexpected fact format: {fact}")
        return [] # Return empty list for invalid facts

    # Remove outer parentheses and split by spaces
    parts = fact[1:-1].split()
    return parts

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    waiting children. It calculates the cost for each unserved child independently
    based on the state of the most readily available suitable sandwich, and sums
    these individual costs. This is an additive heuristic over unserved children.

    # Assumptions
    - Each child requires exactly one sandwich to be served.
    - Ingredient availability in the kitchen is checked to determine if a new
      suitable sandwich *can* be made, but the heuristic does not track the
      depletion of specific ingredients or bread/content portions.
    - Tray availability is not explicitly tracked; the heuristic assumes a tray
      can be used or moved when needed for a sandwich.
    - The cost of moving a tray between any two places is 1 action.
    - The set of all possible sandwich objects is inferred from the 'notexist'
      facts in the initial state.

    # Heuristic Initialization
    The heuristic is initialized by extracting static information from the task:
    - The set of children who need to be served (from goal conditions).
    - The waiting place for each child.
    - The allergy status (allergic_gluten) for each child.
    - The types of bread and content portions that are gluten-free.
    - The set of all possible sandwich objects defined in the problem (inferred
      from initial 'notexist' facts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each child who has not
    yet been served. For each unserved child:

    1.  **Base Cost:** Start with a cost of 1, representing the final 'serve' action.
    2.  **Determine Requirements:** Identify the child's waiting place and whether
        they require a gluten-free sandwich (if allergic).
    3.  **Check Ingredient Availability:** Determine if gluten-free and non-gluten-free
        ingredients are currently available in the kitchen.
    4.  **Check if New Sandwich Can Be Made:** Determine if there is at least one
        unused sandwich object ('notexist') and if suitable ingredients (GF if needed)
        are available in the kitchen. If yes, a new suitable sandwich can be made,
        which would cost 3 actions before serving (make + put_on_tray + move_tray).
        Initialize `min_steps_before_serve` to 3 if a new sandwich can be made,
        otherwise initialize to infinity.
    5.  **Evaluate Existing Sandwiches:** Iterate through all possible sandwich objects.
        For each sandwich 's':
        a.  Check if 's' is suitable for the child (meets GF requirement if needed,
            based on `no_gluten_sandwich` fact in the current state).
        b.  If suitable, determine its current state and the minimum actions needed
            to get it ready for serving at the child's location:
            -   If 's' is `ontray` on tray 't' and 't' is `at` the child's place: 0 steps.
            -   If 's' is `at_kitchen_sandwich`: 2 steps (put_on_tray + move_tray).
            -   If 's' is `ontray` on tray 't' and 't' is `at` the kitchen or any
                other place (not the child's place): 1 step (move_tray).
            -   If 's' is `notexist`: This case's cost (3) is covered by step 4.
            -   If 's' is in any other state (e.g., ontray but tray location unknown/invalid),
                it's not considered a readily available option by this heuristic.
        c.  Update `min_steps_before_serve` with the minimum steps found among
            all suitable existing sandwiches.
    6.  **Handle Unsolvable Case:** After checking all sandwiches (existing and new),
        if `min_steps_before_serve` is still infinity, it means no suitable sandwich
        can be obtained or made for this child. Return infinity for the total heuristic.
    7.  **Add to Total:** Add `min_steps_before_serve` to the child's base cost (1),
        and add this total child cost to the overall heuristic `h`.
    8.  **Return:** After processing all unserved children, return the total heuristic `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find all sandwich objects

        # Extract goal children
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served'}

        # Extract static child information
        self.child_waiting_place = {}
        self.child_is_allergic = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'waiting':
                self.child_waiting_place[parts[1]] = parts[2]
            elif parts and parts[0] == 'allergic_gluten':
                self.child_is_allergic.add(parts[1])

        # Extract static ingredient information
        self.no_gluten_bread_types = {get_parts(f)[1] for f in self.static_facts if get_parts(f)[0] == 'no_gluten_bread'}
        self.no_gluten_content_types = {get_parts(f)[1] for f in self.static_facts if get_parts(f)[0] == 'no_gluten_content'}

        # Infer all possible sandwich objects from the initial state's notexist facts
        # Assuming all sandwich objects are listed with notexist in the initial state
        self.all_sandwich_objects = {get_parts(f)[1] for f in self.initial_state if get_parts(f)[0] == 'notexist'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Pre-calculate suitability (GF status) of existing sandwiches in the current state
        sandwich_is_gf = {} # Map sandwich object to True if GF, False otherwise (for suitable check)
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'no_gluten_sandwich':
                sandwich_is_gf[parts[1]] = True

        # Check if ingredients are available in the kitchen in the current state
        current_breads_at_kitchen = {get_parts(f)[1] for f in state if get_parts(f)[0] == 'at_kitchen_bread'}
        current_contents_at_kitchen = {get_parts(f)[1] for f in state if get_parts(f)[0] == 'at_kitchen_content'}

        has_any_ingredients_in_kitchen = bool(current_breads_at_kitchen and current_contents_at_kitchen)
        has_gf_ingredients_in_kitchen = False
        if has_any_ingredients_in_kitchen:
            has_gf_bread_at_kitchen = any(b in self.no_gluten_bread_types for b in current_breads_at_kitchen)
            has_gf_content_at_kitchen = any(c in self.no_gluten_content_types for c in current_contents_at_kitchen)
            if has_gf_bread_at_kitchen and has_gf_content_at_kitchen:
                has_gf_ingredients_in_kitchen = True

        # Check if there is at least one unused sandwich object available to be made
        has_notexist_sandwich_object = any(f'(notexist {s})' in state for s in self.all_sandwich_objects)


        for child in self.goal_children:
            # If child is already served, no cost for this child
            if f'(served {child})' in state:
                continue

            # Child needs to be served
            child_place = self.child_waiting_place.get(child)
            if child_place is None:
                 # Child is in goal but not waiting anywhere specified in static facts.
                 # This state might be unreachable or indicates a problem definition issue.
                 # For heuristic purposes, treat as unsolvable from here.
                 return float('inf')

            needs_gf = child in self.child_is_allergic

            # Minimum steps required *before* the final serve action for this child
            min_steps_before_serve = float('inf')

            # --- Option 1: Make a new suitable sandwich ---
            can_make_suitable_new_sandwich = False
            if has_notexist_sandwich_object:
                if needs_gf:
                    if has_gf_ingredients_in_kitchen:
                        can_make_suitable_new_sandwich = True
                else:
                    if has_any_ingredients_in_kitchen:
                        can_make_suitable_new_sandwich = True

            if can_make_suitable_new_sandwich:
                 # Cost: make (1) + put (1) + move (1) = 3
                 min_steps_before_serve = min(min_steps_before_serve, 3)

            # --- Option 2: Use an existing sandwich ---
            for s in self.all_sandwich_objects:
                 # Check if this sandwich object 's' is suitable for the child
                 is_suitable = True
                 if needs_gf:
                     # If child needs GF, sandwich must be GF. Check if known to be GF.
                     # If sandwich_is_gf.get(s) is None, it means (no_gluten_sandwich s) is not in state.
                     if sandwich_is_gf.get(s) != True:
                         is_suitable = False
                 # If not needs_gf, any sandwich is suitable, so is_suitable remains True

                 if is_suitable:
                     # Determine the state of this suitable sandwich 's'
                     current_s_steps = float('inf') # Steps needed for this specific sandwich

                     # Check if it's on a tray at the child's location
                     found_on_tray_at_p = False
                     for fact in state:
                         parts = get_parts(fact)
                         if parts and parts[0] == 'ontray' and parts[1] == s:
                             tray = parts[2]
                             if f'(at {tray} {child_place})' in state:
                                 current_s_steps = 0 # Cost: 0 (already there)
                                 found_on_tray_at_p = True
                                 break # Found the best state for this sandwich

                     if found_on_tray_at_p:
                         min_steps_before_serve = min(min_steps_before_serve, current_s_steps)
                         continue # Move to the next sandwich object, found best case for this one

                     # If not on tray at child's location, check other states
                     if f'(at_kitchen_sandwich {s})' in state:
                         # Cost: put (1) + move (1) = 2
                         current_s_steps = 2
                     elif f'(notexist {s})' in state:
                         # This sandwich object hasn't been made yet.
                         # Its cost (3) is covered by the 'can_make_suitable_new_sandwich' check.
                         # If we are here, it means this specific 'notexist' sandwich object *could* be made suitable.
                         # We already updated min_steps_before_serve with 3 if *any* new suitable sandwich can be made.
                         pass # No need to update min_steps_before_serve based on this specific notexist sandwich object state
                     else: # Must be ontray somewhere else (not at_kitchen_sandwich, not notexist, not ontray at child_place)
                         # Find the tray and its location
                         tray_on_s = None
                         tray_loc = None
                         for fact in state:
                             parts = get_parts(fact)
                             if parts and parts[0] == 'ontray' and parts[1] == s:
                                 tray_on_s = parts[2]
                                 # Find tray location
                                 for loc_fact in state:
                                     loc_parts = get_parts(loc_fact)
                                     if loc_parts and loc_parts[0] == 'at' and loc_parts[1] == tray_on_s:
                                         tray_loc = loc_parts[2]
                                         break
                                 break # Found the tray and its location

                         if tray_on_s and tray_loc:
                             # Cost: move (1)
                             current_s_steps = 1
                         # If tray_on_s is true but tray_loc is not found, or if the state is inconsistent,
                         # current_s_steps remains infinity, and won't update min_steps_before_serve.

                     min_steps_before_serve = min(min_steps_before_serve, current_s_steps)


            # After checking all sandwich objects and the possibility of making a new one
            if min_steps_before_serve == float('inf'):
                 # This means no suitable sandwich exists in any state (notexist, at_kitchen, ontray)
                 # and we couldn't make a new one.
                 return float('inf') # Problem is unsolvable from this state

            # Add the cost for this child: steps before serve + the serve action itself
            h += min_steps_before_serve + 1

        return h
