import sys
import os
from fnmatch import fnmatch

# Ensure the heuristics directory is in the Python path
# This might be needed if running the heuristic standalone or from a different directory structure
# Adjust the path ('..') as necessary based on your project structure
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the above path adjustment doesn't work or is not needed
    # Assumes heuristic_base.py is in a package named 'heuristics' accessible in PYTHONPATH
    print("Attempting fallback import for Heuristic base class.")
    # This path adjustment might be necessary if the script is run directly
    # and the 'heuristics' directory is one level up.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from heuristics.heuristic_base import Heuristic
    except ImportError:
        sys.exit("Error: Could not import Heuristic base class. Make sure the heuristics package is in your PYTHONPATH.")


# Helper functions
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    """
    return fact[1:-1].split()

def match(fact, *pattern):
    """
    Checks if a fact string matches a pattern with optional wildcards ('*').
    Example: match("(at tray1 kitchen)", "at", "*", "kitchen") -> True
    """
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    according to their dietary needs (gluten allergy) and waiting locations.
    It calculates the cost for each unserved child based on the state of sandwiches
    (location, type) and available ingredients, attempting to minimize cost by
    using existing resources first and accounting for resource consumption. The
    estimated cost for serving one child ranges from 1 (if a suitable sandwich is
    ready on a tray at the location) to 4 (if a sandwich needs to be made from
    scratch).

    # Assumptions
    - Each 'make' action requires one bread and one content portion and costs 1.
    - Each 'put_on_tray' action costs 1.
    - Each 'move_tray' action costs 1 (moving between any two locations).
    - Each 'serve' action costs 1.
    - The heuristic prioritizes using existing suitable sandwiches over making new ones.
    - It accounts for the consumption of specific sandwiches and ingredients within the
      heuristic calculation for a single state to avoid overestimating available resources.
    - If ingredients are needed but unavailable (already marked 'used' for another child
      in this state evaluation), it still assigns the standard cost of making (as part of
      the 4-action sequence), assuming the state is solvable perhaps through a different
      ordering or resource allocation not captured by this greedy assignment.

    # Heuristic Initialization
    - Stores the goal facts `(served ?c)`.
    - Extracts static information about children: allergy status (`allergic_gluten`,
      `not_allergic_gluten`) and waiting location (`waiting ?c ?p`).
    - Extracts static information about ingredients: which bread/content portions are
      gluten-free (`no_gluten_bread`, `no_gluten_content`).
    - Identifies all tray objects defined in the problem instance.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all unserved children based on the goal `(served ?c)` facts missing
        from the current state. If all children are served, the heuristic value is 0.
    2.  Parse the current state to determine:
        - Location (`at_kitchen_sandwich`, `ontray`) and gluten-free status
          (`no_gluten_sandwich`) of all existing sandwiches.
        - Location (`at`) of all trays.
        - Availability (`at_kitchen_bread`, `at_kitchen_content`) and gluten-free
          status (from static info) of bread and content portions at the kitchen.
    3.  Initialize a total heuristic cost to 0. Mark all resources (sandwiches,
        ingredients) in the current state as 'unused' for this heuristic calculation.
    4.  Iterate through the unserved children (in a fixed order, e.g., sorted name,
        for determinism):
        a. Determine if the child needs a gluten-free sandwich and their waiting location.
        b. Find the cheapest way to get a suitable sandwich to the child, checking in
           order of increasing estimated cost:
            i.   **Level 1 (Cost 1: Serve):** Is there an unused suitable sandwich
                 on a tray already at the child's location?
            ii.  **Level 2 (Cost 2: Move + Serve):** Is there an unused suitable
                 sandwich on a tray at a different location?
            iii. **Level 3 (Cost 3: Put + Move + Serve):** Is there an unused
                 suitable sandwich at the kitchen? (Assumes a tray is available or
                 can be brought to the kitchen).
            iv.  **Level 4 (Cost 4: Make + Put + Move + Serve):** If no suitable
                 existing sandwich is found, can one be made? Check for available
                 unused suitable bread and content portions at the kitchen.
        c. Select the lowest cost option found (Level 1 to 4).
        d. If an option is found:
            - Add the corresponding cost (1, 2, 3, or 4) to the total heuristic value.
            - Mark the specific resource used (the sandwich object, or the bread and
              content objects) as 'used' so it cannot satisfy another child's need
              in this same state evaluation.
        e. If no option is found (e.g., Level 4 fails because required ingredients
           are already marked 'used'), add the base cost of 4.
    5.  Return the total accumulated heuristic cost.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_info = task.static

        self.child_info = {} # child -> {'allergic': bool, 'waiting_at': place}
        self.gluten_free_bread_static = set()
        self.gluten_free_content_static = set()
        self.trays = set() # Store tray names

        # --- Extract static information ---
        all_objects = set()
        facts_for_obj_scan = task.initial_state.union(task.static)
        for fact in facts_for_obj_scan:
             parts = get_parts(fact)
             all_objects.update(parts[1:])
        # Consider objects mentioned in operators too, although less common for static types
        # for op in task.operators:
        #      parts = get_parts(op.name) # op.name is like "(action-name p1 p2 ...)"
        #      all_objects.update(parts[1:])

        # Identify trays - look for objects used as trays in relevant predicates/actions
        potential_trays = set()
        for fact in facts_for_obj_scan:
             parts = get_parts(fact)
             # Trays appear in (at tray place) and (ontray sandwich tray)
             if parts[0] == 'at' and len(parts) == 3:
                 # Could be a tray at a place
                 potential_trays.add(parts[1])
             elif parts[0] == 'ontray' and len(parts) == 3:
                 # The third argument is the tray
                 potential_trays.add(parts[2])

        # Refine tray identification by checking action definitions
        for obj in potential_trays:
             is_tray = False
             # Check if used as tray parameter in actions
             for op in task.operators:
                 op_parts = get_parts(op.name) # op.name is like "(action-name p1 p2 ...)"
                 action_name = op_parts[0]
                 params = op_parts[1:]
                 if action_name == 'move_tray' and obj == params[0]: is_tray = True; break
                 if action_name == 'put_on_tray' and obj == params[1]: is_tray = True; break
                 if action_name.startswith('serve_sandwich') and obj == params[2]: is_tray = True; break
             # Also check initial state/static facts again specifically for ontray
             if not is_tray:
                 for fact in facts_for_obj_scan:
                     parts = get_parts(fact)
                     if parts[0] == 'ontray' and len(parts) == 3 and parts[2] == obj:
                         is_tray = True
                         break
             if is_tray:
                 self.trays.add(obj)

        # Fallback: if type information isn't perfectly parsable, assume objects named 'tray*' are trays
        if not self.trays:
            for obj in all_objects:
                if obj.startswith('tray'):
                    self.trays.add(obj)
        # print(f"Identified trays: {self.trays}") # For debugging

        # Extract child info, ingredient properties
        for fact in self.static_info:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "waiting":
                child, place = parts[1], parts[2]
                if child not in self.child_info: self.child_info[child] = {}
                self.child_info[child]['waiting_at'] = place
            elif predicate == "allergic_gluten":
                child = parts[1]
                if child not in self.child_info: self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                if child not in self.child_info: self.child_info[child] = {}
                self.child_info[child]['allergic'] = False
            elif predicate == "no_gluten_bread":
                self.gluten_free_bread_static.add(parts[1])
            elif predicate == "no_gluten_content":
                self.gluten_free_content_static.add(parts[1])

        # Ensure all children mentioned in goals have complete info (add defaults if necessary)
        goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        for child in goal_children:
            if child not in self.child_info:
                print(f"Warning: Child {child} from goal not found in static waiting/allergy info. Assuming defaults.")
                self.child_info[child] = {'waiting_at': 'kitchen', 'allergic': False} # Default: wait at kitchen, not allergic
            else:
                if 'waiting_at' not in self.child_info[child]:
                     print(f"Warning: Child {child} missing waiting location. Assuming 'kitchen'.")
                     self.child_info[child]['waiting_at'] = 'kitchen'
                if 'allergic' not in self.child_info[child]:
                     print(f"Warning: Child {child} missing allergy info. Assuming not allergic.")
                     self.child_info[child]['allergic'] = False


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        unserved_goals = self.goals - state
        if not unserved_goals:
            # Goal state reached
            return 0

        # --- State Parsing: Extract dynamic information ---
        # Store resource info with a 'used' flag for this heuristic calculation
        kitchen_sandwiches = {} # {sandwich_name: {'name': str, 'is_gluten_free': bool, 'used': bool}}
        sandwiches_on_trays = {} # {tray_name: [{'name': s, 'is_gluten_free': bool, 'used': bool}]}
        tray_locations = {} # {tray_name: place_name}
        gluten_free_sandwiches_state = set() # {sandwich_name}
        available_bread_kitchen = {} # {bread_name: {'name': str, 'is_gluten_free': bool, 'used': bool}}
        available_content_kitchen = {} # {content_name: {'name': str, 'is_gluten_free': bool, 'used': bool}}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at_kitchen_sandwich":
                s = parts[1]
                # Initialize as not GF, update later if no_gluten_sandwich fact exists
                kitchen_sandwiches[s] = {'name': s, 'is_gluten_free': False, 'used': False}
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                if t not in sandwiches_on_trays:
                    sandwiches_on_trays[t] = []
                # Initialize as not GF, update later
                sandwiches_on_trays[t].append({'name': s, 'is_gluten_free': False, 'used': False})
            elif predicate == "at":
                obj, loc = parts[1], parts[2]
                # Check if the object at a location is one of the identified trays
                if obj in self.trays:
                     tray_locations[obj] = loc
            elif predicate == "no_gluten_sandwich":
                # Mark which sandwiches are gluten-free in this state
                gluten_free_sandwiches_state.add(parts[1])
            elif predicate == "at_kitchen_bread":
                bread = parts[1]
                # Check static info for GF status
                is_gf = bread in self.gluten_free_bread_static
                available_bread_kitchen[bread] = {'name': bread, 'is_gluten_free': is_gf, 'used': False}
            elif predicate == "at_kitchen_content":
                content = parts[1]
                # Check static info for GF status
                is_gf = content in self.gluten_free_content_static
                available_content_kitchen[content] = {'name': content, 'is_gluten_free': is_gf, 'used': False}

        # Update gluten-free status for existing sandwiches based on state facts
        for s_info in kitchen_sandwiches.values():
            if s_info['name'] in gluten_free_sandwiches_state:
                s_info['is_gluten_free'] = True
        for t in sandwiches_on_trays:
            for sandwich_info in sandwiches_on_trays[t]:
                if sandwich_info['name'] in gluten_free_sandwiches_state:
                    sandwich_info['is_gluten_free'] = True

        # --- Heuristic Calculation ---
        total_heuristic = 0

        # Process children deterministically (e.g., sorted by name) to ensure consistency
        # Use the original goal set for iteration order consistency
        sorted_unserved_goals = sorted([g for g in self.goals if g not in state])

        for goal_fact in sorted_unserved_goals:
            child = get_parts(goal_fact)[1]

            # Get child requirements (allergy, location) from precomputed info
            child_reqs = self.child_info.get(child)
            if not child_reqs: continue # Should not happen due to __init__ checks

            child_needs_gluten_free = child_reqs['allergic']
            child_location = child_reqs['waiting_at']

            # --- Find the cheapest way to satisfy this child's goal ---
            best_option_level = 5 # 1=serve, 2=move+serve, 3=put+move+serve, 4=make+..., 5=impossible/default
            resource_to_use = None # Stores the dict of the sandwich to mark used
            bread_to_use = None    # Stores the dict of the bread to mark used
            content_to_use = None  # Stores the dict of the content to mark used

            # Check Level 1: Suitable sandwich on tray at location
            found_level1 = False
            for tray, sandwiches in sandwiches_on_trays.items():
                # Check if tray is at the child's location
                if tray_locations.get(tray) == child_location:
                    for sandwich_info in sandwiches:
                        if sandwich_info['used']: continue # Skip if already assigned
                        # Check if sandwich type is suitable
                        is_suitable = (not child_needs_gluten_free) or sandwich_info['is_gluten_free']
                        if is_suitable:
                            best_option_level = 1
                            resource_to_use = sandwich_info # Mark this sandwich as used
                            found_level1 = True
                            break # Found best option for this level
                if found_level1: break # Found best option for this level
            if found_level1:
                 total_heuristic += 1
                 resource_to_use['used'] = True
                 continue # Move to the next child

            # Check Level 2: Suitable sandwich on tray elsewhere
            found_level2 = False
            for tray, sandwiches in sandwiches_on_trays.items():
                 tray_loc = tray_locations.get(tray)
                 # Check if tray exists and is NOT at the child's location
                 if tray_loc is not None and tray_loc != child_location:
                     for sandwich_info in sandwiches:
                         if sandwich_info['used']: continue
                         is_suitable = (not child_needs_gluten_free) or sandwich_info['is_gluten_free']
                         if is_suitable:
                             best_option_level = 2
                             resource_to_use = sandwich_info
                             found_level2 = True
                             break
                 if found_level2: break
            if found_level2:
                 total_heuristic += 2
                 resource_to_use['used'] = True
                 continue

            # Check Level 3: Suitable sandwich at kitchen
            found_level3 = False
            for s_info in kitchen_sandwiches.values(): # Iterate through sandwiches at kitchen
                if s_info['used']: continue
                is_suitable = (not child_needs_gluten_free) or s_info['is_gluten_free']
                if is_suitable:
                    best_option_level = 3
                    resource_to_use = s_info
                    found_level3 = True
                    break
            if found_level3:
                 total_heuristic += 3
                 resource_to_use['used'] = True
                 continue

            # Check Level 4: Make sandwich from available ingredients
            found_bread_info = None
            found_content_info = None

            # Find suitable unused bread
            for b_info in available_bread_kitchen.values():
                if b_info['used']: continue
                # Bread must be GF if child needs GF. Any bread works otherwise.
                if child_needs_gluten_free:
                    if b_info['is_gluten_free']:
                        found_bread_info = b_info
                        break # Found suitable GF bread
                else:
                    found_bread_info = b_info
                    break # Found suitable (any type) bread

            # Find suitable unused content (only if suitable bread was found)
            if found_bread_info:
                for c_info in available_content_kitchen.values():
                    if c_info['used']: continue
                    # Content must be GF if child needs GF. Any content works otherwise.
                    if child_needs_gluten_free:
                        if c_info['is_gluten_free']:
                            found_content_info = c_info
                            break # Found suitable GF content
                    else:
                        found_content_info = c_info
                        break # Found suitable (any type) content

            # If both suitable bread and content were found:
            if found_bread_info and found_content_info:
                best_option_level = 4
                total_heuristic += 4
                # Mark ingredients as used for this state evaluation
                found_bread_info['used'] = True
                found_content_info['used'] = True
                continue # Move to next child
            else:
                # Cannot make sandwich with available unused ingredients.
                # This might mean the state requires a different assignment, or is
                # potentially unsolvable with current resources. Add the default
                # cost of 4, assuming solvability.
                total_heuristic += 4

        return total_heuristic

