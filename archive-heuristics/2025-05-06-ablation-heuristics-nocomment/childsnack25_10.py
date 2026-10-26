from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Helper function to split a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
        This heuristic estimates the number of actions required to reach a goal state
        by summing up the estimated costs of four main processes:
        1. Serving each unserved child.
        2. Making sandwiches that are needed but do not exist.
        3. Putting sandwiches (that are or will be at the kitchen) onto trays.
        4. Moving trays to waiting locations that currently lack a tray.
        It is designed for greedy best-first search and is not admissible, but aims
        to be informative and efficiently computable.

    Assumptions:
        - Each action has a cost of 1.
        - Trays have unlimited capacity for sandwiches (for the purpose of counting).
        - Ingredients (bread and content) are sufficient to make needed sandwiches,
          limited primarily by the availability of 'notexist' sandwich objects and
          secondarily by the counts of ingredients currently at the kitchen.
        - The cost of moving trays is simplified: we only count a move action
          for each waiting location (outside the kitchen) that currently has no tray.
          This assumes one move is sufficient to bring *a* tray there, and doesn't
          account for needing specific sandwiches on that tray or potential
          bottlenecks with tray availability/movement between multiple locations.
        - The heuristic is non-admissible.

    Heuristic Initialization:
        The constructor extracts static information from the task:
        - `all_children`: Set of all child objects.
        - `Child_waiting_place`: Dictionary mapping each child to their waiting place.
        - `Child_allergy`: Dictionary mapping each child to their allergy status ('allergic_gluten' or 'not_allergic_gluten').
        - `Bread_gluten_status`: Dictionary mapping each bread object to its status ('no_gluten_bread' or implicitly regular).
        - `Content_gluten_status`: Dictionary mapping each content object to its status ('no_gluten_content' or implicitly regular).
        - `all_trays`: Set of all tray objects.
        - `all_places`: Set of all place objects (including 'kitchen').
        - `all_sandwiches`: Set of all sandwich objects.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the heuristic is computed as follows:

        1.  **Cost for Serving:** Count the number of children who are not yet served. Each unserved child requires a final 'serve' action. Add this count to the heuristic. If no children are unserved, the state is a goal state, and the heuristic is 0.

        2.  **Cost for Making Sandwiches:**
            - Determine the total number of gluten-free and regular sandwiches needed across all unserved children's requirements.
            - Count the total number of gluten-free and regular sandwiches currently available anywhere in the state (at kitchen or on trays).
            - Calculate the deficit for each type: the number of sandwiches of that type that *must* be made (`max(0, needed - available)`).
            - Count the available ingredients (gluten-free bread/content, regular bread/content) and 'notexist' sandwich slots at the kitchen.
            - Calculate how many sandwiches of each type can *actually* be made, limited by the deficit, available ingredients of the correct type, and available 'notexist' slots (prioritizing gluten-free if slots are a bottleneck).
            - Add the number of sandwiches actually made to the heuristic (each 'make' action costs 1).

        3.  **Cost for Putting Sandwiches on Trays:**
            - Count the number of sandwiches that are currently at the kitchen (`at_kitchen_sandwich`) in the current state.
            - Add the number of sandwiches that were just calculated as needing to be made (from step 2). These sandwiches will appear at the kitchen after being made.
            - This total represents the number of sandwiches that need a 'put_on_tray' action to get onto a tray at the kitchen. Add this total to the heuristic (each 'put_on_tray' action costs 1).

        4.  **Cost for Moving Trays:**
            - Identify all locations where unserved children are waiting.
            - For each such waiting location (excluding the 'kitchen'), check if there is currently at least one tray present.
            - Count the number of these waiting locations (outside the kitchen) that have zero trays.
            - Add this count to the heuristic. This estimates the number of 'move_tray' actions needed to bring a tray to each location that requires one for serving (each 'move_tray' action costs 1). This is a simplification and assumes a tray is available elsewhere to be moved.

        The total heuristic value is the sum of the costs from these four components.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.Child_waiting_place = {}
        self.Child_allergy = {}
        self.Bread_gluten_status = {}
        self.Content_gluten_status = {}
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant place
        self.all_sandwiches = set()
        self.all_children = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'waiting':
                child, place = parts[1:3]
                self.Child_waiting_place[child] = place
                self.all_children.add(child)
            elif pred in ['allergic_gluten', 'not_allergic_gluten']:
                child = parts[1]
                self.Child_allergy[child] = pred
                self.all_children.add(child)
            elif pred == 'no_gluten_bread':
                bread = parts[1]
                self.Bread_gluten_status[bread] = pred
            elif pred == 'no_gluten_content':
                content = parts[1]
                self.Content_gluten_status[content] = pred

        # Parse initial state facts to find objects and places not in static
        # This is a fallback; ideally, object types would be parsed from the problem file.
        # Assuming objects relevant to the heuristic appear in initial state or goals.
        for fact in task.initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             if pred in ['at_kitchen_bread', 'at_kitchen_content', 'at_kitchen_sandwich', 'notexist']:
                 # These predicates involve bread, content, or sandwich objects
                 obj = parts[1]
                 if 'bread' in obj: # Simple heuristic for type based on name
                     pass # Already handled bread status from static
                 elif 'content' in obj: # Simple heuristic for type based on name
                     pass # Already handled content status from static
                 elif 'sandw' in obj: # Simple heuristic for type based on name
                     self.all_sandwiches.add(obj)
             elif pred == 'ontray':
                 s, t = parts[1:3]
                 if 'sandw' in s: self.all_sandwiches.add(s)
                 if 'tray' in t: self.all_trays.add(t)
             elif pred == 'at':
                 obj, place = parts[1:3]
                 if 'tray' in obj: self.all_trays.add(obj)
                 self.all_places.add(place)

        # Add children from goals if any were missed (e.g., not in waiting initially)
        for goal in self.goals:
            if match(goal, 'served', '*'):
                child = get_parts(goal)[1]
                self.all_children.add(child)


    def __call__(self, node):
        state = node.state

        # --- Step 1: Cost for Serving ---
        Unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}
        if not Unserved_children:
            return 0 # Goal state reached

        h = len(Unserved_children) # Each unserved child needs a 'serve' action

        # Collect state information relevant to subsequent steps
        Waiting_locations = set()
        Unserved_at_location = defaultdict(lambda: {'gf': 0, 'reg': 0})
        for c in Unserved_children:
            p = self.Child_waiting_place.get(c) # Use .get in case a child object exists but wasn't in waiting (shouldn't happen based on domain)
            if p:
                Waiting_locations.add(p)
                status = self.Child_allergy.get(c, 'not_allergic_gluten') # Default to not allergic
                Unserved_at_location[p]['gf' if 'allergic' in status else 'reg'] += 1

        Sandwiches_at_kitchen = {'gf': 0, 'reg': 0}
        Sandwiches_on_trays_at = defaultdict(lambda: {'gf': 0, 'reg': 0})
        Trays_at = defaultdict(int)
        Sandwich_slots = 0
        Bread_at_kitchen = {'gf': 0, 'reg': 0}
        Content_at_kitchen = {'gf': 0, 'reg': 0}

        # Pre-calculate sandwich gluten status for existing sandwiches
        Sandwich_gluten_status_in_state = {}
        for fact in state:
             if match(fact, 'no_gluten_sandwich', '*'):
                 Sandwich_gluten_status_in_state[get_parts(fact)[1]] = 'gf'

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at_kitchen_sandwich':
                s = parts[1]
                is_gf = Sandwich_gluten_status_in_state.get(s, 'reg') == 'gf'
                Sandwiches_at_kitchen['gf' if is_gf else 'reg'] += 1
            elif pred == 'ontray':
                s, t = parts[1:3]
                tray_loc = None
                # Find tray location for this tray
                for f_tray_at in state:
                    if match(f_tray_at, 'at', t, '*'):
                        tray_loc = get_parts(f_tray_at)[2]
                        break
                if tray_loc:
                     is_gf = Sandwich_gluten_status_in_state.get(s, 'reg') == 'gf'
                     Sandwiches_on_trays_at[tray_loc]['gf' if is_gf else 'reg'] += 1
            elif pred == 'at':
                obj, p = parts[1:3]
                if obj in self.all_trays: # Check if the object is a tray
                     Trays_at[p] += 1
            elif pred == 'notexist':
                Sandwich_slots += 1
            elif pred == 'at_kitchen_bread':
                b = parts[1]
                is_gf = self.Bread_gluten_status.get(b, 'reg') == 'no_gluten_bread'
                Bread_at_kitchen['gf' if is_gf else 'reg'] += 1
            elif pred == 'at_kitchen_content':
                c = parts[1]
                is_gf = self.Content_gluten_status.get(c, 'reg') == 'no_gluten_content'
                Content_at_kitchen['gf' if is_gf else 'reg'] += 1

        # --- Step 2: Cost for Making Sandwiches ---
        Total_needed_gf = sum(Unserved_at_location[p]['gf'] for p in Waiting_locations)
        Total_needed_reg = sum(Unserved_at_location[p]['reg'] for p in Waiting_locations)

        Total_available_gf = Sandwiches_at_kitchen['gf'] + sum(Sandwiches_on_trays_at[p]['gf'] for p in self.all_places)
        Total_available_reg = Sandwiches_at_kitchen['reg'] + sum(Sandwiches_on_trays_at[p]['reg'] for p in self.all_places)

        Must_make_gf = max(0, Total_needed_gf - Total_available_gf)
        Must_make_reg = max(0, Total_needed_reg - Total_available_reg)

        Can_make_gf_by_ingredients = min(Bread_at_kitchen['gf'], Content_at_kitchen['gf'])
        Can_make_reg_by_ingredients = min(Bread_at_kitchen['reg'], Content_at_kitchen['reg'])

        # Prioritize making GF sandwiches if slots are limited
        Actually_make_gf = min(Must_make_gf, Can_make_gf_by_ingredients, Sandwich_slots)
        Remaining_slots_after_gf = Sandwich_slots - Actually_make_gf
        Actually_make_reg = min(Must_make_reg, Can_make_reg_by_ingredients, Remaining_slots_after_gf)

        h += Actually_make_gf + Actually_make_reg # Cost for 'make_sandwich' actions

        # --- Step 3: Cost for Putting Sandwiches on Trays ---
        # Count sandwiches that are currently at the kitchen (before adding made ones)
        Sandwiches_at_kitchen_orig_gf = Sandwiches_at_kitchen['gf']
        Sandwiches_at_kitchen_orig_reg = Sandwiches_at_kitchen['reg']

        # Total sandwiches that are or will be at the kitchen and need 'put_on_tray'
        Sandwiches_needing_put_on_tray = (Sandwiches_at_kitchen_orig_gf + Sandwiches_at_kitchen_orig_reg) + (Actually_make_gf + Actually_make_reg)
        h += Sandwiches_needing_put_on_tray # Cost for 'put_on_tray' actions

        # --- Step 4: Cost for Moving Trays ---
        # Count waiting locations (outside kitchen) that have no tray
        Locations_need_tray_moved_to = {p for p in Waiting_locations if p != 'kitchen' and Trays_at[p] == 0}
        h += len(Locations_need_tray_moved_to) # Cost for 'move_tray' actions

        return h
