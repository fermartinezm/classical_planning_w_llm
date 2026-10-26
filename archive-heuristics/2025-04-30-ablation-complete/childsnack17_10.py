from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

# Define helper functions outside the class as they are general utilities
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string representation of a PDDL predicate fact like "(predicate arg1 arg2)"
    # Basic validation: check if it's a string and looks like a fact.
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Should not happen with valid PDDL states from a standard planner
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
    This heuristic estimates the number of actions required to serve all children.
    It does this by considering each unserved child and estimating the minimum
    cost to get a suitable sandwich to them, prioritizing sandwiches that are
    closer to being served (already at the child's location, then on a tray
    in the kitchen, then in the kitchen, then needing to be made). It greedily
    matches available sandwiches to unserved children based on this proximity.

    # Assumptions
    - Each unserved child requires exactly one suitable sandwich.
    - A suitable sandwich is gluten-free for allergic children and any sandwich
      for non-allergic children.
    - The cost to serve a child with a suitable sandwich already on a tray
      at their location is 1 (the serve action).
    - The cost to serve a child with a suitable sandwich on a tray in the kitchen
      is 2 (move_tray + serve).
    - The cost to serve a child with a suitable sandwich in the kitchen
      (`at_kitchen_sandwich`) is 3 (put_on_tray + move_tray + serve).
    - The cost to serve a child with a suitable sandwich that needs to be made
      (`notexist` + ingredients) is 4 (make_sandwich + put_on_tray + move_tray + serve).
    - Tray movements are simplified; moving a tray from the kitchen to a child's
      location is assumed to cost 1 action, regardless of the specific path
      or other children at that location.
    - Resource availability (ingredients, notexist sandwich objects, trays) is
      checked but not optimized for sharing beyond the greedy matching.
    - If a child cannot be covered by any available or creatable sandwich with
      available ingredients/notexist objects, the state is considered unsolvable
      from this point with finite cost.

    # Heuristic Initialization
    - Extracts static information about which children are allergic and which
      bread/content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state to identify:
        - Children who are already served.
        - The location of each waiting child.
        - The location of each tray.
        - Sandwiches that are `at_kitchen_sandwich`.
        - Sandwiches that are `ontray` and which tray they are on.
        - Sandwiches that `notexist`.
        - Sandwiches that are `no_gluten_sandwich`.
        - Bread and content portions that are `at_kitchen_bread` or `at_kitchen_content`.
    2.  Identify all unserved children (waiting but not served) and gather their location and allergy status.
    3.  Categorize available sandwiches based on their type (gluten-free or regular)
        and their current location/status:
        - On trays at child locations.
        - On trays in the kitchen.
        - `at_kitchen_sandwich`.
        - `notexist` (available to be made).
    4.  Count available gluten-free and regular ingredients in the kitchen based on state facts and static gluten status.
    5.  Initialize heuristic value `h = 0` and a set `covered_children` to track
        children whose sandwich needs have been accounted for.
    6.  Greedily match unserved children to available suitable sandwiches in stages,
        adding the estimated cost for that stage to `h` and marking the child as covered:
        - **Stage 4 (Cost 1):** Match children to suitable sandwiches already on trays
          at their specific waiting location.
        - **Stage 3 (Cost 2):** Match remaining children to suitable sandwiches
          on trays located in the kitchen.
        - **Stage 2 (Cost 3):** Match remaining children to suitable sandwiches
          that are `at_kitchen_sandwich`.
        - **Stage 1 (Cost 4):** Match remaining children to suitable sandwiches
          that can be made from available `notexist` objects and ingredients.
    7.  If any children remain unserved after exhausting all available and creatable
        sandwiches, return infinity, indicating the state is likely unsolvable
        from this point with the current resources.
    8.  Otherwise, return the accumulated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        static_facts = task.static

        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.gf_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.gf_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Step 1: Parse the current state ---
        served_children = set()
        child_location = {} # child -> place
        tray_location = {} # tray -> place
        at_kitchen_sandwich_list = [] # list of sandwich names
        sandwich_on_tray = {} # sandwich -> tray
        notexist_sandwiches_list = [] # list of sandwich names
        state_gf_sandwiches = set() # set of GF sandwich names currently existing
        state_at_kitchen_bread_facts = set() # facts like (at_kitchen_bread bread1)
        state_at_kitchen_content_facts = set() # facts like (at_kitchen_content content1)

        # Collect all children mentioned in waiting facts
        all_waiting_children = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "served":
                served_children.add(args[0])
            elif predicate == "waiting":
                child, place = args
                child_location[child] = place
                all_waiting_children.add(child)
            elif predicate == "at":
                obj_name, place = args
                if obj_name.startswith('tray'): # Assuming objects starting with 'tray' are trays.
                     tray_location[obj_name] = place
            elif predicate == "at_kitchen_bread":
                state_at_kitchen_bread_facts.add(fact)
            elif predicate == "at_kitchen_content":
                state_at_kitchen_content_facts.add(fact)
            elif predicate == "at_kitchen_sandwich":
                at_kitchen_sandwich_list.append(args[0])
            elif predicate == "ontray":
                sandwich_on_tray[args[0]] = args[1]
            elif predicate == "notexist":
                notexist_sandwiches_list.append(args[0])
            elif predicate == "no_gluten_sandwich":
                 state_gf_sandwiches.add(args[0])

        # --- Step 2: Identify unserved children and their info ---
        # Consider all children who are waiting but not served.
        unserved_children = {c for c in all_waiting_children if c not in served_children}

        if not unserved_children:
            return 0 # Goal reached

        child_info = {} # child -> (location, is_allergic)
        for c in unserved_children:
             loc = child_location.get(c)
             # Assuming valid states where waiting children have locations.
             if loc is None:
                 # This indicates a potentially malformed state or problem definition
                 # where a waiting child has no specified location.
                 # Returning infinity seems appropriate as we cannot serve them.
                 return float('inf')
             is_allergic = c in self.allergic_children
             child_info[c] = (loc, is_allergic)


        # --- Step 3 & 4: Categorize available sandwiches and ingredients ---
        available_gf_at_loc = defaultdict(list) # place -> list of GF sandwiches
        available_reg_at_loc = defaultdict(list) # place -> list of Reg sandwiches
        available_gf_ontray_kitchen = [] # list of GF sandwiches
        available_reg_ontray_kitchen = [] # list of Reg sandwiches
        available_gf_at_kitchen_sandwich = [] # list of GF sandwiches
        available_reg_at_kitchen_sandwich = [] # list of Reg sandwiches
        available_notexist = list(notexist_sandwiches_list) # list of sandwich names

        # Sandwiches at_kitchen_sandwich
        for s in at_kitchen_sandwich_list:
            if s in state_gf_sandwiches:
                available_gf_at_kitchen_sandwich.append(s)
            else:
                available_reg_at_kitchen_sandwich.append(s)

        # Sandwiches on trays
        for s, t in sandwich_on_tray.items():
            p = tray_location.get(t)
            # If a tray has no location, we cannot use the sandwich on it.
            if p is None:
                 continue # Skip this sandwich/tray

            if p == 'kitchen':
                if s in state_gf_sandwiches:
                    available_gf_ontray_kitchen.append(s)
                else:
                    available_reg_ontray_kitchen.append(s)
            else: # Tray is at a child location (not kitchen)
                if s in state_gf_sandwiches:
                    available_gf_at_loc[p].append(s)
                else:
                    available_reg_at_loc[p].append(s)

        # Available ingredients in kitchen
        num_at_kitchen_gf_bread = sum(1 for fact in state_at_kitchen_bread_facts if get_parts(fact)[1] in self.gf_bread)
        num_at_kitchen_gf_content = sum(1 for fact in state_at_kitchen_content_facts if get_parts(fact)[1] in self.gf_content)
        available_gf_ingredients = min(num_at_kitchen_gf_bread, num_at_kitchen_gf_content)

        num_at_kitchen_reg_bread = sum(1 for fact in state_at_kitchen_bread_facts if get_parts(fact)[1] not in self.gf_bread)
        num_at_kitchen_reg_content = sum(1 for fact in state_at_kitchen_content_facts if get_parts(fact)[1] not in self.gf_content)
        available_reg_ingredients = min(num_at_kitchen_reg_bread, num_at_kitchen_reg_content)


        # --- Step 5 & 6: Greedy matching and cost calculation ---
        h = 0
        covered_children = set()
        # Use a list copy to iterate over children who still need covering
        children_to_cover = list(unserved_children)

        # Stage 4 (Cost 1: Serve) - Sandwiches already on trays at child location
        newly_covered = []
        for c in children_to_cover:
            p, is_allergic = child_info[c]
            if is_allergic:
                if available_gf_at_loc[p]:
                    available_gf_at_loc[p].pop() # Use one sandwich
                    h += 1
                    newly_covered.append(c)
            else: # not allergic
                if available_reg_at_loc[p]:
                    available_reg_at_loc[p].pop() # Use one sandwich
                    h += 1
                    newly_covered.append(c)
        covered_children.update(newly_covered)
        children_to_cover = [c for c in children_to_cover if c not in covered_children]

        # Stage 3 (Cost 2: Move + Serve) - Sandwiches on trays in kitchen
        newly_covered = []
        for c in children_to_cover:
            p, is_allergic = child_info[c]
            if is_allergic:
                if available_gf_ontray_kitchen:
                    available_gf_ontray_kitchen.pop()
                    h += 2
                    newly_covered.append(c)
            else: # not allergic
                if available_reg_ontray_kitchen:
                    available_reg_ontray_kitchen.pop()
                    h += 2
                    newly_covered.append(c)
        covered_children.update(newly_covered)
        children_to_cover = [c for c in children_to_cover if c not in covered_children]

        # Stage 2 (Cost 3: Put + Move + Serve) - Sandwiches at_kitchen_sandwich
        newly_covered = []
        for c in children_to_cover:
            p, is_allergic = child_info[c]
            if is_allergic:
                if available_gf_at_kitchen_sandwich:
                    available_gf_at_kitchen_sandwich.pop()
                    h += 3
                    newly_covered.append(c)
            else: # not allergic
                if available_reg_at_kitchen_sandwich:
                    available_reg_at_kitchen_sandwich.pop()
                    h += 3
                    newly_covered.append(c)
        covered_children.update(newly_covered)
        children_to_cover = [c for c in children_to_cover if c not in covered_children]

        # Stage 1 (Cost 4: Make + Put + Move + Serve) - Sandwiches needing to be made
        newly_covered = []
        for c in children_to_cover:
            p, is_allergic = child_info[c]
            if is_allergic:
                if available_notexist and available_gf_ingredients >= 1:
                    available_notexist.pop()
                    available_gf_ingredients -= 1
                    h += 4
                    newly_covered.append(c)
            else: # not allergic
                if available_notexist and available_reg_ingredients >= 1:
                    available_notexist.pop()
                    available_reg_ingredients -= 1
                    h += 4
                    newly_covered.append(c)
        covered_children.update(newly_covered)
        children_to_cover = [c for c in children_to_cover if c not in covered_children]

        # --- Step 7: Check if all children are covered ---
        if children_to_cover:
            # If there are still children not covered by any available/creatable sandwich
            return float('inf') # Unsolvable with available resources

        # --- Step 8: Return heuristic value ---
        return h
