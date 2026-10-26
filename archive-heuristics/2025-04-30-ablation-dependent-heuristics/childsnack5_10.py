import sys

# Assuming the Operator and Task classes are available in the environment
# from task import Operator, Task # Not needed for the heuristic code itself, but good context

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    (all children served) by summing up the estimated costs for different
    stages of the snack delivery process: making missing sandwiches, putting
    sandwiches onto trays from the kitchen, moving trays to the children's
    locations, and finally serving the children. It counts the number of
    items or tasks that are not yet in the state required for the next stage.

    Assumptions:
    - The problem instance is solvable with the available resources (bread,
      content, sandwich objects, trays). The heuristic does not explicitly
      check for resource sufficiency beyond counting the need for new sandwiches.
    - Tray capacity is sufficient (implicitly treated as 1 sandwich per tray
      for counting purposes, or simply counting trays needed).
    - The heuristic is non-admissible and designed for greedy best-first search,
      prioritizing states closer to having all children served through the
      various intermediate steps.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition to
    store information about child allergies, child waiting locations, and
    gluten-free status of bread and content. It also identifies all possible
    child, tray, sandwich, and place objects in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  **Count Unserved Children:** Identify all children who are not yet
        in the `(served ?c)` state. The number of unserved children is a
        lower bound on the number of `serve_sandwich` actions needed.
        (Component: `num_unserved`)
    2.  **Determine Sandwich Requirements:** Based on the allergies of the
        unserved children, determine the total number of gluten-free and
        regular sandwiches required.
    3.  **Identify Available Sandwiches:** Count the number of existing
        sandwiches (either `at_kitchen_sandwich` or `ontray`) and determine
        their gluten-free status based on the `(no_gluten_sandwich ?s)` facts
        present in the current state.
    4.  **Calculate New Sandwiches to Make:** Compare the required number of
        gluten-free and regular sandwiches with the available ones to determine
        the minimum number of new sandwiches of each type that must be created.
        The total number of new sandwiches is an estimate of the `make_sandwich`
        actions needed.
        (Component: `M_new`)
    5.  **Count Sandwiches Needing Put on Tray:** Count the number of existing
        suitable sandwiches that are currently `at_kitchen_sandwich`. These
        sandwiches need to be moved onto trays using the `put_on_tray` action.
        (Component: `N_put_on_tray_existing_kitchen`)
    6.  **Count Trays Needing Move:** Identify trays that currently hold a
        suitable sandwich (i.e., a sandwich needed by at least one unserved
        child) but are not located at any place where an unserved child
        who needs a sandwich from that specific tray is waiting. Each such
        tray needs at least one `move_tray` action to get to a useful location.
        (Component: `N_move_tray`)
    7.  **Sum Components:** The total heuristic value is the sum of the
        estimated actions from each stage: `M_new` (make) +
        `N_put_on_tray_existing_kitchen` (put) + `N_move_tray` (move) +
        `num_unserved` (serve). This sum represents the total number of
        "tasks" across the different stages that need to be completed.
    """
    def __init__(self, task):
        self.task = task
        # Extract static information
        self.child_allergy = {} # {child_name: True/False}
        self.child_location = {} # {child_name: place_name}
        self.bread_gluten_free = {} # {bread_name: True/False}
        self.content_gluten_free = {} # {content_name: True/False}
        self.all_children = set() # Set of all child names
        self.all_trays = set() # Set of all tray names
        self.all_sandwiches = set() # Set of all sandwich names
        self.all_places = set() # Set of all place names

        # Extract objects and static predicates from static facts
        for fact in task.static:
            objs = self._parse_fact_objects(fact)
            for obj in objs:
                 if obj.startswith('child'): self.all_children.add(obj)
                 elif obj.startswith('tray'): self.all_trays.add(obj)
                 elif obj.startswith('sandw'): self.all_sandwiches.add(obj)
                 elif obj.startswith('table') or obj == 'kitchen': self.all_places.add(obj)

            if fact.startswith('(allergic_gluten '):
                child = objs[0]
                self.child_allergy[child] = True
            elif fact.startswith('(not_allergic_gluten '):
                child = objs[0]
                self.child_allergy[child] = False
            elif fact.startswith('(waiting '):
                child, place = objs
                self.child_location[child] = place
            elif fact.startswith('(no_gluten_bread '):
                bread = objs[0]
                self.bread_gluten_free[bread] = True
            elif fact.startswith('(no_gluten_content '):
                content = objs[0]
                self.content_gluten_free[content] = True

        # Extract objects from initial state as well to be comprehensive
        for fact in task.initial_state:
             objs = self._parse_fact_objects(fact)
             for obj in objs:
                 if obj.startswith('child'): self.all_children.add(obj)
                 elif obj.startswith('tray'): self.all_trays.add(obj)
                 elif obj.startswith('sandw'): self.all_sandwiches.add(obj)
                 elif obj.startswith('table') or obj == 'kitchen': self.all_places.add(obj)


    def _parse_fact_objects(self, fact_string):
        """Helper to extract objects from a PDDL fact string."""
        # Example: '(at tray1 kitchen)' -> ['tray1', 'kitchen']
        # Example: '(served child1)' -> ['child1']
        # Remove surrounding brackets and split by space
        parts = fact_string[1:-1].split()
        # The first part is the predicate name, the rest are objects
        return parts[1:]

    def __call__(self, state):
        # 1. Count Unserved Children
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}
        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0 # Goal state

        # 2. Determine Sandwich Requirements
        U_g = sum(1 for c in unserved_children if self.child_allergy.get(c, False))
        U_ng = num_unserved - U_g

        # 3. Identify Available Sandwiches and their status/location
        available_sandwiches = {} # {sandwich_name: is_gluten_free}
        sandwiches_on_tray = {} # {sandwich_name: tray_name}
        sandwiches_in_kitchen = set()
        tray_location = {} # {tray_name: place_name}
        
        # Determine gluten-free status of sandwiches based on state facts
        gf_sandwiches_in_state = {self._parse_fact_objects(fact)[0] for fact in state if fact.startswith('(no_gluten_sandwich ')}
        
        # Identify sandwiches that currently exist (are not 'notexist')
        existing_sandwiches_in_state = {s for s in self.all_sandwiches if '(notexist ' + s + ')' not in state}

        for s in existing_sandwiches_in_state:
             available_sandwiches[s] = s in gf_sandwiches_in_state

        for fact in state:
            if fact.startswith('(at_kitchen_sandwich '):
                s = self._parse_fact_objects(fact)[0]
                sandwiches_in_kitchen.add(s)
            elif fact.startswith('(ontray '):
                s, t = self._parse_fact_objects(fact)
                sandwiches_on_tray[s] = t
            elif fact.startswith('(at '):
                t, p = self._parse_fact_objects(fact)
                tray_location[t] = p

        # Count available suitable sandwiches (kitchen or on tray)
        S_g_avail = sum(1 for s, is_gf in available_sandwiches.items() if is_gf and (s in sandwiches_in_kitchen or s in sandwiches_on_tray))
        S_reg_avail = sum(1 for s, is_gf in available_sandwiches.items() if not is_gf and (s in sandwiches_in_kitchen or s in sandwiches_on_tray))

        # 4. Calculate New Sandwiches to Make (M_new)
        M_g_new = max(0, U_g - S_g_avail)
        # For non-allergic, use available regular + surplus GF
        available_for_ng = S_reg_avail + max(0, S_g_avail - U_g)
        M_reg_new = max(0, U_ng - available_for_ng)
        M_new = M_g_new + M_reg_new

        # 5. Count Suitable Sandwiches in Kitchen Needing Put on Tray
        # Count suitable sandwiches currently in the kitchen
        SK_g = sum(1 for s in sandwiches_in_kitchen if available_sandwiches.get(s, False))
        SK_reg = sum(1 for s in sandwiches_in_kitchen if not available_sandwiches.get(s, False))
        
        # These are the existing sandwiches in the kitchen that need the 'put_on_tray' action.
        N_put_on_tray_existing_kitchen = SK_g + SK_reg

        # 6. Count Trays Needing Move
        # Identify trays holding suitable sandwiches for *any* unserved child
        # A sandwich is suitable if it's GF and there's an unserved allergic child (U_g > 0),
        # OR if there's an unserved non-allergic child (U_ng > 0).
        suitable_sandwiches_for_any_unserved = {s for s in available_sandwiches if (available_sandwiches[s] and U_g > 0) or U_ng > 0}
        
        trays_with_suitable_sandwiches = {sandwiches_on_tray[s] for s in suitable_sandwiches_for_any_unserved if s in sandwiches_on_tray}

        # Count trays with suitable sandwiches that are NOT at any location where
        # a child needing a sandwich from that tray is waiting.
        N_move_tray = 0
        trays_counted_for_move = set()

        for t in trays_with_suitable_sandwiches:
            current_loc = tray_location.get(t)
            if current_loc is None: continue # Should not happen if tray exists and has sandwich

            # Find all unserved children this tray *could* potentially serve at their location
            is_at_needed_location = False
            for s, tray_of_s in sandwiches_on_tray.items():
                if tray_of_s == t and s in suitable_sandwiches_for_any_unserved: # This sandwich is on this tray and is suitable for *some* unserved child
                    s_is_gf = available_sandwiches.get(s, False)
                    
                    # Check if this sandwich is needed by any unserved child at the tray's current location
                    for child in unserved_children:
                        child_loc = self.child_location.get(child)
                        child_is_allergic = self.child_allergy.get(child, False)
                        
                        if child_loc == current_loc:
                            # Check if sandwich s is suitable for this specific child
                            if (child_is_allergic and s_is_gf) or (not child_is_allergic):
                                is_at_needed_location = True
                                break # Found a child at this location who can use this sandwich/tray
                if is_at_needed_location: break # This tray is useful at its current location

            if not is_at_needed_location:
                 # This tray holds a needed sandwich but is not at any location where it's needed.
                 # It needs to move.
                 if t not in trays_counted_for_move:
                     N_move_tray += 1
                     trays_counted_for_move.add(t)

        # 7. Final Heuristic Calculation
        # Sum of estimated actions for each stage:
        # - Making sandwiches that don't exist but are needed.
        # - Putting existing kitchen sandwiches onto trays.
        # - Moving trays with suitable sandwiches to locations where they are needed.
        # - Serving the children once everything is in place.
        h_val = M_new + N_put_on_tray_existing_kitchen + N_move_tray + num_unserved

        return h_val
