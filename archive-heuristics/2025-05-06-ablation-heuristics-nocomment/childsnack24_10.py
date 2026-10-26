from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to match a fact against a pattern."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (serving all children)
    by summing up the estimated costs for each unserved child. The cost for
    an unserved child is estimated based on the "stage" of the closest
    available suitable sandwich (matching allergy status) relative to the
    child's waiting location. The stages represent the minimum number of
    actions required to get a suitable sandwich onto a tray at the child's
    location and then serve it, assuming necessary resources (like trays,
    ingredients, and notexist sandwich objects) are available when needed
    at that stage. The stages and their estimated costs are:
    1. Suitable sandwich is already on a tray at the child's location (cost 1: serve).
    2. Suitable sandwich is on a tray at a different location (cost 2: move_tray + serve).
    3. Suitable sandwich is in the kitchen (cost 3: put_on_tray + move_tray + serve).
    4. Suitable sandwich needs to be made (cost 4: make + put_on_tray + move_tray + serve).

    The heuristic assigns each unserved child to the lowest-cost stage for which
    a suitable sandwich *potentially* exists or can be made. It then sums the
    costs corresponding to the assigned stages for all unserved children. It
    does not perfectly model resource contention (e.g., multiple children needing
    the same sandwich or tray, limited trays at the kitchen, limited ingredients
    or notexist sandwich objects beyond the simple makeable counts), but provides
    a reasonable non-admissible estimate to guide greedy best-first search.

    Assumptions:
    - Each action has a cost of 1.
    - Trays are generally available when needed for put_on_tray or can be moved
      to the kitchen in one step if needed for put_on_tray.
    - Ingredients and notexist sandwich objects are sufficient to make needed
      sandwiches up to the calculated makeable counts.
    - The goal is always to serve all children.
    - The heuristic is non-admissible and designed to minimize expanded nodes.

    Heuristic Initialization:
    The constructor extracts static information from the task:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - Where each child is waiting.
    - Which bread portions are no-gluten.
    - Which content portions are no-gluten.
    This information is stored in dictionaries and sets for efficient lookup during
    heuristic computation. It maps each child to their allergy status and waiting place.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children that are not yet served based on the goal facts and the current state. If no children are unserved, the heuristic value is 0.
    2. For each unserved child, determine their waiting location and whether they are allergic to gluten (from the static information stored during initialization).
    3. Analyze the current state to identify available sandwiches:
       - Create mappings for sandwiches on trays (`ontray_sandwiches: s -> t`) and tray locations (`at_tray_location: t -> p`).
       - Identify sets of sandwiches that are no-gluten (`is_no_gluten_sandwich`), in the kitchen (`is_at_kitchen_sandwich`), or do not exist yet (`is_notexist_sandwich`).
       - Identify sets of bread and content portions in the kitchen, distinguishing between no-gluten and regular based on static facts.
    4. Calculate the number of gluten-free and regular sandwiches that can potentially be made, limited by the counts of `notexist` sandwich objects and available ingredients in the kitchen. Prioritize making gluten-free sandwiches when ingredients can be used for either type.
    5. Initialize the total heuristic value `h = 0`.
    6. Create sets to group unserved children based on the *best* stage (1, 2, 3, or 4) at which a suitable sandwich could potentially be acquired and delivered to them.
    7. Iterate through each unserved child:
       - Determine if a suitable sandwich exists that is already on a tray at the child's waiting location (Stage 1 potential). If yes, add the child to the Stage 1 set and move to the next child.
       - If not Stage 1, determine if a suitable sandwich exists that is on a tray at a different location (Stage 2 potential). If yes, add the child to the Stage 2 set and move on.
       - If not Stage 1 or 2, determine if a suitable sandwich exists in the kitchen (Stage 3 potential). If yes, add the child to the Stage 3 set and move on.
       - If not Stage 1, 2, or 3, determine if a suitable sandwich can be made based on the calculated makeable counts (Stage 4 potential). If yes, add the child to the Stage 4 set and move on.
       - If no suitable sandwich is found or makeable at any stage, the child is still added to the Stage 4 set, assuming it represents the highest cost/effort required.
    8. Calculate the number of children assigned to each stage (Stage 1, Stage 2, Stage 3, Stage 4) by taking the set difference to ensure each child is counted only at their *best* stage.
    9. Compute the total heuristic value by summing the counts for each stage multiplied by their respective costs: `h = (count_stage1 * 1) + (count_stage2 * 2) + (count_stage3 * 3) + (count_stage4 * 4)`.
    10. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.child_info = {} # child -> {'allergic': bool, 'place': place}
        self.static_no_gluten_bread = {fact for fact in static_facts if get_parts(fact)[0] == 'no_gluten_bread'}
        self.static_no_gluten_content = {fact for fact in static_facts if get_parts(fact)[0] == 'no_gluten_content'}

        # Extract child allergy status and waiting places from static facts
        waiting_places_map = {} # child -> place
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_info[child] = {'allergic': True}
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.child_info[child] = {'allergic': False}
            elif parts[0] == 'waiting':
                child = parts[1]
                place = parts[2]
                waiting_places_map[child] = place

        # Add place information to child_info
        for child in self.child_info:
            if child in waiting_places_map:
                self.child_info[child]['place'] = waiting_places_map[child]
            # else: Child exists in static (allergic/not_allergic) but no waiting fact?
            # This is unexpected PDDL. Assume they are not part of the goal or handle defensively.
            # For robustness, ensure all children in goals have place info.
            # If a child is in goals but not in static allergic/not_allergic/waiting,
            # it's a malformed problem instance, but we try to handle it.

        # Ensure all children mentioned in goals have place info (and allergy info if missing)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'served':
                 child = parts[1]
                 if child not in self.child_info:
                      # Child in goal but not in static allergic/not_allergic. Assume not allergic.
                      self.child_info[child] = {'allergic': False}
                 if 'place' not in self.child_info[child]:
                      # Find waiting place from static facts if not already found
                      waiting_fact = next((f for f in static_facts if match(f, 'waiting', child, '*')), None)
                      if waiting_fact:
                           place = get_parts(waiting_fact)[2]
                           self.child_info[child]['place'] = place
                      else:
                           # Child in goal but no waiting fact in static? Problematic instance.
                           # Assign a default place like 'kitchen'.
                           # print(f"Warning: Child {child} in goal but no waiting fact in static. Assigning kitchen.")
                           self.child_info[child]['place'] = 'kitchen' # Default place


    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.child_info if '(served ' + c + ')' not in state}

        if not unserved_children:
            return 0 # Goal reached

        # 3. Identify available sandwiches, trays, and ingredients
        ontray_sandwiches = {} # s -> t
        at_tray_location = {} # t -> p
        is_no_gluten_sandwich = set()
        is_at_kitchen_sandwich = set()
        is_notexist_sandwich = set()
        at_kitchen_bread_facts = set()
        at_kitchen_content_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                ontray_sandwiches[parts[1]] = parts[2]
            elif parts[0] == 'at' and parts[1].startswith('tray'):
                at_tray_location[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_sandwich':
                is_no_gluten_sandwich.add(parts[1])
            elif parts[0] == 'at_kitchen_sandwich':
                is_at_kitchen_sandwich.add(parts[1])
            elif parts[0] == 'notexist':
                is_notexist_sandwich.add(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                 at_kitchen_bread_facts.add(fact)
            elif parts[0] == 'at_kitchen_content':
                 at_kitchen_content_facts.add(fact)

        # 4. Count available ingredients
        num_avail_gf_bread = sum(1 for b_fact in at_kitchen_bread_facts if b_fact in self.static_no_gluten_bread)
        num_avail_reg_bread = sum(1 for b_fact in at_kitchen_bread_facts if b_fact not in self.static_no_gluten_bread)
        num_avail_gf_content = sum(1 for c_fact in at_kitchen_content_facts if c_fact in self.static_no_gluten_content)
        num_avail_reg_content = sum(1 for c_fact in at_kitchen_content_facts if c_fact not in self.static_no_gluten_content)
        num_notexist = len(is_notexist_sandwich)

        # 5. Calculate makeable sandwiches
        makeable_gf_slots = min(num_notexist, num_avail_gf_bread, num_avail_gf_content)

        rem_notexist = num_notexist - makeable_gf_slots
        rem_gf_bread = num_avail_gf_bread - makeable_gf_slots
        rem_gf_content = num_avail_gf_content - makeable_gf_slots

        avail_bread_for_reg = max(0, rem_gf_bread) + num_avail_reg_bread
        avail_content_for_reg = max(0, rem_gf_content) + num_avail_reg_content

        makeable_reg_slots = min(rem_notexist, avail_bread_for_reg, avail_content_for_reg)
        makeable_reg_slots = max(0, makeable_reg_slots)


        # 6. Create sets to group unserved children by best potential stage
        children_s1_potential = set() # Sandwich on tray at child's location
        children_s2_potential = set() # Sandwich on tray at wrong location
        children_s3_potential = set() # Sandwich in kitchen
        children_s4_potential = set() # Sandwich needs making

        # 7. Iterate through each unserved child to determine their best stage
        for child in unserved_children:
            p_c = self.child_info[child]['place']
            needs_gf = self.child_info[child]['allergic']

            found_s1 = False
            found_s2 = False
            found_s3 = False
            can_make_s4 = False

            # Check Stage 1: on tray at child's location
            for s in ontray_sandwiches:
                t = ontray_sandwiches[s]
                p = at_tray_location.get(t)
                if p == p_c:
                    is_gf_sandwich = s in is_no_gluten_sandwich
                    if (needs_gf and is_gf_sandwich) or (not needs_gf):
                         children_s1_potential.add(child)
                         found_s1 = True
                         break # Found one for this child at this stage

            if found_s1: continue # This child's best stage is 1

            # Check Stage 2: on tray at wrong location
            for s in ontray_sandwiches:
                 t = ontray_sandwiches[s]
                 p = at_tray_location.get(t)
                 if p is not None and p != p_c: # Check if tray has a location and it's NOT at the child's location
                     is_gf_sandwich = s in is_no_gluten_sandwich
                     if (needs_gf and is_gf_sandwich) or (not needs_gf):
                         children_s2_potential.add(child)
                         found_s2 = True
                         break # Found one for this child at this stage

            if found_s2: continue # This child's best stage is 2

            # Check Stage 3: in kitchen
            for s in is_at_kitchen_sandwich:
                 is_gf_sandwich = s in is_no_gluten_sandwich
                 if (needs_gf and is_gf_sandwich) or (not needs_gf):
                     children_s3_potential.add(child)
                     found_s3 = True
                     break # Found one for this child at this stage

            if found_s3: continue # This child's best stage is 3

            # Check Stage 4: makeable
            if needs_gf:
                if makeable_gf_slots > 0:
                    children_s4_potential.add(child)
                    can_make_s4 = True
            else: # Needs regular
                # Can use remaining makeable GF slots or makeable Reg slots
                if (makeable_gf_slots > 0 or makeable_reg_slots > 0):
                    children_s4_potential.add(child)
                    can_make_s4 = True

            if can_make_s4: continue # This child's best stage is 4

            # If we reach here, the child might be unservable with current resources
            # For this heuristic, we'll assign the highest cost stage (4)
            # This happens if no suitable sandwich exists anywhere and cannot be made
            # due to lack of ingredients or notexist objects.
            children_s4_potential.add(child)


        # 8. Calculate the number of children assigned to each stage (best stage)
        num_s1 = len(children_s1_potential)
        num_s2 = len(children_s2_potential - children_s1_potential)
        num_s3 = len(children_s3_potential - children_s1_potential - children_s2_potential)
        num_s4 = len(children_s4_potential - children_s1_potential - children_s2_potential - children_s3_potential)

        # 9. Compute the total heuristic value
        h = (num_s1 * 1) + (num_s2 * 2) + (num_s3 * 3) + (num_s4 * 4)

        # 10. Return the total heuristic value
        return h
