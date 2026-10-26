import heapq
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to reach the goal state by summing
    the estimated minimum cost for each unserved child. It prioritizes serving
    children using sandwiches that are already closest to their destination,
    assigning costs based on the stages required:
    1. Sandwich is already on a tray at the child's location (cost 1: serve).
    2. Sandwich is on a tray elsewhere (cost 2: move_tray + serve).
    3. Sandwich is in the kitchen (cost 3: put_on_tray + move_tray + serve).
    4. Sandwich needs to be made (cost 4: make + put_on_tray + move_tray + serve).
    It greedily assigns available sandwiches from the cheapest category first,
    prioritizing gluten-free sandwiches for allergic children. It checks for
    ingredient availability for sandwiches that need to be made and returns
    infinity if they cannot be produced.

    Assumptions:
    - The predicates 'waiting', 'allergic_gluten', 'not_allergic_gluten',
      'no_gluten_bread', and 'no_gluten_content' are static or effectively
      static for the purpose of determining child needs and ingredient types.
    - Tray movements are always possible between any two places if a tray exists.
    - Enough trays exist in total to perform necessary movements (checked partially).
    - Ingredient counts are sufficient for making sandwiches unless explicitly
      determined otherwise, in which case infinity is returned.
    - The goal is always a set of (served ?c) facts.

    Heuristic Initialization:
    The constructor parses the task definition to extract static information
    such as the set of all objects by type (children, trays, places, etc.),
    which children are allergic, which ingredients are gluten-free, and where
    each child is waiting. It also identifies the set of children that need
    to be served based on the goal state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of children who are in the goal state but are not yet
       marked as 'served' in the current state. These are the unserved children.
    2. Separate the unserved children into those who are allergic to gluten
       and those who are not, based on the static facts.
    3. Count the available resources in the current state:
       - Sandwiches on trays at each place, categorized by gluten status.
       - Sandwiches in the kitchen, categorized by gluten status.
       - Bread and content portions in the kitchen, categorized by gluten status.
       - 'notexist' facts for potential new sandwiches.
       - Trays at each place.
    4. Initialize the heuristic value to 0.
    5. Greedily assign available sandwiches to unserved children, prioritizing
       the cheapest source and GF sandwiches for allergic children:
       a. Count children who can be served by sandwiches already on a tray
          at their waiting place (Cost 1: serve). Mark these children and
          the used sandwiches/trays as covered.
       b. Count remaining children who can be served by sandwiches on trays
          at *other* places (Cost 2: move_tray + serve). Mark these children
          and used sandwiches/trays. Prioritize GF for allergic.
       c. Count remaining children who can be served by sandwiches currently
          in the kitchen (Cost 3: put_on_tray + move_tray + serve). Mark these
          children and used sandwiches. Prioritize GF for allergic. This step
          also implicitly requires trays to be available in the kitchen for
          the 'put_on_tray' action.
    6. Count the remaining children who still need a sandwich. These children
       require a sandwich to be made (Cost 4: make + put_on_tray + move_tray + serve).
    7. Check if there are enough ingredients (GF bread, GF content, Any bread,
       Any content, notexist facts) available in the kitchen to make the required
       number of GF and Any sandwiches for the remaining children. If not, the
       state is likely unsolvable through these actions, and the heuristic
       returns infinity.
    8. If ingredients are sufficient, calculate the total heuristic value by
       summing the costs for the children in each category identified in step 5
       and 6 (count * cost).
    9. The heuristic value is 0 if and only if there are no unserved children
       (i.e., the goal state is reached).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        # Helper to parse fact strings
        def parse_fact(fact_str):
            """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
            # Remove parentheses and split by space
            parts = fact_str.strip('()').split()
            return tuple(parts)

        # Extract all objects by type
        self.all_objects = set()
        for fact_str in task.initial_state | task.goals | task.static:
            parts = parse_fact(fact_str)
            for obj in parts[1:]:
                self.all_objects.add(obj)

        self.all_places = {obj for obj in self.all_objects if obj.startswith('table') or obj == 'kitchen'}
        self.all_children = {obj for obj in self.all_objects if obj.startswith('child')}
        self.all_trays = {obj for obj in self.all_objects if obj.startswith('tray')}
        self.all_sandwiches = {obj for obj in self.all_objects if obj.startswith('sandw')}
        self.all_bread = {obj for obj in self.all_objects if obj.startswith('bread')}
        self.all_content = {obj for obj in self.all_objects if obj.startswith('content')}

        # Extract static information
        self.waiting_children_at_place = {} # child -> place
        self.allergic_children = set()
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        for fact_str in task.static | task.initial_state:
            parts = parse_fact(fact_str)
            if parts[0] == 'waiting':
                self.waiting_children_at_place[parts[1]] = parts[2]
            elif parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                 # Also useful to know non-allergic children
                 pass # We can infer non-allergic from all_children - allergic_children
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_bread.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_content.add(parts[1])

        # Identify children who need to be served based on goals
        self.children_to_serve = {parse_fact(g)[1] for g in task.goals if parse_fact(g)[0] == 'served'}

    def __call__(self, node):
        state = node.state

        # Helper to parse fact strings (can be defined inside __call__ or as a method)
        def parse_fact(fact_str):
            """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
            parts = fact_str.strip('()').split()
            return tuple(parts)

        # 1. Identify unserved children
        unserved_children = [c for c in self.children_to_serve if '(served ' + c + ')' not in state]

        if not unserved_children:
            return 0 # Goal reached

        # Separate unserved children by allergy status
        unserved_allergic = [c for c in unserved_children if c in self.allergic_children]
        unserved_non_allergic = [c for c in unserved_children if c not in self.allergic_children]

        # 2. Count available resources in current state
        available_gf_sandwiches_ready_at_place = {p: [] for p in self.all_places}
        available_any_sandwiches_ready_at_place = {p: [] for p in self.all_places}
        available_gf_sandwiches_kitchen = []
        available_any_sandwiches_kitchen = []
        available_gf_sandwiches_ontray_elsewhere = []
        available_any_sandwiches_ontray_elsewhere = []
        available_gf_bread = []
        available_any_bread = []
        available_gf_content = []
        available_any_content = []
        available_notexist_sandwiches = []
        available_trays_at_place = {p: [] for p in self.all_places}

        sandwich_is_gf = {} # Map sandwich name to True/False
        sandwich_on_tray = {} # Map sandwich name to tray name (if on tray)
        tray_at_place = {} # Map tray name to place name

        # Populate basic location/status maps
        for fact_str in state:
            parts = parse_fact(fact_str)
            if parts[0] == 'at':
                if parts[1] in self.all_trays and parts[2] in self.all_places:
                     tray_at_place[parts[1]] = parts[2]
                     available_trays_at_place[parts[2]].append(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                if parts[1] in self.no_gluten_bread:
                    available_gf_bread.append(parts[1])
                else:
                    available_any_bread.append(parts[1])
            elif parts[0] == 'at_kitchen_content':
                if parts[1] in self.no_gluten_content:
                    available_gf_content.append(parts[1])
                else:
                    available_any_content.append(parts[1])
            elif parts[0] == 'at_kitchen_sandwich':
                sandwich_name = parts[1]
                # GF status is indicated by (no_gluten_sandwich S) fact
                if '(no_gluten_sandwich ' + sandwich_name + ')' in state:
                     sandwich_is_gf[sandwich_name] = True
                     available_gf_sandwiches_kitchen.append(sandwich_name)
                else:
                     sandwich_is_gf[sandwich_name] = False
                     available_any_sandwiches_kitchen.append(sandwich_name)
            elif parts[0] == 'ontray':
                sandwich_name = parts[1]
                tray_name = parts[2]
                sandwich_on_tray[sandwich_name] = tray_name
                # GF status is indicated by (no_gluten_sandwich S) fact
                if '(no_gluten_sandwich ' + sandwich_name + ')' in state:
                     sandwich_is_gf[sandwich_name] = True
                else:
                     sandwich_is_gf[sandwich_name] = False
            elif parts[0] == 'notexist':
                available_notexist_sandwiches.append(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                 # This fact confirms GF status for sandwiches already made
                 sandwich_name = parts[1]
                 sandwich_is_gf[sandwich_name] = True


        # Populate available_sandwiches_ready_at_place and available_sandwiches_ontray_elsewhere
        places_with_unserved_children = {self.waiting_children_at_place[c] for c in unserved_children}

        for s, t in sandwich_on_tray.items():
            if t in tray_at_place:
                p = tray_at_place[t]
                is_gf = sandwich_is_gf.get(s, False) # Default to False if GF status unknown (shouldn't happen for made sandwiches)

                if p in places_with_unserved_children:
                    # Sandwich is on a tray at a place where an unserved child is waiting
                    if is_gf:
                         available_gf_sandwiches_ready_at_place[p].append(s)
                    else:
                         available_any_sandwiches_ready_at_place[p].append(s)
                elif p != 'kitchen':
                    # Sandwich is on a tray elsewhere (not kitchen, not a target place)
                    if is_gf:
                         available_gf_sandwiches_ontray_elsewhere.append(s)
                    else:
                         available_any_sandwiches_ontray_elsewhere.append(s)
                # If p is kitchen, it's covered by available_sandwiches_kitchen (which is wrong, ontray is not at_kitchen_sandwich)
                # Sandwiches on trays in the kitchen are already counted in sandwich_on_tray, not at_kitchen_sandwich.
                # Let's correct the kitchen count:
                # available_gf_sandwiches_kitchen should only count (at_kitchen_sandwich S) where S is GF
                # Sandwiches on tray in kitchen should be a separate category, or added to ontray_elsewhere if we consider kitchen as just another place.
                # The action put_on_tray requires (at ?t kitchen). So sandwiches move from at_kitchen_sandwich to ontray ?t where ?t is at kitchen.
                # Let's count sandwiches on trays *in the kitchen* separately.
                # available_gf_sandwiches_ontray_kitchen = []
                # available_any_sandwiches_ontray_kitchen = []
                # ... repopulate ...
                # for s, t in sandwich_on_tray.items():
                #    if t in tray_at_place:
                #        p = tray_at_place[t]
                #        is_gf = sandwich_is_gf.get(s, False)
                #        if p in places_with_unserved_children: ... ready ...
                #        elif p == 'kitchen':
                #            if is_gf: available_gf_sandwiches_ontray_kitchen.append(s)
                #            else: available_any_sandwiches_ontray_kitchen.append(s)
                #        else: ... elsewhere ...

        # Let's simplify: group all sandwiches on trays NOT at a child's place together.
        available_gf_sandwiches_ontray_not_ready = []
        available_any_sandwiches_ontray_not_ready = []

        for s, t in sandwich_on_tray.items():
            if t in tray_at_place:
                p = tray_at_place[t]
                is_gf = sandwich_is_gf.get(s, False)
                if p in places_with_unserved_children:
                    if is_gf:
                         available_gf_sandwiches_ready_at_place[p].append(s)
                    else:
                         available_any_sandwiches_ready_at_place[p].append(s)
                else:
                    # Sandwich is on a tray, but not at a place where an unserved child is waiting
                    if is_gf:
                         available_gf_sandwiches_ontray_not_ready.append(s)
                    else:
                         available_any_sandwiches_ontray_not_ready.append(s)

        # 3. Greedy assignment and cost calculation
        h = 0
        used_ready_s = set()
        used_ready_t = set()
        used_ontray_not_ready_s = set()
        used_ontray_not_ready_t = set() # Need to track trays too for ontray_not_ready? Yes, move_tray uses the tray.
        used_kitchen_s = set()
        used_make_bread_gf = set()
        used_make_content_gf = set()
        used_make_bread_any = set()
        used_make_content_any = set()
        used_make_notexist = set()

        # Use lists for available resources so we can pop them
        avail_gf_ready_list = []
        avail_any_ready_list = []
        for p in places_with_unserved_children:
             avail_gf_ready_list.extend([(s, sandwich_on_tray[s]) for s in available_gf_sandwiches_ready_at_place.get(p, []) if s in sandwich_on_tray])
             avail_any_ready_list.extend([(s, sandwich_on_tray[s]) for s in available_any_sandwiches_ready_at_place.get(p, []) if s in sandwich_on_tray])

        avail_gf_ontray_not_ready_list = [(s, sandwich_on_tray[s]) for s in available_gf_sandwiches_ontray_not_ready if s in sandwich_on_tray]
        avail_any_ontray_not_ready_list = [(s, sandwich_on_tray[s]) for s in available_any_sandwiches_ontray_not_ready if s in sandwich_on_tray]

        avail_gf_kitchen_list = list(available_gf_sandwiches_kitchen)
        avail_any_kitchen_list = list(available_any_sandwiches_kitchen)

        avail_gf_bread_list = list(available_gf_bread)
        avail_any_bread_list = list(available_any_bread)
        avail_gf_content_list = list(available_gf_content)
        avail_any_content_list = list(available_any_content)
        avail_notexist_list = list(available_notexist_sandwiches)


        # Children needing sandwiches (copy list to iterate and remove)
        children_needing_sandwich = list(unserved_children)

        # Category 1: Served by ready sandwiches (Cost 1)
        served_by_ready_count = 0
        children_served_in_pass = []
        # Prioritize GF ready for allergic
        for child in children_needing_sandwich:
            if child in self.allergic_children:
                place = self.waiting_children_at_place[child]
                # Find a ready GF sandwich at this child's place
                found_s_t = None
                for s_t in avail_gf_ready_list:
                    s, t = s_t
                    if tray_at_place.get(t) == place and s not in used_ready_s and t not in used_ready_t:
                        found_s_t = s_t
                        break
                if found_s_t:
                    s, t = found_s_t
                    served_by_ready_count += 1
                    used_ready_s.add(s)
                    used_ready_t.add(t)
                    avail_gf_ready_list.remove(found_s_t)
                    children_served_in_pass.append(child)

        # Then use Any/remaining GF ready for non-allergic
        for child in children_needing_sandwich:
            if child not in self.allergic_children:
                place = self.waiting_children_at_place[child]
                found_s_t = None
                # Try Any ready first
                for s_t in avail_any_ready_list:
                    s, t = s_t
                    if tray_at_place.get(t) == place and s not in used_ready_s and t not in used_ready_t:
                        found_s_t = s_t
                        break
                if found_s_t:
                    s, t = found_s_t
                    served_by_ready_count += 1
                    used_ready_s.add(s)
                    used_ready_t.add(t)
                    avail_any_ready_list.remove(found_s_t)
                    children_served_in_pass.append(child)
                else:
                    # Try remaining GF ready next
                    found_s_t = None
                    for s_t in avail_gf_ready_list:
                        s, t = s_t
                        if tray_at_place.get(t) == place and s not in used_ready_s and t not in used_ready_t:
                            found_s_t = s_t
                            break
                    if found_s_t:
                        s, t = found_s_t
                        served_by_ready_count += 1
                        used_ready_s.add(s)
                        used_ready_t.add(t)
                        avail_gf_ready_list.remove(found_s_t)
                        children_served_in_pass.append(child)

        # Remove children served in this pass
        for child in children_served_in_pass:
             children_needing_sandwich.remove(child)

        h += served_by_ready_count * 1

        # Category 2: Served by sandwiches on tray elsewhere (Cost 2)
        served_by_elsewhere_count = 0
        children_served_in_pass = []
        # Prioritize GF elsewhere for remaining allergic
        for child in children_needing_sandwich:
            if child in self.allergic_children:
                found_s_t = None
                for s_t in avail_gf_ontray_not_ready_list:
                    s, t = s_t
                    if s not in used_ontray_not_ready_s and t not in used_ontray_not_ready_t:
                        found_s_t = s_t
                        break
                if found_s_t:
                    s, t = found_s_t
                    served_by_elsewhere_count += 1
                    used_ontray_not_ready_s.add(s)
                    used_ontray_not_ready_t.add(t)
                    avail_gf_ontray_not_ready_list.remove(found_s_t)
                    children_served_in_pass.append(child)

        # Then use Any/remaining GF elsewhere for remaining non-allergic
        for child in children_needing_sandwich:
            if child not in self.allergic_children:
                found_s_t = None
                # Try Any elsewhere first
                for s_t in avail_any_ontray_not_ready_list:
                    s, t = s_t
                    if s not in used_ontray_not_ready_s and t not in used_ontray_not_ready_t:
                        found_s_t = s_t
                        break
                if found_s_t:
                    s, t = found_s_t
                    served_by_elsewhere_count += 1
                    used_ontray_not_ready_s.add(s)
                    used_ontray_not_ready_t.add(t)
                    avail_any_ontray_not_ready_list.remove(found_s_t)
                    children_served_in_pass.append(child)
                else:
                    # Try remaining GF elsewhere next
                    found_s_t = None
                    for s_t in avail_gf_ontray_not_ready_list:
                        s, t = s_t
                        if s not in used_ontray_not_ready_s and t not in used_ontray_not_ready_t:
                            found_s_t = s_t
                            break
                    if found_s_t:
                        s, t = found_s_t
                        served_by_elsewhere_count += 1
                        used_ontray_not_ready_s.add(s)
                        used_ontray_not_ready_t.add(t)
                        avail_gf_ontray_not_ready_list.remove(found_s_t)
                        children_served_in_pass.append(child)

        # Remove children served in this pass
        for child in children_served_in_pass:
             children_needing_sandwich.remove(child)

        h += served_by_elsewhere_count * 2

        # Category 3: Served by sandwiches in kitchen (Cost 3)
        served_by_kitchen_count = 0
        children_served_in_pass = []
        # Prioritize GF kitchen for remaining allergic
        for child in children_needing_sandwich:
            if child in self.allergic_children:
                found_s = None
                for s in avail_gf_kitchen_list:
                    if s not in used_kitchen_s:
                        found_s = s
                        break
                if found_s:
                    served_by_kitchen_count += 1
                    used_kitchen_s.add(found_s)
                    avail_gf_kitchen_list.remove(found_s)
                    children_served_in_pass.append(child)

        # Then use Any/remaining GF kitchen for remaining non-allergic
        for child in children_needing_sandwich:
            if child not in self.allergic_children:
                found_s = None
                # Try Any kitchen first
                for s in avail_any_kitchen_list:
                    if s not in used_kitchen_s:
                        found_s = s
                        break
                if found_s:
                    served_by_kitchen_count += 1
                    used_kitchen_s.add(found_s)
                    avail_any_kitchen_list.remove(found_s)
                    children_served_in_pass.append(child)
                else:
                    # Try remaining GF kitchen next
                    found_s = None
                    for s in avail_gf_kitchen_list:
                        if s not in used_kitchen_s:
                            found_s = s
                            break
                    if found_s:
                        served_by_kitchen_count += 1
                        used_kitchen_s.add(found_s)
                        avail_gf_kitchen_list.remove(found_s)
                        children_served_in_pass.append(child)

        # Remove children served in this pass
        for child in children_served_in_pass:
             children_needing_sandwich.remove(child)

        h += served_by_kitchen_count * 3

        # Category 4: Served by making sandwiches (Cost 4)
        children_needing_make = list(children_needing_sandwich) # Remaining children

        needed_make_gf = len([c for c in children_needing_make if c in self.allergic_children])
        needed_make_any = len([c for c in children_needing_make if c not in self.allergic_children])

        # Check ingredient availability for making
        can_make_gf_count = min(len(avail_gf_bread_list), len(avail_gf_content_list), len(avail_notexist_list))
        actual_make_gf = min(needed_make_gf, can_make_gf_count)

        # Remaining ingredients after making GF
        rem_gf_bread = len(avail_gf_bread_list) - actual_make_gf
        rem_gf_content = len(avail_gf_content_list) - actual_make_gf
        rem_notexist = len(avail_notexist_list) - actual_make_gf

        avail_any_bread_for_any = len(avail_any_bread_list) + rem_gf_bread
        avail_any_content_for_any = len(avail_any_content_list) + rem_gf_content

        can_make_any_count = min(avail_any_bread_for_any, avail_any_content_for_any, rem_notexist)
        actual_make_any = min(needed_make_any, can_make_any_count)

        # If we cannot make all needed sandwiches, return infinity
        if actual_make_gf < needed_make_gf or actual_make_any < needed_make_any:
            return float('inf')

        served_by_make_count = needed_make_gf + needed_make_any
        h += served_by_make_count * 4

        # The total number of children served by these categories should equal the initial number of unserved children
        # assert served_by_ready_count + served_by_elsewhere_count + served_by_kitchen_count + served_by_make_count == len(unserved_children)

        return h

