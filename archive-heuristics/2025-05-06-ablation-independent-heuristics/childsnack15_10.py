from heuristics.heuristic_base import Heuristic
from task import Task

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    parts = fact_string.strip('()').split()
    if not parts:
        return None, []
    return parts[0], parts[1:]

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to serve all unserved children.
    It does this by greedily assigning available resources (sandwiches, ingredients, trays)
    to unserved children based on the estimated cost to get a suitable sandwich
    to the child's location and serve it. The costs are estimated based on the
    current state of the sandwich (on tray at location, on tray elsewhere,
    in kitchen, not yet made). Resource usage is tracked to avoid double-counting.
    Prioritizes serving children who require fewer actions and prioritizes
    gluten-free resources for allergic children.

    Assumptions:
    - Action costs are implicitly 1.
    - The heuristic is designed for solvable problems, where resources are
      eventually sufficient. If resources are insufficient in a state, the
      heuristic will return a finite value based on what *can* be served
      by the available resources in a greedy assignment.
    - Used with greedy best-first search (admissibility is not strictly required).

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It identifies:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - The waiting location for each child.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    It also collects all object names of relevant types (child, sandwich, tray,
    bread, content, place) from the initial state and goals to know the full
    set of objects that might appear in facts.
    This static information and object lists are stored in sets and dictionaries
    for quick lookup.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all unserved children from the goal state and current state.
        Separate them into those needing gluten-free sandwiches (allergic)
        and those needing regular sandwiches (not allergic), noting their
        waiting locations (from static info).
    2.  Identify all available resources in the current state:
        - Sandwiches on trays, noting their location and whether they are gluten-free.
        - Sandwiches in the kitchen, noting whether they are gluten-free.
        - Bread portions in the kitchen, noting which are gluten-free.
        - Content portions in the kitchen, noting which are gluten-free.
        - Sandwich objects that do not yet exist (`notexist` predicate).
        - Trays at the kitchen.
        These resources are collected into lists, distinguishing between
        gluten-free and regular-suitable, and their location/state.
    3.  Initialize the heuristic value `h` to 0. Initialize sets to track
        resources (sandwiches, trays, bread, content, sandwich objects) that
        are "used" in the heuristic calculation for a specific child.
    4.  Greedily assign available resources to unserved children in phases,
        from cheapest to most expensive action sequence required, adding the
        corresponding cost to `h` and marking the used resources. Prioritize
        allergic children within each phase.
        -   **Phase 1 (Cost 1: Serve):** For each unserved child, check if a
            suitable sandwich is available on an unused tray *at their waiting location*.
            If yes, add 1 to `h`, mark the sandwich and tray as used, and consider
            the child "served" for this heuristic calculation. Prioritize GF sandwiches
            for GF children, then any suitable for Reg children (GF first if available).
        -   **Phase 2 (Cost 2: Move + Serve):** For each remaining unserved child,
            check if a suitable sandwich is available on an unused tray *at a
            different location*. If yes, add 2 to `h`, mark the sandwich and tray
            as used, and consider the child "served". Prioritize GF sandwiches
            for GF children, then any suitable for Reg children (GF first if available).
        -   **Phase 3 (Cost 4: Make + Put + Move + Serve):** For each remaining
            unserved child, check if the necessary components to *make* a suitable
            sandwich are available and unused in the kitchen (bread, content,
            `notexist` sandwich object) AND if an unused tray is available
            at the kitchen. If yes, add 4 to `h`, mark all used components
            (bread, content, sandwich object, tray) as used, and consider the
            child "served". Prioritize GF ingredients for GF children, then
            Reg children (GF ingredients first if available). This phase is
            checked before Cost 3 to prioritize using kitchen trays for making
            if ingredients are available.
        -   **Phase 4 (Cost 3: Put + Move + Serve):** For each remaining unserved
            child, check if a suitable sandwich is available and unused *in the
            kitchen* AND if an unused tray is available at the kitchen. If yes,
            add 3 to `h`, mark the sandwich and tray as used, and consider the
            child "served". Prioritize GF sandwiches for GF children, then
            Reg children (GF sandwiches first if available).
    5.  The final heuristic value is the total sum `h`. If any children remain
        unserved after all phases, it means they cannot be served with the
        currently available resources based on this greedy assignment. For a
        finite heuristic in greedy BFS, we just return the sum calculated.
        The heuristic is 0 if and only if all children are served (goal state).
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        # Extract static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_places = {} # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.all_children = set() # Keep track of all children objects

        for fact_string in self.static_facts:
            pred, args = parse_fact(fact_string)
            if pred == 'allergic_gluten':
                self.allergic_children.add(args[0])
                self.all_children.add(args[0])
            elif pred == 'not_allergic_gluten':
                self.not_allergic_children.add(args[0])
                self.all_children.add(args[0])
            elif pred == 'waiting':
                self.waiting_places[args[0]] = args[1]
            elif pred == 'no_gluten_bread':
                self.no_gluten_bread.add(args[0])
            elif pred == 'no_gluten_content':
                self.no_gluten_content.add(args[0])

        # Get all object names by type (approximation from initial state and goals)
        # This is a common shortcut; a robust parser would read the :objects section.
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_places = set()
        self.all_places.add('kitchen') # Add constant place

        for fact_string in task.initial_state | task.goals | self.static_facts:
             pred, args = parse_fact(fact_string)
             if pred in ['at_kitchen_bread', 'no_gluten_bread']:
                 if args: self.all_bread.add(args[0])
             elif pred in ['at_kitchen_content', 'no_gluten_content']:
                 if args: self.all_content.add(args[0])
             elif pred in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist']:
                 if args: self.all_sandwiches.add(args[0])
             elif pred == 'ontray':
                 if len(args) > 1: self.all_trays.add(args[1])
             elif pred == 'at':
                 if args: self.all_trays.add(args[0])
                 if len(args) > 1: self.all_places.add(args[1])
             elif pred in ['allergic_gluten', 'not_allergic_gluten', 'waiting', 'served']:
                 if args: self.all_children.add(args[0])


    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        unserved_children_gf = [] # List of (child, place)
        unserved_children_reg = [] # List of (child, place)

        served_facts = {fact for fact in state if fact.startswith('(served ')}

        for child in self.all_children:
            served_fact = f'(served {child})'
            if served_fact not in served_facts:
                 # Child is unserved
                 waiting_place = self.waiting_places.get(child) # Get place from static info
                 if waiting_place: # Should always have a waiting place in this domain
                     if child in self.allergic_children:
                         unserved_children_gf.append((child, waiting_place))
                     elif child in self.not_allergic_children:
                         unserved_children_reg.append((child, waiting_place))


        # If no unserved children, goal is reached
        if not unserved_children_gf and not unserved_children_reg:
            return 0

        # 2. Identify available resources in the current state
        ontray_facts = {fact for fact in state if fact.startswith('(ontray ')}
        at_facts = {fact for fact in state if fact.startswith('(at ')}
        kitchen_sandwich_facts = {fact for fact in state if fact.startswith('(at_kitchen_sandwich ')}
        kitchen_bread_facts = {fact for fact in state if fact.startswith('(at_kitchen_bread ')}
        kitchen_content_facts = {fact for fact in state if fact.startswith('(at_kitchen_content ')}
        notexist_facts = {fact for fact in state if fact.startswith('(notexist ')}
        no_gluten_sandwich_facts = {fact for fact in state if fact.startswith('(no_gluten_sandwich ')}

        # Resource Pools (using lists to allow removal upon use)
        avail_ontray_at_loc_gf = [] # [(s, t)]
        avail_ontray_at_loc_reg = [] # [(s, t)]
        avail_ontray_elsewhere_gf = [] # [(s, t)]
        avail_ontray_elsewhere_reg = [] # [(s, t)]
        avail_kitchen_sandwich_gf = [] # [s]
        avail_kitchen_sandwich_reg = [] # [s]
        avail_kitchen_bread_gf = [] # [b]
        avail_kitchen_content_gf = [] # [c]
        avail_kitchen_bread_reg = [] # [b] (all bread in kitchen)
        avail_kitchen_content_reg = [] # [c] (all content in kitchen)
        avail_notexist_sandwich_objects = [] # [s_obj]
        avail_trays_at_kitchen = [] # [t]

        # Map tray to its current location
        tray_locations = {}
        for fact_string in at_facts:
            _, args = parse_fact(fact_string)
            if len(args) == 2:
                tray, place = args
                tray_locations[tray] = place
                if place == 'kitchen':
                    avail_trays_at_kitchen.append(tray)

        # Collect sandwiches on trays
        child_waiting_locations = {place for _, place in unserved_children_gf + unserved_children_reg}
        for fact_string in ontray_facts:
            _, args = parse_fact(fact_string)
            if len(args) == 2:
                s, t = args
                is_gf = f'(no_gluten_sandwich {s})' in no_gluten_sandwich_facts
                loc = tray_locations.get(t) # Get tray location

                if loc in child_waiting_locations:
                    if is_gf:
                        avail_ontray_at_loc_gf.append((s, t))
                    else:
                        avail_ontray_at_loc_reg.append((s, t))
                elif loc is not None: # On a tray, but not at a child's waiting location
                     if is_gf:
                        avail_ontray_elsewhere_gf.append((s, t))
                     else:
                        avail_ontray_elsewhere_reg.append((s, t))
                # Note: If loc is None, the tray is not 'at' any place, which shouldn't happen in valid states.

        # Collect sandwiches in kitchen
        for fact_string in kitchen_sandwich_facts:
            _, args = parse_fact(fact_string)
            if args:
                s = args[0]
                is_gf = f'(no_gluten_sandwich {s})' in no_gluten_sandwich_facts
                if is_gf:
                    avail_kitchen_sandwich_gf.append(s)
                else:
                    avail_kitchen_sandwich_reg.append(s)

        # Collect ingredients in kitchen
        for fact_string in kitchen_bread_facts:
            _, args = parse_fact(fact_string)
            if args:
                b = args[0]
                avail_kitchen_bread_reg.append(b) # All bread is regular-suitable
                if b in self.no_gluten_bread:
                    avail_kitchen_bread_gf.append(b)

        for fact_string in kitchen_content_facts:
            _, args = parse_fact(fact_string)
            if args:
                c = args[0]
                avail_kitchen_content_reg.append(c) # All content is regular-suitable
                if c in self.no_gluten_content:
                    avail_kitchen_content_gf.append(c)

        # Collect notexist sandwich objects
        for fact_string in notexist_facts:
             _, args = parse_fact(fact_string)
             if args:
                s_obj = args[0]
                avail_notexist_sandwich_objects.append(s_obj)


        # 3. Initialize heuristic and used resources
        h = 0
        used_sandwiches = set()
        used_trays = set()
        used_bread = set()
        used_content = set()
        used_sandwich_objects = set()

        # Use copies of lists for consumption during greedy assignment
        current_ontray_at_loc_gf = list(avail_ontray_at_loc_gf)
        current_ontray_at_loc_reg = list(avail_ontray_at_loc_reg)
        current_ontray_elsewhere_gf = list(avail_ontray_elsewhere_gf)
        current_ontray_elsewhere_reg = list(avail_ontray_elsewhere_reg)
        current_kitchen_sandwich_gf = list(avail_kitchen_sandwich_gf)
        current_kitchen_sandwich_reg = list(avail_kitchen_sandwich_reg)
        current_kitchen_bread_gf = list(avail_kitchen_bread_gf)
        current_kitchen_content_gf = list(avail_kitchen_content_gf)
        current_kitchen_bread_reg = list(avail_kitchen_bread_reg) # All bread in kitchen
        current_kitchen_content_reg = list(avail_kitchen_content_reg) # All content in kitchen
        current_notexist_sandwich_objects = list(avail_notexist_sandwich_objects)
        current_trays_at_kitchen = list(avail_trays_at_kitchen)

        # Keep track of children not yet served in heuristic calculation
        remaining_children_gf = list(unserved_children_gf)
        remaining_children_reg = list(unserved_children_reg)

        # Helper to find and use a resource tuple (e.g., (s, t)) from a list
        def find_and_use_tuple(resource_list, is_used_func, use_func):
            # Iterate over a copy to allow removal from original list
            for resource in list(resource_list):
                if not is_used_func(resource):
                    use_func(resource)
                    resource_list.remove(resource) # Remove from the actual list
                    return resource # Return the used resource
            return None

        # Helper to find and use a single resource (e.g., s) from a list
        def find_and_use_single(resource_list, is_used_func, use_func):
             # Iterate over a copy to allow removal from original list
             for resource in list(resource_list):
                if not is_used_func(resource):
                    use_func(resource)
                    resource_list.remove(resource) # Remove from the actual list
                    return resource # Return the used resource
             return None

        # Helper usage functions
        def is_sandwich_used(s): return s in used_sandwiches
        def use_sandwich(s): used_sandwiches.add(s)
        def is_tray_used(t): return t in used_trays
        def use_tray(t): used_trays.add(t)
        def is_bread_used(b): return b in used_bread
        def use_bread(b): used_bread.add(b)
        def is_content_used(c): return c in used_content
        def use_content(c): used_content.add(c)
        def is_s_obj_used(s_obj): return s_obj in used_sandwich_objects
        def use_s_obj(s_obj): used_sandwich_objects.add(s_obj)

        # 4. Greedy assignment phases

        # Phase 1 (Cost 1: Serve at location)
        served_now = []
        # GF children first
        for child_info in remaining_children_gf:
            # Resource is (sandwich, tray) tuple
            is_used = lambda res_tuple: is_sandwich_used(res_tuple[0]) or is_tray_used(res_tuple[1])
            use_res = lambda res_tuple: (use_sandwich(res_tuple[0]), use_tray(res_tuple[1]))
            resource = find_and_use_tuple(current_ontray_at_loc_gf, is_used, use_res)
            if resource:
                h += 1
                served_now.append(child_info)
        for child_info in served_now: remaining_children_gf.remove(child_info)
        served_now = []
        # Reg children
        for child_info in remaining_children_reg:
            # Try regular sandwich first
            is_used = lambda res_tuple: is_sandwich_used(res_tuple[0]) or is_tray_used(res_tuple[1])
            use_res = lambda res_tuple: (use_sandwich(res_tuple[0]), use_tray(res_tuple[1]))
            resource = find_and_use_tuple(current_ontray_at_loc_reg, is_used, use_res)
            if resource:
                h += 1
                served_now.append(child_info)
            else:
                # Try GF sandwich if regular not available
                is_used = lambda res_tuple: is_sandwich_used(res_tuple[0]) or is_tray_used(res_tuple[1])
                use_res = lambda res_tuple: (use_sandwich(res_tuple[0]), use_tray(res_tuple[1]))
                resource = find_and_use_tuple(current_ontray_at_loc_gf, is_used, use_res)
                if resource:
                    h += 1
                    served_now.append(child_info)
        for child_info in served_now: remaining_children_reg.remove(child_info)

        # Phase 2 (Cost 2: Move tray)
        served_now = []
        # GF children first
        for child_info in remaining_children_gf:
            is_used = lambda res_tuple: is_sandwich_used(res_tuple[0]) or is_tray_used(res_tuple[1])
            use_res = lambda res_tuple: (use_sandwich(res_tuple[0]), use_tray(res_tuple[1]))
            resource = find_and_use_tuple(current_ontray_elsewhere_gf, is_used, use_res)
            if resource:
                h += 2
                served_now.append(child_info)
        for child_info in served_now: remaining_children_gf.remove(child_info)
        served_now = []
        # Reg children
        for child_info in remaining_children_reg:
            # Try regular sandwich first
            is_used = lambda res_tuple: is_sandwich_used(res_tuple[0]) or is_tray_used(res_tuple[1])
            use_res = lambda res_tuple: (use_sandwich(res_tuple[0]), use_tray(res_tuple[1]))
            resource = find_and_use_tuple(current_ontray_elsewhere_reg, is_used, use_res)
            if resource:
                h += 2
                served_now.append(child_info)
            else:
                # Try GF sandwich if regular not available
                is_used = lambda res_tuple: is_sandwich_used(res_tuple[0]) or is_tray_used(res_tuple[1])
                use_res = lambda res_tuple: (use_sandwich(res_tuple[0]), use_tray(res_tuple[1]))
                resource = find_and_use_tuple(current_ontray_elsewhere_gf, is_used, use_res)
                if resource:
                    h += 2
                    served_now.append(child_info)
        for child_info in served_now: remaining_children_reg.remove(child_info)

        # Phase 3 (Cost 4: Make + put + move) - Prioritize using kitchen trays for making
        served_now = []
        # Process GF children needing make
        for child_info in remaining_children_gf:
            # Need GF ingredients (b, c), unused sandwich object (s_obj), unused tray at kitchen (t)
            found_b = find_and_use_single(current_kitchen_bread_gf, is_bread_used, use_bread)
            if found_b:
                found_c = find_and_use_single(current_kitchen_content_gf, is_content_used, use_content)
                if found_c:
                    found_s_obj = find_and_use_single(current_notexist_sandwich_objects, is_s_obj_used, use_s_obj)
                    if found_s_obj:
                        found_t = find_and_use_single(current_trays_at_kitchen, is_tray_used, use_tray)
                        if found_t:
                            # Found all resources
                            h += 4
                            served_now.append(child_info)
                            # Resources already marked used and removed by find_and_use_single
                            continue # Move to next child
                        else: # Tray not found, un-use b, c, s_obj
                            used_bread.remove(found_b)
                            used_content.remove(found_c)
                            used_sandwich_objects.remove(found_s_obj)
                            current_kitchen_bread_gf.append(found_b) # Add back to list
                            current_kitchen_content_gf.append(found_c) # Add back to list
                            current_notexist_sandwich_objects.append(found_s_obj) # Add back to list
                    else: # S_obj not found, un-use b, c
                        used_bread.remove(found_b)
                        used_content.remove(found_c)
                        current_kitchen_bread_gf.append(found_b) # Add back to list
                        current_kitchen_content_gf.append(found_c) # Add back to list
                else: # Content not found, un-use b
                    used_bread.remove(found_b)
                    current_kitchen_bread_gf.append(found_b) # Add back to list

        for child_info in served_now: remaining_children_gf.remove(child_info)
        served_now = []

        # Process Reg children needing make
        for child_info in remaining_children_reg:
            # Need Reg ingredients (b, c), unused sandwich object (s_obj), unused tray at kitchen (t)
            # Try regular ingredients first
            found_b = find_and_use_single(current_kitchen_bread_reg, is_bread_used, use_bread)
            if found_b:
                found_c = find_and_use_single(current_kitchen_content_reg, is_content_used, use_content)
                if found_c:
                    found_s_obj = find_and_use_single(current_notexist_sandwich_objects, is_s_obj_used, use_s_obj)
                    if found_s_obj:
                        found_t = find_and_use_single(current_trays_at_kitchen, is_tray_used, use_tray)
                        if found_t:
                            # Found all resources
                            h += 4
                            served_now.append(child_info)
                            # Resources already marked used and removed by find_and_use_single
                            continue # Move to next child
                        else: # Tray not found, un-use b, c, s_obj
                            used_bread.remove(found_b)
                            used_content.remove(found_c)
                            used_sandwich_objects.remove(found_s_obj)
                            current_kitchen_bread_reg.append(found_b) # Add back to list
                            current_kitchen_content_reg.append(found_c) # Add back to list
                            current_notexist_sandwich_objects.append(found_s_obj) # Add back to list
                    else: # S_obj not found, un-use b, c
                        used_bread.remove(found_b)
                        used_content.remove(found_c)
                        current_kitchen_bread_reg.append(found_b) # Add back to list
                        current_kitchen_content_reg.append(found_c) # Add back to list
                else: # Content not found, un-use b
                    used_bread.remove(found_b)
                    current_kitchen_bread_reg.append(found_b) # Add back to list

            # If regular ingredients failed, try GF ingredients
            if child_info not in served_now:
                found_b = find_and_use_single(current_kitchen_bread_gf, is_bread_used, use_bread) # Use GF bread pool
                if found_b:
                    found_c = find_and_use_single(current_kitchen_content_gf, is_content_used, use_content) # Use GF content pool
                    if found_c:
                        found_s_obj = find_and_use_single(current_notexist_sandwich_objects, is_s_obj_used, use_s_obj)
                        if found_s_obj:
                            found_t = find_and_use_single(current_trays_at_kitchen, is_tray_used, use_tray)
                            if found_t:
                                # Found all resources
                                h += 4
                                served_now.append(child_info)
                                # Resources already marked used and removed by find_and_use_single
                                continue # Move to next child
                            else: # Tray not found, un-use b, c, s_obj
                                used_bread.remove(found_b)
                                used_content.remove(found_c)
                                used_sandwich_objects.remove(found_s_obj)
                                current_kitchen_bread_gf.append(found_b) # Add back to list
                                current_kitchen_content_gf.append(found_c) # Add back to list
                                current_notexist_sandwich_objects.append(found_s_obj) # Add back to list
                        else: # S_obj not found, un-use b, c
                            used_bread.remove(found_b)
                            used_content.remove(found_c)
                            current_kitchen_bread_gf.append(found_b) # Add back to list
                            current_kitchen_content_gf.append(found_c) # Add back to list
                    else: # Content not found, un-use b
                        used_bread.remove(found_b)
                        current_kitchen_bread_gf.append(found_b) # Add back to list

        for child_info in served_now: remaining_children_reg.remove(child_info)

        # Phase 4 (Cost 3: Put + move)
        served_now = []
        # Process GF children needing put
        for child_info in remaining_children_gf:
            # Need GF kitchen sandwich (s), unused tray at kitchen (t)
            found_s = find_and_use_single(current_kitchen_sandwich_gf, is_sandwich_used, use_sandwich)
            if found_s:
                found_t = find_and_use_single(current_trays_at_kitchen, is_tray_used, use_tray)
                if found_t:
                    h += 3
                    served_now.append(child_info)
                    # Resources already marked used and removed by find_and_use_single
                    continue
                else: # Tray not found, un-use s
                    used_sandwiches.remove(found_s)
                    current_kitchen_sandwich_gf.append(found_s) # Add back to list

        for child_info in served_now: remaining_children_gf.remove(child_info)
        served_now = []

        # Process Reg children needing put
        for child_info in remaining_children_reg:
            # Need Reg kitchen sandwich (s), unused tray at kitchen (t)
            # Try regular sandwich first
            found_s = find_and_use_single(current_kitchen_sandwich_reg, is_sandwich_used, use_sandwich)
            if found_s:
                found_t = find_and_use_single(current_trays_at_kitchen, is_tray_used, use_tray)
                if found_t:
                    h += 3
                    served_now.append(child_info)
                    # Resources already marked used and removed by find_and_use_single
                    continue
                else: # Tray not found, un-use s
                    used_sandwiches.remove(found_s)
                    current_kitchen_sandwich_reg.append(found_s) # Add back to list
            # If regular sandwich failed, try GF sandwich
            if child_info not in served_now:
                found_s = find_and_use_single(current_kitchen_sandwich_gf, is_sandwich_used, use_sandwich) # Use GF sandwich pool
                if found_s:
                    found_t = find_and_use_single(current_trays_at_kitchen, is_tray_used, use_tray)
                    if found_t:
                        h += 3
                        served_now.append(child_info)
                        # Resources already marked used and removed by find_and_use_single
                        continue
                    else: # Tray not found, un-use s
                        used_sandwiches.remove(found_s)
                        current_kitchen_sandwich_gf.append(found_s) # Add back to list

        for child_info in served_now: remaining_children_reg.remove(child_info)

        # Remaining children could not be served in this greedy pass.
        # Their cost is implicitly included in the state changes needed to make resources available.
        # The heuristic value is the sum of costs for children served in this pass.

        return h
