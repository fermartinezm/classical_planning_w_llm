from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are waiting and not yet served. It calculates the total number of
    sandwiches (gluten-free and regular) needed based on the unserved children's
    dietary requirements. It then counts the available sandwiches in different
    states (not made, in kitchen, on tray elsewhere, on tray at the child's location)
    and assigns a cost to each based on the minimum number of actions needed to get
    it to the served state (make, put on tray, move tray, serve). The heuristic
    sums these minimum costs, prioritizing the use of gluten-free sandwiches for
    allergic children and using sandwiches that are closer to being served first.
    It also includes basic checks for resource availability (sandwich slots and
    ingredients) to return infinity for clearly unsolvable states.

    # Assumptions
    - All children specified in the goal predicate `(served ?c)` who are not yet
      served are assumed to be currently `(waiting ?c ?p)` at some location `?p`.
    - Ingredient and sandwich object resources, if available in the initial state,
      are sufficient to make any required sandwiches, unless explicitly counted
      and found insufficient.
    - Trays can hold any number of sandwiches (implicit, as `ontray` is a predicate
      per sandwich, not a capacity).
    - A tray can be moved between any two places.
    - Gluten-free ingredients can be used to make regular sandwiches if needed.

    # Heuristic Initialization
    The heuristic extracts the following information from the task definition:
    - The set of all children, trays, places, bread, content, and sandwich objects.
    - The set of static facts, specifically identifying which children are allergic
      to gluten and which bread/content portions are gluten-free.
    - The set of children who need to be served according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are in the goal state but are not yet served.
    2. For each unserved child, determine their waiting location and whether they
       require a gluten-free sandwich (based on static allergy facts).
    3. Calculate the total number of gluten-free and regular sandwiches needed
       (`total_needed_gf`, `total_needed_reg`) based on the unserved children's
       requirements.
    4. Identify all sandwich objects defined in the problem instance.
    5. Categorize each sandwich object based on its current state in the given state:
       - `not_made`: `(notexist s)` is true.
       - `in_kitchen`: `(at_kitchen_sandwich s)` is true.
       - `on_tray_elsewhere`: `(ontray s t)` is true for some tray `t`, and tray `t`
         is `(at t p)` where `p` is *not* a location where unserved children are waiting.
       - `on_tray_at_place`: `(ontray s t)` is true for some tray `t`, and tray `t`
         is `(at t p)` where `p` *is* a location where unserved children are waiting.
       Also, determine if the sandwich is gluten-free (`(no_gluten_sandwich s)`).
       Count the number of sandwiches in each state category, separated by type (GF/Reg).
    6. Calculate the number of gluten-free and regular sandwiches that still need
       to be made (`to_make_gf`, `to_make_reg`) by comparing the total needed
       sandwiches with the total number of made sandwiches (in kitchen or on trays).
       Prioritize satisfying GF needs with available made GF sandwiches first.
    7. Check for basic unsolvability:
       - If the total number of sandwiches to be made (`to_make_gf + to_make_reg`)
         exceeds the number of available `notexist` sandwich objects, return infinity.
       - Count available gluten-free and regular bread/content portions in the kitchen.
       - Check if enough ingredients exist to make `to_make_gf` and `to_make_reg`
         sandwiches. If not, return infinity. (Assumes GF ingredients can be used
         for regular sandwiches if needed).
    8. Calculate the heuristic cost by assigning the `total_needed_gf` and
       `total_needed_reg` sandwich requirements to the available sandwiches,
       starting from the ones that require the fewest actions to be served:
       - Cost 1: Sandwiches `on_tray_at_place`. Requires 1 `serve` action.
       - Cost 2: Sandwiches `on_tray_elsewhere`. Requires 1 `move_tray` + 1 `serve` actions.
       - Cost 3: Sandwiches `in_kitchen`. Requires 1 `put_on_tray` + 1 `move_tray` + 1 `serve` actions.
       - Cost 4: Sandwiches `not_made` that need to be made (`to_make_gf`, `to_make_reg`).
         Requires 1 `make_sandwich` + 1 `put_on_tray` + 1 `move_tray` + 1 `serve` actions.
       Assign sandwiches greedily from lowest cost to highest cost, prioritizing
       GF sandwiches for GF needs, then remaining GF for Reg needs, then Reg for Reg needs.
    9. The total heuristic value is the sum of the costs for all assigned sandwiches.
       If the state is a goal state (all children to serve are served), the total
       needed sandwiches will be zero, and the heuristic will correctly return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.task_objects = task.objects
        self.static_facts_set = set(task.static)
        self.goals = task.goals

        # Extract objects by type
        self.all_children = {obj for obj, type in self.task_objects if type == 'child'}
        self.all_trays = {obj for obj, type in self.task_objects if type == 'tray'}
        self.all_places = {obj for obj, type in self.task_objects if type == 'place'}
        self.all_sandwiches = {obj for obj, type in self.task_objects if type == 'sandwich'}
        self.all_bread = {obj for obj, type in self.task_objects if type == 'bread-portion'}
        self.all_content = {obj for obj, type in self.task_objects if type == 'content-portion'}

        # Extract static information
        self.allergic_children_facts_set = {f for f in self.static_facts_set if f.startswith('(allergic_gluten ')}
        self.not_allergic_children_facts_set = {f for f in self.static_facts_set if f.startswith('(not_allergic_gluten ')}
        self.no_gluten_bread_set = {get_parts(f)[1] for f in self.static_facts_set if f.startswith('(no_gluten_bread ')}
        self.no_gluten_content_set = {get_parts(f)[1] for f in self.static_facts_set if f.startswith('(no_gluten_content ')}

        # Identify children who need to be served from the goal
        self.children_to_serve = {get_parts(f)[1] for f in self.goals if get_parts(f)[0] == 'served'}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state_set = set(node.state)

        # Create lookup structures for state facts
        served_children_set = {get_parts(f)[1] for f in state_set if f.startswith('(served ')}
        waiting_children_dict = {get_parts(f)[1]: get_parts(f)[2] for f in state_set if f.startswith('(waiting ')}
        at_kitchen_sandwich_set = {get_parts(f)[1] for f in state_set if f.startswith('(at_kitchen_sandwich ')}
        no_gluten_sandwich_set = {get_parts(f)[1] for f in state_set if f.startswith('(no_gluten_sandwich ')}
        notexist_sandwich_set = {get_parts(f)[1] for f in state_set if f.startswith('(notexist ')}
        ontray_dict = {get_parts(f)[1]: get_parts(f)[2] for f in state_set if f.startswith('(ontray ')} # sandwich -> tray
        tray_locations_dict = {get_parts(f)[1]: get_parts(f)[2] for f in state_set if f.startswith('(at tray')}
        at_kitchen_bread_set = {get_parts(f)[1] for f in state_set if f.startswith('(at_kitchen_bread ')}
        at_kitchen_content_set = {get_parts(f)[1] for f in state_set if f.startswith('(at_kitchen_content ')}


        # 1. Identify unserved children and their needs/places
        unserved_children_details = [] # list of (child, place, needs_gf)
        places_with_unserved = set()
        for child in self.children_to_serve:
            if child not in served_children_set:
                place = waiting_children_dict.get(child)
                # Assuming children in goal are always waiting if not served
                if place is None:
                     # This state might be unreachable or invalid, treat as infinite cost
                     return math.inf

                needs_gf = (f'(allergic_gluten {child})' in self.allergic_children_facts_set)
                unserved_children_details.append((child, place, needs_gf))
                places_with_unserved.add(place)

        # If no unserved children, we are in a goal state
        if not unserved_children_details:
            return 0

        # 2. Calculate needed sandwiches at each place
        needed_counts_at_place = {} # { P: {'gf': count, 'reg': count} }
        for child, place, needs_gf in unserved_children_details:
            if place not in needed_counts_at_place:
                needed_counts_at_place[place] = {'gf': 0, 'reg': 0}
            if needs_gf:
                needed_counts_at_place[place]['gf'] += 1
            else:
                needed_counts_at_place[place]['reg'] += 1

        total_needed_gf = sum(counts['gf'] for counts in needed_counts_at_place.values())
        total_needed_reg = sum(counts['reg'] for counts in needed_counts_at_place.values())

        # 3. Count available sandwiches by state and type
        counts = {'not_made': {'gf': 0, 'reg': 0},
                  'in_kitchen': {'gf': 0, 'reg': 0},
                  'on_tray_elsewhere': {'gf': 0, 'reg': 0},
                  'on_tray_at_place': {'gf': 0, 'reg': 0}}

        for s in self.all_sandwiches:
            is_not_made = (s in notexist_sandwich_set)
            is_in_kitchen = (s in at_kitchen_sandwich_set)
            is_on_tray = (s in ontray_dict)
            is_gf = (s in no_gluten_sandwich_set) # This predicate only exists if made with GF ingredients

            if is_not_made:
                # Cannot determine type yet. Count as potential GF and Reg slots.
                counts['not_made']['gf'] += 1
                counts['not_made']['reg'] += 1
                continue

            # If made, determine type
            sandwich_type = 'gf' if is_gf else 'reg'

            if is_in_kitchen:
                counts['in_kitchen'][sandwich_type] += 1
            elif is_on_tray:
                tray_s_is_on = ontray_dict[s]
                tray_location = tray_locations_dict.get(tray_s_is_on)
                # Check if the tray is at a place where unserved children are waiting
                if tray_location and tray_location in places_with_unserved:
                     # Count as 'on_tray_at_place' if the tray is at *any* place
                     # with unserved children. The assignment logic handles matching needs.
                     counts['on_tray_at_place'][sandwich_type] += 1
                else:
                    counts['on_tray_elsewhere'][sandwich_type] += 1

        # 4. Calculate sandwiches that still need to be made
        available_made_gf = counts['in_kitchen']['gf'] + counts['on_tray_elsewhere']['gf'] + counts['on_tray_at_place']['gf']
        available_made_reg = counts['in_kitchen']['reg'] + counts['on_tray_elsewhere']['reg'] + counts['on_tray_at_place']['reg']

        to_make_gf = max(0, total_needed_gf - available_made_gf)
        # Use surplus made GF sandwiches for regular needs before counting needed regular makes
        available_made_gf_for_reg = max(0, available_made_gf - total_needed_gf)
        to_make_reg = max(0, total_needed_reg - available_made_reg - available_made_gf_for_reg)

        # 5. Check for basic unsolvability (sandwich slots and ingredients)
        # Check sandwich slots
        total_to_make = to_make_gf + to_make_reg
        if total_to_make > counts['not_made']['gf']: # counts['not_made']['gf'] == counts['not_made']['reg']
            return math.inf

        # Check ingredients
        available_gf_bread_count = sum(1 for b in self.all_bread if b in at_kitchen_bread_set and b in self.no_gluten_bread_set)
        available_gf_content_count = sum(1 for c in self.all_content if c in at_kitchen_content_set and c in self.no_gluten_content_set)
        available_reg_bread_count = sum(1 for b in self.all_bread if b in at_kitchen_bread_set and b not in self.no_gluten_bread_set)
        available_reg_content_count = sum(1 for c in self.all_content if c in at_kitchen_content_set and c not in self.no_gluten_content_set)

        # Can we make 'to_make_gf' GF sandwiches?
        if to_make_gf > min(available_gf_bread_count, available_gf_content_count):
            return math.inf

        # Can we make 'to_make_reg' Reg sandwiches using available ingredients (including surplus GF)?
        remaining_gf_bread = available_gf_bread_count - to_make_gf
        remaining_gf_content = available_gf_content_count - to_make_gf
        available_ing_for_reg = min(available_reg_bread_count + remaining_gf_bread, available_reg_content_count + remaining_gf_content)

        if to_make_reg > available_ing_for_reg:
            return math.inf

        # 6. Calculate heuristic cost using assignments
        h = 0
        needed_gf = total_needed_gf
        needed_reg = total_needed_reg

        # Assign sandwiches from cheapest state first (Cost 1: on_tray_at_place)
        # Use GF for GF first
        assign = min(needed_gf, counts['on_tray_at_place']['gf'])
        h += assign * 1
        needed_gf -= assign
        available_c1_gf_for_reg = counts['on_tray_at_place']['gf'] - assign

        # Use Reg for Reg
        assign = min(needed_reg, counts['on_tray_at_place']['reg'])
        h += assign * 1
        needed_reg -= assign

        # Use remaining GF for Reg
        assign = min(needed_reg, available_c1_gf_for_reg)
        h += assign * 1
        needed_reg -= assign

        # Assign sandwiches from next cheapest state (Cost 2: on_tray_elsewhere)
        # Use GF for GF first
        assign = min(needed_gf, counts['on_tray_elsewhere']['gf'])
        h += assign * 2
        needed_gf -= assign
        available_c2_gf_for_reg = counts['on_tray_elsewhere']['gf'] - assign

        # Use Reg for Reg
        assign = min(needed_reg, counts['on_tray_elsewhere']['reg'])
        h += assign * 2
        needed_reg -= assign

        # Use remaining GF for Reg
        assign = min(needed_reg, available_c2_gf_for_reg)
        h += assign * 2
        needed_reg -= assign

        # Assign sandwiches from next cheapest state (Cost 3: in_kitchen)
        # Use GF for GF first
        assign = min(needed_gf, counts['in_kitchen']['gf'])
        h += assign * 3
        needed_gf -= assign
        available_c3_gf_for_reg = counts['in_kitchen']['gf'] - assign

        # Use Reg for Reg
        assign = min(needed_reg, counts['in_kitchen']['reg'])
        h += assign * 3
        needed_reg -= assign

        # Use remaining GF for Reg
        assign = min(needed_reg, available_c3_gf_for_reg)
        h += assign * 3
        needed_reg -= assign

        # Assign sandwiches from most expensive state (Cost 4: to_make)
        # Use GF for GF first
        assign = min(needed_gf, to_make_gf)
        h += assign * 4
        needed_gf -= assign
        available_c4_gf_for_reg = to_make_gf - assign

        # Use Reg for Reg
        assign = min(needed_reg, to_make_reg)
        h += assign * 4
        needed_reg -= assign

        # Use remaining GF for Reg
        assign = min(needed_reg, available_c4_gf_for_reg)
        h += assign * 4
        needed_reg -= assign

        # If needed_gf or needed_reg is still > 0, it implies an issue in logic
        # or an unsolvable state not caught by resource checks.
        # In a solvable state, these should be 0 here.
        # If needed_gf > 0 or needed_reg > 0: return math.inf # Should not be needed if checks are correct

        return h
