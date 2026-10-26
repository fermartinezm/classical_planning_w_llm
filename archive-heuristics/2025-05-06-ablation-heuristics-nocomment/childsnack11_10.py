from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to match a fact against a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the cost to reach the goal (all children served) by summing up
    the estimated costs for each unserved child. For each unserved child,
    it calculates the minimum number of actions required to get an appropriate
    sandwich onto a tray at the child's location, plus the final 'serve' action.
    It prioritizes using existing sandwiches that are closer to the child's
    location (on tray elsewhere > in kitchen > needs to be made).

    Assumptions:
    - The problem is solvable (e.g., enough ingredients exist eventually).
    - Tray moves between any two places cost 1.
    - Put on tray from kitchen costs 1.
    - Make sandwich costs 1.
    - Serve sandwich costs 1.
    - There are always enough trays available to perform necessary moves and put_on_tray actions.
    - Waiting locations and allergy statuses are static.

    Heuristic Initialization:
    - Extracts all child, tray, sandwich, bread, content, and place objects mentioned
      in the initial state, static facts, goals, and operator definitions.
    - Stores static facts like allergy status and waiting locations for quick lookup.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all unserved children. If none, the heuristic is 0 (goal state).
    2. Identify all existing sandwiches and classify them by type (gluten-free or regular)
       and current location state (at_kitchen_sandwich, ontray).
    3. Identify which unserved children are *not* yet covered by a delivered sandwich
       (an appropriate sandwich already on a tray at their waiting location). These are
       the children needing delivery.
    4. The base heuristic cost is the number of children needing delivery (representing
       the final 'serve' action for each).
    5. For the children needing delivery, count how many appropriate sandwiches are
       available from different sources:
       - On a tray elsewhere (cost 1: move_tray)
       - In the kitchen (cost 2: put_on_tray + move_tray)
       - Need to be made (cost 3: make + put_on_tray + move_tray)
       These counts consider sandwich type requirements (GF for allergic, Any for non-allergic).
    6. Greedily assign the available sandwiches to the children needing delivery,
       prioritizing the cheapest source (cost 1 > cost 2 > cost 3) and required type
       (GF for allergic first, then Any for non-allergic).
    7. Add the calculated costs for the delivery stages (move, put, make) to the base
       heuristic cost.
    8. The total sum is the heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        # Extract objects by type from all facts in the task definition
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        # Predicates and the index of the object argument(s) and their types
        # This mapping helps extract objects reliably
        obj_info = {
            'allergic_gluten': [(0, 'child')],
            'not_allergic_gluten': [(0, 'child')],
            'served': [(0, 'child')],
            'waiting': [(0, 'child'), (1, 'place')],
            'at_kitchen_bread': [(0, 'bread-portion')],
            'no_gluten_bread': [(0, 'bread-portion')],
            'at_kitchen_content': [(0, 'content-portion')],
            'no_gluten_content': [(0, 'content-portion')],
            'at_kitchen_sandwich': [(0, 'sandwich')],
            'ontray': [(0, 'sandwich'), (1, 'tray')],
            'no_gluten_sandwich': [(0, 'sandwich')],
            'notexist': [(0, 'sandwich')],
            'at': [(0, 'tray'), (1, 'place')],
        }

        def add_object(obj_name, obj_type):
            if obj_type == 'child': self.all_children.add(obj_name)
            elif obj_type == 'tray': self.all_trays.add(obj_name)
            elif obj_type == 'sandwich': self.all_sandwiches.add(obj_name)
            elif obj_type == 'bread-portion': self.all_bread.add(obj_name)
            elif obj_type == 'content-portion': self.all_content.add(obj_name)
            elif obj_type == 'place': self.all_places.add(obj_name)

        # Iterate through all facts in initial state, static, goals, and operators
        all_facts = set(task.initial_state) | set(task.static) | set(task.goals)
        for op in task.operators:
             all_facts |= set(op.preconditions) | set(op.add_effects) | set(op.del_effects)

        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate in obj_info:
                for idx, obj_type in obj_info[predicate]:
                    if len(args) > idx:
                        add_object(args[idx], obj_type)

        # Store static info for quick lookup
        self.is_allergic = {c for c in self.all_children if '(allergic_gluten ' + c + ')' in self.static_facts}
        self.waiting_locations = {c: p for c in self.all_children for p in self.all_places if '(waiting ' + c + ' ' + p + ')' in self.static_facts}


    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}

        if not unserved_children:
            return 0 # Goal state reached

        # 2. Identify existing sandwiches by type and location state
        existing_sandwiches = {s for s in self.all_sandwiches if '(notexist ' + s + ')' not in state}
        existing_gf_sandwiches = {s for s in existing_sandwiches if '(no_gluten_sandwich ' + s + ')' in state}
        existing_reg_sandwiches = existing_sandwiches - existing_gf_sandwiches

        # Group existing sandwiches by location state
        ak_sandwiches = {s for s in existing_sandwiches if '(at_kitchen_sandwich ' + s + ')' in state}
        ot_sandwiches_with_tray = {(s, t) for s in existing_sandwiches for t in self.all_trays if '(ontray ' + s + ' ' + t + ')' in state}
        ot_sandwiches = {s for s, t in ot_sandwiches_with_tray}

        ak_gf = ak_sandwiches & existing_gf_sandwiches
        ak_reg = ak_sandwiches & existing_reg_sandwiches
        ot_gf = ot_sandwiches & existing_gf_sandwiches
        ot_reg = ot_sandwiches & existing_reg_sandwiches

        # 3. Identify children needing delivery
        children_needing_delivery = set()

        # First, find sandwiches already delivered to the correct location
        delivered_sandwiches_at_location = {} # Map location -> set of sandwiches delivered there
        for s, t in ot_sandwiches_with_tray:
            tray_location = None
            for p in self.all_places:
                if '(at ' + t + ' ' + p + ')' in state:
                    tray_location = p
                    break
            if tray_location:
                delivered_sandwiches_at_location.setdefault(tray_location, set()).add(s)

        for child in unserved_children:
            loc = self.waiting_locations.get(child)
            # If a child is unserved, they must be waiting at a location
            if loc is None: continue

            is_delivered = False
            if loc in delivered_sandwiches_at_location:
                for s in delivered_sandwiches_at_location[loc]:
                    is_appropriate = (child in self.is_allergic and s in existing_gf_sandwiches) or \
                                     (child not in self.is_allergic) # Non-allergic can take any
                    if is_appropriate:
                        is_delivered = True
                        break
            if not is_delivered:
                children_needing_delivery.add(child)

        num_children_needing_delivery = len(children_needing_delivery)

        # 4. Base heuristic cost: 1 action (serve) per child needing delivery
        h = num_children_needing_delivery

        if num_children_needing_delivery == 0:
             return h # Should be 0 if unserved_children was 0

        # 5. Count available appropriate sandwiches by source state for children needing delivery
        # These counts represent the *pools* of sandwiches available at different costs.

        # Available GF on tray elsewhere (cost 1 move_tray) needed by allergic children needing delivery
        avail_gf_ot_wp_sandwiches = {s for s, t in ot_sandwiches_with_tray if s in existing_gf_sandwiches and any('(at ' + t + ' ' + p + ')' in state and p != self.waiting_locations.get(c) for p in self.all_places for c in children_needing_delivery if c in self.is_allergic)}

        # Available Reg on tray elsewhere (cost 1 move_tray) needed by non-allergic children needing delivery
        avail_reg_ot_wp_sandwiches = {s for s, t in ot_sandwiches_with_tray if s in existing_reg_sandwiches and any('(at ' + t + ' ' + p + ')' in state and p != self.waiting_locations.get(c) for p in self.all_places for c in children_needing_delivery if c not in self.is_allergic)}

        # Available GF in kitchen (cost 2 put_on_tray + move_tray) needed by allergic children needing delivery
        avail_gf_ak_sandwiches = {s for s in ak_gf if any(c in self.is_allergic for c in children_needing_delivery)}

        # Available Reg in kitchen (cost 2 put_on_tray + move_tray) needed by non-allergic children needing delivery
        avail_reg_ak_sandwiches = {s for s in ak_reg if any(c not in self.is_allergic for c in children_needing_delivery)}

        # Available GF on tray elsewhere (cost 1 move_tray) needed by non-allergic children needing delivery (surplus GF)
        avail_gf_ot_wp_for_reg_sandwiches = {s for s, t in ot_sandwiches_with_tray if s in existing_gf_sandwiches and any('(at ' + t + ' ' + p + ')' in state and p != self.waiting_locations.get(c) for p in self.all_places for c in children_needing_delivery if c not in self.is_allergic)}

        # Available GF in kitchen (cost 2 put_on_tray + move_tray) needed by non-allergic children needing delivery (surplus GF)
        avail_gf_ak_for_reg_sandwiches = {s for s in ak_gf if any(c not in self.is_allergic for c in children_neerving_delivery)}


        # 6. Greedily assign available sandwiches to needed slots, prioritizing source cost and type
        needed_gf_delivery = len([c for c in children_needing_delivery if c in self.is_allergic])
        needed_reg_delivery = len([c for c in children_needing_delivery if c not in self.is_allergic])

        delivery_cost = 0

        # 6a. Satisfy allergic children with GF from OT_WP (cost 1)
        use_ot_wp_gf_allergic = min(needed_gf_delivery, len(avail_gf_ot_wp_sandwiches))
        delivery_cost += use_ot_wp_gf_allergic * 1
        needed_gf_delivery -= use_ot_wp_gf_allergic
        avail_gf_ot_wp_rem = len(avail_gf_ot_wp_sandwiches) - use_ot_wp_gf_allergic # Remaining GF OT_WP can serve non-allergic

        # 6b. Satisfy allergic children with GF from AK (cost 2)
        use_ak_gf_allergic = min(needed_gf_delivery, len(avail_gf_ak_sandwiches))
        delivery_cost += use_ak_gf_allergic * 2
        needed_gf_delivery -= use_ak_gf_allergic
        avail_gf_ak_rem = len(avail_gf_ak_sandwiches) - use_ak_gf_allergic # Remaining GF AK can serve non-allergic

        # 6c. Allergic children still needing GF must have them made (cost 3)
        make_gf_allergic = needed_gf_delivery
        delivery_cost += make_gf_allergic * 3

        # 6d. Now for non-allergic children needing delivery
        # Available Any from OT_WP = Reg OT_WP + remaining GF OT_WP
        avail_any_ot_wp_for_reg = len(avail_reg_ot_wp_sandwiches) + avail_gf_ot_wp_rem
        # Available Any from AK = Reg AK + remaining GF AK
        avail_any_ak_for_reg = len(avail_reg_ak_sandwiches) + avail_gf_ak_rem

        # 6e. Satisfy non-allergic children with Any from OT_WP (cost 1)
        use_ot_wp_any_reg = min(needed_reg_delivery, avail_any_ot_wp_for_reg)
        delivery_cost += use_ot_wp_any_reg * 1
        needed_reg_delivery -= use_ot_wp_any_reg

        # 6f. Satisfy non-allergic children with Any from AK (cost 2)
        use_ak_any_reg = min(needed_reg_delivery, avail_any_ak_for_reg)
        delivery_cost += use_ak_any_reg * 2
        needed_reg_delivery -= use_ak_any_reg

        # 6g. Non-allergic children still needing Any must have them made (cost 3)
        make_any_reg = needed_reg_delivery
        delivery_cost += make_any_reg * 3

        # 7. Total heuristic = cost of delivery stages + cost of serve actions
        h += delivery_cost

        return h
