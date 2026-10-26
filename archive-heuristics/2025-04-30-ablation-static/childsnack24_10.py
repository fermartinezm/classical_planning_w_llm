from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact.startswith('(') or not fact.endswith(')'):
         return [] # Should not happen with valid PDDL facts
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the number of unserved children and adds estimated costs based on the
    current state of available sandwiches, prioritizing sandwiches that are closer
    to being served (e.g., already on a tray at the child's location, then on a tray
    at the kitchen, then in the kitchen not on a tray, then makeable). It accounts
    for gluten allergies by prioritizing gluten-free sandwiches for allergic children.

    # Assumptions
    - Each unserved child requires one sandwich.
    - Gluten-free sandwiches can serve both allergic and non-allergic children.
    - Regular sandwiches can only serve non-allergic children.
    - Making a sandwich requires one bread, one content, and one 'notexist' slot.
    - Putting a sandwich on a tray requires the sandwich in the kitchen and a tray in the kitchen.
    - Moving a tray requires the tray to be at the kitchen and needs to go to a child's location.
    - Serving requires a suitable sandwich on a tray at the child's location.
    - Resource counts (bread, content, notexist slots, trays) are simplified for makeable sandwiches.
    - Tray capacity is not explicitly modeled; assume enough capacity or that trays are reusable.
    - The heuristic counts the number of "tasks" (like 'make', 'put', 'move', 'serve') needed
      to get enough sandwiches through the pipeline to serve all children, assuming each task costs 1 action.

    # Heuristic Initialization
    - Extracts static information: child allergy status, child waiting locations,
      lists of all objects (children, sandwiches, trays, places, bread, content),
      and which bread/content items are gluten-free. This information is derived
      from the initial state facts provided in the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children and count the total (`U`), allergic (`U_allergic`),
       and non-allergic (`U_not_allergic`). If `U` is 0, the heuristic is 0.
    2. Identify the location of all trays.
    3. Identify the gluten status of all existing sandwiches.
    4. Count available sandwiches/resources at different stages of processing, categorized by gluten status:
       - `Ready_to_serve_gf_children`: Number of allergic children who can be served immediately by GF sandwiches on trays at their location.
       - `Ready_to_serve_reg_children`: Number of non-allergic children who can be served immediately by regular (or remaining GF) sandwiches on trays at their location.
       - `Ontray_kitchen_gf`: GF sandwiches on trays at the kitchen.
       - `Ontray_kitchen_reg`: Regular sandwiches on trays at the kitchen.
       - `Kitchen_notray_gf`: GF sandwiches in the kitchen, not on trays.
       - `Kitchen_notray_reg`: Regular sandwiches in the kitchen, not on trays.
       - `Makeable_gf`: GF sandwiches that can be made (based on `notexist` slots and GF ingredients in kitchen).
       - `Makeable_reg`: Regular sandwiches that can be made (based on remaining `notexist` slots and any remaining ingredients in kitchen, after accounting for GF sandwich needs).
    5. Calculate the number of children still needing a GF sandwich (`needed_gf = U_allergic`)
       and those needing any sandwich (`needed_reg = U_not_allergic`).
    6. Greedily fulfill the needed sandwiches using available resources from the cheapest
       stages first (Cost 1 -> Cost 2 -> Cost 3 -> Cost 4), consuming resources as they are used.
       Prioritize using GF sandwiches for `needed_gf` first, then use remaining GF and REG
       sandwiches for `needed_reg`. Add the cost (1, 2, 3, or 4) multiplied by the number
       of children/sandwiches consumed from that stage to the total heuristic.
    7. If any children still need sandwiches after exhausting all makeable options, add a penalty
       equal to the remaining count multiplied by the highest stage cost (4).
    8. The total heuristic value is the sum of costs from each stage.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        # Static facts from task.static are not used in this domain based on examples.
        # We extract necessary static-like info from the initial state.
        initial_state = task.initial_state

        self.child_allergy_status = {}
        self.child_waiting_location = {}
        self.all_children = set() # Use sets during parsing to avoid duplicates
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_bread = set()
        self.all_content = set()
        self.gluten_free_bread = set()
        self.gluten_free_content = set()

        # Extract static information from initial state facts
        for fact in initial_state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            pred = parts[0]
            if pred == "allergic_gluten":
                child = parts[1]
                self.child_allergy_status[child] = 'allergic'
                self.all_children.add(child)
            elif pred == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy_status[child] = 'not_allergic'
                self.all_children.add(child)
            elif pred == "waiting":
                child, place = parts[1:]
                self.child_waiting_location[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif pred == "no_gluten_bread":
                 bread = parts[1]
                 self.gluten_free_bread.add(bread)
                 self.all_bread.add(bread)
            elif pred == "no_gluten_content":
                 content = parts[1]
                 self.gluten_free_content.add(content)
                 self.all_content.add(content)
            elif pred == "at_kitchen_bread":
                 self.all_bread.add(parts[1])
            elif pred == "at_kitchen_content":
                 self.all_content.add(parts[1])
            elif pred == "at_kitchen_sandwich":
                 self.all_sandwiches.add(parts[1])
            elif pred == "ontray":
                 self.all_sandwiches.add(parts[1])
                 self.all_trays.add(parts[2])
            elif pred == "at" and parts[1].startswith('tray'):
                 self.all_trays.add(parts[1])
                 self.all_places.add(parts[2])
            elif pred == "notexist":
                 self.all_sandwiches.add(parts[1])
            elif pred == "served":
                 self.all_children.add(parts[1])

        # Ensure kitchen is included in places
        self.all_places.add('kitchen')

        # Convert sets to lists for consistent attribute types if needed, or keep as sets.
        # Keeping as sets might be slightly more efficient for lookups.
        # Let's keep them as sets.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = [c for c in self.all_children if c not in served_children_in_state]
        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0 # Goal reached

        unserved_allergic = [c for c in unserved_children if self.child_allergy_status.get(c) == 'allergic']
        unserved_not_allergic = [c for c in unserved_children if self.child_allergy_status.get(c) == 'not_allergic']
        needed_gf = len(unserved_allergic)
        needed_reg = len(unserved_not_allergic)

        # 2. Identify state of sandwiches and trays
        ontray_facts = [(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "ontray", "*", "*")]
        at_tray_facts = [(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "at", "tray*", "*")]
        at_kitchen_sandwich_facts = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        no_gluten_sandwich_facts = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        notexist_sandwich_facts = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        at_kitchen_bread_facts = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        at_kitchen_content_facts = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        tray_location = {t: p for t, p in at_tray_facts}
        is_gf_sandwich = {s: s in no_gluten_sandwich_facts for s in self.all_sandwiches if s not in notexist_sandwich_facts}

        # 3. Count available sandwiches/resources at different stages

        # Stage 1: Ready to serve (on tray at child's location) - Count children who can be served
        Ready_to_serve_gf_children = 0
        Ready_to_serve_reg_children = 0
        ontray_at_child_loc = defaultdict(lambda: {'gf': 0, 'reg': 0})
        child_locations = {self.child_waiting_location[c] for c in unserved_children}

        for s, t in ontray_facts:
            loc = tray_location.get(t)
            if loc in child_locations:
                s_type = 'gf' if is_gf_sandwich.get(s) else 'reg'
                ontray_at_child_loc[loc][s_type] += 1

        # Count how many children at each location can be served by existing sandwiches at that location
        for loc in child_locations:
            allergic_at_p = [c for c in unserved_allergic if self.child_waiting_location[c] == loc]
            not_allergic_at_p = [c for c in unserved_not_allergic if self.child_waiting_location[c] == loc]
            avail_gf_at_p = ontray_at_child_loc.get(loc, {}).get('gf', 0)
            avail_reg_at_p = ontray_at_child_loc.get(loc, {}).get('reg', 0)

            # Serve allergic first with GF at this location
            can_serve_gf_at_p = min(len(allergic_at_p), avail_gf_at_p)
            Ready_to_serve_gf_children += can_serve_gf_at_p
            avail_gf_at_p -= can_serve_gf_at_p

            # Serve non-allergic with remaining GF and REG at this location
            can_serve_reg_at_p = min(len(not_allergic_at_p), avail_reg_at_p + avail_gf_at_p)
            Ready_to_serve_reg_children += can_serve_reg_at_p


        # Stage 2: On tray at kitchen (Count sandwiches)
        Ontray_kitchen_gf = 0
        Ontray_kitchen_reg = 0
        for s, t in ontray_facts:
            loc = tray_location.get(t)
            if loc == 'kitchen':
                s_type = 'gf' if is_gf_sandwich.get(s) else 'reg'
                if s_type == 'gf': Ontray_kitchen_gf += 1
                else: Ontray_kitchen_reg += 1

        # Stage 3: Kitchen not on tray (Count sandwiches)
        Kitchen_notray_gf = 0
        Kitchen_notray_reg = 0
        for s in at_kitchen_sandwich_facts:
             s_type = 'gf' if is_gf_sandwich.get(s) else 'reg'
             if s_type == 'gf': Kitchen_notray_gf += 1
             else: Kitchen_notray_reg += 1

        # Stage 4: Makeable (Simplified resource counting)
        num_notexist = len(notexist_sandwich_facts)
        num_avail_bread = len(at_kitchen_bread_facts)
        num_avail_content = len(at_kitchen_content_facts)
        num_avail_gf_bread = len(self.gluten_free_bread.intersection(at_kitchen_bread_facts))
        num_avail_gf_content = len(self.gluten_free_content.intersection(at_kitchen_content_facts))

        Makeable_gf = min(num_notexist, num_avail_gf_bread, num_avail_gf_content)
        remaining_notexist = num_notexist - Makeable_gf
        remaining_bread = num_avail_bread - Makeable_gf # Assume GF bread used for GF sandwiches
        remaining_content = num_avail_content - Makeable_gf # Assume GF content used for GF sandwiches

        Makeable_reg = min(remaining_notexist, remaining_bread, remaining_content)


        # 6. Greedily fulfill needed sandwiches from cheapest stages

        h = 0

        # Stage 1: Ready to serve (Cost 1: serve) - Fulfill children directly
        can_serve_gf = min(needed_gf, Ready_to_serve_gf_children)
        h += can_serve_gf * 1
        needed_gf -= can_serve_gf
        # No sandwich counts to consume here, as we counted children served by them.

        can_serve_reg = min(needed_reg, Ready_to_serve_reg_children) # Note: Ready_to_serve_reg_children already accounts for GF used for REG at location
        h += can_serve_reg * 1
        needed_reg -= can_serve_reg


        # Stage 2: On tray at kitchen (Cost 2: move + serve) - Fulfill remaining children using sandwiches
        can_move_gf = min(needed_gf, Ontray_kitchen_gf)
        h += can_move_gf * 2
        needed_gf -= can_move_gf
        Ontray_kitchen_gf -= can_move_gf # Consume GF

        can_move_reg_from_reg = min(needed_reg, Ontray_kitchen_reg)
        h += can_move_reg_from_reg * 2
        needed_reg -= can_move_reg_from_reg
        Ontray_kitchen_reg -= can_move_reg_from_reg # Consume REG

        can_move_reg_from_gf = min(needed_reg, Ontray_kitchen_gf) # Use remaining GF for REG needs
        h += can_move_reg_from_gf * 2
        needed_reg -= can_move_reg_from_gf
        Ontray_kitchen_gf -= can_move_reg_from_gf # Consume GF


        # Stage 3: Kitchen not on tray (Cost 3: put + move + serve) - Fulfill remaining children using sandwiches
        can_put_gf = min(needed_gf, Kitchen_notray_gf)
        h += can_put_gf * 3
        needed_gf -= can_put_gf
        Kitchen_notray_gf -= can_put_gf # Consume GF

        can_put_reg_from_reg = min(needed_reg, Kitchen_notray_reg)
        h += can_put_reg_from_reg * 3
        needed_reg -= can_put_reg
        Kitchen_notray_reg -= can_put_reg # Consume REG

        can_put_reg_from_gf = min(needed_reg, Kitchen_notray_gf) # Use remaining GF for REG needs
        h += can_put_reg_from_gf * 3
        needed_reg -= can_put_reg
        Kitchen_notray_gf -= can_put_reg_from_gf # Consume GF


        # Stage 4: Makeable (Cost 4: make + put + move + serve) - Fulfill remaining children using sandwiches
        can_make_gf = min(needed_gf, Makeable_gf)
        h += can_make_gf * 4
        needed_gf -= can_make_gf
        Makeable_gf -= can_make_gf # Consume GF

        can_make_reg_from_reg = min(needed_reg, Makeable_reg)
        h += can_make_reg_from_reg * 4
        needed_reg -= can_make_reg
        Makeable_reg -= can_make_reg # Consume REG

        can_make_reg_from_gf = min(needed_reg, Makeable_gf) # Use remaining GF for REG needs
        h += can_make_reg_from_gf * 4
        needed_reg -= can_make_reg
        Makeable_gf -= can_make_reg_from_gf # Consume GF

        # 7. Add penalty for remaining needed sandwiches
        h += (needed_gf + needed_reg) * 4 # Penalty for unmet needs

        return h
