from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have the same number of parts as the pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    unserved children. It does this by counting the number of unserved children
    (separated by allergy status) and matching them greedily with the cheapest
    available suitable sandwiches. The cost of a sandwich is estimated based on
    its current location/stage (ready to serve, on tray elsewhere, in kitchen,
    needs making).

    # Assumptions
    - Each action has a cost of 1.
    - Making a sandwich requires one 'notexist' slot, one bread, and one content.
    - Gluten-free sandwiches require gluten-free bread and content. Regular
      sandwiches can use any bread/content (though typically non-gluten-free).
      This heuristic assumes distinct pools of ingredients for GF and Regular makes.
    - Trays can move between any two places (kitchen and waiting places).
    - Resource contention (e.g., limited trays, limited robot capacity) is
      partially handled by counting available sandwiches at different stages,
      but not fully modeled (e.g., path conflicts for trays are ignored).
    - The problem is solvable with the initial resources provided.

    # Heuristic Initialization
    The heuristic pre-processes the static facts and initial state goals to
    identify:
    - Which children are allergic to gluten.
    - Which children are initially waiting and at which place.
    - Which bread and content items are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic computes the estimated cost as follows:

    1.  **Identify Unserved Children:** Determine which children from the initial
        'waiting' list are not yet marked as 'served' in the current state.
        Count the total number of unserved allergic and non-allergic children.
        If no children are unserved, the heuristic is 0.

    2.  **Count Available Sandwiches by Stage and Type:** Iterate through the
        current state facts to count sandwiches based on their status:
        -   **Cost 1 (Ready to Serve):** Sandwiches that are `ontray` and the
            tray is `at` a place where an unserved child is waiting. Separate
            into gluten-free (GF) and regular (Reg).
        -   **Cost 2 (On Tray Elsewhere):** Sandwiches that are `ontray` but the
            tray is `at` a place where *no* unserved child is waiting (e.g., kitchen).
            Separate into GF and Reg.
        -   **Cost 3 (In Kitchen):** Sandwiches that are `at_kitchen_sandwich`.
            Separate into GF and Reg.
        -   **Cost 4 (Creatable):** Count the number of available `notexist`
            sandwich slots. Count available GF bread/content and regular
            bread/content in the kitchen. Estimate the number of new GF and
            Regular sandwiches that can be made, limited by both ingredients
             and `notexist` slots.

    3.  **Greedy Assignment and Cost Summation:** Assign the available sandwiches
        to the unserved children, prioritizing:
        -   Allergic children needing GF sandwiches.
        -   Sandwiches requiring fewer actions (Cost 1 < Cost 2 < Cost 3 < Cost 4).
        -   For non-allergic children, use available GF sandwiches first (if any
            remain after serving allergic children), then regular sandwiches.
        Sum the estimated cost for each assigned sandwich based on its stage:
        -   Cost 1 sandwiches contribute 1 to the heuristic.
        -   Cost 2 sandwiches contribute 2 (move_tray + serve).
        -   Cost 3 sandwiches contribute 3 (put_on_tray + move_tray + serve).
        -   Cost 4 sandwiches contribute 4 (make_sandwich + put_on_tray + move_tray + serve).

    4.  **Return Total Cost:** The sum calculated in step 3 is the heuristic value.
        If, after exhausting all counted resources, there are still unserved
        children, this indicates a potential issue (e.g., unsolvable problem
        or heuristic limitation), but for solvable problems, this count should
        reach zero.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        and ingredients.
        """
        self.goals = task.goals  # Goal conditions are needed to identify served children
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }

        # Extract initial waiting children and their places from static facts
        self.initial_waiting_children = {} # child -> place
        self.waiting_places = set() # set of places where children wait
        for fact in static_facts:
             if match(fact, "waiting", "*", "*"):
                 child, place = get_parts(fact)[1:]
                 self.initial_waiting_children[child] = place
                 self.waiting_places.add(place)

        # Extract gluten-free ingredient types from static facts
        self.gf_bread_objects = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "no_gluten_bread", "*")
        }
        self.gf_content_objects = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {c for c in self.initial_waiting_children if c not in served_children}

        if not unserved_children:
            return 0 # Goal reached

        unserved_allergic_count = sum(1 for c in unserved_children if c in self.allergic_children)
        unserved_non_allergic_count = len(unserved_children) - unserved_allergic_count

        # 2. Count Available Sandwiches by Stage and Type
        sandwich_is_gf = {get_parts(fact)[1]: True for fact in state if match(fact, "no_gluten_sandwich", "*")}
        sandwich_location = {} # sandwich -> 'kitchen' | tray_object
        tray_location = {} # tray -> place_object
        available_new_sandwich_slots = 0
        bread_in_kitchen = set()
        content_in_kitchen = set()

        # Parse state facts to build location maps and count ingredients/slots
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]

            if predicate == "ontray" and len(parts) == 3:
                s, t = parts[1:]
                sandwich_location[s] = t
            elif predicate == "at" and len(parts) == 3:
                 # Check if it's a tray location fact
                 # We don't have type info here, assume anything 'at' a place is a tray for this heuristic
                 # A more robust parser would use type info from the task object
                 # For this domain, only trays have (at ?t ?p) predicate
                 t, p = parts[1:]
                 tray_location[t] = p
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                s = parts[1]
                sandwich_location[s] = 'kitchen'
            elif predicate == "notexist" and len(parts) == 2:
                available_new_sandwich_slots += 1
            elif predicate == "at_kitchen_bread" and len(parts) == 2:
                b = parts[1]
                bread_in_kitchen.add(b)
            elif predicate == "at_kitchen_content" and len(parts) == 2:
                c = parts[1]
                content_in_kitchen.add(c)

        # Count sandwiches by cost level
        C1_gf = 0; C1_reg = 0 # On tray at a waiting place
        C2_gf = 0; C2_reg = 0 # On tray elsewhere
        C3_gf = 0; C3_reg = 0 # At kitchen

        waiting_places_with_unserved = {self.initial_waiting_children[c] for c in unserved_children}

        for sandwich, loc in sandwich_location.items():
            is_gf = sandwich_is_gf.get(sandwich, False)
            if loc == 'kitchen':
                if is_gf: C3_gf += 1
                else: C3_reg += 1
            else: # It's on a tray 'loc'
                tray = loc
                if tray in tray_location:
                    place = tray_location[tray]
                    if place in waiting_places_with_unserved:
                        if is_gf: C1_gf += 1
                        else: C1_reg += 1
                    else:
                        if is_gf: C2_gf += 1
                        else: C2_reg += 1
                # else: tray location unknown, ignore? (Shouldn't happen in valid states)

        # Calculate ingredient makes potential
        num_gf_bread_kitchen = len(bread_in_kitchen.intersection(self.gf_bread_objects))
        num_gf_content_kitchen = len(content_in_kitchen.intersection(self.gf_content_objects))
        num_reg_bread_kitchen = len(bread_in_kitchen) - num_gf_bread_kitchen
        num_reg_content_kitchen = len(content_in_kitchen) - num_gf_content_objects # Corrected: reg content is total - gf content

        available_gf_makes = min(num_gf_bread_kitchen, num_gf_content_kitchen)
        available_reg_makes = min(num_reg_bread_kitchen, num_reg_content_kitchen)

        # Calculate C4 potential (limited by slots and ingredients)
        C4_gf_potential = min(available_new_sandwich_slots, available_gf_makes)
        # Regular makes use remaining slots and regular ingredients
        C4_reg_potential = min(available_new_sandwich_slots - C4_gf_potential, available_reg_makes)


        # 3. Greedy Assignment and Cost Summation
        h = 0
        N_al = unserved_allergic_count
        N_nal = unserved_non_allergic_count

        # Cost 1: Serve with sandwiches on tray at a waiting place
        use_c1_gf = min(N_al, C1_gf)
        h += use_c1_gf * 1
        N_al -= use_c1_gf
        C1_gf -= use_c1_gf

        use_c1_gf_for_nal = min(N_nal, C1_gf) # Use remaining C1_gf for non-allergic
        h += use_c1_gf_for_nal * 1
        N_nal -= use_c1_gf_for_nal
        C1_gf -= use_c1_gf_for_nal

        use_c1_reg_for_nal = min(N_nal, C1_reg)
        h += use_c1_reg_for_nal * 1
        N_nal -= use_c1_reg_for_nal
        C1_reg -= use_c1_reg_for_nal

        # Cost 2: Serve with sandwiches on tray elsewhere (need move + serve)
        use_c2_gf = min(N_al, C2_gf)
        h += use_c2_gf * 2
        N_al -= use_c2_gf
        C2_gf -= use_c2_gf

        use_c2_gf_for_nal = min(N_nal, C2_gf) # Use remaining C2_gf for non-allergic
        h += use_c2_gf_for_nal * 2
        N_nal -= use_c2_gf_for_nal
        C2_gf -= use_c2_gf_for_nal

        use_c2_reg_for_nal = min(N_nal, C2_reg)
        h += use_c2_reg_for_nal * 2
        N_nal -= use_c2_reg_for_nal
        C2_reg -= use_c2_reg_for_nal

        # Cost 3: Serve with sandwiches in kitchen (need put_on_tray + move + serve)
        use_c3_gf = min(N_al, C3_gf)
        h += use_c3_gf * 3
        N_al -= use_c3_gf
        C3_gf -= use_c3_gf

        use_c3_gf_for_nal = min(N_nal, C3_gf) # Use remaining C3_gf for non-allergic
        h += use_c3_gf_for_nal * 3
        N_nal -= use_c3_gf_for_nal
        C3_gf -= use_c3_gf_for_nal

        use_c3_reg_for_nal = min(N_nal, C3_reg)
        h += use_c3_reg_for_nal * 3
        N_nal -= use_c3_reg_for_nal
        C3_reg -= use_c3_reg_for_nal

        # Cost 4: Serve with new sandwiches (need make + put_on_tray + move + serve)
        use_c4_gf = min(N_al, C4_gf_potential)
        h += use_c4_gf * 4
        N_al -= use_c4_gf
        C4_gf_potential -= use_c4_gf
        available_new_sandwich_slots -= use_c4_gf # Decrement shared slot resource

        use_c4_gf_for_nal = min(N_nal, C4_gf_potential) # Use remaining GF potential
        h += use_c4_gf_for_nal * 4
        N_nal -= use_c4_gf_for_nal
        C4_gf_potential -= use_c4_gf_for_nal
        available_new_sandwich_slots -= use_c4_gf_for_nal # Decrement shared slot resource

        use_c4_reg_for_nal = min(N_nal, available_new_sandwich_slots, C4_reg_potential) # Use remaining slots and Reg potential
        h += use_c4_reg_for_nal * 4
        N_nal -= use_c4_reg_for_nal
        available_new_sandwich_slots -= use_c4_reg_for_nal
        # C4_reg_potential doesn't need decrementing further

        # If N_al > 0 or N_nal > 0 here, it means not enough resources were counted
        # or the problem is potentially unsolvable. For solvable problems, this should be 0.
        # We can add a large penalty, but assuming solvable problems, this isn't strictly needed
        # for correctness, only potentially for performance if search explores unsolvable branches.
        # h += (N_al + N_nal) * 1000 # Optional penalty

        return h
