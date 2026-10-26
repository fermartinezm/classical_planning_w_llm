import math

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove surrounding parentheses and split by spaces
    parts = fact_string.strip("()").split()
    return tuple(parts)

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing the estimated costs of the necessary actions: serve, make_sandwich,
    put_on_tray, and move_tray. It prioritizes satisfying gluten-allergic children's
    needs first. It considers the current state of sandwiches (on trays at location,
    on trays elsewhere, at kitchen, or needing to be made from ingredients)
    and the locations of trays. It returns infinity if the current state
    makes the goal unreachable due to lack of ingredients at the kitchen.

    Assumptions:
    - The heuristic assumes enough sandwiches and trays exist as objects
      in the PDDL problem definition to satisfy all children, provided
      ingredients are available.
    - The heuristic assumes trays can carry multiple sandwiches.
    - The constant 'kitchen' is the only place where ingredients are located
      and sandwiches are initially made and put on trays.

    Heuristic Initialization:
    The constructor processes the static facts from the Task object.
    It identifies which children are allergic or not, and where each child
    is waiting. It also identifies which bread and content portions are
    gluten-free. This information is stored in sets and dictionaries
    for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  **Goal Check:** If all children are already served in the current state,
        the heuristic value is 0.
    2.  **Identify Unserved Children and Needs:** Determine which children are
        not yet served and their waiting locations and allergy status
        (allergic requires GF sandwich, non-allergic can take any).
        Count unserved allergic and non-allergic children per waiting location.
        Identify all waiting places where children still need serving.
    3.  **Parse State Resources:** Extract information about available ingredients
        at the kitchen, sandwiches at the kitchen (not on trays), sandwiches
        on trays at various locations, tray locations, and gluten-free status
        of sandwiches.
    4.  **Calculate Sandwich Deficits at Locations:** For each waiting location,
        calculate the deficit of GF sandwiches needed (for allergic children)
        and the deficit of *any* sandwiches needed (for all children at that
        location), based on suitable sandwiches already available on trays
        at that specific location.
    5.  **Calculate Total Sandwiches to Deliver:** Sum the GF deficits and
        any-sandwiches deficits across all waiting locations. These represent
        the minimum number of GF and any sandwiches that must be brought
        to the waiting locations from elsewhere (kitchen, other trays, or made).
    6.  **Count Available Sandwiches Not at Destination:** Count GF and regular
        sandwiches available at the kitchen (not on trays) or on trays at
        locations where no children are waiting. These are potential sources
        to cover the delivery deficits.
    7.  **Estimate Sandwiches to Make:** Based on the total sandwiches to deliver
        and the available sandwiches not at destination, calculate how many GF
        sandwiches and how many *additional* regular sandwiches (or GF) must
        be made from ingredients at the kitchen.
    8.  **Check Ingredient Feasibility:** Verify if the current state has enough
        GF and regular ingredients at the kitchen to make the required sandwiches.
        If not, the state is considered effectively unsolvable, and the heuristic
        returns infinity.
    9.  **Calculate Action Costs:**
        -   **Serve Cost:** Add 1 for each unserved child.
        -   **Make Cost:** Add 1 for each sandwich that needs to be made.
        -   **Put on Tray Cost:** Add 1 for each sandwich that needs to be put
            on a tray (i.e., sandwiches made + sandwiches at kitchen not already
            on a tray).
        -   **Move Tray Cost:** Add 1 for each waiting location that needs
            sandwiches delivered (deficit > 0) and does not currently have a tray
            present in that location.
    10. **Sum Costs:** The total heuristic value is the sum of the costs from
        step 9.1 to 9.4.
    """
    def __init__(self, task):
        self.task = task
        self.allergic_children_set = set()
        self.non_allergic_children_set = set()
        self.waiting_places_map = {} # child -> place
        self.gf_bread_set = set()
        self.gf_content_set = set()
        self.all_children = set()

        # Process static facts
        for fact_string in task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'allergic_gluten':
                self.allergic_children_set.add(fact[1])
                self.all_children.add(fact[1])
            elif fact[0] == 'not_allergic_gluten':
                self.non_allergic_children_set.add(fact[1])
                self.all_children.add(fact[1])
            elif fact[0] == 'waiting':
                self.waiting_places_map[fact[1]] = fact[2]
            elif fact[0] == 'no_gluten_bread':
                self.gf_bread_set.add(fact[1])
            elif fact[0] == 'no_gluten_content':
                self.gf_content_set.add(fact[1])

        self.waiting_places = set(self.waiting_places_map.values())
        self.kitchen_place = 'kitchen' # Assuming 'kitchen' is always the constant as per domain file


    def __call__(self, state):
        # 1. Goal Check
        served_children_set = set()
        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'served':
                served_children_set.add(fact[1])

        if served_children_set == self.all_children:
             return 0

        # 2. Identify Unserved Children and Needs
        unserved_children_set = self.all_children - served_children_set

        # U_allergic[p]: count of unserved allergic children waiting at p
        # U_non_allergic[p]: count of unserved non-allergic children waiting at p
        U_allergic = {p: 0 for p in self.waiting_places}
        U_non_allergic = {p: 0 for p in self.waiting_places}

        for child in unserved_children_set:
            place = self.waiting_places_map[child]
            if child in self.allergic_children_set:
                U_allergic[place] += 1
            else:
                U_non_allergic[place] += 1

        waiting_places_with_children = {p for p in self.waiting_places if U_allergic[p] > 0 or U_non_allergic[p] > 0}

        # 3. Parse State Resources
        at_kitchen_bread_set = set()
        at_kitchen_content_set = set()
        at_kitchen_sandwich_set = set()
        ontray_map = {} # tray -> set of sandwiches
        tray_location_map = {} # tray -> place
        no_gluten_sandwich_set = set()

        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at_kitchen_bread':
                at_kitchen_bread_set.add(fact[1])
            elif fact[0] == 'at_kitchen_content':
                at_kitchen_content_set.add(fact[1])
            elif fact[0] == 'at_kitchen_sandwich':
                at_kitchen_sandwich_set.add(fact[1])
            elif fact[0] == 'ontray':
                sandwich, tray = fact[1], fact[2]
                ontray_map.setdefault(tray, set()).add(sandwich)
            elif fact[0] == 'at':
                tray, place = fact[1], fact[2]
                tray_location_map[tray] = place
            elif fact[0] == 'no_gluten_sandwich':
                no_gluten_sandwich_set.add(fact[1])

        # Available ingredients at kitchen
        I_gf_bread = len(at_kitchen_bread_set.intersection(self.gf_bread_set))
        I_reg_bread = len(at_kitchen_bread_set) - I_gf_bread
        I_gf_content = len(at_kitchen_content_set.intersection(self.gf_content_set))
        I_reg_content = len(at_kitchen_content_set) - I_gf_content

        # Available sandwiches at kitchen not on tray
        avail_gf_kitchen_not_ontray = len(at_kitchen_sandwich_set.intersection(no_gluten_sandwich_set))
        avail_reg_kitchen_not_ontray = len(at_kitchen_sandwich_set) - avail_gf_kitchen_not_ontray

        # Available sandwiches on trays at locations
        avail_gf_ontray_at_waiting_place = {p: 0 for p in self.waiting_places}
        avail_reg_ontray_at_waiting_place = {p: 0 for p in self.waiting_places}
        avail_gf_ontray_not_at_waiting_place = 0
        avail_reg_ontray_not_at_waiting_place = 0

        for tray, sandwiches_on_it in ontray_map.items():
            location = tray_location_map.get(tray)
            if location is None: continue # Should not happen in valid states

            for s in sandwiches_on_it:
                is_gf = s in no_gluten_sandwich_set
                if location in self.waiting_places:
                    if is_gf:
                        avail_gf_ontray_at_waiting_place[location] += 1
                    else:
                        avail_reg_ontray_at_waiting_place[location] += 1
                else: # Location is not a waiting place
                     if is_gf:
                         avail_gf_ontray_not_at_waiting_place += 1
                     else:
                         avail_reg_ontray_not_at_waiting_place += 1

        # Places that currently have a tray
        Places_with_trays = set(tray_location_map.values())

        # 4. Calculate Sandwich Deficits at Locations
        Total_deficit_gf_at_p = 0
        Total_deficit_any_at_p = 0
        Deficit_any_at_p_map = {} # Store for move cost calculation

        for p in waiting_places_with_children:
            Needed_at_p_gf = U_allergic[p]
            Needed_at_p_any = U_allergic[p] + U_non_allergic[p]
            Avail_at_p_gf = avail_gf_ontray_at_waiting_place.get(p, 0)
            Avail_at_p_any = Avail_at_p_gf + avail_reg_ontray_at_waiting_place.get(p, 0)

            Deficit_at_p_gf = max(0, Needed_at_p_gf - Avail_at_p_gf)
            # Any deficit can be covered by GF or Reg sandwiches
            Deficit_at_p_any = max(0, Needed_at_p_any - Avail_at_p_any)

            Total_deficit_gf_at_p += Deficit_at_p_gf
            Total_deficit_any_at_p += Deficit_at_p_any
            Deficit_any_at_p_map[p] = Deficit_at_p_any


        # 5. Calculate Total Sandwiches to Deliver (same as Total_deficit_gf_at_p and Total_deficit_any_at_p)
        Total_deliver_gf = Total_deficit_gf_at_p
        Total_deliver_any = Total_deficit_any_at_p

        # 6. Count Available Sandwiches Not at Destination
        Avail_gf_not_at_waiting = avail_gf_ontray_not_at_waiting_place + avail_gf_kitchen_not_ontray
        Avail_reg_not_at_waiting = avail_reg_ontray_not_at_waiting_place + avail_reg_kitchen_not_ontray
        Avail_any_not_at_waiting = Avail_gf_not_at_waiting + Avail_reg_not_at_waiting

        # 7. Estimate Sandwiches to Make
        # Need to make GF sandwiches if Avail_gf_not_at_waiting is not enough
        Make_gf_needed = max(0, Total_deliver_gf - Avail_gf_not_at_waiting)
        # Need to make additional sandwiches (can be GF or Reg) if Avail_any_not_at_waiting is not enough
        Make_any_needed = max(0, Total_deliver_any - Avail_any_not_at_waiting)
        # The number of additional regular sandwiches to make is the total 'any' needed minus the GF ones that must be made
        Make_reg_additional_needed = max(0, Make_any_needed - Make_gf_needed)


        # 8. Check Ingredient Feasibility
        # Can we make Make_gf_needed GF sandwiches?
        Make_gf_possible = min(I_gf_bread, I_gf_content)
        if Make_gf_needed > Make_gf_possible:
            return math.inf # Cannot make required GF sandwiches

        # After hypothetically using ingredients for GF, check if enough for additional Reg/Any
        Rem_I_gf_bread = I_gf_bread - Make_gf_needed
        Rem_I_gf_content = I_gf_content - Make_gf_needed
        Rem_I_reg_bread = I_reg_bread
        Rem_I_reg_content = I_reg_content

        Make_reg_additional_possible = min(Rem_I_gf_bread + Rem_I_reg_bread, Rem_I_gf_content + Rem_I_reg_content)
        if Make_reg_additional_needed > Make_reg_additional_possible:
             return math.inf # Cannot make required additional sandwiches

        # If feasible, the actual number to make is the needed amount
        Make_gf_actual = Make_gf_needed
        Make_reg_additional_actual = Make_reg_additional_needed

        # 9. Calculate Action Costs
        cost = 0

        # 9.1 Serve Cost
        cost += len(unserved_children_set)

        # 9.2 Make Cost
        cost += Make_gf_actual + Make_reg_additional_actual

        # 9.3 Put on Tray Cost
        # Sandwiches made need to be put on tray.
        # Sandwiches at kitchen not on tray need to be put on tray.
        Sandwiches_to_put = Make_gf_actual + Make_reg_additional_actual + avail_gf_kitchen_not_ontray + avail_reg_kitchen_not_ontray
        cost += Sandwiches_to_put

        # 9.4 Move Tray Cost
        # Count places p in waiting_places_with_children where deficit > 0 and no tray is currently at p
        move_cost = 0
        for p in waiting_places_with_children:
            if Deficit_any_at_p_map.get(p, 0) > 0: # Check if any sandwiches are needed at this location
                if p not in Places_with_trays:
                    move_cost += 1
        cost += move_cost

        # 10. Sum Costs
        return cost
