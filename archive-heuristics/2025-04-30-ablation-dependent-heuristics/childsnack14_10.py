from collections import defaultdict

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state where all children are served. It does this by counting the number
    of unserved children and adding costs based on the stage of preparation
    and location of suitable sandwiches needed to serve them. Sandwiches
    ready on trays at the child's location are cheapest (cost 1), followed
    by sandwiches on trays at the kitchen or on trays at other waiting places
    (cost 2), sandwiches at the kitchen not on trays (cost 3), and finally,
    sandwiches that need to be made from ingredients (cost 4). Allergic
    children's needs for gluten-free sandwiches are prioritized.

    Assumptions:
    - The problem is solvable.
    - There are enough trays available at the kitchen or can be moved there
      to support sandwich preparation and delivery (stages 1-3). The heuristic
      does not explicitly model tray availability or movement *to* the kitchen.
    - Ingredient availability is simplified; it counts the potential number
      of sandwiches that can be made based on current ingredients and 'notexist'
      sandwich objects, assuming ingredients can be combined freely (with
      priority for GF makes).

    Heuristic Initialization:
    The constructor extracts static information from the task, specifically
    which children are allergic and where each child is waiting. This information
    is stored for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify dynamic facts: served children,
        sandwiches at kitchen, sandwiches on trays, sandwich types (gluten-free),
        tray locations, available ingredients (bread, content), and 'notexist'
        sandwich objects.
    2.  Identify unserved children and group them by their waiting place and
        allergy status (allergic needing GF, non-allergic needing Reg/GF).
    3.  If no children are unserved, the state is a goal state, and the heuristic is 0.
    4.  Count the total number of GF sandwiches needed (sum of unserved allergic children).
    5.  Count the total number of Regular sandwiches needed (sum of unserved non-allergic children).
    6.  Count available sandwiches categorized by type (GF/Reg) and stage of readiness/location:
        - Stage 4 (Cost 1: Serve): On tray at a waiting place *where a child needing this specific type waits*.
        - Stage 3 (Cost 2: Move + Serve): On tray at the kitchen OR on tray at a waiting place *where no child needing this specific type waits*.
        - Stage 2 (Cost 3: Put + Move + Serve): At the kitchen, not on a tray.
        - Stage 1 (Cost 4: Make + Put + Move + Serve): Can be made from available ingredients and 'notexist' sandwich objects.
        Ingredient availability for making sandwiches is calculated by determining the maximum
        number of GF sandwiches that can be made, consuming GF ingredients and 'notexist'
        objects, and then determining the maximum number of Regular sandwiches that can
        be made from remaining ingredients (including any remaining GF ingredients) and
        'notexist' objects.
    7.  Initialize the heuristic value to 0.
    8.  Satisfy the needs for GF sandwiches first, using available GF sandwiches from the lowest cost stage upwards (Stage 4 -> Stage 3 -> Stage 2 -> Stage 1). For each sandwich used from a stage, add the corresponding cost (1, 2, 3, or 4) to the heuristic. Track the remaining GF needs and remaining available GF sandwiches in each stage.
    9.  Satisfy the needs for Regular sandwiches using the remaining available GF sandwiches (from all stages) and the available Regular sandwiches from the lowest cost stage upwards (Stage 4 -> Stage 3 -> Stage 2 -> Stage 1). For each sandwich used, add the corresponding cost (1, 2, 3, or 4) to the heuristic. Note that remaining GF sandwiches from Stage 4 (at places with allergic needs) and Stage 3 (at kitchen or places without allergic needs) are considered available for Regular needs at Stage 3 cost (2), as they require a move action.
    10. The final heuristic value is the sum of costs accumulated in steps 8 and 9. This represents an estimate of the total actions needed to get suitable sandwiches to the children and serve them.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object.
        """
        self.allergic_children = set()
        self.waiting_places = {} # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        # Parse static facts
        for fact_str in task.static:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif predicate == 'waiting':
                self.waiting_places[parts[1]] = parts[2]
            elif predicate == 'no_gluten_bread':
                self.no_gluten_bread.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.no_gluten_content.add(parts[1])

        # Store all children names for easy iteration
        self.all_children = set(self.waiting_places.keys())


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal.
        """
        # Parse dynamic facts from the state
        served_children = set()
        at_kitchen_bread = set()
        at_kitchen_content = set()
        at_kitchen_sandwich = set()
        ontray = defaultdict(set) # tray -> set of sandwiches
        sandwich_on_tray_map = {} # sandwich -> tray
        no_gluten_sandwich = set()
        at_tray_place = {} # tray -> place
        notexist_sandwiches = set()

        for fact_str in state:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'at_kitchen_bread':
                at_kitchen_bread.add(parts[1])
            elif predicate == 'at_kitchen_content':
                at_kitchen_content.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwich.add(parts[1])
            elif predicate == 'ontray':
                s, t = parts[1], parts[2]
                ontray[t].add(s)
                sandwich_on_tray_map[s] = t
            elif predicate == 'no_gluten_sandwich':
                no_gluten_sandwich.add(parts[1])
            elif predicate == 'at':
                t, p = parts[1], parts[2]
                at_tray_place[t] = p
            elif predicate == 'notexist':
                notexist_sandwiches.add(parts[1])

        # Identify unserved children
        unserved_children = self.all_children - served_children

        # Goal state reached
        if not unserved_children:
            return 0

        # Count needs
        unserved_allergic_at_P = defaultdict(int)
        unserved_non_allergic_at_P = defaultdict(int)
        for child in unserved_children:
            place = self.waiting_places[child]
            if child in self.allergic_children:
                unserved_allergic_at_P[place] += 1
            else:
                unserved_non_allergic_at_P[place] += 1

        total_gf_needed = sum(unserved_allergic_at_P.values())
        total_reg_needed = sum(unserved_non_allergic_at_P.values())

        # Count available sandwiches by type and stage
        # Stage 4 (Cost 1: Serve): On tray at a waiting place *where a child needing this type waits*.
        # Stage 3 (Cost 2: Move + Serve): On tray at the kitchen OR on tray at a waiting place *where no child needing this type waits*.
        # Stage 2 (Cost 3: Put + Move + Serve): At the kitchen, not on a tray.
        # Stage 1 (Cost 4: Make + Put + Move + Serve): Can be made.

        gf_s4_available = 0 # GF on tray at place with allergic need
        reg_s4_available = 0 # Reg on tray at place with non-allergic need
        gf_s3_available = 0 # GF on tray at kitchen or place without allergic need
        reg_s3_available = 0 # Reg on tray at kitchen or place without non-allergic need
        gf_s2_available = 0 # GF at kitchen, not on tray
        reg_s2_available = 0 # Reg at kitchen, not on tray

        # Count sandwiches at kitchen (not on tray)
        for s in at_kitchen_sandwich:
            if s in no_gluten_sandwich:
                gf_s2_available += 1
            else:
                reg_s2_available += 1

        # Count sandwiches on trays
        for tray, sandwiches_on_this_tray in ontray.items():
            place = at_tray_place.get(tray) # Tray might not have a location yet
            if place:
                for s in sandwiches_on_this_tray:
                    is_gf = s in no_gluten_sandwich
                    if place == 'kitchen':
                        if is_gf: gf_s3_available += 1
                        else: reg_s3_available += 1
                    else: # It's at a waiting place
                        # Check if there is an unserved child needing this type at this place
                        needs_gf_at_P = unserved_allergic_at_P.get(place, 0) > 0
                        needs_reg_at_P = unserved_non_allergic_at_P.get(place, 0) > 0

                        if is_gf:
                            if needs_gf_at_P:
                                gf_s4_available += 1 # GF at place with allergic need (Cost 1)
                            else:
                                gf_s3_available += 1 # GF at place without allergic need (Cost 2)
                        else: # Regular sandwich
                            if needs_reg_at_P:
                                reg_s4_available += 1 # Reg at place with non-allergic need (Cost 1)
                            else:
                                reg_s3_available += 1 # Reg at place without non-allergic need (Cost 2)


        # Count available ingredients and notexist sandwiches
        num_gf_bread = len([b for b in at_kitchen_bread if b in self.no_gluten_bread])
        num_reg_bread = len(at_kitchen_bread) - num_gf_bread
        num_gf_content = len([c for c in at_kitchen_content if c in self.no_gluten_content])
        num_reg_content = len(at_kitchen_content) - num_gf_content
        num_notexist = len(notexist_sandwiches)

        # Calculate potential makes (Stage 1)
        potential_gf_makes = min(num_gf_bread, num_gf_content, num_notexist)
        remaining_notexist_after_gf = num_notexist - potential_gf_makes
        remaining_gf_bread_after_gf = num_gf_bread - potential_gf_makes
        remaining_gf_content_after_gf = num_gf_content - potential_gf_makes

        potential_reg_makes = min(num_reg_bread + remaining_gf_bread_after_gf,
                                  num_reg_content + remaining_gf_content_after_gf,
                                  remaining_notexist_after_gf)

        gf_s1_available = potential_gf_makes
        reg_s1_available = potential_reg_makes


        # Calculate heuristic by satisfying needs from lowest cost stages first
        heuristic = 0
        gf_needs_remaining = total_gf_needed
        reg_needs_remaining = total_reg_needed

        # --- Satisfy GF needs (allergic children) ---

        # Stage 4 (Cost 1: Serve)
        num_from_s4_gf = min(gf_needs_remaining, gf_s4_available)
        heuristic += num_from_s4_gf * 1
        gf_needs_remaining -= num_from_s4_gf
        # Remaining GF s4 sandwiches are already accounted for in gf_s3_available

        # Stage 3 (Cost 2: Move + Serve)
        num_from_s3_gf = min(gf_needs_remaining, gf_s3_available)
        heuristic += num_from_s3_gf * 2
        gf_needs_remaining -= num_from_s3_gf
        gf_s3_remaining_for_reg = gf_s3_available - num_from_s3_gf # Remaining GF s3 can serve reg

        # Stage 2 (Cost 3: Put + Move + Serve)
        num_from_s2_gf = min(gf_needs_remaining, gf_s2_available)
        heuristic += num_from_s2_gf * 3
        gf_needs_remaining -= num_from_s2_gf
        gf_s2_remaining_for_reg = gf_s2_available - num_from_s2_gf # Remaining GF s2 can serve reg

        # Stage 1 (Cost 4: Make + Put + Move + Serve)
        num_from_s1_gf = min(gf_needs_remaining, gf_s1_available)
        heuristic += num_from_s1_gf * 4
        gf_needs_remaining -= num_from_s1_gf
        gf_s1_remaining_for_reg = gf_s1_available - num_from_s1_gf # Remaining GF s1 can serve reg


        # --- Satisfy Reg needs (non-allergic children) ---

        # Stage 4 (Cost 1: Serve)
        num_from_s4_reg = min(reg_needs_remaining, reg_s4_available)
        heuristic += num_from_s4_reg * 1
        reg_needs_remaining -= num_from_s4_reg

        # Stage 3 (Cost 2: Move + Serve) - Includes remaining GF from stages 4 and 3
        # Remaining GF from Stage 4 (at places with allergic needs) are NOT in gf_s3_available.
        # They were counted in gf_s4_available. We need to add them here.
        gf_s4_remaining_total = gf_s4_available - num_from_s4_gf # These need a move to a place with reg need
        reg_s3_total_available = reg_s3_available + gf_s3_remaining_for_reg + gf_s4_remaining_total
        num_from_s3_reg = min(reg_needs_remaining, reg_s3_total_available)
        heuristic += num_from_s3_reg * 2
        reg_needs_remaining -= num_from_s3_reg

        # Stage 2 (Cost 3: Put + Move + Serve) - Includes remaining GF from stage 2
        reg_s2_total_available = reg_s2_available + gf_s2_remaining_for_reg
        num_from_s2_reg = min(reg_needs_remaining, reg_s2_total_available)
        heuristic += num_from_s2_reg * 3
        reg_needs_remaining -= num_from_s2_reg

        # Stage 1 (Cost 4: Make + Put + Move + Serve) - Includes remaining GF from stage 1
        reg_s1_total_available = reg_s1_available + gf_s1_remaining_for_reg
        num_from_s1_reg = min(reg_needs_remaining, reg_s1_total_available)
        heuristic += num_from_s1_reg * 4
        reg_needs_remaining -= num_from_s1_reg

        # If gf_needs_remaining > 0 or reg_needs_remaining > 0, it means we couldn't find enough potential sandwiches.
        # This state might be unsolvable or the heuristic underestimated.
        # For a non-admissible heuristic, just returning the calculated cost is fine.

        return heuristic

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into a list of parts."""
        # Remove surrounding parentheses and split by spaces
        return fact_str.strip('()').split()
