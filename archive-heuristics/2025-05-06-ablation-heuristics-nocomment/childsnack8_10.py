# Assuming heuristics.heuristic_base.Heuristic is available
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Helper function to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to serve all waiting children.
    The heuristic considers the steps needed to get the correct type of
    sandwich (regular or gluten-free) onto a tray at the child's location,
    plus the final serving action for each child. It prioritizes using
    sandwiches that are closer to being served (already on trays at wrong
    locations < in kitchen < need to be made from ingredients).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts from the task.

        Heuristic Initialization:
        Extracts information about child allergies, waiting locations,
        gluten-free ingredients, and identifies all children and places.
        This information is stored in instance variables for efficient lookup
        during heuristic computation.
        """
        super().__init__(task) # Call base class constructor

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children = {} # child -> place
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.all_children = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        # Process static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'allergic_gluten':
                child = args[0]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = args[0]
                self.not_allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'waiting':
                child, place = args
                self.waiting_children[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif predicate == 'no_gluten_bread':
                self.no_gluten_breads.add(args[0])
            elif predicate == 'no_gluten_content':
                self.no_gluten_contents.add(args[0])

        # Add children from goals who might not be in static allergy/waiting facts
        # (though typically they would be in waiting facts)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served':
                 self.all_children.add(parts[1])


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer estimate of the remaining actions to reach a goal state.
            Returns 0 if the state is a goal state.

        Step-By-Step Thinking for Computing Heuristic:

        1.  Identify Unserved Children: Determine which children in the goal
            are not yet marked as 'served' in the current state. If all are
            served, the heuristic is 0.
        2.  Categorize Unserved Children: Count how many unserved children
            are allergic (need gluten-free sandwiches) and how many are not
            allergic (need regular sandwiches).
        3.  Gather State Information: Iterate through the current state facts
            to count available resources and their locations:
            -   Number of bread/content portions in the kitchen (categorized by gluten-free status).
            -   Number of sandwich slots that 'notexist'.
            -   Number of sandwiches already made, categorized by gluten-free
                status and location (in kitchen, or on a tray).
            -   Location of each tray.
            -   Which sandwich is on which tray.
            -   Which sandwich is gluten-free.
        4.  Calculate Makable Sandwiches: Determine how many gluten-free and
            regular sandwiches can be made from available ingredients and
            'notexist' slots. Slots are a shared resource.
        5.  Calculate Sandwiches Needed at Correct Locations: For each
            unserved child, they need a suitable sandwich on a tray at their
            waiting location. Count how many sandwiches of each type (GF/Reg)
            are *still needed* on trays at the correct locations, after
            accounting for suitable sandwiches already on trays at those locations.
            This is done by iterating through unserved children and trying to match
            them with suitable sandwiches already on trays at their location,
            keeping track of used sandwiches to avoid double counting.
        6.  Estimate Cost to Get Sandwiches to Correct Trays: The remaining
            needed sandwiches must come from available sources:
            -   Source 1 (Cost 1 per sandwich): Sandwiches already on trays
                but at the wrong location (requires 1 'move_tray' action).
            -   Source 2 (Cost 2 per sandwich): Sandwiches in the kitchen
                (requires 1 'put_on_tray' + 1 'move_tray' action). Assumes
                a tray is available in the kitchen when needed.
            -   Source 3 (Cost 3 per sandwich): Sandwiches that need to be made
                (requires 1 'make' + 1 'put_on_tray' + 1 'move_tray' action).
                Requires ingredients, a slot, and a tray in the kitchen.
            Greedily assign needed sandwiches to the cheapest available source
            until all needed sandwiches are accounted for. Sum the costs.
        7.  Add Serving Cost: Each unserved child requires a final 'serve'
            action, which costs 1. Add the total number of unserved children
            to the cost calculated in step 6.
        8.  Return Total Cost: The sum represents the estimated minimum
            actions remaining.
        """
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = {c for c in self.all_children if f'(served {c})' not in state}

        if not unserved_children:
            return 0 # Goal state

        # 2. Categorize Unserved Children (Total counts needed)
        N_gf = sum(1 for c in unserved_children if c in self.allergic_children)
        N_reg = len(unserved_children) - N_gf

        # 3. Gather State Information
        B_gf_k = 0
        B_reg_k = 0
        C_gf_k = 0
        C_reg_k = 0
        Sandwich_slots_notexist = 0
        Sandwich_is_gf = {} # map sandwich -> bool
        Sandwich_ontray_map = {} # map sandwich -> tray
        Tray_locations = {} # map tray -> place
        Sandwiches_in_kitchen = set() # set of sandwiches at_kitchen_sandwich

        # Collect basic facts first
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_kitchen_bread':
                bread = args[0]
                if bread in self.no_gluten_breads: B_gf_k += 1
                else: B_reg_k += 1
            elif predicate == 'at_kitchen_content':
                content = args[0]
                if content in self.no_gluten_contents: C_gf_k += 1
                else: C_reg_k += 1
            elif predicate == 'notexist':
                Sandwich_slots_notexist += 1
            elif predicate == 'no_gluten_sandwich':
                Sandwich_is_gf[args[0]] = True
            elif predicate == 'ontray':
                sandwich, tray = args
                Sandwich_ontray_map[sandwich] = tray
            elif predicate == 'at':
                tray, place = args
                Tray_locations[tray] = place
            elif predicate == 'at_kitchen_sandwich':
                 Sandwiches_in_kitchen.add(args[0])


        # Categorize made sandwiches by location (kitchen vs on tray) and type (GF vs Reg)
        S_gf_k = 0 # GF sandwiches in kitchen
        S_reg_k = 0 # Regular sandwiches in kitchen
        S_gf_t = 0 # GF sandwiches on trays (any location)
        S_reg_t = 0 # Regular sandwiches on trays (any location)

        # Need to find all sandwich objects that exist (not notexist)
        # These are the ones that are either at_kitchen_sandwich or ontray
        existing_sandwiches = set(Sandwich_is_gf.keys()) | set(Sandwich_ontray_map.keys()) | Sandwiches_in_kitchen

        for s in existing_sandwiches:
            is_gf = Sandwich_is_gf.get(s, False)
            is_k = s in Sandwiches_in_kitchen
            is_t = s in Sandwich_ontray_map

            if is_k:
                if is_gf: S_gf_k += 1
                else: S_reg_k += 1
            elif is_t:
                if is_gf: S_gf_t += 1
                else: S_reg_t += 1
            # Sandwiches that are neither in kitchen nor on tray are assumed served or otherwise unavailable.


        # 4. Calculate Makable Sandwiches
        avail_ingredients_gf = min(B_gf_k, C_gf_k)
        avail_ingredients_reg = min(B_reg_k, C_reg_k)

        # Slots are consumed sequentially
        avail_gf_makable = min(avail_ingredients_gf, Sandwich_slots_notexist)
        remaining_slots = Sandwich_slots_notexist - avail_gf_makable
        avail_reg_makable = min(avail_ingredients_reg, remaining_slots)


        # 5. Calculate Sandwiches Needed at Correct Locations
        # Count sandwiches already on trays at the correct location for an unserved child
        served_by_ontray_at_loc_gf = 0
        served_by_ontray_at_loc_reg = 0
        used_ontray_sandwiches = set() # Ensure each sandwich serves at most one child's need

        # Group unserved children by location and allergy
        unserved_needs = {} # (place, is_allergic) -> count
        for child in unserved_children:
            place = self.waiting_children.get(child) # Should always exist based on domain
            is_allergic = child in self.allergic_children
            key = (place, is_allergic)
            unserved_needs[key] = unserved_needs.get(key, 0) + 1

        # Match available on-tray sandwiches at location to needs
        # Iterate through needs groups
        for (place, is_allergic), count_needed in unserved_needs.items():
             needed_type_is_gf = is_allergic
             found_count = 0
             # Find suitable sandwiches on trays at this place
             for s, t in Sandwich_ontray_map.items():
                 if s in used_ontray_sandwiches: continue
                 if Tray_locations.get(t) == place: # Tray is at child's location
                     is_gf_s = Sandwich_is_gf.get(s, False)
                     if (needed_type_is_gf and is_gf_s) or (not needed_type_is_gf and not is_gf_s): # Sandwich is suitable
                         # Found one suitable sandwich at the correct location
                         found_count += 1
                         used_ontray_sandwiches.add(s)
                         if found_count == count_needed: break # Found enough for this group

             if needed_type_is_gf: served_by_ontray_at_loc_gf += found_count
             else: served_by_ontray_at_loc_reg += found_count


        # Total sandwiches of each type needed on trays at correct locations
        needed_gf_loc = max(0, N_gf - served_by_ontray_at_loc_gf)
        needed_reg_loc = max(0, N_reg - served_by_ontray_at_loc_reg)

        # 6. Estimate Cost to Get Sandwiches to Correct Trays
        total_cost = 0

        # Calculate available sandwiches by source, excluding those already assigned above
        # Total on tray - those used for served_by_ontray_at_loc
        avail_gf_wrong_tray = S_gf_t - served_by_ontray_at_loc_gf
        avail_reg_wrong_tray = S_reg_t - served_by_ontray_at_loc_reg

        avail_gf_kitchen = S_gf_k
        avail_reg_kitchen = S_reg_k

        # Available makable calculated earlier: avail_gf_makable, avail_reg_makable

        # Source 1 (cost 1): Sandwiches on trays at wrong locations (move tray)
        use_gf_wrong_tray = min(needed_gf_loc, avail_gf_wrong_tray)
        total_cost += use_gf_wrong_tray * 1
        needed_gf_loc -= use_gf_wrong_tray

        use_reg_wrong_tray = min(needed_reg_loc, avail_reg_wrong_tray)
        total_cost += use_reg_wrong_tray * 1
        needed_reg_loc -= use_reg_wrong_tray

        # Source 2 (cost 2): Sandwiches in the kitchen (put on tray + move tray)
        use_gf_kitchen = min(needed_gf_loc, avail_gf_kitchen)
        total_cost += use_gf_kitchen * 2
        needed_gf_loc -= use_gf_kitchen

        use_reg_kitchen = min(needed_reg_loc, avail_reg_kitchen)
        total_cost += use_reg_kitchen * 2
        needed_reg_loc -= use_reg_kitchen

        # Source 3 (cost 3): Sandwiches to be made (make + put on tray + move tray)
        use_gf_makable = min(needed_gf_loc, avail_gf_makable)
        total_cost += use_gf_makable * 3
        needed_gf_loc -= use_gf_makable

        use_reg_makable = min(needed_reg_loc, avail_reg_makable)
        total_cost += use_reg_makable * 3
        needed_reg_loc -= use_reg_makable

        # If needed_gf_loc > 0 or needed_reg_loc > 0 here, it means we couldn't find enough
        # potential sandwiches. This shouldn't happen in solvable instances if
        # the initial state and goals are consistent with domain rules, but
        # returning infinity is a safe fallback.
        # For a greedy search, returning a large finite number is also an option.
        # Let's assume solvable instances and needed_loc becomes 0.
        # if needed_gf_loc > 0 or needed_reg_loc > 0:
        #     # This state is likely unsolvable or heuristic underestimated makable
        #     # For greedy search, returning a large number is fine.
        #     # A simple approach is to add a large penalty, or just rely on search failure.
        #     # Let's assume solvable and continue.

        # 7. Add Serving Cost
        # Each unserved child needs one final 'serve' action.
        total_cost += len(unserved_children)

        # 8. Return Total Cost
        return total_cost
