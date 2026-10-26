from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    unserved children. It does this by counting the number of sandwiches of each
    type (gluten-free or regular) needed and the "stage" of completion for
    available sandwiches (ready to serve, on tray wrong place, in kitchen, or
    needs making). It assumes sufficient trays are available for transport and
    serving.

    # Assumptions
    - Each child requires exactly one sandwich.
    - Allergic children require gluten-free sandwiches. Non-allergic children
      can accept any sandwich.
    - The cost of making, putting on tray, moving tray, and serving are 1 action each.
    - Sufficient trays are available to move sandwiches when needed.
    - Ingredient availability is checked only for sandwiches that need to be made.
    - The set of all possible sandwich names is implicitly defined by the names
      appearing in initial state facts (at_kitchen_sandwich, ontray, notexist).

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the goal.
    - Extracts static information about children (allergy status, waiting place)
      and ingredients (gluten-free bread/content) from static facts.
    - Extracts all possible place names.
    - Initializes attributes to store all bread and content names, which are
      collected dynamically from the state during the first heuristic call.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of unserved children from the goal and current state.
    2. Separate unserved children into those needing gluten-free sandwiches and
       those needing regular sandwiches. Count the total number of each type needed.
    3. Identify all existing sandwiches in the current state and determine if
       each is gluten-free or regular based on the `no_gluten_sandwich` predicate.
    4. Categorize available sandwiches (existing sandwiches) into stages based on their location
       relative to the unserved children's waiting places and their suitability:
       - Stage 4 (Cost 1): A sandwich suitable for a child of a certain type (GF or Reg)
         that is currently on a tray at the location of *any* unserved child of that type.
       - Stage 3 (Cost 2): A sandwich suitable for a child of a certain type,
         that is currently on a tray anywhere, but *not* counted in Stage 4 for that type.
       - Stage 2 (Cost 3): A sandwich suitable for a child of a certain type,
         that is currently in the kitchen, but *not* counted in Stage 4 or 3 for that type.
    5. Determine the number of sandwiches that still need to be made (Stage 1, Cost 4)
       by subtracting the counts of available sandwiches in Stages 2, 3, and 4
       from the total number needed for each type (GF/Reg).
    6. Check if enough sandwich names marked `notexist` are available and if
       necessary ingredients are in the kitchen to make the required Stage 1 sandwiches.
       If not, the state is likely unsolvable, return infinity.
    7. Calculate the total heuristic cost by greedily satisfying the sandwich needs
       (GF needs first, then Reg needs using remaining GF and available Reg)
       by drawing from the highest stages (lowest cost) first. Sum the costs
       (Stage 4: 1, Stage 3: 2, Stage 2: 3, Stage 1: 4) for the number of sandwiches
       taken from each stage.
    8. If, after allocating all available and makeable sandwiches, there are still
       unmet needs, the state is unsolvable, return infinity.
    9. The heuristic value is the total calculated cost. It is 0 if and only if
       all children are served (needed counts are 0).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal children
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract static child info
        self.child_allergic = {}
        self.child_waiting_place = {}
        self.all_places = set()

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergic[get_parts(fact)[1]] = True
            elif match(fact, "not_allergic_gluten", "*"):
                self.child_allergic[get_parts(fact)[1]] = False
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_waiting_place[child] = place
                self.all_places.add(place)
            # Add places mentioned as objects if any exist in static facts (less common)
            elif match(fact, "place", "*"):
                 self.all_places.add(get_parts(fact)[1])

        # Ensure kitchen is included as a place
        self.all_places.add("kitchen")

        # Extract static ingredient info (only GF status is static)
        self.gf_breads = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.gf_contents = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

        # All bread and content names are collected dynamically from the state
        self._all_breads = None
        self._all_contents = None


    def _collect_ingredient_names(self, state):
         """Collect all bread and content names from the current state."""
         if self._all_breads is None:
             self._all_breads = set()
             self._all_contents = set()
             for fact in state:
                 if match(fact, "at_kitchen_bread", "*"):
                     self._all_breads.add(get_parts(fact)[1])
                 elif match(fact, "at_kitchen_content", "*"):
                     self._all_contents.add(get_parts(fact)[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Collect ingredient names if not already done
        self._collect_ingredient_names(state)

        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if f"(served {c})" not in state}

        # If all children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        # 2. Separate unserved children by need
        unserved_children_gf = {c for c in unserved_children if self.child_allergic.get(c, False)}
        unserved_children_reg = {c for c in unserved_children if not self.child_allergic.get(c, True)} # Default to not allergic if status unknown

        needed_gf = len(unserved_children_gf)
        needed_reg = len(unserved_children_reg)

        # 3. Identify existing sandwiches and their type
        gf_sandwiches_in_state = {get_parts(f)[1] for f in state if match(f, "no_gluten_sandwich", "*")}
        existing_sandwiches = {get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "*") or match(f, "ontray", "*", "*")}
        reg_sandwiches_in_state = {s for s in existing_sandwiches if s not in gf_sandwiches_in_state}

        # 4. Categorize available sandwiches by stage and type suitability
        gf_s4_set = set() # GF sandwich on tray at GF child's location
        reg_s4_set = set() # Reg sandwich on tray at Reg child's location (can also be GF)
        gf_s3_set = set() # GF sandwich on tray elsewhere, suitable for unserved GF
        reg_s3_set = set() # Reg sandwich on tray elsewhere, suitable for unserved Reg
        gf_s2_set = set() # GF sandwich in kitchen, suitable for unserved GF
        reg_s2_set = set() # Reg sandwich in kitchen, suitable for unserved Reg

        # Stage 4: On tray at correct location for *any* unserved child needing that type
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                is_gf_sandwich = (s in gf_sandwiches_in_state)
                is_reg_sandwich = (s in reg_sandwiches_in_state)

                for child in unserved_children:
                    child_place = self.child_waiting_place.get(child)
                    if child_place and match(f"(at {t} {child_place})", "at", t, child_place):
                        needs_gf = self.child_allergic.get(child, False)
                        if needs_gf and is_gf_sandwich:
                            gf_s4_set.add(s)
                        elif not needs_gf and (is_reg_sandwich or is_gf_sandwich): # Reg child can take any
                            reg_s4_set.add(s)

        # Stage 3: On tray anywhere, not S4, suitable for *any* unserved child needing that type
        ontray_sandwiches = {get_parts(f)[1] for f in state if match(fact, "ontray", "*", "*")}
        for s in ontray_sandwiches:
            # Check if already accounted for in Stage 4 for either type
            if s in gf_s4_set or s in reg_s4_set:
                continue

            is_gf_sandwich = (s in gf_sandwiches_in_state)
            is_reg_sandwich = (s in reg_sandwiches_in_state)

            # Is suitable for *any* unserved GF child?
            suitable_for_any_unserved_gf = is_gf_sandwich and needed_gf > 0
            # Is suitable for *any* unserved Reg child?
            suitable_for_any_unserved_reg = (is_reg_sandwich or is_gf_sandwich) and needed_reg > 0

            if suitable_for_any_unserved_gf:
                gf_s3_set.add(s)
            if suitable_for_any_unserved_reg:
                reg_s3_set.add(s)

        # Stage 2: In kitchen, not S4/S3, suitable for *any* unserved child needing that type
        kitchen_sandwiches = {get_parts(f)[1] for f in state if match(fact, "at_kitchen_sandwich", "*")}
        for s in kitchen_sandwiches:
             # Check if already accounted for in higher stages
            if s in gf_s4_set or s in reg_s4_set or s in gf_s3_set or s in reg_s3_set:
                continue

            is_gf_sandwich = (s in gf_sandwiches_in_state)
            is_reg_sandwich = (s in reg_sandwiches_in_state)

            # Is suitable for *any* unserved GF child?
            suitable_for_any_unserved_gf = is_gf_sandwich and needed_gf > 0
            # Is suitable for *any* unserved Reg child?
            suitable_for_any_unserved_reg = (is_reg_sandwich or is_gf_sandwich) and needed_reg > 0

            if suitable_for_any_unserved_gf:
                gf_s2_set.add(s)
            if suitable_for_any_unserved_reg:
                reg_s2_set.add(s)

        # Count unique sandwiches in each stage, prioritizing higher stages
        # A sandwich is counted in the highest stage it qualifies for, for its *specific type* (GF or Reg suitability).
        # Note: A GF sandwich suitable for a Reg child at their location counts towards reg_s4_set.
        # A GF sandwich suitable for a GF child at their location counts towards gf_s4_set.
        # If a GF sandwich is at a location with both types, it's in both sets. This is fine,
        # the allocation logic handles that it can only be used once.

        avail_gf_s4 = len(gf_s4_set)
        avail_reg_s4 = len(reg_s4_set)

        # Sandwiches in S3 are those on trays not in S4, suitable for their type.
        # Ensure we don't double count sandwiches already in S4 sets.
        avail_gf_s3 = len(gf_s3_set - gf_s4_set)
        avail_reg_s3 = len(reg_s3_set - reg_s4_set)

        # Sandwiches in S2 are those in kitchen not in S4/S3, suitable for their type.
        # Ensure we don't double count sandwiches already in S4 or S3 sets.
        avail_gf_s2 = len(gf_s2_set - gf_s4_set - gf_s3_set)
        avail_reg_s2 = len(reg_s2_set - reg_s4_set - reg_s3_set)


        # Stage 1: Makeable
        num_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))
        has_gf_bread = any(match(f, "at_kitchen_bread", b) for f in state for b in self.gf_breads)
        has_gf_content = any(match(f, "at_kitchen_content", c) for f in state for c in self.gf_contents)
        has_any_bread = any(match(f, "at_kitchen_bread", "*") for f in state)
        has_any_content = any(match(f, "at_kitchen_content", "*") for f in state)

        can_make_gf_ing = has_gf_bread and has_gf_content
        can_make_reg_ing = has_any_bread and has_any_content # Reg can use any ingredients

        # 5. Calculate needed sandwiches from Stage 1
        needed_gf_from_s1 = max(0, needed_gf - (avail_gf_s4 + avail_gf_s3 + avail_gf_s2))
        # Reg needs can be met by any sandwich. Count total available suitable sandwiches (GF+Reg)
        total_avail_s4 = avail_gf_s4 + avail_reg_s4
        total_avail_s3 = avail_gf_s3 + avail_reg_s3
        total_avail_s2 = avail_gf_s2 + avail_reg_s2

        needed_reg_from_s1 = max(0, needed_reg - (total_avail_s4 + total_avail_s3 + total_avail_s2))


        # Check if makeable
        makeable_gf_s1_count = 0
        if can_make_gf_ing:
            makeable_gf_s1_count = min(needed_gf_from_s1, num_notexist)

        makeable_reg_s1_count = 0
        if can_make_reg_ing:
             # Use remaining notexist names after potentially making GF ones
            makeable_reg_s1_count = min(needed_reg_from_s1, num_notexist - makeable_gf_s1_count)

        # 6. Check if unsolvable based on makeable sandwiches
        if needed_gf_from_s1 > makeable_gf_s1_count or needed_reg_from_s1 > makeable_reg_s1_count:
             return float('inf')

        # 7. Calculate total heuristic cost by allocating sandwiches
        total_heuristic = 0

        # Satisfy GF needs first
        use_gf_s4 = min(needed_gf, avail_gf_s4)
        total_heuristic += use_gf_s4 * 1
        needed_gf -= use_gf_s4

        use_gf_s3 = min(needed_gf, avail_gf_s3)
        total_heuristic += use_gf_s3 * 2
        needed_gf -= use_gf_s3

        use_gf_s2 = min(needed_gf, avail_gf_s2)
        total_heuristic += use_gf_s2 * 3
        needed_gf -= use_gf_s2

        use_gf_s1 = min(needed_gf, makeable_gf_s1_count)
        total_heuristic += use_gf_s1 * 4
        needed_gf -= use_gf_s1
        # num_notexist -= use_gf_s1 # Don't modify state counts here

        # If needed_gf > 0, should have returned inf earlier, but double check
        if needed_gf > 0: return float('inf')

        # Satisfy Reg needs using remaining GF and available Reg
        # Recalculate remaining available sandwiches after GF allocation
        rem_gf_s4 = avail_gf_s4 - use_gf_s4
        rem_gf_s3 = avail_gf_s3 - use_gf_s3
        rem_gf_s2 = avail_gf_s2 - use_gf_s2
        rem_gf_s1 = makeable_gf_s1_count - use_gf_s1

        # Reg needs can use remaining GF or available Reg
        # Prioritize remaining GF first
        use_rem_gf_s4 = min(needed_reg, rem_gf_s4)
        total_heuristic += use_rem_gf_s4 * 1
        needed_reg -= use_rem_gf_s4

        use_rem_gf_s3 = min(needed_reg, rem_gf_s3)
        total_heuristic += use_rem_gf_s3 * 2
        needed_reg -= use_rem_gf_s3

        use_rem_gf_s2 = min(needed_reg, rem_gf_s2)
        total_heuristic += use_rem_gf_s2 * 3
        needed_reg -= use_rem_gf_s2

        use_rem_gf_s1 = min(needed_reg, rem_gf_s1)
        total_heuristic += use_rem_gf_s1 * 4
        needed_reg -= use_rem_gf_s1

        # Now use Reg sandwiches for remaining Reg needs
        use_reg_s4 = min(needed_reg, avail_reg_s4)
        total_heuristic += use_reg_s4 * 1
        needed_reg -= use_reg_s4

        use_reg_s3 = min(needed_reg, avail_reg_s3)
        total_heuristic += use_reg_s3 * 2
        needed_reg -= use_reg_s3

        use_reg_s2 = min(needed_reg, avail_reg_s2)
        total_heuristic += use_reg_s2 * 3
        needed_reg -= use_reg_s2

        # Use makeable Reg
        # Need to calculate makeable_reg_s1_count again based on remaining notexist names
        remaining_notexist = num_notexist - use_gf_s1
        makeable_reg_s1_count_actual = 0
        if can_make_reg_ing:
             makeable_reg_s1_count_actual = min(needed_reg, remaining_notexist)

        use_reg_s1 = min(needed_reg, makeable_reg_s1_count_actual)
        total_heuristic += use_reg_s1 * 4
        needed_reg -= use_reg_s1


        # 8. Final check for unsolvable state
        if needed_reg > 0:
             return float('inf')

        # 9. Return total heuristic cost
        return total_heuristic
