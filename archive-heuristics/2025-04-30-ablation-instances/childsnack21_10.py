from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Depending on expected input robustness, could raise an error
         return [] # Indicate invalid fact
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    It counts the number of suitable sandwiches needed and sums the estimated minimum
    actions to get a sandwich from its current state (not existing, in kitchen,
    on tray at kitchen, on tray at child's location) to the point of being served.
    It prioritizes using sandwiches that are closer to being served.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can receive any sandwich (but gluten-free ones are prioritized for allergic children).
    - The costs assumed for moving sandwiches through the process stages are fixed:
        - Serve (sandwich on tray at child's location): 1 action.
        - Move tray + Serve (sandwich on tray at kitchen): 2 actions.
        - Put on tray + Move tray + Serve (sandwich in kitchen): 3 actions.
        - Make + Put on tray + Move tray + Serve (sandwich not existing): 4 actions.
    - Sufficient trays are available or can be moved without significant extra cost beyond the steps counted for the sandwich itself.
    - Sufficient ingredients are available to make any sandwich that doesn't exist, up to the count of 'notexist' sandwiches, limited by available ingredients.

    # Heuristic Initialization
    - Identify all children and their allergy status from static facts.
    - Identify all goal conditions (all children served).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the goal is reached (all children served). If yes, heuristic is 0.
    2. Count the number of unserved children, separating them by allergy status (gluten-allergic vs. not allergic). Let these counts be `needed_gf` and `needed_reg`.
    3. Count the number of suitable sandwiches available in different stages based on the current state:
       - `available_gf_ready`: Gluten-free sandwiches on trays at any location *other than* the kitchen (i.e., at a child's waiting place).
       - `available_reg_ready`: Regular sandwiches (or GF ones not needed by allergic children) on trays at any location *other than* the kitchen.
       - `available_gf_on_tray_kitchen`: Gluten-free sandwiches on trays at the kitchen.
       - `available_reg_on_tray_kitchen`: Regular sandwiches on trays at the kitchen.
       - `available_gf_kitchen`: Gluten-free sandwiches in the kitchen (not on a tray).
       - `available_reg_kitchen`: Regular sandwiches in the kitchen (not on a tray).
       - `num_notexist`: Sandwiches that do not exist yet.
       - Count available ingredients in the kitchen: `num_gf_bread`, `num_gf_content`, `num_reg_bread`, `num_reg_content`.
    4. Calculate the maximum number of new sandwiches that can be made based on available ingredients and `notexist` facts:
       - `can_make_gf`: Limited by GF bread, GF content, and `num_notexist`.
       - `can_make_reg`: Limited by remaining bread (Reg + any leftover GF), remaining content (Reg + any leftover GF), and remaining `num_notexist`. Prioritize using GF ingredients for GF sandwiches first.
    5. Calculate the heuristic cost by greedily assigning available sandwiches/make actions to needed sandwiches, starting with the cheapest options:
       - Satisfy `needed_gf` using `available_gf_ready` (cost 1).
       - Satisfy remaining `needed_gf` using `available_gf_on_tray_kitchen` (cost 2).
       - Satisfy remaining `needed_gf` using `available_gf_kitchen` (cost 3).
       - Satisfy remaining `needed_gf` using `can_make_gf` (cost 4).
       - Satisfy `needed_reg` using `available_reg_ready` (cost 1).
       - Satisfy remaining `needed_reg` using `available_reg_on_tray_kitchen` (cost 2).
       - Satisfy remaining `needed_reg` using `available_reg_kitchen` (cost 3).
       - Satisfy remaining `needed_reg` using `can_make_reg` (cost 4).
    6. Sum up the costs from each stage. If any children still need serving after exhausting all potential sandwich sources, the heuristic returns the calculated cost based on available resources.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children's
        allergy status and goal conditions.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store allergy status for each child
        self.child_allergy = {}
        # Store waiting place for each child (useful for 'ready' sandwiches)
        self.child_place = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] in ["allergic_gluten", "not_allergic_gluten"]:
                child_name = parts[1]
                self.child_allergy[child_name] = (parts[0] == "allergic_gluten") # True if allergic
            elif parts[0] == "waiting":
                 child_name = parts[1]
                 place_name = parts[2]
                 self.child_place[child_name] = place_name


        # Store the set of children who need to be served (all children in the problem)
        # We can infer the set of all children from the static facts about allergies or waiting.
        # Let's use the keys from child_allergy, assuming all children mentioned in allergies are the ones to be served.
        self.all_children = set(self.child_allergy.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Count unserved children by allergy status
        needed_gf = 0
        needed_reg = 0
        served_children = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "served":
                 served_children.add(parts[1])

        unserved_children = self.all_children - served_children

        for child in unserved_children:
            # Default to not allergic if status unknown (shouldn't happen with typical PDDL)
            if self.child_allergy.get(child, False):
                needed_gf += 1
            else:
                needed_reg += 1

        # If no children need serving, but goal isn't reached, something is wrong or goal is complex.
        # Assuming goals are ONLY served predicates for all children.
        if needed_gf == 0 and needed_reg == 0:
             # This should only happen if self.goals <= state is True, but handle defensively.
             return 0

        # 2. Count available sandwiches/ingredients in different states
        num_available_gf_ready = 0 # on tray at child's place
        num_available_reg_ready = 0 # on tray at child's place
        num_available_gf_on_tray_kitchen = 0
        num_available_reg_on_tray_kitchen = 0
        num_available_gf_kitchen = 0 # not on tray
        num_available_reg_kitchen = 0 # not on tray
        num_gf_bread = 0
        num_gf_content = 0
        num_reg_bread = 0
        num_reg_content = 0
        num_notexist = 0

        # Helper to check if a sandwich is gluten-free based on state fact
        is_gf_sandwich_in_state = lambda s, current_state: f"(no_gluten_sandwich {s})" in current_state

        # Helper to find tray location for a sandwich on a tray
        def get_tray_location(sandwich, current_state):
             for fact in current_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == "ontray" and parts[1] == sandwich:
                     tray = parts[2]
                     for tray_fact in current_state:
                         t_parts = get_parts(tray_fact)
                         if t_parts and t_parts[0] == "at" and t_parts[1] == tray:
                             return t_parts[2] # Return place name
             return None # Not on a tray or tray location unknown

        # Helper to check if a sandwich on a tray is at a place where an unserved child is waiting
        def is_at_unserved_child_place(sandwich, current_state, unserved_children_set, child_place_map):
            tray_loc = get_tray_location(sandwich, current_state)
            if tray_loc and tray_loc != "kitchen":
                # Check if any unserved child is waiting at this location
                for child in unserved_children_set:
                    if child_place_map.get(child) == tray_loc:
                        return True
            return False


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == "at_kitchen_bread":
                bread = parts[1]
                # Check static facts for no_gluten status
                is_gf = any(match(static_fact, "no_gluten_bread", bread) for static_fact in self.static)
                if is_gf:
                     num_gf_bread += 1
                else:
                     num_reg_bread += 1
            elif predicate == "at_kitchen_content":
                 content = parts[1]
                 # Check static facts for no_gluten status
                 is_gf = any(match(static_fact, "no_gluten_content", content) for static_fact in self.static)
                 if is_gf:
                     num_gf_content += 1
                 else:
                     num_reg_content += 1
            elif predicate == "at_kitchen_sandwich":
                 sandwich = parts[1]
                 if is_gf_sandwich_in_state(sandwich, state):
                     num_available_gf_kitchen += 1
                 else:
                     num_available_reg_kitchen += 1
            elif predicate == "ontray":
                 sandwich = parts[1]
                 tray_loc = get_tray_location(sandwich, state)
                 if tray_loc: # Should always find location if ontray
                     is_gf = is_gf_sandwich_in_state(sandwich, state)
                     if tray_loc == "kitchen":
                         if is_gf:
                             num_available_gf_on_tray_kitchen += 1
                         else:
                             num_available_reg_on_tray_kitchen += 1
                     else: # Tray is at a place other than kitchen
                         # Check if this sandwich/tray combination is at a place where an unserved child is waiting
                         # This counts sandwiches ready *at any* unserved child's location, not necessarily the specific one needing it.
                         # This is a simplification, but okay for non-admissible.
                         if is_at_unserved_child_place(sandwich, state, unserved_children, self.child_place):
                             if is_gf:
                                 num_available_gf_ready += 1
                             else:
                                 num_available_reg_ready += 1
                         # Note: Sandwiches on trays at places where no unserved child is waiting are not counted as "ready".
                         # They would need a move action, similar to sandwiches on trays at the kitchen.
                         # Let's add them to the kitchen count for simplicity in this heuristic.
                         else:
                             if is_gf:
                                 num_available_gf_on_tray_kitchen += 1 # Treat as if at kitchen for cost calculation
                             else:
                                 num_available_reg_on_tray_kitchen += 1 # Treat as if at kitchen for cost calculation


            elif predicate == "notexist":
                 num_notexist += 1

        # 3. Calculate potential make actions (prioritize GF)
        can_make_gf = min(num_gf_bread, num_gf_content, num_notexist)
        remaining_notexist = num_notexist - can_make_gf
        remaining_gf_bread = num_gf_bread - can_make_gf
        remaining_gf_content = num_gf_content - can_make_gf

        # Regular sandwiches can use regular ingredients or leftover GF ingredients
        can_make_reg = min(num_reg_bread + remaining_gf_bread, num_reg_content + remaining_gf_content, remaining_notexist)


        # 4. Calculate heuristic cost by assigning resources greedily
        cost = 0

        # Stage 1: Use sandwiches already at children's locations (Cost 1: Serve)
        # These are sandwiches on trays at places where unserved children are waiting.
        use_gf = min(needed_gf, num_available_gf_ready)
        cost += use_gf * 1
        needed_gf -= use_gf

        use_reg = min(needed_reg, num_available_reg_ready)
        cost += use_reg * 1
        needed_reg -= use_reg

        # Stage 2: Use sandwiches on trays at kitchen (or other non-child places, Cost 2: Move + Serve)
        use_gf = min(needed_gf, num_available_gf_on_tray_kitchen)
        cost += use_gf * 2
        needed_gf -= use_gf

        use_reg = min(needed_reg, num_available_reg_on_tray_kitchen)
        cost += use_reg * 2
        needed_reg -= use_reg

        # Stage 3: Use sandwiches in kitchen (not on tray) (Cost 3: Put on tray + Move + Serve)
        use_gf = min(needed_gf, num_available_gf_kitchen)
        cost += use_gf * 3
        needed_gf -= use_gf

        use_reg = min(needed_reg, num_available_reg_kitchen)
        cost += use_reg * 3
        needed_reg -= use_reg

        # Stage 4: Make new sandwiches (Cost 4: Make + Put on tray + Move + Serve)
        use_gf = min(needed_gf, can_make_gf)
        cost += use_gf * 4
        needed_gf -= use_gf
        # can_make_gf -= use_gf # Consume make actions (not strictly needed for heuristic value)

        # Note: Regular sandwiches can use remaining can_make_reg actions.
        # The remaining can_make_reg calculation already accounted for leftover GF ingredients.
        use_reg = min(needed_reg, can_make_reg)
        cost += use_reg * 4
        needed_reg -= use_reg
        # can_make_reg -= use_reg # Consume make actions (not strictly needed for heuristic value)


        # If needed_gf > 0 or needed_reg > 0 here, it means we don't have enough
        # notexist sandwiches or ingredients to make all needed sandwiches.
        # The heuristic simply returns the cost calculated so far based on available resources.
        # This is acceptable for a non-admissible heuristic.

        return cost

