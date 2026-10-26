from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
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
    This heuristic estimates the number of actions required to serve all unserved
    children. It calculates the minimum steps needed to get a suitable sandwich
    onto a tray at each child's table and then serve them, greedily assigning
    available resources (sandwiches, trays, ingredients, make opportunities)
    to children requiring the cheapest options first.

    # Assumptions
    - Each child needs exactly one suitable sandwich to be served.
    - A sandwich must be on a tray at the child's table to be served.
    - Suitable ingredients (bread, content) must be at the kitchen to make a sandwich.
    - A tray must be at the kitchen to put a sandwich onto it from the kitchen.
    - Trays can be moved between the kitchen and any table location (cost 1).
    - If a sandwich needs to be made, a corresponding `notexist` fact exists.
    - Resource availability (sandwiches, trays, ingredients, notexist slots) is tracked greedily for heuristic calculation.

    # Heuristic Initialization
    - Extracts static information: which children are allergic to gluten, which
      table each child is waiting at, and which ingredients are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who have not yet been served based on the goal state
       and the current state.
    2. Collect available resources in the current state:
       - Sandwiches currently on trays (and their tray's location).
       - Sandwiches currently at the kitchen (not on trays).
       - Sandwiches that do not yet exist (`notexist` facts).
       - Trays currently at the kitchen.
       - Trays currently at tables.
       - Available ingredients (bread and content) at the kitchen, categorized by
         gluten status.
       - Sandwiches marked as `no_gluten_sandwich`.
    3. Initialize the total heuristic cost to 0.
    4. Initialize sets to track resources (sandwiches, trays, notexist slots,
       ingredients) that are conceptually "used" by the heuristic calculation
       for already considered children, to avoid double-counting.
    5. Iterate through the unserved children in a fixed order (e.g., sorted by name)
       to ensure deterministic heuristic values.
    6. For each unserved child:
       a. Determine the child's waiting table and gluten allergy status.
       b. Find the minimum cost to get a *suitable* sandwich onto a tray at
          this child's table, considering the current state and available resources.
          Prioritize cheaper options:
          - Cost 0: A suitable sandwich is already on a tray at the child's table.
          - Cost 1: A suitable sandwich is on a tray at the kitchen (needs 1 move_tray).
          - Cost 1: A suitable sandwich is on a tray at another table (needs 1 move_tray).
          - Cost 2: A suitable sandwich is at the kitchen (not on tray) (needs 1 put_on_tray + 1 move_tray). Requires an available kitchen tray.
          - Cost 3: A suitable sandwich needs to be made (needs 1 make_sandwich + 1 put_on_tray + 1 move_tray). Requires an available `notexist` slot, available suitable ingredients at the kitchen, and an available kitchen tray.
       c. Select the cheapest available option for this child using resources
          that have not been "used" by previous children in this heuristic calculation pass.
       d. If a path is found (cost is not infinity):
          - Add the cost of preparing/moving the sandwich (0, 1, 1, 2, or 3) plus
            the cost of serving (1) to the total heuristic.
          - Mark the resources used by this child's plan (e.g., the specific
            sandwich, tray, notexist slot, ingredients) as "used" so they
            are not available for subsequent children in this calculation pass.
    7. Return the total calculated heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals # Goal conditions, e.g., frozenset({'(served child1)', ...})
        static_facts = task.static # Facts that are not affected by actions.

        # Extract static information
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.waiting_tables = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts if match(fact, "waiting", "*", "*")
        }
        # Ingredients with no_gluten property are static
        self.no_gluten_bread = {
             get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
             get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # 1. Identify unserved children
        unserved_children = sorted([
            get_parts(goal)[1] for goal in self.goals
            if match(goal, "served", "*") and goal not in state
        ])

        # If all children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        # 2. Collect available resources in the current state
        available_trays_at_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at", "*", "kitchen") and match(fact, "tray", get_parts(fact)[1])
        }
         # Collect trays at tables, mapping tray to table location
        available_trays_at_tables = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in state
            if match(fact, "at", "*", "*") and match(fact, "tray", get_parts(fact)[1]) and get_parts(fact)[2] != "kitchen"
        }

        available_sandwiches_at_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")
        }
        available_sandwiches_on_trays = {
            (get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "ontray", "*", "*")
        }
        available_notexist_sandwiches = {
            get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")
        }
        # Sandwiches that are marked as no_gluten (this is a dynamic fact, effect of make_sandwich)
        no_gluten_sandwiches_in_state = {
            get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")
        }

        # Available ingredients at kitchen
        gf_bread_at_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] in self.no_gluten_bread
        }
        any_bread_at_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")
        }
        gf_content_at_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] in self.no_gluten_content
        }
        any_content_at_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")
        }


        # 3. Initialize total heuristic and used resources tracking
        total_heuristic = 0
        used_sandwiches = set()
        used_trays = set()
        used_notexist = set()
        used_ingredients = set() # Track ingredients used for making


        # 5. Iterate through unserved children
        for child in unserved_children:
            table = self.waiting_tables.get(child)
            if table is None: # Should not happen in valid problems
                 continue

            is_allergic = child in self.allergic_children

            best_cost_prep_move = float('inf')
            assigned_resource = None # Store which resource was assigned to this child

            # 6b. Find minimum cost path for this child

            # Pool 1: Suitable sandwich on tray at child's table (Cost 0)
            current_pool_cost = 0
            for s, tr in available_sandwiches_on_trays:
                if s not in used_sandwiches and tr not in used_trays:
                    is_suitable = (s in no_gluten_sandwiches_in_state) if is_allergic else True
                    if is_suitable:
                        # Check if tray is at this child's table
                        if any(match(f, "at", tr, table) for f in state):
                            if current_pool_cost < best_cost_prep_move:
                                best_cost_prep_move = current_pool_cost
                                assigned_resource = ('ontray_table', s, tr)


            # Pool 2: Suitable sandwich on tray at kitchen (Cost 1)
            current_pool_cost = 1
            if current_pool_cost < best_cost_prep_move: # Only consider if potentially better
                for s, tr in available_sandwiches_on_trays:
                    if s not in used_sandwiches and tr not in used_trays:
                        is_suitable = (s in no_gluten_sandwiches_in_state) if is_allergic else True
                        if is_suitable:
                            # Check if tray is at kitchen
                            if any(match(f, "at", tr, "kitchen") for f in state):
                                if current_pool_cost < best_cost_prep_move:
                                    best_cost_prep_move = current_pool_cost # move_tray
                                    assigned_resource = ('ontray_kitchen', s, tr)

            # Pool 3: Suitable sandwich on tray at another table (Cost 1)
            current_pool_cost = 1
            if current_pool_cost < best_cost_prep_move: # Only consider if potentially better
                 for s, tr in available_sandwiches_on_trays:
                    if s not in used_sandwiches and tr not in used_trays:
                        is_suitable = (s in no_gluten_sandwiches_in_state) if is_allergic else True
                        if is_suitable:
                            # Check if tray is at *another* table
                            tray_loc = available_trays_at_tables.get(tr)
                            if tray_loc is not None and tray_loc != table:
                                if current_pool_cost < best_cost_prep_move: # Cost is 1 (move T' -> T)
                                    best_cost_prep_move = current_pool_cost
                                    assigned_resource = ('ontray_other_table', s, tr)


            # Pool 4: Suitable sandwich at kitchen (not on tray) (Cost 2)
            current_pool_cost = 2
            if current_pool_cost < best_cost_prep_move: # Only consider if potentially better
                for s in available_sandwiches_at_kitchen:
                    if s not in used_sandwiches:
                        is_suitable = (s in no_gluten_sandwiches_in_state) if is_allergic else True
                        if is_suitable:
                            # Need an unused tray at kitchen
                            available_unused_trays_at_kitchen = [t for t in available_trays_at_kitchen if t not in used_trays]
                            if available_unused_trays_at_kitchen:
                                if current_pool_cost < best_cost_prep_move:
                                    best_cost_prep_move = current_pool_cost # put_on_tray + move_tray
                                    assigned_resource = ('kitchen_sandwich', s, available_unused_trays_at_kitchen[0]) # Greedily pick a tray

            # Pool 5: Makeable (from notexist) (Cost 3)
            current_pool_cost = 3
            if current_pool_cost < best_cost_prep_move: # Only consider if potentially better
                # Need an unused notexist slot
                available_unused_notexist = [s for s in available_notexist_sandwiches if s not in used_notexist]
                # Need an unused tray at kitchen
                available_unused_trays_at_kitchen = [t for t in available_trays_at_kitchen if t not in used_trays]

                if available_unused_notexist and available_unused_trays_at_kitchen:
                    # Check if suitable ingredients are available at kitchen (not used)
                    can_make_suitable = False
                    ingredient_pair = None
                    if is_allergic:
                        # Need a GF bread and GF content that haven't been used for making
                        available_unused_gf_bread = [b for b in gf_bread_at_kitchen if b not in used_ingredients]
                        available_unused_gf_content = [c for c in gf_content_at_kitchen if c not in used_ingredients]
                        if available_unused_gf_bread and available_unused_gf_content:
                            can_make_suitable = True
                            ingredient_pair = (available_unused_gf_bread[0], available_unused_gf_content[0]) # Greedily pick ingredients
                    else: # Not allergic
                         # Need any bread and any content that haven't been used for making
                        available_unused_any_bread = [b for b in any_bread_at_kitchen if b not in used_ingredients]
                        available_unused_any_content = [c for c in any_content_at_kitchen if c not in used_ingredients]
                        if available_unused_any_bread and available_unused_any_content:
                            can_make_suitable = True
                            ingredient_pair = (available_unused_any_bread[0], available_unused_any_content[0]) # Greedily pick ingredients

                    if can_make_suitable:
                         if current_pool_cost < best_cost_prep_move:
                            best_cost_prep_move = current_pool_cost # make + put_on_tray + move_tray
                            assigned_resource = ('make', available_unused_notexist[0], available_unused_trays_at_kitchen[0], ingredient_pair)


            # 6d. Add cost and mark resources used
            if best_cost_prep_move != float('inf'):
                total_heuristic += best_cost_prep_move + 1 # +1 for serve_child

                if assigned_resource:
                    res_type = assigned_resource[0]
                    if res_type == 'ontray_table' or res_type == 'ontray_kitchen' or res_type == 'ontray_other_table':
                        s, tr = assigned_resource[1], assigned_resource[2]
                        used_sandwiches.add(s)
                        used_trays.add(tr) # Mark specific tray as used
                    elif res_type == 'kitchen_sandwich':
                        s, tr = assigned_resource[1], assigned_resource[2]
                        used_sandwiches.add(s)
                        used_trays.add(tr) # Mark specific tray as used
                    elif res_type == 'make':
                        s_notexist, tr, (b, c) = assigned_resource[1], assigned_resource[2], assigned_resource[3]
                        used_notexist.add(s_notexist)
                        used_trays.add(tr) # Mark specific tray as used
                        used_ingredients.add(b)
                        used_ingredients.add(c)
                        # The made sandwich doesn't exist yet, so no need to add to used_sandwiches

            # Note: If best_cost_prep_move remains infinity, it implies this child cannot be served
            # with the currently available resources and make opportunities.
            # For solvable problems, this shouldn't happen unless all resources are exhausted.
            # The heuristic will simply not count this child, which is acceptable for a non-admissible heuristic.
            # If the problem is unsolvable, the heuristic might return a value < infinity,
            # but the search won't find a goal.

        return total_heuristic
