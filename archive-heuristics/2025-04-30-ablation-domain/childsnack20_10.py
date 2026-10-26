from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

def is_suitable(sandwich, needs_gf, sandwich_is_gf_map):
    """
    Checks if a sandwich is suitable for a child.
    A sandwich is suitable if the child does not need GF or the sandwich is GF.
    """
    if not needs_gf:
        return True # Any sandwich is suitable for a non-allergic child
    # Child needs GF, check if sandwich is GF
    return sandwich_is_gf_map.get(sandwich, False) # Default to False if status unknown


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all unserved
    children. It does this by summing the minimum estimated steps required for
    each unserved child, greedily assigning available resources (sandwiches,
    ingredients, trays) based on their proximity to the child's waiting table.

    # Assumptions
    - Children specified in the goal are initially in a 'waiting' state at a specific table.
    - Trays can move directly between the kitchen and any table where a child is waiting.
    - A tray can carry multiple sandwiches, but the heuristic simplifies by considering tray availability per child's required delivery path.
    - A sandwich's gluten-free status is explicitly indicated by the '(no_gluten_sandwich s)' predicate in the state.
    - Gluten-free bread and content portions are identified by '(no_gluten_bread b)' and '(no_gluten_content c)' static facts.
    - Regular bread and content portions are those not marked as gluten-free.
    - A gluten-free sandwich can be served to a non-allergic child.
    - Resource availability (trays, ingredients) is checked and consumed greedily for children requiring resources at increasing 'distances' (action costs).

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Which children are allergic to gluten.
    - Which table each child is waiting at.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    This information is stored in dictionaries and sets for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify the set of children who are part of the goal and are currently in a 'waiting' state but not yet 'served'.
    2. For each unserved child, determine their waiting table and whether they require a gluten-free sandwich based on their allergy status (pre-computed in __init__).
    3. Inventory all relevant resources available in the current state:
       - Sandwiches at children's tables.
       - Sandwiches on trays at children's tables.
       - Sandwiches on trays in the kitchen.
       - Sandwiches in the kitchen (not on trays).
       - Gluten-free and regular bread portions in the kitchen.
       - Gluten-free and regular content portions in the kitchen.
       - Trays in the kitchen.
       - Trays at children's tables.
       Also, determine the gluten-free status of each existing sandwich.
    4. Calculate the number of available gluten-free and regular ingredient pairs in the kitchen.
    5. Initialize the total heuristic cost `h = 0`. Keep track of specific sandwiches, trays, and ingredient pairs that are considered 'used' by the heuristic calculation to satisfy a child's need.
    6. Iterate through the unserved children and available resources in phases, corresponding to the estimated minimum actions required to get a suitable sandwich to the child and serve it, from closest to furthest:
       - **Phase 1 (Cost 1: Serve):** For each unserved child, check if there is an unused suitable sandwich already located `at_table` at their waiting spot. If yes, add 1 to `h`, mark the child as covered, and mark the sandwich as used.
       - **Phase 2 (Cost 2: Take + Serve):** For remaining unserved children, check if there is an unused suitable sandwich on an unused tray located `at` their waiting table. If yes, add 2 to `h`, mark the child as covered, and mark the sandwich and tray as used.
       - **Phase 3 (Cost 3: Move + Take + Serve):** For remaining unserved children, check if there is an unused suitable sandwich on an unused tray located `at` the kitchen. If yes, add 3 to `h`, mark the child as covered, and mark the sandwich and tray as used. This assumes the tray can be moved directly to the child's table.
       - **Phase 4 (Cost 4: Put + Move + Take + Serve):** For remaining unserved children, check if there is an unused suitable sandwich `at_kitchen_sandwich` and if an unused tray is available (either in the kitchen or at a table). If yes, add 4 to `h`, mark the child as covered, and mark the sandwich and tray as used.
       - **Phase 5 (Cost 5: Make + Put + Move + Take + Serve):** For remaining unserved children, check if suitable, unused ingredients are available in the kitchen to make a sandwich and if an unused tray is available. If yes, add 5 to `h`, mark the child as covered, and mark the ingredients and tray as used. When making a regular sandwich, prioritize using regular ingredients before using gluten-free ingredients if both are available.
    7. After processing all phases, if there are still unserved children, it implies that the current state lacks the necessary resources (sandwiches, ingredients, or trays) to serve them according to this greedy assignment strategy. In this case, return `float('inf')` to indicate a likely unsolvable state from this point.
    8. Otherwise, return the total accumulated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The task object contains goal conditions and static facts.
        self.goals = task.goals
        static_facts = task.static

        # Extract child allergy status from static facts
        self.child_allergy = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                 child = get_parts(fact)[1]
                 self.child_allergy[child] = False

        # Extract child waiting table from static facts
        self.child_table = {}
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                child, table = get_parts(fact)[1], get_parts(fact)[2]
                self.child_table[child] = table

        # Extract gluten-free ingredients from static facts
        self.gf_bread = set()
        for fact in static_facts:
            if match(fact, "no_gluten_bread", "*"):
                self.gf_bread.add(get_parts(fact)[1])

        self.gf_content = set()
        for fact in static_facts:
            if match(fact, "no_gluten_content", "*"):
                self.gf_content.add(get_parts(fact)[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of fact strings.

        # 1. Identify unserved children who are waiting and are goal children
        goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        unserved = set()
        child_needs = {} # {child: (table, needs_gf)}

        for child in goal_children:
            # Check if child is waiting and not served
            is_waiting = False
            child_table = None
            if child in self.child_table:
                 child_table = self.child_table[child]
                 # Check if the specific waiting fact exists in the state
                 if '(waiting ' + child + ' ' + child_table + ')' in state:
                     is_waiting = True

            is_served = '(served ' + child + ')' in state

            # Only consider children who are currently waiting and not yet served
            if is_waiting and not is_served:
                unserved.add(child)
                # Determine if the child needs a gluten-free sandwich
                needs_gf = self.child_allergy.get(child, False) # Default to not allergic if info missing
                child_needs[child] = (child_table, needs_gf)

        # If no unserved children, the goal is reached
        if not unserved:
            return 0

        # 3. Inventory available resources from the current state
        # Initialize resource containers. Use sets for easy removal/checking existence.
        # Only track resources at tables where children are waiting.
        waiting_tables = set(self.child_table.values())

        current_s_attable = {table: set() for table in waiting_tables}
        current_s_ontray_attable = {table: set() for table in waiting_tables} # Store (s, tray) tuples
        current_s_ontray_kitchen = set() # Store (s, tray) tuples
        current_s_kitchen = set()
        current_trays_k = set()
        current_trays_t = {table: set() for table in waiting_tables}
        current_sandwich_is_gf = {} # {sandwich_name: True/False}
        all_sandwiches = set() # Collect all sandwich objects mentioned in the state

        # Temporary storage for tray locations and ingredient counts
        tray_locations = {}
        bread_gf_count = 0
        content_gf_count = 0
        bread_reg_count = 0
        content_reg_count = 0

        # First pass: Collect locations, sandwich GF status, and ingredient counts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('sandw'):
                    all_sandwiches.add(obj)
                    # Add to s_attable later, after GF status is known
                elif obj.startswith('tray'):
                    tray_locations[obj] = loc
                    if loc == 'kitchen': current_trays_k.add(obj)
                    elif loc in current_trays_t: current_trays_t[loc].add(obj)
            elif parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                all_sandwiches.add(s)
                current_s_kitchen.add(s)
            elif parts[0] == 'at_kitchen_bread':
                b = parts[1]
                if b in self.gf_bread: bread_gf_count += 1
                else: bread_reg_count += 1
            elif parts[0] == 'at_kitchen_content':
                c = parts[1]
                if c in self.gf_content: content_gf_count += 1
                else: content_reg_count += 1
            elif parts[0] == 'no_gluten_sandwich':
                s = parts[1]
                all_sandwiches.add(s)
                current_sandwich_is_gf[s] = True

        # Ensure all sandwiches in state have a GF status entry (default to False)
        for s in all_sandwiches:
            if s not in current_sandwich_is_gf:
                current_sandwich_is_gf[s] = False

        # Second pass: Populate sandwich location sets now that GF status and tray locations are known
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue

             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj.startswith('sandw'):
                     if loc in current_s_attable: current_s_attable[loc].add(obj)
             elif parts[0] == 'ontray':
                 s, tray = parts[1], parts[2]
                 if tray in tray_locations:
                     loc = tray_locations[tray]
                     if loc == 'kitchen': current_s_ontray_kitchen.add((s, tray))
                     elif loc in current_s_ontray_attable: current_s_ontray_attable[loc].add((s, tray))


        # 4. Calculate available ingredient pairs
        current_ing_gf_pairs = min(bread_gf_count, content_gf_count)
        current_ing_reg_pairs = min(bread_reg_count, content_reg_count)

        # 5. Initialize heuristic and used resources
        h = 0
        used_sandwiches = set()
        used_trays = set()
        used_ing_gf = 0
        used_ing_reg = 0

        # 6. Greedy assignment loop by distance
        # Process children needing resources at increasing distances (estimated action costs)
        # Distances: 1 (Serve), 2 (Take+Serve), 3 (Move+Take+Serve), 4 (Put+Move+Take+Serve), 5 (Make+Put+Move+Take+Serve)
        distances = [1, 2, 3, 4, 5]

        for dist in distances:
            served_at_dist = set()
            # Iterate over a copy of the unserved set as we modify it
            for c in list(unserved):
                table_c, needs_gf = child_needs[c]
                suitable_resource = None # Represents the sandwich/ingredients/tray used

                if dist == 1: # Serve (sandwich at table)
                    # Find a suitable, unused sandwich at the child's table
                    for s in list(current_s_attable[table_c]): # Iterate over copy
                        if s not in used_sandwiches and is_suitable(s, needs_gf, current_sandwich_is_gf):
                            suitable_resource = s
                            break
                    if suitable_resource:
                        h += dist
                        served_at_dist.add(c)
                        used_sandwiches.add(suitable_resource)

                elif dist == 2: # Take + Serve (sandwich on tray at table)
                     # Find a suitable, unused sandwich on an unused tray at the child's table
                     for item in list(current_s_ontray_attable[table_c]): # Iterate over copy
                        s, tray = item
                        if s not in used_sandwiches and tray not in used_trays and is_suitable(s, needs_gf, current_sandwich_is_gf):
                            suitable_resource = item
                            break
                     if suitable_resource:
                        s, tray = suitable_resource
                        h += dist
                        served_at_dist.add(c)
                        used_sandwiches.add(s)
                        used_trays.add(tray) # Tray used for this delivery path

                elif dist == 3: # Move + Take + Serve (sandwich on tray in kitchen)
                     # Find a suitable, unused sandwich on an unused tray in the kitchen
                     for item in list(current_s_ontray_kitchen): # Iterate over copy
                        s, tray = item
                        if s not in used_sandwiches and tray not in used_trays and is_suitable(s, needs_gf, current_sandwich_is_gf):
                             suitable_resource = item
                             break
                     if suitable_resource:
                        s, tray = suitable_resource
                        h += dist
                        served_at_dist.add(c)
                        used_sandwiches.add(s)
                        used_trays.add(tray) # Tray used for this delivery path

                elif dist == 4: # Put + Move + Take + Serve (sandwich in kitchen)
                     # Find a suitable, unused sandwich in the kitchen and an unused tray anywhere
                     suitable_s = None
                     for s in list(current_s_kitchen): # Iterate over copy
                        if s not in used_sandwiches and is_suitable(s, needs_gf, current_sandwich_is_gf):
                            suitable_s = s
                            break

                     if suitable_s:
                        # Find an available tray anywhere (kitchen or table)
                        available_tray = None
                        for t in list(current_trays_k): # Iterate over copy
                            if t not in used_trays:
                                available_tray = t
                                break
                        if not available_tray:
                             for table, trays_set in current_trays_t.items():
                                 for t in list(trays_set): # Iterate over copy
                                     if t not in used_trays:
                                         available_tray = t
                                         break
                                 if available_tray: break

                        if available_tray:
                            suitable_resource = (suitable_s, available_tray)

                     if suitable_resource:
                        s, tray = suitable_resource
                        h += dist
                        served_at_dist.add(c)
                        used_sandwiches.add(s)
                        used_trays.add(tray) # Tray used for this delivery path

                elif dist == 5: # Make + Put + Move + Take + Serve (needs ingredients)
                     # Check if suitable, unused ingredients and an unused tray are available
                     suitable_ingredients_available = False
                     if needs_gf:
                         if used_ing_gf < current_ing_gf_pairs:
                             suitable_ingredients_available = True
                     else: # Needs Reg or GF
                         # Check if regular ingredients are available
                         if used_ing_reg < current_ing_reg_pairs:
                             suitable_ingredients_available = True
                         # Else, check if GF ingredients are available (can be used for Reg)
                         elif used_ing_gf < current_ing_gf_pairs:
                             suitable_ingredients_available = True

                     available_tray = None
                     if suitable_ingredients_available:
                         # Find an available tray anywhere (kitchen or table)
                         for t in list(current_trays_k): # Iterate over copy
                             if t not in used_trays:
                                 available_tray = t
                                 break
                         if not available_tray:
                              for table, trays_set in current_trays_t.items():
                                  for t in list(trays_set): # Iterate over copy
                                      if t not in used_trays:
                                          available_tray = t
                                          break
                                  if available_tray: break

                     if suitable_ingredients_available and available_tray:
                        h += dist
                        served_at_dist.add(c)
                        used_trays.add(available_tray) # Tray used for this delivery path
                        # Consume ingredients
                        if needs_gf:
                            used_ing_gf += 1
                        else: # Needs Reg
                            # Prioritize using Reg ingredients first
                            if used_ing_reg < current_ing_reg_pairs:
                                used_ing_reg += 1
                            elif used_ing_gf < current_ing_gf_pairs:
                                used_ing_gf += 1
                            else:
                                 # This case should ideally not be reached if suitable_ingredients_available was True
                                 pass # Error state in logic?

            # Remove children served at this distance from the unserved set
            unserved -= served_at_dist

        # 7. If any children remain unserved after trying all resource levels,
        # they cannot be served with current resources according to this strategy.
        if unserved:
            return float('inf')

        # 8. Return the total accumulated heuristic value
        return h
