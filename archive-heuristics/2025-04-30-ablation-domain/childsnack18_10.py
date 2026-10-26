from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It calculates the minimum number of steps needed for each unserved child independently,
    based on the current location and state of suitable sandwiches relative to the child's table.
    The total heuristic is the sum of these minimum per-child costs.

    # Assumptions
    - Each child needs one suitable sandwich.
    - Trays have sufficient capacity.
    - Ingredients are consumed when making sandwiches.
    - The cost of actions is uniform (e.g., 1).
    - The heuristic assumes the "closest" available suitable sandwich will be used for a child.
    - It might overestimate costs by not fully accounting for shared tray movements or
      sandwiches suitable for multiple children at the same table.
    - Children remain waiting at their initial tables.

    # Heuristic Initialization
    The heuristic extracts static information and object lists from the task:
    - Which children are allergic to gluten.
    - Which bread and content portions are gluten-free.
    - Which table each child is waiting at.
    - Lists of all objects (children, sandwiches, bread, content, trays, tables) mentioned in the initial state or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all children who are goals (i.e., need to be served) but are not yet served in the current state. These are the unserved children.
    2. For each unserved child:
       a. Determine if the child is allergic to gluten using static facts.
       b. Find all existing sandwiches (those not marked `notexist`) that are "suitable" for this child. A sandwich is suitable if it is `made_sandwich` and meets the allergy requirement (any made sandwich for non-allergic children, only `no_gluten_sandwich` for allergic children).
       c. Determine the "best stage" (representing the minimum number of actions) required to get a suitable sandwich to this child's table:
          - Stage 1 (Cost 1: serve): A suitable sandwich is currently on a tray located at the child's waiting table.
          - Stage 2 (Cost 2: move_tray, serve): No suitable sandwich is on a tray at the child's table, but at least one suitable sandwich is on a tray located elsewhere (either in the kitchen or at another table).
          - Stage 3 (Cost 3: put_on_tray, move_tray, serve): No suitable sandwich is on any tray, but at least one suitable sandwich exists and is located `at_kitchen_sandwich` (not on a tray).
          - Stage 4 (Cost 4: make_sandwich, put_on_tray, move_tray, serve): No suitable sandwich exists (all potential suitable sandwiches are `notexist` or not made), but ingredients are available in the kitchen to make a suitable one.
          - Stage 5 (Cost 1000: unsolvable): No suitable sandwich exists or can be made with available ingredients.
       d. The child is assigned to the lowest-cost stage among all suitable sandwiches (existing or makeable).
    3. Count the number of unserved children assigned to each stage (N_stage1, N_stage2, N_stage3, N_stage4, N_stage5).
    4. The total heuristic value is the sum of (N_stage1 * 1) + (N_stage2 * 2) + (N_stage3 * 3) + (N_stage4 * 4) + (N_stage5 * 1000).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and object lists."""
        # Goal conditions (subset of facts that must be true)
        self.goals = task.goals
        # Static facts (facts that do not change during planning)
        static_facts = task.static
        # Initial state facts (used to extract all possible objects)
        initial_state = task.initial_state

        # Extract object lists from initial state and static facts
        self.all_children = set()
        self.all_tables = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_trays = set()

        # Objects from initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'waiting':
                 if len(parts) > 2: self.all_children.add(parts[1]); self.all_tables.add(parts[2])
             elif parts[0] == 'notexist':
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
             elif parts[0] == 'at_kitchen_bread':
                 if len(parts) > 1: self.all_bread.add(parts[1])
             elif parts[0] == 'at_kitchen_content':
                 if len(parts) > 1: self.all_content.add(parts[1])
             elif parts[0] == 'at' and len(parts) == 3 and parts[2] == 'kitchen' and parts[1].startswith('tray'):
                 self.all_trays.add(parts[1])
             elif parts[0] == 'ontray': # Sandwiches already on trays in init
                 if len(parts) > 2: self.all_sandwiches.add(parts[1]); self.all_trays.add(parts[2])
             elif parts[0] == 'at_kitchen_sandwich': # Sandwiches already made in init
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
             elif parts[0] == 'made_sandwich': # Sandwiches already made in init
                 if len(parts) > 3: self.all_sandwiches.add(parts[1]); self.all_bread.add(parts[2]); self.all_content.add(parts[3])


        # Objects from static facts (might include objects not in initial state facts)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] in ['allergic_gluten', 'not_allergic_gluten']:
                 if len(parts) > 1: self.all_children.add(parts[1])
             elif parts[0] == 'waiting':
                 if len(parts) > 2: self.all_children.add(parts[1]); self.all_tables.add(parts[2])
             elif parts[0] == 'no_gluten_bread':
                 if len(parts) > 1: self.all_bread.add(parts[1])
             elif parts[0] == 'no_gluten_content':
                 if len(parts) > 1: self.all_content.add(parts[1])
             # Sandwiches, trays, tables might also appear in static facts depending on domain definition,
             # but the initial state facts usually list all instances. The above covers common cases.


        # Extract static relationships for quick lookup
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")}
        self.no_gluten_bread_set = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.no_gluten_content_set = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}
        # Map child to their waiting table (assuming this is static)
        self.waiting_tables = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")}

        # Store static facts as a set for efficient lookup in helper functions
        self.static_facts_set = static_facts


    def is_suitable(self, sandwich, child, state):
        """Check if a made sandwich is suitable for a child based on allergies."""
        # Sandwich must exist and be made
        if f'(notexist {sandwich})' in state:
            return False

        # Check if the sandwich is made
        is_made = any(match(fact, "made_sandwich", sandwich, "*", "*") for fact in state)
        if not is_made:
             return False

        # Check allergy status
        if child in self.allergic_children:
            # Allergic children need gluten-free sandwiches
            # Check if the sandwich is marked as no_gluten_sandwich
            return f'(no_gluten_sandwich {sandwich})' in state

        elif child in self.not_allergic_children:
            # Non-allergic children can eat any made sandwich
            return True
        else:
            # Child allergy status unknown - should not happen in valid problems
            return False


    def can_make_suitable(self, child, state):
        """Check if ingredients exist in the kitchen to make a suitable sandwich."""
        available_bread_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        available_content_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        if child in self.allergic_children:
            # Need at least one no-gluten bread and at least one no-gluten content in the kitchen
            has_gf_bread_kitchen = any(b in self.no_gluten_bread_set for b in available_bread_kitchen)
            has_gf_content_kitchen = any(c in self.no_gluten_content_set for c in available_content_kitchen)
            return has_gf_bread_kitchen and has_gf_content_kitchen
        elif child in self.not_allergic_children:
            # Need at least one of any bread and at least one of any content in the kitchen
            return len(available_bread_kitchen) > 0 and len(available_content_kitchen) > 0
        else:
            # Child allergy status unknown
            return False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their tables
        unserved_children = {}
        # Iterate through all children that are part of the goal
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'served':
                child = parts[1]
                # Check if the child is NOT served in the current state
                if goal_fact not in state:
                    # Find where the child is waiting (should be in static facts)
                    table = self.waiting_tables.get(child)
                    if table:
                         unserved_children[child] = table
                    # Note: If a child is a goal but not in waiting_tables, something is wrong with the problem definition.
                    # We only consider children listed in waiting_tables for heuristic calculation.


        if not unserved_children:
            return 0 # Goal state reached (all required children served)

        # Classify unserved children by the "best stage" of suitable sandwiches
        children_by_stage = {1: set(), 2: set(), 3: set(), 4: set(), 5: set()} # Stage 5 is unmakeable

        # Identify existing sandwiches (those not marked 'notexist')
        existing_sandwiches = {s for s in self.all_sandwiches if f'(notexist {s})' not in state}

        # Pre-calculate locations and tray status of existing sandwiches for efficiency
        sandwich_info = {} # Map sandwich -> {'loc': location, 'ontray': tray_name or None}
        for s in existing_sandwiches:
             info = {'loc': None, 'ontray': None}
             # Check if on a tray
             ontray_facts = [get_parts(fact) for fact in state if match(fact, "ontray", s, "*")]
             if ontray_facts:
                  tray = ontray_facts[0][2] # Tray name
                  info['ontray'] = tray
                  # Find tray location
                  at_facts = [get_parts(fact) for fact in state if match(fact, "at", tray, "*")]
                  if at_facts:
                       info['loc'] = at_facts[0][2] # Tray location (kitchen or table)
             elif f'(at_kitchen_sandwich {s})' in state:
                  info['loc'] = 'kitchen' # Sandwich is in kitchen, not on tray
             # Store findings
             sandwich_info[s] = info


        for child, table in unserved_children.items():
            best_stage_for_child = 5 # Assume unmakeable initially

            # Find suitable sandwiches for this child among existing ones
            suitable_existing_s = {s for s in existing_sandwiches if self.is_suitable(s, child, state)}

            if not suitable_existing_s:
                 # No suitable sandwich exists, check if one can be made
                 if self.can_make_suitable(child, state):
                      best_stage_for_child = 4 # Needs making
                 else:
                      best_stage_for_child = 5 # Unmakeable
            else:
                 # Suitable sandwiches exist, find the one closest to the child's table
                 min_stage = 5 # Initialize with worst stage
                 for s in suitable_existing_s:
                      info = sandwich_info.get(s)
                      if info is None: continue # Should not happen if s is in existing_sandwiches

                      s_loc = info['loc']
                      s_ontray = info['ontray']

                      if s_loc == table and s_ontray is not None:
                           min_stage = min(min_stage, 1) # On tray at child's table
                      elif (s_loc in self.all_tables or s_loc == 'kitchen') and s_ontray is not None:
                           min_stage = min(min_stage, 2) # On tray elsewhere
                      elif s_loc == 'kitchen' and s_ontray is None:
                           min_stage = min(min_stage, 3) # At kitchen, not on tray

                 # If no existing suitable sandwich is found in a standard location (on tray or at kitchen),
                 # it might be in an unexpected state (e.g., carried by robot if that action existed, or at a non-table/non-kitchen location).
                 # In such cases, we fall back to checking if a new one can be made, as that might be the only path forward.
                 # If min_stage is still 5, it means no suitable existing sandwich was found in a usable location.
                 if min_stage == 5:
                     if self.can_make_suitable(child, state):
                         min_stage = 4 # Needs making
                     else:
                         min_stage = 5 # Unmakeable


                 best_stage_for_child = min_stage # Update best stage based on existing sandwiches or makeability


            children_by_stage[best_stage_for_child].add(child)


        # Calculate heuristic based on counts in each stage
        h = 0
        # Cost for stage 1: serve (1 action)
        h += len(children_by_stage[1]) * 1
        # Cost for stage 2: move_tray, serve (2 actions)
        h += len(children_by_stage[2]) * 2
        # Cost for stage 3: put_on_tray, move_tray, serve (3 actions)
        h += len(children_by_stage[3]) * 3
        # Cost for stage 4: make_sandwich, put_on_tray, move_tray, serve (4 actions)
        h += len(children_by_stage[4]) * 4
        # Cost for stage 5: unsolvable (large number)
        h += len(children_by_stage[5]) * 1000 # Use a large number for unsolvable

        return h
