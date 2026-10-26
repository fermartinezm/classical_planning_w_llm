from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its parts."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Use fnmatch for flexible matching (e.g., '*' wildcard)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to serve all
    unserved children. It calculates the total number of no-gluten and
    regular sandwiches needed based on the unserved children's allergy
    status. It then counts the available sandwiches at different stages
    of readiness (notexist, in kitchen, on tray in kitchen, on tray at
    child's location) and assigns a cost to bring a sandwich from that
    stage to being served (1 to 4 actions). The heuristic greedily
    satisfies the needed sandwiches, prioritizing no-gluten sandwiches
    for allergic children and using the cheapest available sources first.
    The total heuristic value is the sum of the costs to obtain and
    deliver the required number of sandwiches.

    Assumptions:
    - This heuristic is designed for greedy best-first search and is
      not admissible.
    - It assumes that enough trays are available in the kitchen to put
      sandwiches on if needed.
    - It assumes trays can be moved freely between any two places.
    - It assumes that ingredients and 'notexist' sandwich slots, if
      available, can be used to make sandwiches. It accounts for the
      limited quantity of these resources when calculating the number
      of sandwiches that can be made.

    Heuristic Initialization:
    The constructor extracts the following static information from the task:
    - The set of all children that need to be served (goal_children).
    - The allergy status for each child (is_allergic dictionary).
    - The waiting place for each child (waiting_place dictionary).
    - The set of bread portions that are no-gluten (no_gluten_breads).
    - The set of content portions that are no-gluten (no_gluten_contents).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children who are in the goal state but are not yet
       marked as 'served' in the current state.
    2. Categorize these unserved children by their allergy status (allergic
       or non-allergic) and their waiting place. Count the total number
       of unserved allergic children (who need no-gluten sandwiches) and
       unserved non-allergic children (who can accept any sandwich).
    3. If no children are unserved, the heuristic is 0 (goal state).
    4. Count the available sandwich resources in the current state,
       categorized by type (no-gluten/regular) and their current status/location:
       - Sandwiches on trays at places where unserved children are waiting (cost 1 to serve).
       - Sandwiches on trays in the kitchen (cost 2: move tray + serve).
       - Sandwiches in the kitchen but not on trays (cost 3: put on tray + move tray + serve).
       - 'notexist' sandwich slots, available bread, and content portions
         in the kitchen (cost 4: make + put on tray + move tray + serve).
    5. Count available ingredients (no-gluten bread/content, regular bread/content)
       and 'notexist' sandwich slots in the kitchen.
    6. Initialize the total heuristic cost to 0.
    7. Satisfy the demand for no-gluten sandwiches (for allergic children)
       by using the available no-gluten sandwiches from the cheapest source
       (cost 1) to the most expensive source (cost 4). For each sandwich used,
       add the corresponding cost (1, 2, 3, or 4) to the total heuristic cost
       and decrement the available resource counts (especially for making
       sandwiches, which consumes 'notexist' slots and ingredients).
    8. Satisfy the demand for regular sandwiches (for non-allergic children)
       by using the available regular sandwiches and any remaining no-gluten
       sandwiches, again from the cheapest source (cost 1) to the most
       expensive source (cost 4). For each sandwich used, add the corresponding
       cost to the total heuristic cost and update resource counts.
    9. The final total cost accumulated is the heuristic value.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}

        self.is_allergic = {}
        self.waiting_place = {}
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.is_allergic[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.is_allergic[parts[1]] = False
            elif parts[0] == 'waiting':
                self.waiting_place[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if '(served ' + c + ')' not in state}

        # 3. Goal state check
        if not unserved_children:
            return 0

        # 2. Categorize unserved children and count needed sandwiches
        num_allergic_unserved = 0
        num_non_allergic_unserved = 0
        places_with_unserved_children = set()

        for child in unserved_children:
            place = self.waiting_place.get(child) # Get place from static info
            if place: # Child must have a waiting place
                places_with_unserved_children.add(place) # Mark this place as having unserved children
                if self.is_allergic.get(child, False): # Default to not allergic if info missing
                    num_allergic_unserved += 1
                else:
                    num_non_allergic_unserved += 1

        needed_ng = num_allergic_unserved
        needed_reg = num_non_allergic_unserved # Non-allergic can take Reg or NG

        # 4. Count available sandwich resources by type and stage
        ng_at_place_ontray = 0
        reg_at_place_ontray = 0
        ng_kitchen_ontray = 0
        reg_kitchen_ontray = 0
        ng_kitchen = 0
        reg_kitchen = 0
        notexist_count = 0

        trays_at = {}
        sandwiches_ontray = set()

        # First pass to find tray locations and sandwiches on trays
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('tray'):
                trays_at[parts[1]] = parts[2]
            elif parts[0] == 'ontray':
                sandwiches_ontray.add(parts[1])

        # Second pass to count sandwiches by status and location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                if s not in sandwiches_ontray: # Count only if not already on a tray
                    if '(no_gluten_sandwich ' + s + ')' in state:
                        ng_kitchen += 1
                    else:
                        reg_kitchen += 1
            elif parts[0] == 'ontray':
                s = parts[1]
                t = parts[2]
                p = trays_at.get(t)
                if p: # Tray location is known
                    is_ng = '(no_gluten_sandwich ' + s + ')' in state
                    if p == 'kitchen':
                        if is_ng:
                            ng_kitchen_ontray += 1
                        else:
                            reg_kitchen_ontray += 1
                    elif p in places_with_unserved_children: # Only count if tray is at a place needing service
                         if is_ng:
                             ng_at_place_ontray += 1
                         else:
                             reg_at_place_ontray += 1
                    # Sandwiches on trays at places *without* unserved children don't directly help
                    # without a move action, which is covered by cost 2 logic if moved to kitchen first,
                    # or implicitly by needing a new sandwich source.
            elif parts[0] == 'notexist':
                notexist_count += 1

        # 5. Count available ingredients in kitchen
        available_bread_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        available_content_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        available_ng_bread_kitchen = len(available_bread_kitchen.intersection(self.no_gluten_breads))
        available_reg_bread_kitchen = len(available_bread_kitchen) - available_ng_bread_kitchen
        available_ng_content_kitchen = len(available_content_kitchen.intersection(self.no_gluten_contents))
        available_reg_content_kitchen = len(available_content_kitchen) - available_ng_content_kitchen

        # 6. Initialize total cost
        total_cost = 0
        served_ng_count = 0
        served_reg_count = 0

        # Resources used for making
        used_notexist = 0
        used_ng_bread = 0
        used_reg_bread = 0
        used_ng_content = 0
        used_reg_content = 0

        # 7. Satisfy NG needs (for allergic children) from cheapest sources
        # Cost 1: NG on tray at place
        use_ng_c1 = min(needed_ng - served_ng_count, ng_at_place_ontray)
        total_cost += use_ng_c1 * 1
        served_ng_count += use_ng_c1

        # Cost 2: NG on tray in kitchen
        use_ng_c2 = min(needed_ng - served_ng_count, ng_kitchen_ontray)
        total_cost += use_ng_c2 * 2
        served_ng_count += use_ng_c2

        # Cost 3: NG in kitchen
        use_ng_c3 = min(needed_ng - served_ng_count, ng_kitchen)
        total_cost += use_ng_c3 * 3
        served_ng_count += use_ng_c3

        # Cost 4: Make NG
        can_make_ng = min(notexist_count - used_notexist,
                          available_ng_bread_kitchen - used_ng_bread,
                          available_ng_content_kitchen - used_ng_content)
        use_ng_c4 = min(needed_ng - served_ng_count, can_make_ng)
        total_cost += use_ng_c4 * 4
        served_ng_count += use_ng_c4
        used_notexist += use_ng_c4
        used_ng_bread += use_ng_c4
        used_ng_content += use_ng_c4

        # 8. Satisfy Reg needs (for non-allergic children) from cheapest sources (Reg or remaining NG)
        # Cost 1: Reg on tray at place
        use_reg_c1 = min(needed_reg - served_reg_count, reg_at_place_ontray)
        total_cost += use_reg_c1 * 1
        served_reg_count += use_reg_c1

        # Cost 1: Remaining NG on tray at place for Reg
        remaining_ng_c1 = ng_at_place_ontray - use_ng_c1
        use_ng_c1_for_reg = min(needed_reg - served_reg_count, remaining_ng_c1)
        total_cost += use_ng_c1_for_reg * 1
        served_reg_count += use_ng_c1_for_reg

        # Cost 2: Reg on tray in kitchen
        use_reg_c2 = min(needed_reg - served_reg_count, reg_kitchen_ontray)
        total_cost += use_reg_c2 * 2
        served_reg_count += use_reg_c2

        # Cost 2: Remaining NG on tray in kitchen for Reg
        remaining_ng_c2 = ng_kitchen_ontray - use_ng_c2
        use_ng_c2_for_reg = min(needed_reg - served_reg_count, remaining_ng_c2)
        total_cost += use_ng_c2_for_reg * 2
        served_reg_count += use_ng_c2_for_reg

        # Cost 3: Reg in kitchen
        use_reg_c3 = min(needed_reg - served_reg_count, reg_kitchen)
        total_cost += use_reg_c3 * 3
        served_reg_count += use_reg_c3

        # Cost 3: Remaining NG in kitchen for Reg
        remaining_ng_c3 = ng_kitchen - use_ng_c3
        use_ng_c3_for_reg = min(needed_reg - served_reg_count, remaining_ng_c3)
        total_cost += use_ng_c3_for_reg * 3
        served_reg_count += use_ng_c3_for_reg

        # Cost 4: Make Reg
        can_make_reg = min(notexist_count - used_notexist,
                           available_reg_bread_kitchen - used_reg_bread,
                           available_reg_content_kitchen - used_reg_content)
        use_reg_c4 = min(needed_reg - served_reg_count, can_make_reg)
        total_cost += use_reg_c4 * 4
        served_reg_count += use_reg_c4
        used_notexist += use_reg_c4
        used_reg_bread += use_reg_c4
        used_reg_content += use_reg_c4

        # Cost 4: Make remaining NG for Reg
        can_make_ng_for_reg = min(notexist_count - used_notexist,
                                  available_ng_bread_kitchen - used_ng_bread,
                                  available_ng_content_kitchen - used_ng_content)
        use_ng_c4_for_reg = min(needed_reg - served_reg_count, can_make_ng_for_reg)
        total_cost += use_ng_c4_for_reg * 4
        served_reg_count += use_ng_c4_for_reg
        # Note: Ingredients and notexist slots used here are already accounted for
        # in the previous 'used' variables.

        # 9. Return total cost
        return total_cost
