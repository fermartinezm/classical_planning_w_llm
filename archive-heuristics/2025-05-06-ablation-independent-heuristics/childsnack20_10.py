# Helper function to parse a PDDL fact string
def parse_fact(fact_string):
    # Removes parentheses and splits by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Assuming Heuristic base class is available from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # This line should be uncommented in the actual planner environment

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task=None):
        pass
    def __call__(self, node):
        raise NotImplementedError


class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served). It does this by summing the cost of the final
    'serve' action for each unserved child and the estimated minimum cost to
    get a suitable sandwich onto a tray at the child's location. The cost to
    get a sandwich to the required location stage depends on its current stage:
    on a tray elsewhere (cost 1: move), in the kitchen (cost 2: put + move),
    or needing to be made (cost 3: make + put + move). It prioritizes using
    sandwiches that are closer to the final stage.

    Assumptions:
    - Children wait at non-kitchen places (implied by domain examples).
    - Sufficient tray capacity is available (heuristic doesn't model tray capacity).
    - Sufficient unique sandwich objects exist initially (heuristic relies on `notexist` count).
    - The problem is solvable given initial resources. If not enough resources
      (bread, content, notexist sandwiches) exist to make the required sandwiches,
      the heuristic returns infinity.
    - Static predicates like `allergic_gluten`, `not_allergic_gluten`, `no_gluten_bread`, `no_gluten_content`
      are present in `task.static`. Dynamic predicates like `waiting`, `served`,
      `at_kitchen_bread`, etc., are in the state.

    Heuristic Initialization:
    The constructor processes the static facts from `task.static` to identify
    which children are allergic, and which bread and content types are gluten-free.
    This information is stored for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify the status of all relevant predicates:
       - Which children are served?
       - Which children are waiting and where?
       - Which sandwiches are on which trays?
       - Where are the trays located?
       - Which sandwiches are in the kitchen?
       - Which bread and content portions are in the kitchen?
       - Which sandwich objects `notexist`?
       - Which sandwiches are marked as gluten-free?
    2. Identify the set of unserved children and their waiting locations.
    3. If there are no unserved children, the state is a goal state, return 0.
    4. Count the total number of unserved children (`num_unserved`). This contributes `num_unserved * 1` to the heuristic (for the final 'serve' action).
    5. Categorize unserved children into those needing gluten-free sandwiches (`num_unserved_gf`) and those needing regular sandwiches (`num_unserved_reg`), based on the static allergy information.
    6. For each place where children are waiting, count the number of unserved children needing GF and Reg sandwiches (`needed_gf_at_p`, `needed_reg_at_p`).
    7. For each place where children are waiting, count the number of available GF and Reg sandwiches already on trays at that location (`available_gf_ontray_at_p`, `available_reg_ontray_at_p`). These sandwiches are considered "at location".
    8. Calculate the deficit of suitable sandwiches at each waiting location (`deficit_gf_at_p = max(0, needed_gf_at_p - available_gf_ontray_at_p)`, `deficit_reg_at_p = max(0, needed_reg_at_p - available_reg_ontray_at_p)`).
    9. Sum the deficits across all waiting locations to get the total number of GF and Reg sandwiches that need to be brought to the "ontray at location" stage (`Total_deficit_gf_at_location`, `Total_deficit_reg_at_location`).
    10. Count the available GF and Reg sandwiches at stages *before* "ontray at location":
        - On trays not in the kitchen (`Total_gf_ontray_not_kitchen`, `Total_reg_ontray_not_kitchen`). These need 1 'move_tray' action.
        - In the kitchen (`C_gf_kitchen`, `C_reg_kitchen`). These need 1 'put_on_tray' + 1 'move_tray' actions (cost 2).
        - Makable from available bread, content, and `notexist` objects in the kitchen (`C_gf_makable`, `C_reg_makable`). These need 1 'make_sandwich' + 1 'put_on_tray' + 1 'move_tray' actions (cost 3). The calculation for makable sandwiches considers the current stock of bread, content, and `notexist` items, prioritizing GF sandwich making first if `notexist` is the bottleneck.
    11. Calculate the cost to fulfill the `Total_deficit_gf_at_location` and `Total_deficit_reg_at_location` by drawing from the available sandwiches at earlier stages, prioritizing the cheapest sources (ontray elsewhere, then kitchen, then makable). The number of sandwiches available from 'ontray elsewhere' is the total ontray not in kitchen minus those already counted as 'available_ontray_at_p'.
    12. If, after exhausting all available sandwiches from earlier stages, there is still a deficit, it implies the state is likely unsolvable with current resources (or the heuristic is an underestimate), so return `float('inf')`.
    13. The total heuristic value is the sum of the base cost (unserved children) and the calculated costs to bring the necessary GF and Reg sandwiches to the locations where they are needed.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gf_bread_types = set()
        self.gf_content_types = set()

        # Process static facts
        for fact_string in task.static:
            predicate, args = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(args[0])
            elif predicate == 'no_gluten_bread':
                self.gf_bread_types.add(args[0])
            elif predicate == 'no_gluten_content':
                self.gf_content_types.add(args[0])

    def __call__(self, node):
        state = node.state

        # 1. Parse the current state
        served_children = set()
        waiting_children_locations = {} # child -> place
        ontray_sandwiches = {} # sandwich -> tray
        tray_locations = {} # tray -> place
        kitchen_sandwiches = set()
        kitchen_bread = set()
        kitchen_content = set()
        notexist_sandwiches = set()
        gf_sandwiches_in_state = set() # Sandwiches currently marked as GF

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'served':
                served_children.add(args[0])
            elif predicate == 'waiting':
                waiting_children_locations[args[0]] = args[1]
            elif predicate == 'ontray':
                ontray_sandwiches[args[0]] = args[1]
            elif predicate == 'at':
                tray_locations[args[0]] = args[1]
            elif predicate == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(args[0])
            elif predicate == 'at_kitchen_bread':
                kitchen_bread.add(args[0])
            elif predicate == 'at_kitchen_content':
                kitchen_content.add(args[0])
            elif predicate == 'notexist':
                notexist_sandwiches.add(args[0])
            elif predicate == 'no_gluten_sandwich':
                gf_sandwiches_in_state.add(args[0])

        # 2. Identify unserved children and their waiting locations
        unserved_children_waiting = {
            c: p for c, p in waiting_children_locations.items()
            if c not in served_children
        }

        # 3. If no unserved children, return 0
        num_unserved = len(unserved_children_waiting)
        if num_unserved == 0:
            return 0

        # 4. Base cost (serve action for each unserved child)
        heuristic_value = num_unserved

        # 5. Categorize unserved children by allergy (already done in step 2 implicitly)
        # 6. & 7. Count needed and available ontray at each waiting location
        needed_gf_at_p = {} # place -> count
        needed_reg_at_p = {} # place -> count
        available_gf_ontray_at_p = {} # place -> count
        available_reg_ontray_at_p = {} # place -> count
        waiting_places = set(unserved_children_waiting.values())

        for p in waiting_places:
            needed_gf_at_p[p] = 0
            needed_reg_at_p[p] = 0
            available_gf_ontray_at_p[p] = 0
            available_reg_ontray_at_p[p] = 0

        for child, place in unserved_children_waiting.items():
            if child in self.allergic_children:
                needed_gf_at_p[place] += 1
            elif child in self.not_allergic_children:
                needed_reg_at_p[place] += 1

        # Count sandwiches on trays at waiting locations
        sandwiches_ontray_at_waiting_place = set() # Keep track to avoid double counting in 'ontray elsewhere'
        for s, t in ontray_sandwiches.items():
            if t in tray_locations:
                tray_place = tray_locations[t]
                if tray_place in waiting_places:
                    if s in gf_sandwiches_in_state:
                        available_gf_ontray_at_p[tray_place] += 1
                        sandwiches_ontray_at_waiting_place.add(s)
                    else: # Assume non-GF if not marked GF
                        available_reg_ontray_at_p[tray_place] += 1
                        sandwiches_ontray_at_waiting_place.add(s)

        # 8. & 9. Calculate total deficit at location stage
        Total_deficit_gf_at_location = 0
        Total_deficit_reg_at_location = 0

        for p in waiting_places:
            Total_deficit_gf_at_location += max(0, needed_gf_at_p[p] - available_gf_ontray_at_p[p])
            Total_deficit_reg_at_location += max(0, needed_reg_at_p[p] - available_reg_ontray_at_p[p])


        # 10. Count available sandwiches at stages before "ontray at location"

        # Count sandwiches on trays not in kitchen and not already counted at a waiting place
        C_gf_ontray_elsewhere = 0
        C_reg_ontray_elsewhere = 0

        for s, t in ontray_sandwiches.items():
             if s not in sandwiches_ontray_at_waiting_place: # Exclude those already counted as 'at location'
                if t in tray_locations:
                    tray_place = tray_locations[t]
                    if tray_place != 'kitchen':
                         if s in gf_sandwiches_in_state:
                             C_gf_ontray_elsewhere += 1
                         else:
                             C_reg_ontray_elsewhere += 1


        C_gf_kitchen = len([s for s in kitchen_sandwiches if s in gf_sandwiches_in_state])
        C_reg_kitchen = len([s for s in kitchen_sandwiches if s not in gf_sandwiches_in_state])

        # Makable sandwiches (relaxation)
        N_ne_current = len(notexist_sandwiches)
        N_gb_current = len([b for b in kitchen_bread if b in self.gf_bread_types])
        N_gc_current = len([c for c in kitchen_content if c in self.gf_content_types])
        N_rb_current = len([b for b in kitchen_bread if b not in self.gf_bread_types])
        N_rc_current = len([c for c in kitchen_content if c not in self.gf_content_types])

        can_make_gf_by_ingredients = min(N_gb_current, N_gc_current)
        can_make_reg_by_ingredients = min(N_rb_current, N_rc_current)

        # Prioritize making GF if notexist is the bottleneck
        C_gf_makable = min(N_ne_current, can_make_gf_by_ingredients)
        remaining_ne = N_ne_current - C_gf_makable
        C_reg_makable = min(remaining_ne, can_make_reg_by_ingredients)


        # 11. Calculate cost to fulfill deficits
        cost_gf = 0
        N_gf_to_location = Total_deficit_gf_at_location

        # Use sandwiches ontray elsewhere (cost 1: move)
        from_ontray_elsewhere = min(N_gf_to_location, C_gf_ontray_elsewhere)
        cost_gf += from_ontray_elsewhere * 1
        rem_needed_gf = N_gf_to_location - from_ontray_elsewhere

        # Use sandwiches in kitchen (cost 2: put + move)
        from_kitchen = min(rem_needed_gf, C_gf_kitchen)
        cost_gf += from_kitchen * 2
        rem_needed_gf -= from_kitchen

        # Make new sandwiches (cost 3: make + put + move)
        from_makable = min(rem_needed_gf, C_gf_makable)
        cost_gf += from_makable * 3
        rem_needed_gf -= from_makable

        # 12. Check if deficit remains
        if rem_needed_gf > 0:
            return float('inf')

        cost_reg = 0
        N_reg_to_location = Total_deficit_reg_at_location

        # Use sandwiches ontray elsewhere (cost 1: move)
        from_ontray_elsewhere = min(N_reg_to_location, C_reg_ontray_elsewhere)
        cost_reg += from_ontray_elsewhere * 1
        rem_needed_reg = N_reg_to_location - from_ontray_elsewhere

        # Use sandwiches in kitchen (cost 2: put + move)
        from_kitchen = min(rem_needed_reg, C_reg_kitchen)
        cost_reg += from_kitchen * 2
        rem_needed_reg -= from_kitchen

        # Make new sandwiches (cost 3: make + put + move)
        from_makable = min(rem_needed_reg, C_reg_makable)
        cost_reg += from_makable * 3
        rem_needed_reg -= from_makable

        # 12. Check if deficit remains
        if rem_needed_reg > 0:
            return float('inf')

        # 13. Total heuristic
        heuristic_value += cost_gf + cost_reg

        return heuristic_value
