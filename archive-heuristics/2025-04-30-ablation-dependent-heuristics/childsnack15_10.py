import collections

def parse_fact(fact_str):
    """
    Parses a PDDL fact string into a predicate name and a list of objects.
    e.g., '(at tray1 kitchen)' -> ('at', ['tray1', 'kitchen'])
    e.g., '(served child1)' -> ('served', ['child1'])
    """
    # Remove surrounding brackets and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served) by summing up counts of items or conditions
    that are not yet in a state ready for the next step towards the goal.
    It counts:
    1. The number of unserved children (representing the final 'serve' action).
    2. The number of suitable sandwiches that still need to be made.
    3. The number of suitable sandwiches that are currently in the kitchen
       and need to be put on a tray.
    4. The number of locations with unserved children that do not currently
       have a tray.

    Assumptions:
    - The problem instance is solvable, implying sufficient ingredients
      (bread, content) and sandwich objects exist initially to make all
      necessary sandwiches.
    - Sufficient trays exist to eventually deliver sandwiches to all required locations.
    - The heuristic does not account for the specific types of bread/content
      needed beyond the gluten-free requirement for making GF sandwiches.
    - The heuristic is not admissible; it is designed for greedy best-first search.
    - Object types are inferred from the predicates they appear with in the
      initial state and static facts.

    Heuristic Initialization:
    The constructor processes the static facts and initial goal state from the
    Task object to pre-compute information needed for the heuristic calculation.
    This includes:
    - Identifying which children are allergic to gluten.
    - Mapping each child to their waiting location.
    - Identifying the set of children that need to be served according to the goal.
    - Collecting the names of all relevant objects (children, trays, sandwiches,
      places, bread, content) present in the initial state or static facts,
      inferring types based on the predicates they appear with.
    - Identifying which bread and content portions are gluten-free (static).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the set of children who are in the goal state but are not yet
       marked as 'served' in the current state (U).
    2. Calculate the total number of unserved children (N_unserved). This contributes
       N_unserved to the heuristic sum, representing the final 'serve' action
       needed for each.
    3. Determine the number of gluten-free and regular sandwiches required to
       serve all unserved children (needed_gf, needed_reg).
    4. Identify all sandwiches currently existing in the state (either in the
       kitchen or on a tray) and determine their gluten-free status.
    5. Calculate the number of gluten-free and regular sandwiches that still
       need to be made (to_make_gf, to_make_reg) by comparing the number needed
       with the number already existing. The sum (to_make_gf + to_make_reg)
       contributes to the heuristic, representing the 'make_sandwich' actions.
    6. Identify sandwiches currently located in the kitchen. The count of these
       sandwiches (regardless of type) contributes to the heuristic, representing
       the 'put_on_tray' actions needed.
    7. Identify locations (other than the kitchen) where unserved children are
       waiting.
    8. Identify locations (other than the kitchen) where trays are currently
       located.
    9. Calculate the number of locations with unserved children that do not
       currently have a tray. This count contributes to the heuristic, representing
       the 'move_tray' actions needed to bring trays to these locations.
    10. The total heuristic value is the sum of the counts from steps 2, 5, 6, and 9.
        h = N_unserved + (to_make_gf + to_make_reg) + (kitchen_sandwiches_count) + (locations_needing_tray_count).
    """
    def __init__(self, task):
        # Extract static information and collect all objects by type
        self.child_allergic = {}
        self.child_location = {}
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.goal_children = set()

        self.all_children = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()

        # Map predicates to the type of object expected at each argument position
        predicate_arg_types = {
            'allergic_gluten': {0: 'child'},
            'not_allergic_gluten': {0: 'child'},
            'served': {0: 'child'},
            'waiting': {0: 'child', 1: 'place'},
            'at': {0: 'tray', 1: 'place'},
            'at_kitchen_bread': {0: 'bread-portion'},
            'at_kitchen_content': {0: 'content-portion'},
            'at_kitchen_sandwich': {0: 'sandwich'},
            'no_gluten_bread': {0: 'bread-portion'},
            'no_gluten_content': {0: 'content-portion'},
            'ontray': {0: 'sandwich', 1: 'tray'},
            'no_gluten_sandwich': {0: 'sandwich'},
            'notexist': {0: 'sandwich'},
        }

        # Collect objects from initial state and static facts based on predicate arguments
        for fact_str in task.static | task.initial_state:
            predicate, objects = parse_fact(fact_str)
            if predicate in predicate_arg_types:
                for i, obj in enumerate(objects):
                    obj_type = predicate_arg_types[predicate].get(i)
                    if obj_type == 'child': self.all_children.add(obj)
                    elif obj_type == 'tray': self.all_trays.add(obj)
                    elif obj_type == 'place': self.all_places.add(obj)
                    elif obj_type == 'sandwich': self.all_sandwiches.add(obj)
                    elif obj_type == 'bread-portion': self.all_bread.add(obj)
                    elif obj_type == 'content-portion': self.all_content.add(obj)

        # Add 'kitchen' constant explicitly
        self.all_places.add('kitchen')

        # Process static facts
        for fact_str in task.static:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'allergic_gluten' and objects: self.child_allergic[objects[0]] = True
            elif predicate == 'not_allergic_gluten' and objects: self.child_allergic[objects[0]] = False
            elif predicate == 'waiting' and len(objects) == 2: self.child_location[objects[0]] = objects[1]
            elif predicate == 'no_gluten_bread' and objects: self.no_gluten_bread.add(objects[0])
            elif predicate == 'no_gluten_content' and objects: self.no_gluten_content.add(objects[0])

        # Process goal facts
        for fact_str in task.goals:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'served' and objects: self.goal_children.add(objects[0])


    def __call__(self, state):
        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if f'(served {c})' not in state}
        num_unserved = len(unserved_children)

        # If goal is reached, heuristic is 0
        if num_unserved == 0:
            return 0

        # 2. Count sandwiches needed by unserved children
        needed_gf = sum(1 for c in unserved_children if self.child_allergic.get(c, False))
        needed_reg = sum(1 for c in unserved_children if not self.child_allergic.get(c, True))

        # 3. Collect available sandwiches and their status/location from current state
        sandwich_is_gf = {}
        available_gf_kitchen = set()
        available_reg_kitchen = set()
        available_gf_ontray = set()
        available_reg_ontray = set()
        tray_locations = {} # tray -> location

        # Determine GF status of existing sandwiches from state
        for fact_str in state:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'no_gluten_sandwich' and objects:
                sandwich_is_gf[objects[0]] = True

        # Determine location and collect counts
        for fact_str in state:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'at_kitchen_sandwich' and objects:
                s = objects[0]
                if sandwich_is_gf.get(s, False): available_gf_kitchen.add(s)
                else: available_reg_kitchen.add(s)
            elif predicate == 'ontray' and len(objects) == 2:
                s, t = objects
                if sandwich_is_gf.get(s, False): available_gf_ontray.add(s)
                else: available_reg_ontray.add(s)
            elif predicate == 'at' and len(objects) == 2:
                t, p = objects
                tray_locations[t] = p

        # 4. Calculate sandwiches still needing to be made
        available_gf_anywhere = available_gf_kitchen | available_gf_ontray
        available_reg_anywhere = available_reg_kitchen | available_reg_ontray

        to_make_gf = max(0, needed_gf - len(available_gf_anywhere))
        to_make_reg = max(0, needed_reg - len(available_reg_anywhere))
        cost_make = to_make_gf + to_make_reg

        # 5. Calculate sandwiches in kitchen needing put_on_tray
        cost_put_on_tray = len(available_gf_kitchen) + len(available_reg_kitchen)

        # 6. Calculate trays needing to be moved
        locations_with_unserved = {self.child_location.get(c) for c in unserved_children if self.child_location.get(c) != 'kitchen'}
        locations_with_trays = {loc for tray, loc in tray_locations.items() if loc != 'kitchen'}
        locations_needing_tray = locations_with_unserved - locations_with_trays
        cost_move_tray = len(locations_needing_tray)

        # 7. Calculate children needing serve (this is just the number unserved)
        cost_serve = num_unserved

        # Total heuristic is the sum of these components
        heuristic_value = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return heuristic_value
