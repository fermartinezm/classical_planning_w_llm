# Ensure the base Heuristic class and Task class are available
# from heuristics.heuristic_base import Heuristic
# from task import Operator, Task

# Assuming the above imports are handled by the environment where this code runs
# If running standalone, you would need to define or import them.

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove leading '(' and trailing ')'
    fact_string = fact_string[1:-1]
    parts = fact_string.split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for three main components:
    1. Serving each unserved child.
    2. Ensuring each unserved child has a suitable sandwich on a tray.
    3. Ensuring trays are located at the places where children are waiting.

    It counts the number of unserved children (cost +1 per child for the serve action).
    It counts the number of "sandwich on tray" units that still need to be created
    (either by putting an existing kitchen sandwich on a tray, cost +1, or by
    making a new sandwich and putting it on a tray, cost +2), accounting for
    gluten requirements and prioritizing existing sandwiches.
    It counts the number of locations where unserved children are waiting but
    no tray is present (cost +1 per location for a move_tray action).

    Assumptions:
    - The heuristic assumes that sufficient ingredients and 'notexist' sandwich
      objects are available in the initial state to make any required sandwiches
      for a solvable problem. It does not check or count available ingredients
      or 'notexist' objects in the current state beyond what's needed for the
      cost calculation structure.
    - It assumes that trays can be moved freely between locations.
    - It assumes that a single tray at a location can serve all children waiting
      at that location (sequentially).
    - It assumes that a single sandwich on a tray can serve one child.
    - The heuristic is not admissible but aims to be informative for greedy
      best-first search by providing a more detailed estimate than simple
      goal counting or relaxation heuristics might in this domain.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task to identify:
    - Which children are allergic or not allergic.
    - Where each child is waiting.
    - Which bread and content portions are gluten-free.
    This information is stored in sets and dictionaries for efficient lookup
    during heuristic computation. It also stores the set of all children defined
    in the static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state facts to extract relevant dynamic information:
       - Set of children who are currently served.
       - Sets of sandwiches that are in the kitchen or on trays.
       - Set of sandwiches that are gluten-free (based on state predicate `no_gluten_sandwich`).
       - Map of tray locations.

    2. Identify the set of unserved children by comparing the set of all children
       (from static facts) with the set of served children (from state facts).

    3. If the set of unserved children is empty, the goal is reached, and the
       heuristic value is 0.

    4. Calculate the first component of the heuristic: the cost for serving.
       This is simply the number of unserved children, as each requires one
       `serve_sandwich` action.

    5. Calculate the demand for sandwiches on trays:
       - Count unserved children who are allergic (`num_unserved_gf`). They need a GF sandwich on a tray.
       - Count unserved children who are not allergic (`num_unserved_any`). They need any sandwich on a tray.

    6. Count the supply of sandwiches already made:
       - `avail_gf_ontray`: GF sandwiches currently on any tray.
       - `avail_any_ontray`: Non-GF sandwiches currently on any tray.
       - `avail_gf_kitchen`: GF sandwiches currently in the kitchen.
       - `avail_any_kitchen`: Non-GF sandwiches currently in the kitchen.

    7. Calculate the second component of the heuristic: the cost for sandwich
       placement (getting sandwiches onto trays). This is done by determining
       how many "sandwich on tray" units are still needed and the cheapest way
       to create them, prioritizing existing sandwiches and GF requirements.
       - Determine how many GF and Any "sandwich on tray" units are needed after
         accounting for those already on trays (`needed_gf_ontray`, `needed_any_ontray`).
       - Determine how many of these remaining needs can be met by sandwiches
         currently in the kitchen (`avail_gf_kitchen`, `avail_any_kitchen`),
         prioritizing GF sandwiches for GF needs. Each such sandwich requires one
         `put_on_tray` action (cost +1).
       - Determine how many needs still remain after using kitchen sandwiches.
         These must be met by making new sandwiches. Each requires one
         `make_sandwich` action + one `put_on_tray` action (cost +2).
       - Sum the costs from using kitchen sandwiches and making new ones.

    8. Calculate the third component of the heuristic: the cost for tray movements.
       - Identify the set of unique locations where unserved children are waiting
         (excluding the kitchen, as children don't wait there).
       - Identify the set of unique locations where trays are currently present.
       - Count the number of locations needing a tray (from the first set) that
         do not currently have a tray (from the second set). Each such location
         requires one `move_tray` action (cost +1).

    9. The total heuristic value is the sum of the costs calculated in steps 4, 7, and 8.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # Pre-process static information
        self.allergic_children_static = set()
        self.not_allergic_children_static = set()
        self.waiting_children_static = {} # child -> place
        self.gf_bread_static = set()
        self.gf_content_static = set()

        for fact_string in task.static:
            predicate, args = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                self.allergic_children_static.add(args[0])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children_static.add(args[0])
            elif predicate == 'waiting':
                self.waiting_children_static[args[0]] = args[1]
            elif predicate == 'no_gluten_bread':
                self.gf_bread_static.add(args[0])
            elif predicate == 'no_gluten_content':
                self.gf_content_static.add(args[0])

        # Store all children names from static facts
        self.all_children_static = set(self.waiting_children_static.keys())


    def __call__(self, node):
        state = node.state

        # --- Parse State Facts ---
        served_children_in_state = set()
        kitchen_sandwiches = set()
        ontray_sandwiches = set() # Stores sandwich names on trays
        tray_locations = {} # tray -> place
        gf_sandwiches_in_state = set()

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'served':
                served_children_in_state.add(args[0])
            elif predicate == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(args[0])
            elif predicate == 'ontray':
                ontray_sandwiches.add(args[0]) # Store sandwich name
            elif predicate == 'at':
                tray_locations[args[0]] = args[1]
            elif predicate == 'no_gluten_sandwich':
                gf_sandwiches_in_state.add(args[0])


        # --- Calculate Heuristic Components ---

        # 1. Cost for serve actions
        unserved_children = {c for c in self.all_children_static if c not in served_children_in_state}
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached
        if num_unserved == 0:
            return 0

        cost_serve = num_unserved

        # 2. Cost for sandwich placement (make + put_on_tray)
        num_unserved_gf = len({c for c in unserved_children if c in self.allergic_children_static})
        num_unserved_any = num_unserved - num_unserved_gf # Non-allergic children can take any sandwich

        # Count available sandwiches by type and location
        avail_gf_ontray = len({s for s in ontray_sandwiches if s in gf_sandwiches_in_state})
        avail_any_ontray = len(ontray_sandwiches) - avail_gf_ontray # Non-GF ontray
        avail_gf_kitchen = len({s for s in kitchen_sandwiches if s in gf_sandwiches_in_state})
        avail_any_kitchen = len(kitchen_sandwiches) - avail_gf_kitchen # Non-GF kitchen

        # Calculate how many "sandwich on tray" units are needed
        needed_gf_ontray = num_unserved_gf
        needed_any_ontray = num_unserved_any

        # Use ontray sandwiches first (cost 0 for placement)
        # Use GF ontray for GF needs
        used_ontray_gf_for_gf = min(needed_gf_ontray, avail_gf_ontray)
        needed_gf_ontray -= used_ontray_gf_for_gf
        avail_gf_ontray_rem = avail_gf_ontray - used_ontray_gf_for_gf

        # Use Any ontray for Any needs
        used_ontray_any_for_any = min(needed_any_ontray, avail_any_ontray)
        needed_any_ontray -= used_ontray_any_for_any
        # avail_any_ontray_rem = avail_any_ontray - used_ontray_any_for_any # Not needed

        # Use remaining GF ontray for remaining Any needs
        used_ontray_gf_for_any = min(needed_any_ontray, avail_gf_ontray_rem)
        needed_any_ontray -= used_ontray_gf_for_any

        # Remaining needs must be met by kitchen sandwiches or new ones
        remaining_needed_gf_ontray = needed_gf_ontray
        remaining_needed_any_ontray = needed_any_ontray

        cost_sandwich_placement = 0

        # Use kitchen sandwiches (cost 1: put_on_tray)
        # Use GF kitchen for remaining GF needs
        use_kitchen_gf_for_gf = min(remaining_needed_gf_ontray, avail_gf_kitchen)
        remaining_needed_gf_ontray -= use_kitchen_gf_for_gf
        avail_gf_kitchen_rem = avail_gf_kitchen - use_kitchen_gf_for_gf
        cost_sandwich_placement += use_kitchen_gf_for_gf

        # Use Any kitchen for remaining Any needs
        use_kitchen_any_for_any = min(remaining_needed_any_ontray, avail_any_kitchen)
        remaining_needed_any_ontray -= use_kitchen_any_for_any
        # avail_any_kitchen_rem = avail_any_kitchen - use_kitchen_any_for_any # Not needed
        cost_sandwich_placement += use_kitchen_any_for_any

        # Use remaining GF kitchen for remaining Any needs
        use_kitchen_gf_for_any = min(remaining_needed_any_ontray, avail_gf_kitchen_rem)
        remaining_needed_any_ontray -= use_kitchen_gf_for_any
        cost_sandwich_placement += use_kitchen_gf_for_any

        # Make new sandwiches (cost 2: make + put_on_tray)
        make_new_gf = remaining_needed_gf_ontray
        make_new_any = remaining_needed_any_ontray
        cost_sandwich_placement += make_new_gf * 2
        cost_sandwich_placement += make_new_any * 2


        # 3. Cost for tray movements
        locations_needing_tray = {self.waiting_children_static[c] for c in unserved_children if self.waiting_children_static[c] != 'kitchen'}
        locations_with_tray = set(tray_locations.values())

        cost_tray_move = len({p for p in locations_needing_tray if p not in locations_with_tray})

        # Total heuristic value
        h_value = cost_serve + cost_sandwich_placement + cost_tray_move

        return h_value
