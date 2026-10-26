# Helper function to parse PDDL fact string
def parse_fact(fact_str):
    """Parses a PDDL fact string into a predicate and a list of objects."""
    # Remove parentheses and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served) by summing up the estimated costs for
    different stages of the sandwich delivery pipeline for the unserved
    children. It counts how many "items" (sandwiches needed for unserved
    children) still need to pass through the 'make', 'put on tray', and
    'move tray' stages before they can be 'served'.

    Assumptions:
    - The problem instance is solvable.
    - There are enough ingredients (bread/content) and `notexist` sandwich
      predicates available in the kitchen to make any required sandwiches.
    - Tray capacity is not a limiting factor for the number of sandwiches
      that can be placed on or moved with a tray instance in terms of
      heuristic calculation (i.e., we count the number of sandwich-tray
      deliveries needed, not the number of tray instances).
    - Suitability (gluten-free or not) is considered when determining if a
      sandwich can satisfy a child's need for the 'Ready_Deliveries' count.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task definition
    to identify which children are allergic to gluten and where each child
    is waiting. This information is constant throughout the planning process.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children who are waiting but not yet served. The total
       count of these children (`N_unserved`) represents the minimum number
       of 'serve' actions required. This contributes `N_unserved` to the heuristic.
    2. Count the number of sandwiches currently available in the kitchen
       (`Avail_Kitchen`, i.e., `at_kitchen_sandwich`) and the number of
       sandwiches already on trays (`Avail_Ontray`, i.e., `ontray`). The total
       made sandwiches are `Avail_Made = Avail_Kitchen + Avail_Ontray`.
    3. Estimate the number of 'make_sandwich' actions needed. This is the
       number of sandwiches required (`N_unserved`) minus those already made.
       Cost: `max(0, N_unserved - Avail_Made)`.
    4. Estimate the number of 'put_on_tray' actions needed. This is the number
       of sandwiches required (`N_unserved`) minus those already on trays.
       Cost: `max(0, N_unserved - Avail_Ontray)`.
    5. Estimate the number of 'move_tray' actions needed. This is the number
       of sandwich-on-tray deliveries required (`N_unserved`) minus those
       that are already on a tray at the correct location for a suitable
       unserved child (`Ready_Deliveries`).
       To calculate `Ready_Deliveries`: Iterate through each unserved child.
       For a child at place `p`, check if there is any suitable sandwich `s`
       currently on a tray `t` where `t` is located at `p`. Count how many
       unserved children can be matched with such a "ready" sandwich-on-tray
       unit, ensuring each physical sandwich instance is used at most once
       in this count.
       Cost: `max(0, N_unserved - Ready_Deliveries)`.
    6. The total heuristic value is the sum of the costs from steps 1, 3, 4, and 5.
       `h = N_unserved + cost_make + cost_put_on_tray + cost_move_tray`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object.
        """
        self.allergic_children = set()
        self.waiting_children = {} # {child: place}

        # Process static facts
        for fact_str in task.static:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'allergic_gluten':
                self.allergic_children.add(objects[0])
            elif predicate == 'waiting':
                self.waiting_children[objects[0]] = objects[1]

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach a goal state.
        """
        # Parse state facts
        at_kitchen_sandwiches = set()
        ontray_sandwiches = set() # Stores sandwich objects that are on trays
        ontray_map = {} # {sandwich: tray}
        tray_location_map = {} # {tray: place}
        no_gluten_sandwiches_in_state = set() # Sandwiches currently known to be GF

        served_children = set()

        for fact_str in state:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(objects[0])
            elif predicate == 'ontray':
                s, t = objects
                ontray_sandwiches.add(s)
                ontray_map[s] = t
            elif predicate == 'at':
                t, p = objects
                tray_location_map[t] = p
            elif predicate == 'no_gluten_sandwich':
                no_gluten_sandwiches_in_state.add(objects[0])
            elif predicate == 'served':
                served_children.add(objects[0])

        # 1. Count unserved children
        unserved_children_list = [] # List of (child, place) tuples
        for child, place in self.waiting_children.items():
            if child not in served_children:
                unserved_children_list.append((child, place))

        N_unserved = len(unserved_children_list)

        # If no children are unserved, the goal is reached
        if N_unserved == 0:
            return 0

        # 2. Count available made sandwiches
        Avail_Kitchen = len(at_kitchen_sandwiches)
        Avail_Ontray = len(ontray_sandwiches)
        Avail_Made = Avail_Kitchen + Avail_Ontray

        # 3. Calculate Ready_Deliveries
        Ready_Deliveries = 0
        # Keep track of sandwich instances used for ready deliveries to avoid double counting
        used_ontray_sandwiches_for_ready = set()

        # Create a list of (sandwich, tray, location) for sandwiches currently on trays
        sandwiches_ontray_at_loc = []
        for s in ontray_sandwiches:
            t = ontray_map.get(s)
            if t is not None:
                loc = tray_location_map.get(t)
                if loc is not None: # Only consider trays with a known location
                     sandwiches_ontray_at_loc.append((s, t, loc))

        # Match unserved children with available ready deliveries
        for child, child_loc in unserved_children_list:
            child_is_allergic = child in self.allergic_children
            found_ready_sandwich_for_child = False

            for s, t, tray_loc in sandwiches_ontray_at_loc:
                 # Check if this sandwich instance is already used for another child's ready delivery
                if s not in used_ontray_sandwiches_for_ready:
                    if tray_loc == child_loc:
                        sandwich_is_nogluten = s in no_gluten_sandwiches_in_state
                        is_suitable = (child_is_allergic and sandwich_is_nogluten) or (not child_is_allergic)

                        if is_suitable:
                            Ready_Deliveries += 1
                            used_ontray_sandwiches_for_ready.add(s)
                            found_ready_sandwich_for_child = True
                            break # Found a ready sandwich for this child, move to the next unserved child

        # Calculate heuristic components
        cost_serve = N_unserved
        cost_make = max(0, N_unserved - Avail_Made)
        cost_put_on_tray = max(0, N_unserved - Avail_Ontray)
        cost_move_tray = max(0, N_unserved - Ready_Deliveries)

        # Total heuristic is the sum of costs for each stage needed
        h = cost_serve + cost_make + cost_put_on_tray + cost_move_tray

        return h
