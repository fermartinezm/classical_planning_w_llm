# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and a list of objects.
    e.g., '(at tray1 kitchen)' -> ('at', ['tray1', 'kitchen'])
    """
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served) by summing up the estimated costs of
    different stages of the process: making necessary sandwiches, putting
    sandwiches onto trays, moving trays to locations where children are
    waiting, and finally serving the children. It is not admissible but
    aims to guide a greedy best-first search efficiently.

    Assumptions:
    - The problem instance is solvable.
    - Resource constraints (like limited bread/content types for specific
      sandwich types, or tray capacity) are relaxed; we only count the
      *number* of items needed/available.
    - Actions in different stages (make, put, move, serve) are largely
      independent for heuristic calculation purposes, although in reality
      they form a sequence.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It extracts and stores:
    - Which children are allergic or not allergic.
    - The waiting place for each child.
    - Which bread and content items are gluten-free.
    This information is constant throughout the planning process.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic is calculated as the sum of four components:

    1.  Cost to Make Sandwiches (`cost_make`):
        - Identify unserved allergic and non-allergic children.
        - Count existing gluten-free (GF) and regular sandwiches (anywhere).
        - Calculate the number of *new* GF sandwiches needed: `max(0, needed_gf - existing_gf)`.
        - Calculate the number of *new* regular sandwiches needed: `max(0, needed_reg - existing_reg - gf_surplus)`, where `gf_surplus` is any excess of existing GF sandwiches over the GF need.
        - `cost_make` is the sum of new GF and new regular sandwiches needed. This represents the number of `make_sandwich` actions required.

    2.  Cost to Put Sandwiches on Trays (`cost_put_on_tray`):
        - Sandwiches must be on trays to be served. Sandwiches are initially made in the kitchen or might already be in the kitchen.
        - This component counts the number of sandwiches that are currently `at_kitchen_sandwich` plus the number of sandwiches that still need to be made (calculated in `cost_make`). Each of these sandwiches will require a `put_on_tray` action.

    3.  Cost to Move Trays (`cost_move_trays`):
        - Children need to be served at their waiting places. This requires a tray to be present at that location.
        - Identify all locations where unserved children are waiting.
        - Identify all locations where trays are currently present.
        - `cost_move_trays` is the number of distinct locations with unserved children that do *not* currently have a tray. This represents the minimum number of `move_tray` actions needed to bring trays to these locations.

    4.  Cost to Serve Children (`cost_serve`):
        - Each unserved child requires a final `serve_sandwich` action.
        - `cost_serve` is simply the total number of unserved children.

    The total heuristic value is the sum of `cost_make + cost_put_on_tray + cost_move_trays + cost_serve`.
    A value of 0 is returned if and only if all children are served (goal state).
    """
    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts from the task.
        """
        self.task = task
        self.child_allergy = {} # child -> bool (True if allergic)
        self.child_waiting_place = {} # child -> place
        self.gf_bread_types = set() # set of GF bread objects
        self.gf_content_types = set() # set of GF content objects

        # Process static facts
        for fact_string in task.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                child = objects[0]
                self.child_allergy[child] = True
            elif predicate == 'not_allergic_gluten':
                child = objects[0]
                self.child_allergy[child] = False
            elif predicate == 'waiting':
                child, place = objects
                self.child_waiting_place[child] = place
            elif predicate == 'no_gluten_bread':
                bread = objects[0]
                self.gf_bread_types.add(bread)
            elif predicate == 'no_gluten_content':
                content = objects[0]
                self.gf_content_types.add(content)

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # Goal check
        if self.task.goal_reached(state):
            return 0

        # Data structures for state information
        served_children = set()
        kitchen_sandwiches = set()
        sandwich_on_tray = {} # sandwich -> tray
        gf_sandwiches = set() # existing GF sandwiches
        tray_location = {} # tray -> place

        # Process state facts
        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'served':
                served_children.add(objects[0])
            elif predicate == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(objects[0])
            elif predicate == 'ontray':
                sandwich, tray = objects
                sandwich_on_tray[sandwich] = tray
            elif predicate == 'no_gluten_sandwich':
                gf_sandwiches.add(objects[0])
            elif predicate == 'at':
                tray, place = objects
                tray_location[tray] = place
            # Ignore at_kitchen_bread, at_kitchen_content, notexist for this heuristic calculation

        # Identify unserved children and their needs
        unserved_allergic = set()
        unserved_non_allergic = set()
        unserved_children_by_place = {} # place -> set of unserved children

        for child, place in self.child_waiting_place.items():
            if child not in served_children:
                unserved_children_by_place.setdefault(place, set()).add(child)
                # Use .get() with default False for safety, though static should define allergy
                if self.child_allergy.get(child, False):
                    unserved_allergic.add(child)
                else:
                    unserved_non_allergic.add(child)

        # Count existing sandwiches (anywhere)
        existing_sandwiches = kitchen_sandwiches | set(sandwich_on_tray.keys())
        existing_gf_sandwiches = {s for s in existing_sandwiches if s in gf_sandwiches}
        existing_reg_sandwiches = existing_sandwiches - existing_gf_sandwiches # Regular sandwiches

        # Calculate sandwiches to make
        needed_gf = len(unserved_allergic)
        needed_reg = len(unserved_non_allergic)

        # Use existing GF sandwiches for allergic children first
        make_gf = max(0, needed_gf - len(existing_gf_sandwiches))
        gf_surplus = max(0, len(existing_gf_sandwiches) - needed_gf)

        # Use existing regular sandwiches and GF surplus for non-allergic children
        make_reg = max(0, needed_reg - len(existing_reg_sandwiches) - gf_surplus)

        cost_make = make_gf + make_reg

        # Calculate cost to put sandwiches on trays
        # Sandwiches that need putting on trays are those in the kitchen now,
        # plus those that will be made.
        cost_put_on_tray = len(kitchen_sandwiches) + cost_make

        # Calculate cost to move trays
        locations_with_unserved = set(unserved_children_by_place.keys())
        locations_with_trays = set(tray_location.values())
        cost_move_trays = len(locations_with_unserved - locations_with_trays)

        # Calculate cost to serve children
        cost_serve = len(unserved_allergic) + len(unserved_non_allergic)

        # Total heuristic is the sum of independent action counts
        heuristic_value = cost_make + cost_put_on_tray + cost_move_trays + cost_serve

        return heuristic_value
