class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
        This heuristic estimates the number of actions required to reach the goal
        state (all children served) by summing up the estimated number of
        'make_sandwich', 'put_on_tray', 'move_tray', and 'serve_sandwich' actions
        needed for the unserved children. It uses a relaxed approach, primarily
        counting the number of items (sandwiches, trays) that need to transition
        between key states (notexist -> kitchen -> on_tray -> at_location -> served).
        It considers the distinction between gluten-free and regular sandwiches
        when calculating the number of sandwiches that still need to be made or
        put on trays, based on the needs of allergic vs non-allergic children.
        The cost for moving trays is estimated based on the number of locations
        with unserved children that do not currently have a tray.

    Assumptions:
        - The problem instance is solvable.
        - Sufficient ingredients and sandwich slots ('notexist') are available
          to make all necessary sandwiches (the heuristic counts 'make' actions
          needed based on sandwich deficit, but doesn't strictly check ingredient/slot limits).
        - Sufficient trays are available in the kitchen when needed for 'put_on_tray'
          actions (the heuristic counts 'put' actions needed based on sandwich deficit,
          but doesn't strictly check tray availability in kitchen).
        - The cost of moving a tray to a location needing one is 1, regardless of
          its current position or subsequent moves (a weak lower bound).
        - The capacity of trays is sufficient.

    Heuristic Initialization:
        The constructor processes the static facts from the PDDL problem.
        It identifies:
        - Which children are allergic to gluten.
        - The waiting location for each child.
        - Which bread portions are gluten-free.
        - Which content portions are gluten-free.
        This information is stored in sets and dictionaries for efficient lookup
        during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify all children who have not yet been served. If none, the heuristic is 0.
        2. Count the total number of unserved children (`N_unserved`). This is the base cost, representing the minimum number of 'serve_sandwich' actions required. Add `N_unserved` to the heuristic.
        3. Count the total number of sandwiches that have already been made (either in the kitchen or on a tray).
        4. Estimate the number of new sandwiches that still need to be made (`M_total`). This is calculated as the total number of sandwiches required (`N_unserved`) minus the total number already made, minimum 0. Add `M_total` to the heuristic. This step implicitly assumes suitable ingredients and sandwich slots are available.
        5. Count the total number of sandwiches that are already on trays.
        6. Estimate the number of sandwiches that need to be put on trays (`P_total`). This is calculated as the total number of sandwiches required on trays (`N_unserved`) minus the total number already on trays, minimum 0. Add `P_total` to the heuristic. This step implicitly assumes trays are available in the kitchen when needed.
        7. Identify the set of unique locations where unserved children are waiting.
        8. Identify the set of unique locations from step 7 that currently have at least one tray.
        9. Estimate the number of 'move_tray' actions needed. This is calculated as the number of locations from step 7 that are *not* in the set from step 8. This represents a lower bound on the tray movements needed to get trays to all required locations. Add this count to the heuristic.
        10. The total heuristic value is the sum of the costs from steps 2, 5, 7, and 9.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object (instance of the Task class).
        """
        self.allergic_children = set()
        self.waiting_at = {}  # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.all_children = set()
        self.all_places = set()

        # Process static facts
        for fact_str in task.static:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'allergic_gluten':
                self.allergic_children.add(objs[0])
                self.all_children.add(objs[0])
            elif pred == 'not_allergic_gluten':
                 # Also add non-allergic children to the set of all children
                 self.all_children.add(objs[0])
            elif pred == 'waiting':
                child, place = objs
                self.waiting_at[child] = place
                # Children and places from waiting facts are also part of the problem
                self.all_children.add(child)
                self.all_places.add(place)
            elif pred == 'no_gluten_bread':
                self.no_gluten_bread.add(objs[0])
            elif pred == 'no_gluten_content':
                self.no_gluten_content.add(objs[0])

        # Ensure 'kitchen' is included in places if it's not explicitly in waiting facts
        self.all_places.add('kitchen')


    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string."""
        # Remove leading/trailing parentheses and split by space
        parts = fact_str.strip().strip('()').split()
        if not parts:
            return None, [] # Handle empty string case
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            An integer representing the estimated number of actions to reach the goal.
        """
        served_children = set()
        at_kitchen_sandwich = set()
        ontray = set()  # set of (sandwich, tray)
        at_tray = {}  # tray -> place
        # We don't need no_gluten_sandwich from state for this simplified heuristic,
        # as we count total sandwiches needed regardless of type for make/put stages.
        # The GF requirement is implicitly handled by the total count of unserved children.

        # Process state facts
        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'served':
                served_children.add(objs[0])
            elif pred == 'at_kitchen_sandwich':
                at_kitchen_sandwich.add(objs[0])
            elif pred == 'ontray':
                ontray.add((objs[0], objs[1]))
            elif pred == 'at':
                at_tray[objs[0]] = objs[1]
            # We don't need at_kitchen_bread, at_kitchen_content, notexist, no_gluten_sandwich for this heuristic calculation

        unserved_children = self.all_children - served_children
        N_unserved = len(unserved_children)

        # Goal reached
        if N_unserved == 0:
            return 0

        # Heuristic components calculation
        h = 0

        # 1. Cost for serve actions
        # Each unserved child needs one serve action.
        h += N_unserved

        # 2. Cost for make actions
        # Count sandwiches already made (in kitchen or on tray)
        all_made_sandwiches = {s for s in at_kitchen_sandwich} | {s for s, t in ontray}
        N_sandw_avail_made = len(all_made_sandwiches)

        # Number of sandwiches that still need to be made
        # We need N_unserved sandwiches in total.
        M_total = max(0, N_unserved - N_sandw_avail_made)
        h += M_total

        # 3. Cost for put_on_tray actions
        # Count sandwiches already on trays
        N_sandw_avail_ontray = len(ontray)

        # Number of sandwiches that need to be put on trays
        # We need N_unserved sandwiches on trays in total.
        P_total = max(0, N_unserved - N_sandw_avail_ontray)
        h += P_total

        # 4. Cost for move_tray actions
        # Identify places where unserved children are waiting
        # Ensure child is in waiting_at map before accessing it
        Target_places_set = {self.waiting_at[c] for c in unserved_children if c in self.waiting_at}

        # Identify places from Target_places_set that currently have a tray
        Trays_at_target_places_set = {place for tray, place in at_tray.items() if place in Target_places_set}

        # Number of target places that do not currently have a tray
        # Each such place needs at least one tray moved to it.
        N_places_needing_tray = len(Target_places_set - Trays_at_target_places_set)
        h += N_places_needing_tray

        return h
