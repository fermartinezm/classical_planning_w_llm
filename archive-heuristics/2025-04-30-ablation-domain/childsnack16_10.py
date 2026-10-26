from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of children still needing service, the number of sandwiches
    that need to be made, the number of sandwiches that need to be placed on trays,
    and the number of tables with waiting children that need a tray moved to them.

    # Assumptions
    - Each waiting child requires one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can eat any sandwich.
    - Ingredients (bread, content) are available in the kitchen if needed to make sandwiches.
    - Trays are available somewhere to be moved.
    - The cost of moving a tray between any two locations is 1.
    - The cost of making a sandwich, placing it on a tray, and serving a child is 1 each.
    - The heuristic does not account for resource contention (e.g., limited trays, limited ingredients of specific types beyond total count, robot location).

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are allergic to gluten.
    - Which table each child is initially waiting at.
    - Which bread and content portions are gluten-free (though this is not strictly used in the current simplified sandwich-making cost).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:
    1. Identify all children who are currently 'waiting' but not yet 'served'. This gives the total number of children still needing service (`N_waiting_not_served`). Each needs a 'serve' action.
    2. Categorize the waiting, not-served children by allergy status (This categorization helps understand the problem but is not directly used in the simplified counts below).
    3. Identify all sandwiches that currently 'exist' (are not 'notexist'). This includes sandwiches that are 'at_kitchen_sandwich' or 'ontray'. Count the total number of existing sandwiches (`N_existing_any`).
    4. Identify existing sandwiches that are 'no_gluten_sandwich'. Count these (`N_existing_gf`). (This count is noted but not directly used in the simplified counts below).
    5. Calculate the number of new sandwiches that need to be made (`N_total_to_make`). This is the total number of waiting children minus the total number of existing sandwiches, capped at zero (we don't make surplus sandwiches). `N_total_to_make = max(0, N_waiting_not_served - N_existing_any)`. Each needs a 'make-sandwich' action.
    6. Count the number of existing sandwiches that are currently 'at_kitchen_sandwich' (`N_at_kitchen`). These, plus the sandwiches that will be made (`N_total_to_make`), need to be placed on a tray. Calculate the total number of 'place-on-tray' actions needed: `N_to_place = N_at_kitchen + N_total_to_make`.
    7. Identify all distinct tables where children are waiting (from static information).
    8. Check the current state to see which of these waiting tables currently have a tray located at them.
    9. Count the number of waiting tables that do *not* have a tray (`N_tables_needing_tray`). Each needs a 'move-tray' action.
    10. The total heuristic value is the sum of the counts of these necessary actions: `N_waiting_not_served + N_total_to_make + N_to_place + N_tables_needing_tray`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        # Store goals for potential future use, though not strictly needed for this heuristic calculation logic.
        self.goals = task.goals

        # Extract static facts
        self.child_allergy = {} # child_name -> 'gluten' or 'none'
        self.child_waiting_table = {} # child_name -> table_name
        self.gf_bread = set() # set of gluten-free bread names (not used in current heuristic logic)
        self.gf_content = set() # set of gluten-free content names (not used in current heuristic logic)

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'allergic_gluten':
                if len(parts) > 1:
                    self.child_allergy[parts[1]] = 'gluten'
            elif parts[0] == 'not_allergic_gluten':
                 if len(parts) > 1:
                    self.child_allergy[parts[1]] = 'none'
            elif parts[0] == 'waiting':
                # Static waiting facts define initial waiting locations
                if len(parts) > 2:
                    self.child_waiting_table[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_bread':
                if len(parts) > 1:
                    self.gf_bread.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                if len(parts) > 1:
                    self.gf_content.add(parts[1])

        # Identify all children who are initially waiting based on static facts
        self.initial_waiting_children = set(self.child_waiting_table.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify served children
        served_children = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'served' and len(parts) > 1:
                served_children.add(parts[1])

        # Identify children who were initially waiting but are not yet served
        waiting_children_not_served = self.initial_waiting_children - served_children
        N_waiting_not_served = len(waiting_children_not_served)

        # If no children are waiting to be served, the goal is reached.
        if N_waiting_not_served == 0:
            return 0

        # 2. Categorize waiting children by allergy status (not used in simplified counts)
        # waiting_gf_children = {c for c in waiting_children_not_served if self.child_allergy.get(c) == 'gluten'}
        # waiting_any_children = waiting_children_not_served - waiting_gf_children

        # 3. Identify existing sandwiches and their locations/status
        existing_sandwiches = set()
        sandwiches_at_kitchen = set()
        # sandwiches_on_trays = set() # Not directly used in counts
        # gf_sandwiches_in_state = set() # Not directly used in counts

        tray_locations = {} # tray_name -> place_name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'ontray' and len(parts) > 1:
                sandwich_name = parts[1]
                # tray_name = parts[2] # Not directly used
                existing_sandwiches.add(sandwich_name)
                # sandwiches_on_trays.add(sandwich_name) # Not directly used
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) > 1:
                sandwich_name = parts[1]
                existing_sandwiches.add(sandwich_name)
                sandwiches_at_kitchen.add(sandwich_name)
            # elif parts[0] == 'no_gluten_sandwich' and len(parts) > 1:
                 # This predicate indicates the sandwich type, not necessarily existence/location
                 # gf_sandwiches_in_state.add(parts[1]) # Not directly used
            elif parts[0] == 'at' and len(parts) > 2 and parts[1].startswith('tray'):
                 tray_locations[parts[1]] = parts[2]


        N_existing_any = len(existing_sandwiches)
        # N_existing_gf = len(existing_sandwiches.intersection(gf_sandwiches_in_state)) # Not used

        # 5. Calculate sandwiches to make
        # Simple deficit count: total needed vs total existing
        N_total_to_make = max(0, N_waiting_not_served - N_existing_any)

        # 6. Calculate sandwiches to place on trays
        # These are sandwiches currently in the kitchen + sandwiches that will be made
        N_at_kitchen = len(sandwiches_at_kitchen)
        N_to_place = N_at_kitchen + N_total_to_make

        # 7. Identify tables where children are waiting (and not yet served)
        waiting_tables = {self.child_waiting_table[c] for c in waiting_children_not_served if c in self.child_waiting_table}

        # 8. Check which waiting tables have a tray
        tables_with_trays = {loc for tray, loc in tray_locations.items() if loc in waiting_tables}

        # 9. Count tables needing a tray
        N_tables_needing_tray = len(waiting_tables) - len(tables_with_trays)

        # 10. Calculate total heuristic
        # Sum of required actions: serve, make, place, move
        heuristic_value = N_waiting_not_served + N_total_to_make + N_to_place + N_tables_needing_tray

        return heuristic_value
