from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Helper function to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to serve all
    unserved children. It breaks down the problem into four main stages
    for the sandwiches needed to satisfy the remaining children:
    1. Making sandwiches that don't exist yet.
    2. Putting sandwiches (newly made or already in the kitchen) onto trays in the kitchen.
    3. Moving trays from the kitchen to the locations where sandwiches are needed.
    4. Serving the children.
    The heuristic sums the estimated minimum actions for each stage, based
    on the current state and the unsatisfied goals.

    Assumptions:
    - Sufficient bread, content, and 'notexist' sandwich objects are available
      to make any required sandwiches.
    - Sufficient trays are available.
    - Trays are always moved from the kitchen to the destination place.
    - Sandwiches are always put onto trays while the tray is in the kitchen.
    - Trays have infinite capacity for sandwiches.
    - A single tray move can deliver all sandwiches needed at a specific location.

    Heuristic Initialization:
    In the __init__ method, the heuristic pre-processes the static facts
    from the task description to build data structures:
    - Identifies all children and places in the domain.
    - Stores the waiting location for each child.
    - Stores whether each child is allergic to gluten.
    - Identifies all tray objects.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic calculates the sum of estimated actions for four main processes:
    1.  Calculate Unserved Children's Needs:
        Iterate through all children. If a child is not marked as 'served' in the current state,
        identify their waiting location and gluten allergy status (using pre-processed static info).
        Count the number of unserved children needing a gluten-free (GF) sandwich and those
        needing a regular (Reg) sandwich, grouped by their waiting location.

    2.  Calculate Available Sandwiches on Trays at Locations:
        Iterate through facts in the current state to find which trays are at which locations.
        Iterate through facts to find which sandwiches are on which trays.
        Identify which sandwiches are gluten-free based on the 'no_gluten_sandwich' predicate
        in the current state.
        Count the number of GF and Reg sandwiches currently on trays at each location.

    3.  Calculate Deficit at Each Location:
        For each location (excluding the kitchen), compare the unserved children's needs
        with the available sandwiches on trays at that location. Calculate the deficit
        of GF sandwiches and Reg sandwiches needed at that location. Surplus GF sandwiches
        at a location can cover Reg needs at that same location.

    4.  Calculate Total Deficit and Kitchen Inventory:
        Sum the GF and Reg deficits across all locations (excluding the kitchen) to get
        the total number of GF and Reg sandwiches that must eventually be delivered
        from the kitchen.
        Count the number of GF and Reg sandwiches currently in the kitchen
        ('at_kitchen_sandwich' predicate).

    5.  Estimate 'Make Sandwich' Actions:
        Calculate how many GF and Reg sandwiches still need to be made to cover the
        total deficit, considering the sandwiches already available in the kitchen
        (either 'at_kitchen_sandwich' or 'ontray' in the kitchen). This is the number
        of needed sandwiches not yet in the kitchen. Each sandwich made costs 1 action.

    6.  Estimate 'Put On Tray' Actions (in Kitchen):
        Calculate how many sandwiches need to be put on trays in the kitchen to fulfill
        the total deficit at other locations. This is the total deficit minus any
        sandwiches already on trays in the kitchen. Each 'put_on_tray' action costs 1.

    7.  Estimate 'Move Tray' Actions:
        Identify the number of distinct locations (excluding the kitchen) that still
        have a deficit of suitable sandwiches on trays. For each such location,
        at least one tray needs to be moved there from the kitchen. Each 'move_tray'
        action from kitchen costs 1.

    8.  Estimate 'Serve Sandwich' Actions:
        The number of 'serve' actions required is equal to the total number of unserved children.
        Each 'serve' action costs 1.

    9.  Total Heuristic Value:
        The heuristic value is the sum of the estimated actions from steps 5, 6, 7, and 8.
        If the goal (all children served) is reached, the heuristic is 0.
    """
    def __init__(self, task):
        super().__init__(task)
        self.task = task # Store task to access objects later
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-process static facts
        self.waiting_location = {}
        self.is_allergic = {}
        self.all_children = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_location[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif parts[0] == 'allergic_gluten':
                child = parts[1]
                self.is_allergic[child] = True
                self.all_children.add(child)
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.is_allergic[child] = False # Explicitly store False
                self.all_children.add(child)

        # Ensure all children from static facts are in is_allergic map
        # (Children might be mentioned in waiting but not allergy facts if not_allergic is omitted,
        # though PDDL usually requires one or the other). Assume not_allergic if not allergic_gluten.
        for child in self.all_children:
             if child not in self.is_allergic:
                 self.is_allergic[child] = False # Default to not allergic if not specified

        # Get all places from task objects as well
        for obj_str in task.objects:
            parts = obj_str.split(' - ')
            if len(parts) == 2 and parts[1] == 'place':
                self.all_places.add(parts[0])

        # Identify all tray objects
        self.all_trays = {obj.split(' - ')[0] for obj in task.objects if 'tray' in obj}


    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children and their needs (GF/Reg) and locations.
        unserved_children = set()
        served_children_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}

        unserved_needs_at_place = {place: {'gf': 0, 'reg': 0} for place in self.all_places}

        for child in self.all_children:
            if child not in served_children_in_state:
                unserved_children.add(child)
                place = self.waiting_location.get(child) # Get place from static info
                if place: # Should always have a waiting location if unserved
                    if self.is_allergic.get(child, False): # Default to not allergic
                        unserved_needs_at_place[place]['gf'] += 1
                    else:
                        unserved_needs_at_place[place]['reg'] += 1

        # If no unserved children, goal is reached
        if not unserved_children:
            return 0

        # 2. Calculate Available Sandwiches on Trays at Locations
        tray_location = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.all_trays: # Check if object is a tray
                 tray, place = parts[1], parts[2]
                 tray_location[tray] = place

        is_gf_sandwich_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'no_gluten_sandwich'}

        sandwiches_on_trays_at_loc = {place: {'gf': 0, 'reg': 0} for place in self.all_places}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                sandwich, tray = parts[1], parts[2]
                if tray in tray_location: # Only count if the tray's location is known
                    place = tray_location[tray]
                    if sandwich in is_gf_sandwich_in_state:
                        sandwiches_on_trays_at_loc[place]['gf'] += 1
                    else:
                        sandwiches_on_trays_at_loc[place]['reg'] += 1

        # 3. Calculate Deficit at Each Location (excluding kitchen)
        deficit_at_place = {place: {'gf': 0, 'reg': 0} for place in self.all_places if place != 'kitchen'}

        for place in deficit_at_place: # Iterate only over places != kitchen
            needed_gf = unserved_needs_at_place.get(place, {'gf': 0})['gf']
            needed_reg = unserved_needs_at_place.get(place, {'reg': 0})['reg']
            available_gf = sandwiches_on_trays_at_loc.get(place, {'gf': 0})['gf']
            available_reg = sandwiches_on_trays_at_loc.get(place, {'reg': 0})['reg']

            # Calculate GF deficit
            deficit_gf = max(0, needed_gf - available_gf)

            # Calculate Reg deficit, using surplus GF sandwiches
            surplus_gf = max(0, available_gf - needed_gf)
            deficit_reg = max(0, needed_reg - available_reg - surplus_gf)

            deficit_at_place[place] = {'gf': deficit_gf, 'reg': deficit_reg}

        # 4. Calculate Total Deficit and Kitchen Inventory
        total_deficit_gf = sum(d['gf'] for d in deficit_at_place.values())
        total_deficit_reg = sum(d['reg'] for d in deficit_at_place.values())
        total_deficit = total_deficit_gf + total_deficit_reg

        kitchen_sandwiches = {'gf': 0, 'reg': 0}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at_kitchen_sandwich':
                sandwich = parts[1]
                if sandwich in is_gf_sandwich_in_state:
                    kitchen_sandwiches['gf'] += 1
                else:
                    kitchen_sandwiches['reg'] += 1

        kitchen_ontray_at_kitchen = sandwiches_on_trays_at_loc.get('kitchen', {'gf': 0, 'reg': 0})
        kitchen_ontray_total = kitchen_ontray_at_kitchen['gf'] + kitchen_ontray_at_kitchen['reg']

        # 5. Estimate 'Make Sandwich' Actions
        # Sandwiches needed that are not already in the kitchen (either at_kitchen_sandwich or ontray in kitchen)
        available_in_kitchen_gf = kitchen_sandwiches['gf'] + kitchen_ontray_at_kitchen['gf']
        available_in_kitchen_reg = kitchen_sandwiches['reg'] + kitchen_ontray_at_kitchen['reg']

        make_gf = max(0, total_deficit_gf - available_in_kitchen_gf)
        make_reg = max(0, total_deficit_reg - available_in_kitchen_reg - max(0, available_in_kitchen_gf - total_deficit_gf))

        make_cost = make_gf + make_reg

        # 6. Estimate 'Put On Tray' Actions (in Kitchen)
        # Sandwiches that need to be put on trays in the kitchen to fulfill the deficit elsewhere.
        # This is the total deficit minus those already on trays in the kitchen.
        # These sandwiches must pass through the at_kitchen_sandwich state first, then put_on_tray.
        # The number of sandwiches that need to be put on trays in the kitchen is the number
        # that are either newly made or are already at_kitchen_sandwich, and are needed for the deficit.
        # Total sandwiches needed for deficit = total_deficit
        # Sandwiches already on trays in kitchen = kitchen_ontray_total
        # Sandwiches that need to be put on trays in kitchen = max(0, total_deficit - kitchen_ontray_total)
        put_on_tray_cost = max(0, total_deficit - kitchen_ontray_total)


        # 7. Estimate 'Move Tray' Actions
        # Number of distinct places (excluding kitchen) that have a deficit.
        places_needing_delivery = {
            place for place, deficit in deficit_at_place.items()
            if deficit['gf'] > 0 or deficit['reg'] > 0
        }
        move_tray_cost = len(places_needing_delivery)

        # 8. Estimate 'Serve Sandwich' Actions
        serve_cost = len(unserved_children)

        # Total heuristic value
        heuristic_value = make_cost + put_on_tray_cost + move_tray_cost + serve_cost

        return heuristic_value
