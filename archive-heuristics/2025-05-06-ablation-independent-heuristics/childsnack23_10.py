from heuristics.heuristic_base import Heuristic
from task import Task

# Helper function to parse a fact string like '(predicate arg1 arg2)'
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and its arguments.

    Args:
        fact_string: The string representation of a PDDL fact (e.g., '(at obj1 place1)').

    Returns:
        A tuple containing the predicate name (string) and a list of arguments (strings).
    """
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for several key steps in the process
    of serving children:
    1. Serving the children who are ready to be served (counted by the number
       of unserved children).
    2. Making the sandwiches that are needed but not yet made.
    3. Putting the sandwiches that are at the kitchen (either newly made or
       already there) onto trays.
    4. Delivering sandwiches on trays to the locations where unserved children
       are waiting and need them (counted by the number of sandwiches that
       need to arrive at locations).

    Assumptions:
    - Each action (make, put, move, serve) has a cost of 1.
    - Tray capacity is sufficient or not the primary bottleneck.
    - Trays are initially at the kitchen or their initial location.
    - Ingredients (bread, content) are sufficient to make needed sandwiches
      (solvability is not checked by the heuristic being infinite).
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    The constructor processes the static facts from the task description to
    build data structures that allow quick lookup of:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - Where each child is waiting.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    - All children mentioned in the goal (those who need serving).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all unserved children by checking the goal facts against the
       current state facts. Count the total number of unserved children (`N_unserved`).
       This contributes `N_unserved` to the heuristic (minimum serve actions).
       If `N_unserved` is 0, the state is a goal state, and the heuristic is 0.
    2. Determine the total number of gluten-free (GF) and regular sandwiches
       required across all unserved children (`Total_GF_needed`, `Total_Reg_needed`).
    3. Count the number of GF and regular sandwiches that are currently
       `ontray` anywhere (`Available_GF_ontray_anywhere`, `Available_Reg_ontray_anywhere`).
    4. Count the number of GF and regular sandwiches that are currently
       `at_kitchen_sandwich` (`Available_GF_kitchen`, `Available_Reg_kitchen`).
    5. Calculate the number of new GF and regular sandwiches that *must* be made
       to satisfy the total demand, considering those already on trays or at kitchen.
       - `Available_GF_anywhere = Available_GF_ontray_anywhere + Available_GF_kitchen`
       - `Available_Reg_anywhere = Available_Reg_ontray_anywhere + Available_Reg_kitchen`
       - `GF_to_make = max(0, Total_GF_needed - Available_GF_anywhere)`
       - Calculate GF surplus anywhere: `gf_surplus_anywhere = max(0, Available_GF_anywhere - Total_GF_needed)`
       - `Reg_to_make = max(0, Total_Reg_needed - Available_Reg_anywhere - gf_surplus_anywhere)`
       - The cost for making sandwiches is `N_sandwiches_to_make = GF_to_make + Reg_to_make`.
    6. Calculate the number of sandwiches that are either `at_kitchen_sandwich`
       or need to be newly made (`GF_to_make_or_at_kitchen`, `Reg_to_make_or_at_kitchen`).
       These are the sandwiches that are needed but not currently on a tray anywhere.
       They must transition to being on a tray (likely via `put_on_tray` at the kitchen).
       - `GF_to_make_or_at_kitchen = max(0, Total_GF_needed - Available_GF_ontray_anywhere)`
       - `Reg_to_make_or_at_kitchen = max(0, Total_Reg_needed - Available_Reg_ontray_anywhere - max(0, Available_GF_ontray_anywhere - Total_GF_needed))`
       - The cost for putting these on trays is `N_sandwiches_from_kitchen_or_to_make = GF_to_make_or_at_kitchen + Reg_to_make_or_at_kitchen`.
    7. For each location `p` where unserved children are waiting:
       - Count GF and regular sandwiches needed at `p` (`gf_needed_at_p`, `reg_needed_at_p`).
       - Count GF and regular sandwiches available on trays at `p` (`gf_available_at_p`, `reg_available_at_p`).
       - Calculate the deficit of suitable sandwiches at `p`, accounting for GF serving regular needs:
         - `gf_still_needed_at_p = max(0, gf_needed_at_p - gf_available_at_p)`
         - `gf_surplus_at_p = max(0, gf_available_at_p - gf_needed_at_p)`
         - `reg_still_needed_at_p = max(0, reg_needed_at_p - reg_available_at_p - gf_surplus_at_p)`
       - The number of sandwiches that need to be brought to location `p` is `sandwiches_to_bring_to_p = gf_still_needed_at_p + reg_still_needed_at_p`.
    8. Sum `sandwiches_to_bring_to_p` over all locations `p` with unserved children. This gives `N_sandwiches_to_deliver_to_locations`. This contributes `N_sandwiches_to_deliver_to_locations` to the heuristic (proxy for move_tray actions needed to deliver sandwiches).
    9. The total heuristic value is the sum of costs from steps 1, 5, 6, and 8:
       `h = N_unserved + N_sandwiches_to_make + N_sandwiches_from_kitchen_or_to_make + N_sandwiches_to_deliver_to_locations`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Store static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {} # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        for fact_string in self.static:
            predicate, args = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                if len(args) == 1:
                    self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                if len(args) == 1:
                    self.not_allergic_children.add(args[0])
            elif predicate == 'waiting':
                if len(args) == 2: # Ensure correct number of arguments
                    self.child_waiting_place[args[0]] = args[1]
            elif predicate == 'no_gluten_bread':
                if len(args) == 1:
                    self.no_gluten_bread.add(args[0])
            elif predicate == 'no_gluten_content':
                 if len(args) == 1:
                    self.no_gluten_content.add(args[0])

        # Get all children from goals (they are the ones that need serving)
        self.all_children_in_goal = set()
        for goal_fact in self.goals:
             predicate, args = parse_fact(goal_fact)
             if predicate == 'served':
                 if len(args) == 1:
                    self.all_children_in_goal.add(args[0])


    def __call__(self, node):
        state = node.state

        # --- Step 1: Count unserved children ---
        served_children_in_state = set()
        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'served':
                if len(args) == 1:
                    served_children_in_state.add(args[0])

        unserved_children = self.all_children_in_goal - served_children_in_state
        N_unserved = len(unserved_children)

        # If no children are unserved, we are in a goal state
        if N_unserved == 0:
            return 0

        # --- Collect state information ---
        sandwiches_ontray = {} # sandwich -> tray
        sandwiches_at_kitchen = set()
        sandwich_is_gluten_free = set() # set of GF sandwich names
        object_location = {} # object -> place # Collect location for any object with 'at' predicate

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'ontray':
                if len(args) == 2:
                    sandwiches_ontray[args[0]] = args[1]
            elif predicate == 'at_kitchen_sandwich':
                if len(args) == 1:
                    sandwiches_at_kitchen.add(args[0])
            elif predicate == 'no_gluten_sandwich':
                if len(args) == 1:
                    sandwich_is_gluten_free.add(args[0])
            elif predicate == 'at':
                 if len(args) == 2:
                    object_location[args[0]] = args[1]

        # Filter object_location to get only tray locations
        # A tray is an object that appears as the second argument in an 'ontray' fact
        all_trays_in_state = set(sandwiches_ontray.values())
        tray_location = {obj: loc for obj, loc in object_location.items() if obj in all_trays_in_state}


        # Group unserved children by location
        children_at_location = {} # place -> set of children
        for child in unserved_children:
            place = self.child_waiting_place.get(child) # Get place from static info
            if place: # Child must have a waiting place
                if place not in children_at_location:
                    children_at_location[place] = set()
                children_at_location[place].add(child)

        # --- Step 2: Determine total sandwiches needed ---
        Total_GF_needed = len([c for c in unserved_children if c in self.allergic_children])
        Total_Reg_needed = len([c for c in unserved_children if c in self.not_allergic_children])

        # --- Step 3 & 4: Count available sandwiches anywhere & Calculate sandwiches to make ---
        Available_GF_ontray_anywhere = len([s for s in sandwiches_ontray if s in sandwich_is_gluten_free])
        Available_Reg_ontray_anywhere = len([s for s in sandwiches_ontray if s not in sandwich_is_gluten_free])
        Available_GF_kitchen = len([s for s in sandwiches_at_kitchen if s in sandwich_is_gluten_free])
        Available_Reg_kitchen = len([s for s in sandwiches_at_kitchen if s not in sandwich_is_gluten_free])

        Available_GF_anywhere = Available_GF_ontray_anywhere + Available_GF_kitchen
        Available_Reg_anywhere = Available_Reg_ontray_anywhere + Available_Reg_kitchen

        # GF needed must be GF
        GF_to_make = max(0, Total_GF_needed - Available_GF_anywhere)

        # Reg needed can be Reg or GF surplus
        gf_surplus_anywhere = max(0, Available_GF_anywhere - Total_GF_needed)
        Reg_to_make = max(0, Total_Reg_needed - Available_Reg_anywhere - gf_surplus_anywhere)

        N_sandwiches_to_make = GF_to_make + Reg_to_make

        # --- Step 6: Calculate sandwiches from kitchen or to make (need put_on_tray) ---
        # These are sandwiches not currently on a tray anywhere, that are needed.
        # They must transition to being on a tray.
        GF_to_make_or_at_kitchen = max(0, Total_GF_needed - Available_GF_ontray_anywhere)
        Reg_to_make_or_at_kitchen = max(0, Total_Reg_needed - Available_Reg_ontray_anywhere - max(0, Available_GF_ontray_anywhere - Total_GF_needed))

        N_sandwiches_from_kitchen_or_to_make = GF_to_make_or_at_kitchen + Reg_to_make_or_at_kitchen

        # --- Step 7 & 8: Calculate sandwiches to deliver to locations ---
        N_sandwiches_to_deliver_to_locations = 0

        # Group sandwiches on trays by location
        sandwiches_ontray_at_location = {} # place -> set of sandwiches on trays at this place
        for s, t in sandwiches_ontray.items():
            p = tray_location.get(t)
            if p: # Tray must have a location
                if p not in sandwiches_ontray_at_location:
                    sandwiches_ontray_at_location[p] = set()
                sandwiches_ontray_at_location[p].add(s)

        for place, children_at_p in children_at_location.items():
            # Count needed sandwiches at this location
            gf_needed_at_p = len([c for c in children_at_p if c in self.allergic_children])
            reg_needed_at_p = len([c for c in children_at_p if c in self.not_allergic_children])

            # Count available suitable sandwiches on trays at this location
            available_sandwiches_at_p = sandwiches_ontray_at_location.get(place, set())
            gf_available_at_p = len([s for s in available_sandwiches_at_p if s in sandwich_is_gluten_free])
            reg_available_at_p = len([s for s in available_sandwiches_at_p if s not in sandwich_is_gluten_free])

            # Calculate deficit at this location
            # First satisfy GF needs with GF sandwiches
            gf_still_needed_at_p = max(0, gf_needed_at_p - gf_available_at_p)
            gf_surplus_at_p = max(0, gf_available_at_p - gf_needed_at_p)

            # Then satisfy regular needs with regular sandwiches and GF surplus
            reg_still_needed_at_p = max(0, reg_needed_at_p - reg_available_at_p - gf_surplus_at_p)

            sandwiches_to_bring_to_p = gf_still_needed_at_p + reg_still_needed_at_p
            N_sandwiches_to_deliver_to_locations += sandwiches_to_bring_to_p


        # --- Step 9: Calculate total heuristic ---
        h_value = N_unserved + N_sandwiches_to_make + N_sandwiches_from_kitchen_or_to_make + N_sandwiches_to_deliver_to_locations

        return h_value
