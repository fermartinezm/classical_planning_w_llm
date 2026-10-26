from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    children who are currently waiting and not yet served. It calculates the
    cost for each unserved child independently and sums these costs. The cost
    for a child is estimated based on the minimum actions needed to get a
    suitable sandwich onto a tray at their location and then serve it.

    # Assumptions
    - The goal is to serve all children specified in the task goals.
    - Each child needs exactly one sandwich.
    - Children allergic to gluten require a gluten-free sandwich. Non-allergic
      children can receive any sandwich.
    - Sandwiches can be in the kitchen, on a tray, or not yet made.
    - Trays can be in the kitchen or at various places where children are waiting.
    - Making a sandwich requires one bread and one content portion in the kitchen
      and an available sandwich name (notexist). Gluten-free sandwiches require
      gluten-free ingredients.
    - There are always enough trays available for the necessary operations
      (specifically, one can always be used for put_on_tray in the kitchen
      or moved to a child's location). This simplifies tray logistics.
    - The heuristic assumes solvability if resources (ingredients, sandwich names)
      are present in the initial state, even if they are not currently in the
      kitchen. However, the cost calculation is based on *currently available*
      resources in the kitchen or existing sandwiches/trays. If a child cannot
      be served with currently available/makeable resources, a large penalty is applied.

    # Heuristic Initialization
    - Extracts static information from the task: which children are allergic,
      where each child is waiting, which bread/content items are gluten-free.
    - Infers the set of all possible objects (children, bread, content, sandwiches,
      trays, places) from the initial state and static facts. This inference
      might be incomplete if objects only appear in goals or action definitions
      and not in the initial state or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are waiting but not yet served (goal served facts
       minus state served facts).
    2. For each unserved child:
       a. Determine if the child is allergic to gluten and find their waiting place.
       b. Calculate the minimum estimated cost to get a *suitable* sandwich
          onto a tray located at the child's waiting place. This is the 'min_ready_cost'.
          - Initialize `min_ready_cost` to infinity.
          - Check existing sandwiches on trays: For any suitable sandwich `S` on tray `T`
            (`(ontray S T)` in state), find the location `P_T` of tray `T` (`(at T P_T)` in state).
            If `P_T` is the child's place, `min_ready_cost` is at least 0.
            If `P_T` is elsewhere, `min_ready_cost` is at least 1 (for moving the tray).
          - Check existing sandwiches in the kitchen: For any suitable sandwich `S`
            (`(at_kitchen_sandwich S)` in state), the cost to get it ready at the
            child's place is 2 (1 for put_on_tray + 1 for move_tray). Update `min_ready_cost`.
          - Check if a new suitable sandwich can be made: Determine if there is
            an unused sandwich name (`(notexist S_new)` in state) and if suitable
            ingredients (bread and content) are available in the kitchen
            (`(at_kitchen_bread B)` and `(at_kitchen_content C_ing)` in state,
            checking gluten-free status if needed). If yes, the cost to get it
            ready at the child's place is 3 (1 for make + 1 for put_on_tray + 1 for move_tray).
            Update `min_ready_cost`.
       c. If `min_ready_cost` is still infinity after checking all options, it means
          the child cannot be served with currently available resources. Add a large
          penalty to the total heuristic.
       d. Otherwise, the estimated cost for this child is `min_ready_cost + 1` (for the serve action).
    3. Sum the estimated costs for all unserved children.
    4. If the set of unserved children is empty, the heuristic is 0 (goal state).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and inferring object lists.
        """
        self.goals = task.goals  # Goal conditions

        # --- Extract static information ---
        self.allergic_children = set()
        self.waiting_places = {} # child -> place
        self.gf_breads = set()
        self.gf_contents = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == "allergic_gluten":
                if len(parts) > 1: self.allergic_children.add(parts[1])
            elif predicate == "waiting":
                if len(parts) > 2: self.waiting_places[parts[1]] = parts[2]
            elif predicate == "no_gluten_bread":
                if len(parts) > 1: self.gf_breads.add(parts[1])
            elif predicate == "no_gluten_content":
                if len(parts) > 1: self.gf_contents.add(parts[1])

        # --- Infer all objects from initial state and static facts ---
        # This might be incomplete if objects only appear in goals or action definitions.
        self.all_children = set(self.waiting_places.keys()) # Children mentioned in waiting
        self.all_breads = set()
        self.all_contents = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        all_facts = set(task.initial_state) | set(task.static)
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]

             if predicate in ['at_kitchen_bread', 'no_gluten_bread']:
                 if args: self.all_breads.add(args[0])
             elif predicate in ['at_kitchen_content', 'no_gluten_content']:
                 if args: self.all_contents.add(args[0])
             elif predicate in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist']:
                 if args: self.all_sandwiches.add(args[0])
             elif predicate in ['ontray', 'at']:
                 if len(args) > 1:
                     if predicate == 'ontray': self.all_trays.add(args[1])
                     elif predicate == 'at': self.all_trays.add(args[0])
                     self.all_places.add(args[1]) # Location argument
             elif predicate in ['served']:
                 if args: self.all_children.add(args[0])

        # Ensure all children from goals are included, even if not in init/static
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served' and len(parts) > 1:
                 self.all_children.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Pre-process state for faster lookups
        served_children_in_state = set()
        sandwich_locations = {} # sandwich -> 'kitchen' or tray_name
        sandwich_on_tray = {} # sandwich -> tray_name
        tray_locations = {} # tray -> place
        gluten_free_sandwiches_in_state = set()
        kitchen_breads_in_state = set()
        kitchen_contents_in_state = set()
        notexist_sandwiches_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == "served" and args:
                 served_children_in_state.add(args[0])
            elif predicate == "at_kitchen_sandwich" and args:
                sandwich_locations[args[0]] = 'kitchen'
            elif predicate == "ontray" and len(args) > 1:
                sandwich_locations[args[0]] = args[1] # Store tray name as location
                sandwich_on_tray[args[0]] = args[1]
            elif predicate == "at" and len(args) > 1:
                 tray_locations[args[0]] = args[1]
            elif predicate == "no_gluten_sandwich" and args:
                gluten_free_sandwiches_in_state.add(args[0])
            elif predicate == "at_kitchen_bread" and args:
                kitchen_breads_in_state.add(args[0])
            elif predicate == "at_kitchen_content" and args:
                kitchen_contents_in_state.add(args[0])
            elif predicate == "notexist" and args:
                notexist_sandwiches_in_state.add(args[0])

        # Identify unserved children
        unserved_children = {c for c in self.all_children if c not in served_children_in_state}

        # If all children are served, goal reached, heuristic is 0
        if not unserved_children:
            return 0

        total_heuristic = 0
        large_penalty = 1000 # Penalty for a child that cannot be served

        # Pre-calculate ingredient availability for making new sandwiches
        has_notexist_sandwich_name = len(notexist_sandwiches_in_state) > 0
        has_bread_kitchen = len(kitchen_breads_in_state) > 0
        has_content_kitchen = len(kitchen_contents_in_state) > 0
        has_gf_bread_kitchen = any(b in self.gf_breads for b in kitchen_breads_in_state)
        has_gf_content_kitchen = any(c in self.gf_contents for c in kitchen_contents_in_state)


        for child in unserved_children:
            child_place = self.waiting_places.get(child) # Get waiting place from static info
            if child_place is None:
                 # Child is unserved but not waiting? Should not happen in valid problems.
                 # Treat as unservable from this state.
                 total_heuristic += large_penalty
                 continue

            is_allergic = child in self.allergic_children

            min_ready_cost = float('inf') # Min cost to get a suitable sandwich on a tray at child_place

            # Option 1: Use an existing suitable sandwich already on a tray
            # Iterate through sandwiches known to be on trays
            for s in sandwich_on_tray:
                # Check if sandwich S is suitable for the child
                is_suitable = (s in gluten_free_sandwiches_in_state) if is_allergic else True

                if is_suitable:
                    tray = sandwich_on_tray[s]
                    if tray in tray_locations:
                        tray_place = tray_locations[tray]
                        if tray_place == child_place:
                            min_ready_cost = min(min_ready_cost, 0) # Already on tray at place
                        else:
                            min_ready_cost = min(min_ready_cost, 1) # On tray elsewhere, needs 1 move_tray

            # Option 2: Use an existing suitable sandwich in the kitchen
            # Iterate through sandwiches known to be in the kitchen
            for s in sandwich_locations:
                 if sandwich_locations[s] == 'kitchen':
                    # Check if sandwich S is suitable for the child
                    is_suitable = (s in gluten_free_sandwiches_in_state) if is_allergic else True
                    if is_suitable:
                         # Cost: 1 (put_on_tray) + 1 (move_tray from kitchen to child_place) = 2
                         min_ready_cost = min(min_ready_cost, 2)


            # Option 3: Make a new suitable sandwich
            can_make_suitable = False
            if has_notexist_sandwich_name and has_bread_kitchen and has_content_kitchen:
                if is_allergic:
                    if has_gf_bread_kitchen and has_gf_content_kitchen:
                        can_make_suitable = True
                else:
                    can_make_suitable = True # Any ingredients work for non-allergic

            if can_make_suitable:
                 # Cost: 1 (make) + 1 (put_on_tray) + 1 (move_tray from kitchen to child_place) = 3
                 min_ready_cost = min(min_ready_cost, 3)


            # Add cost for this child
            if min_ready_cost == float('inf'):
                # Cannot serve this child with current resources/makeable sandwiches
                total_heuristic += large_penalty
            else:
                # Cost is min_ready_cost + 1 (for the serve action)
                total_heuristic += min_ready_cost + 1

        return total_heuristic
