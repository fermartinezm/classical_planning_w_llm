from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(predicate)' with no arguments
    if len(fact) <= 2:
        return [fact[1:-1]]
    return fact[1:-1].split()

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
    This heuristic estimates the number of actions required to serve each unserved child.
    It calculates the minimum number of steps needed for each child independently,
    considering the current state of suitable sandwiches (on tray at location,
    on tray elsewhere, in kitchen, or needing to be made). The total heuristic
    value is the sum of these minimum costs for all unserved children.

    # Assumptions
    - Each child needs one suitable sandwich to be served.
    - The process for serving a child involves making a sandwich (if needed),
      putting it on a tray, moving the tray to the child's location, and serving.
    - The heuristic assumes that trays and necessary ingredients/sandwich slots
      are available when needed for the calculation stages, without modeling
      resource contention or exact resource availability beyond a simple check
      for the 'make' stage.
    - Action costs are uniform (each action costs 1).

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - For each child, their allergy status (allergic or not) and their waiting place.
    - The set of gluten-free bread and content portions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the cost for each child that is
    not yet served:

    1.  **Identify Unserved Children:** Find all children for whom the predicate
        `(served ?c)` is not true in the current state.

    2.  **For Each Unserved Child:** Determine the minimum number of actions
        required to serve them based on the current state. This minimum is found
        by checking the following stages in increasing order of estimated cost:

        *   **Stage 1 (Cost 1: Serve):** Check if a suitable sandwich (gluten-free
            for allergic children, any for non-allergic) is already on a tray
            that is located at the child's waiting place. If yes, the cost for
            this child is 1 (the serve action).

        *   **Stage 2 (Cost 2: Move Tray + Serve):** If Stage 1 is not met, check
            if a suitable sandwich is on a tray that is located at a *different*
            place than the child's waiting place. If yes, the cost for this child
            is 2 (1 for moving the tray + 1 for serving).

        *   **Stage 3 (Cost 3: Put on Tray + Move Tray + Serve):** If Stages 1 and 2
            are not met, check if a suitable sandwich is currently in the kitchen
            (`at_kitchen_sandwich`). If yes, the cost for this child is 3
            (1 for putting on a tray + 1 for moving the tray + 1 for serving).
            This assumes a tray is available in the kitchen.

        *   **Stage 4 (Cost 4: Make + Put on Tray + Move Tray + Serve):** If Stages
            1, 2, and 3 are not met, check if a suitable sandwich can be made from
            available ingredients and sandwich slots in the kitchen. This requires
            a `notexist` sandwich slot, bread in the kitchen, and content in the
            kitchen. For allergic children, gluten-free bread and content are
            specifically required. If the necessary ingredients and a slot exist,
            the cost for this child is 4 (1 for making the sandwich + 1 for putting
            it on a tray + 1 for moving the tray + 1 for serving). This assumes a
            tray is available in the kitchen.

        *   **Unreachable (Cost 1000):** If none of the above stages can be met
            (e.g., no suitable sandwiches anywhere, and cannot make one due to
            missing ingredients or slots), the child might be unservable from
            this state. A large cost (e.g., 1000) is assigned to penalize such states.

    3.  **Sum Costs:** The total heuristic value for the state is the sum of the
        minimum costs calculated for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Child allergy status and waiting places.
        - Gluten-free bread and content portions.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        self.child_info = {}  # {child_name: {'allergic': bool, 'place': place_name}}
        self.gf_bread = set()
        self.gf_content = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = False
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['place'] = place
            elif parts[0] == 'no_gluten_bread':
                self.gf_bread.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gf_content.add(parts[1])

        # Ensure all children mentioned in static facts are in child_info
        # (Handles cases where a child might only have allergy info but not waiting info initially, though waiting is required for goal)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'served':
                 child = parts[1]
                 if child not in self.child_info:
                      # This child wasn't in static allergy/waiting facts,
                      # but is in the goal. We can't determine allergy/place
                      # from static, which is a problem. Assuming static
                      # contains complete info for all relevant children.
                      # For robustness, could try to infer from initial state
                      # or problem objects, but sticking to static as per example.
                      # If a child is in goal but not static, we can't serve them
                      # based on allergy/place, so they'd get a high cost.
                      self.child_info[child] = {'allergic': False, 'place': 'unknown'} # Default or handle error


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Extract relevant state information
        sandwiches_on_trays = {}  # {sandwich: tray}
        tray_locations = {}       # {tray: place}
        sandwiches_in_kitchen = set()
        notexist_sandwiches = set()
        bread_in_kitchen = set()
        content_in_kitchen = set()
        gf_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                sandwiches_on_trays[parts[1]] = parts[2]
            elif parts[0] == 'at':
                 # Check if it's a tray location fact
                 # Need to know tray objects. Can't get from static/state easily.
                 # Assume any 'at ?t ?p' where ?t is not kitchen is a tray location.
                 # Or, check if ?t is a key in sandwiches_on_trays (meaning it's a tray holding a sandwich).
                 # A more robust way would be to parse object types from the problem file,
                 # but let's rely on the structure of facts.
                 # A fact like '(at tray1 table1)' clearly indicates tray1 is at table1.
                 # Let's assume the first argument of 'at' that isn't 'kitchen' is a tray.
                 # Or, simply collect all 'at ?obj ?place' facts and filter later if needed.
                 # Given the domain, 'at' is only used for trays and the kitchen constant.
                 # So, '(at ?t ?p)' means tray ?t is at place ?p.
                 if len(parts) == 3: # Ensure it's a binary predicate
                     tray, place = parts[1], parts[2]
                     # The kitchen constant is also a 'place', but 'at kitchen' facts
                     # are not tray locations in the sense of serving children.
                     # Tray locations are places where children wait.
                     # However, trays *can* be at the kitchen.
                     # Let's just store all 'at' facts for trays.
                     tray_locations[tray] = place
            elif parts[0] == 'at_kitchen_sandwich':
                sandwiches_in_kitchen.add(parts[1])
            elif parts[0] == 'notexist':
                notexist_sandwiches.add(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                bread_in_kitchen.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                content_in_kitchen.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                gf_sandwiches.add(parts[1])


        total_heuristic = 0

        for child, info in self.child_info.items():
            if child not in served_children:
                is_allergic = info.get('allergic', False) # Default to False if info is incomplete
                child_place = info.get('place')

                if child_place is None:
                    # Child exists but no waiting place found in static.
                    # This child cannot be served according to the domain rules.
                    # Assign a very high cost.
                    total_heuristic += 1000
                    continue # Move to the next child

                child_cost = 1000 # Large initial cost

                # Check Stage 1 (Cost 1: Serve)
                # Is there a suitable sandwich on a tray at the child's place?
                for s, t in sandwiches_on_trays.items():
                    is_gf_s = s in gf_sandwiches
                    is_suitable_s = (is_allergic and is_gf_s) or (not is_allergic)
                    if is_suitable_s and tray_locations.get(t) == child_place:
                        child_cost = min(child_cost, 1)
                        break # Found the cheapest way for this child

                if child_cost > 1:
                    # Check Stage 2 (Cost 2: Move Tray + Serve)
                    # Is there a suitable sandwich on a tray elsewhere?
                    for s, t in sandwiches_on_trays.items():
                        is_gf_s = s in gf_sandwiches
                        is_suitable_s = (is_allergic and is_gf_s) or (not is_allergic)
                        if is_suitable_s and t in tray_locations and tray_locations[t] != child_place:
                             child_cost = min(child_cost, 2)
                             break # Found the cheapest way among move+serve options

                if child_cost > 2:
                    # Check Stage 3 (Cost 3: Put on Tray + Move Tray + Serve)
                    # Is there a suitable sandwich in the kitchen?
                    for s in sandwiches_in_kitchen:
                        is_gf_s = s in gf_sandwiches
                        is_suitable_s = (is_allergic and is_gf_s) or (not is_allergic)
                        if is_suitable_s:
                            child_cost = min(child_cost, 3)
                            break # Found the cheapest way among put+move+serve options

                if child_cost > 3:
                    # Check Stage 4 (Cost 4: Make + Put on Tray + Move Tray + Serve)
                    # Can a suitable sandwich be made?
                    can_make_suitable = False
                    has_notexist = len(notexist_sandwiches) > 0
                    has_bread = len(bread_in_kitchen) > 0
                    has_content = len(content_in_kitchen) > 0

                    if is_allergic:
                        has_gf_bread_in_kitchen = any(b in self.gf_bread for b in bread_in_kitchen)
                        has_gf_content_in_kitchen = any(c in self.gf_content for c in content_in_kitchen)
                        if has_notexist and has_gf_bread_in_kitchen and has_gf_content_in_kitchen:
                             can_make_suitable = True
                    else: # Not allergic
                        # Non-allergic can eat regular or GF.
                        # Can make regular?
                        if has_notexist and has_bread and has_content:
                             can_make_suitable = True
                        # Can make GF? (already checked ingredients above)
                        # has_gf_bread_in_kitchen = any(b in self.gf_bread for b in bread_in_kitchen)
                        # has_gf_content_in_kitchen = any(c in self.gf_content for c in content_in_kitchen)
                        # if has_notexist and has_gf_bread_in_kitchen and has_gf_content_in_kitchen:
                        #      can_make_suitable = True # This is covered if GF ingredients are also counted in has_bread/has_content

                    if can_make_suitable:
                        child_cost = min(child_cost, 4)

                total_heuristic += child_cost

        return total_heuristic

