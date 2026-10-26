from heuristics.heuristic_base import Heuristic
# No other specific imports like fnmatch are needed for this implementation.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle empty/malformed facts defensively
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for invalid facts, or handle as an error depending on expected input robustness
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve each
    unserved child based on the current availability and location of suitable
    sandwiches. The total heuristic is the sum of these individual child costs.

    # Assumptions
    - Each unserved child requires exactly one suitable sandwich.
    - The cost to serve a child depends primarily on the state of the most
      readily available suitable sandwich (ready at the child's location,
      on a tray elsewhere, in the kitchen, or needing to be made).
    - Action costs are assumed to be 1 for make, put, move, and serve actions.
    - Resource constraints (like the exact number of bread/content portions,
      or trays) are simplified: we only check if *any* suitable ingredients
      exist in the kitchen and if *any* `notexist` sandwich object is available
      when estimating the cost of making a sandwich.
    - Children wait at fixed locations.

    # Heuristic Initialization
    - Extracts the set of children that are required to be served from the goal state.
    - Extracts allergy information (allergic or not allergic to gluten) for all
      children from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the total cost by summing the estimated minimum cost
    for each child that has not yet been served according to the goal state.

    For each child `c` that needs to be served:
    1.  Check if the child `c` is already `served` in the current state. If yes,
        the cost for this child is 0, and we proceed to the next child.
    2.  If the child is not served, find their current waiting location `p` from
        the state fact `(waiting c p)`. Determine if the child is allergic to gluten
        using the pre-calculated static information. If the child is not found
        waiting anywhere, the state is considered unsolvable.
    3.  Initialize the minimum estimated cost for this child (`child_cost`) to
        infinity, representing the worst case (potentially unsolvable or requires
        maximum steps).
    4.  **Check for Cost 1 (Serve):** Iterate through all sandwiches `s` currently
        `ontray` `t`. Check if `s` is suitable for child `c` (considering allergies
        and `no_gluten_sandwich` status). If suitable, check if the tray `t` is
        currently `at` the child's location `p`. If such a sandwich and tray are
        found, the minimum cost for this child is 1 (the `serve` action). Update
        `child_cost` to 1 and stop checking lower costs for this child.
    5.  **Check for Cost 2 (Move + Serve):** If cost 1 was not met, iterate through
        all sandwiches `s` currently `ontray` `t`. Check if `s` is suitable for
        child `c`. If suitable, check if the tray `t` is currently `at` *any*
        location (which implies it's not at the child's location, otherwise cost 1
        would have been met). If such a sandwich and tray are found, the minimum
        cost is 2 (a `move_tray` action followed by a `serve` action). Update
        `child_cost` to 2 and stop checking lower costs.
    6.  **Check for Cost 3 (Put + Move + Serve):** If costs 1 and 2 were not met,
        iterate through all sandwiches `s` currently `at_kitchen_sandwich`. Check
        if `s` is suitable for child `c`. If suitable, the minimum cost is 3 (a
        `put_on_tray` action, followed by a `move_tray` action, followed by a
        `serve` action, assuming a tray is available at the kitchen). Update
        `child_cost` to 3 and stop checking lower costs.
    7.  **Check for Cost 4 (Make + Put + Move + Serve):** If not cost 1, 2, or 3,
        check if it is possible to `make` a suitable sandwich. This requires
        checking if a `notexist` sandwich object is available and if suitable
        ingredients (bread and content, considering gluten-free requirements) are
        `at_kitchen`. If a suitable sandwich can be made, the minimum cost is 4
        (a `make_sandwich` action, followed by `put_on_tray`, `move_tray`, and
        `serve` actions). Update `child_cost` to 4.
    8.  **Check for Unsolvable:** If after checking all possibilities (Costs 1-4),
        `child_cost` is still infinity, it means the child cannot be served from
        this state (e.g., no suitable sandwich exists anywhere, and none can be
        made). In this case, the state is likely unsolvable, and the heuristic
        returns `float('inf')` immediately for the entire state.
    9.  Add the determined `child_cost` to the `total_heuristic`.
    10. After iterating through all children that need serving, return the
        `total_heuristic`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract children who need serving from goals
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served' and len(parts) == 2:
                self.children_to_serve.add(parts[1])

        # Extract allergy info from static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.allergic_children.add(parts[1])
            elif parts and parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                self.not_allergic_children.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_heuristic = 0

        # Convert frozenset to set for faster 'in' checks
        state_facts = set(state)

        # Pre-process state for faster lookups
        tray_locations = {}
        sandwiches_on_trays = {} # {sandwich: tray}
        sandwiches_in_kitchen = set()
        nogluten_sandwiches = set()
        bread_in_kitchen = set()
        content_in_kitchen = set()
        can_make_sandwich_object = False
        child_waiting_locations = {} # {child: location}


        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'at' and len(parts) == 3: # (at ?t ?p)
                 tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'ontray' and len(parts) == 3: # (ontray ?s ?t)
                 sandwiches_on_trays[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2: # (at_kitchen_sandwich ?s)
                 sandwiches_in_kitchen.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich' and len(parts) == 2: # (no_gluten_sandwich ?s)
                 nogluten_sandwiches.add(parts[1])
            elif parts[0] == 'at_kitchen_bread' and len(parts) == 2: # (at_kitchen_bread ?b)
                bread_in_kitchen.add(parts[1])
            elif parts[0] == 'at_kitchen_content' and len(parts) == 2: # (at_kitchen_content ?c)
                content_in_kitchen.add(parts[1])
            elif parts[0] == 'notexist' and len(parts) == 2: # (notexist ?s)
                can_make_sandwich_object = True # Found at least one
            elif parts[0] == 'waiting' and len(parts) == 3: # (waiting ?c ?p)
                 child_waiting_locations[parts[1]] = parts[2]


        # Get gluten-free ingredients (need to check static facts)
        # Filter the bread/content in kitchen based on static gluten-free facts
        nogluten_bread_in_kitchen = {b for b in bread_in_kitchen if '(no_gluten_bread ' + b + ')' in self.static}
        nogluten_content_in_kitchen = {c for c in content_in_kitchen if '(no_gluten_content ' + c + ')' in self.static}


        # Iterate through children who need serving
        for child in self.children_to_serve:
            if '(served ' + child + ')' in state_facts:
                continue # Child is already served, cost is 0 for this child

            # Find child's waiting location
            child_location = child_waiting_locations.get(child)

            if child_location is None:
                 # Child is supposed to be served but is not waiting anywhere?
                 # This state might be invalid or unsolvable.
                 return float('inf')

            is_allergic = child in self.allergic_children

            child_cost = float('inf') # Initialize with highest possible cost

            # Helper to check if a sandwich is suitable for the current child
            def is_suitable_sandwich(s):
                 if is_allergic:
                     return s in nogluten_sandwiches
                 else: # Not allergic
                     return True # Any sandwich is suitable

            # Check Cost 1: Suitable sandwich on tray at child's location?
            found_cost_1 = False
            for s, t in sandwiches_on_trays.items():
                if is_suitable_sandwich(s):
                    if t in tray_locations and tray_locations[t] == child_location:
                         child_cost = 1
                         found_cost_1 = True
                         break

            if found_cost_1:
                total_heuristic += child_cost
                continue # Move to next child

            # Check Cost 2: Suitable sandwich on tray elsewhere?
            found_cost_2 = False
            for s, t in sandwiches_on_trays.items():
                if is_suitable_sandwich(s):
                    # Check if tray t is anywhere (we already know it's not at child_location if cost 1 wasn't met)
                    if t in tray_locations: # Tray is somewhere
                         child_cost = 2
                         found_cost_2 = True
                         break

            if found_cost_2:
                total_heuristic += child_cost
                continue # Move to next child

            # Check Cost 3: Suitable sandwich in kitchen?
            found_cost_3 = False
            for s in sandwiches_in_kitchen:
                if is_suitable_sandwich(s):
                    child_cost = 3
                    found_cost_3 = True
                    break

            if found_cost_3:
                total_heuristic += child_cost
                continue # Move to next child

            # Check Cost 4: Can make a suitable sandwich?
            can_make_suitable = False
            if can_make_sandwich_object: # Check if a sandwich object is available to be made
                if is_allergic:
                    if nogluten_bread_in_kitchen and nogluten_content_in_kitchen:
                        can_make_suitable = True
                else: # Not allergic
                    if bread_in_kitchen and content_in_kitchen:
                        can_make_suitable = True

            if can_make_suitable:
                child_cost = 4
            else:
                # Cannot make a suitable sandwich (no object or no ingredients)
                # This child cannot be served.
                child_cost = float('inf') # Unsolvable for this child

            # If child_cost is still infinity, the problem is likely unsolvable from this state.
            # The heuristic should return infinity.
            if child_cost == float('inf'):
                return float('inf') # Return infinity immediately if any child is unservable

            total_heuristic += child_cost

        return total_heuristic
