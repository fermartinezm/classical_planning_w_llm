from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Return an empty list for non-fact strings or malformed facts
         return []

    # Remove parentheses and split into individual elements.
    # Handle potential empty fact string after stripping
    content = fact[1:-1].strip()
    if not content:
        return []
    return content.split()


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
    This heuristic estimates the minimum number of actions required to serve each unserved child independently. The total heuristic is the sum of these minimum costs. It considers the current location of appropriate sandwiches (on tray at child's place, on tray elsewhere, in kitchen) and the cost of making a new sandwich if none exist, taking into account tray availability.

    # Assumptions
    - Each unserved child requires exactly one appropriate sandwich (gluten-free if allergic, any otherwise).
    - The cost to serve a child is the minimum cost among all available appropriate sandwiches.
    - Resource conflicts (e.g., multiple children needing the same sandwich or tray) are ignored (relaxation).
    - Sufficient ingredients (bread, content) are assumed to be available to make a sandwich if at least one of the required type is present in the kitchen.
    - A tray can always be moved to the kitchen or a child's place if at least one tray exists anywhere.
    - The cost of each action is 1.

    # Heuristic Initialization
    - Extracts static information from the task: which children are allergic, which ingredients are gluten-free, and where each child is waiting. Identifies all children.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who have not yet been served by checking the `(served ?c)` facts in the current state.
    2. If there are no unserved children, the goal is reached, and the heuristic value is 0.
    3. Extract dynamic information from the current state:
        - The current location of each tray (`(at ?t ?p)`).
        - Which sandwiches are on which trays (`(ontray ?s ?t)`).
        - Which sandwiches are currently in the kitchen (`(at_kitchen_sandwich ?s)`).
        - Which sandwiches are gluten-free (`(no_gluten_sandwich ?s)`).
        - Which bread and content portions are in the kitchen (`(at_kitchen_bread ?b)`, `(at_kitchen_content ?c)`).
    4. Determine the availability of ingredients in the kitchen (gluten-free and regular bread/content) and trays (in the kitchen or anywhere).
    5. Initialize the total heuristic cost to 0. Define a large cost (`BLOCKED_COST`) to represent seemingly unsolvable subproblems for a child.
    6. For each unserved child `c`:
        a. Determine if the child is allergic to gluten and the place `p` where they are waiting.
        b. Initialize the minimum cost to serve this child (`min_cost_child`) to infinity.
        c. Iterate through all sandwiches currently on trays:
            - If a sandwich `s` is appropriate for child `c` (gluten-free if needed):
                - Find the tray `t` it's on and the current location `p'` of that tray.
                - If `p'` is the child's waiting place `p`, the cost to serve is 1 (just the serve action). Update `min_cost_child = min(min_cost_child, 1)`.
                - If `p'` is different from `p`, the cost is 2 (move tray + serve). Update `min_cost_child = min(min_cost_child, 2)`.
        d. Iterate through all sandwiches currently in the kitchen:
            - If a sandwich `s` is appropriate for child `c`:
                - If there is a tray available in the kitchen, the cost is 3 (put on tray + move tray + serve). Update `min_cost_child = min(min_cost_child, 3)`.
                - If no tray is in the kitchen but at least one tray exists elsewhere, the cost is 4 (move tray to kitchen + put on tray + move tray + serve). Update `min_cost_child = min(min_cost_child, 4)`.
                - If no tray exists anywhere, this path is blocked for this sandwich.
        e. If `min_cost_child` is still infinity (meaning no appropriate sandwich was found on a tray or in the kitchen):
            - Check if it's possible to make a new appropriate sandwich based on ingredient availability in the kitchen.
            - If a new appropriate sandwich can be made:
                - If there is a tray available in the kitchen, the cost is 4 (make + put on tray + move tray + serve). Update `min_cost_child = min(min_cost_child, 4)`.
                - If no tray is in the kitchen but at least one tray exists elsewhere, the cost is 5 (make + move tray to kitchen + put on tray + move tray + serve). Update `min_cost_child = min(min_cost_child, 5)`.
                - If no tray exists anywhere, this path is blocked.
            - If a new appropriate sandwich cannot be made (ingredients missing), this path is blocked.
        f. If after checking all options, `min_cost_child` is still infinity (or >= `BLOCKED_COST`), it indicates a likely unsolvable state for this child. Return `BLOCKED_COST` immediately as the total heuristic.
        g. Add `min_cost_child` to the `total_heuristic`.
    7. Return the `total_heuristic`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        @param task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions (e.g., served children)

        # Extract static facts
        self.allergic_children = {get_parts(f)[1] for f in task.static if match(f, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(f)[1] for f in task.static if match(f, "not_allergic_gluten", "*")}
        self.all_children = self.allergic_children | self.not_allergic_children # Set of all children objects

        self.gf_bread_static = {get_parts(f)[1] for f in task.static if match(f, "no_gluten_bread", "*")}
        self.gf_content_static = {get_parts(f)[1] for f in task.static if match(f, "no_gluten_content", "*")}

        # Map child to their waiting place
        self.waiting_places = {get_parts(f)[1]: get_parts(f)[2] for f in task.static if match(f, "waiting", "*", "*")}

        # Note: We don't strictly need to store all object names like bread, content, etc.
        # as we can identify them by iterating through relevant predicates in the state.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.

        @param node: The search node containing the current state.
        @return: The estimated heuristic cost.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify unserved children
        served_children = {get_parts(f)[1] for f in state if match(f, "served", "*")}
        unserved_children = self.all_children - served_children

        # 2. If no unserved children, goal is reached
        if not unserved_children:
            return 0

        # 3. Extract dynamic state information
        # Map tray object name to its current place object name
        # Assuming 'at' predicate is only for trays based on domain analysis
        tray_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('tray'):
                 tray_locations[parts[1]] = parts[2]

        # Map sandwich object name to the tray object name it's on
        sandwiches_on_trays = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'ontray':
                 sandwiches_on_trays[parts[1]] = parts[2]

        # Set of sandwich object names currently in the kitchen
        sandwiches_in_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "*")}
        # Set of gluten-free sandwich object names
        gf_sandwiches_in_state = {get_parts(f)[1] for f in state if match(f, "no_gluten_sandwich", "*")}
        # Set of bread object names currently in the kitchen
        bread_in_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        # Set of content object names currently in the kitchen
        content_in_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}

        # 4. Determine availability of resources
        gf_bread_available = any(b in self.gf_bread_static for b in bread_in_kitchen)
        gf_content_available = any(c in self.gf_content_static for c in content_in_kitchen)
        reg_bread_available = len(bread_in_kitchen) > 0 # Any bread in kitchen
        reg_content_available = len(content_in_kitchen) > 0 # Any content in kitchen

        tray_at_kitchen = 'kitchen' in tray_locations.values()
        any_tray_exists = len(tray_locations) > 0

        # 5. Initialize total heuristic and blocked cost
        total_heuristic = 0
        BLOCKED_COST = 1000 # Large number indicating a likely dead end

        # 6. Compute minimum cost for each unserved child
        for child in unserved_children:
            waiting_place = self.waiting_places.get(child)
            if waiting_place is None:
                 # Child is waiting at an unknown place? Should not happen in valid PDDL.
                 # Treat as blocked.
                 return BLOCKED_COST

            is_allergic = child in self.allergic_children

            min_cost_child = float('inf')

            # a. Check sandwiches on trays
            for sandwich, tray in sandwiches_on_trays.items():
                # Check if sandwich is appropriate
                is_appropriate = (not is_allergic) or (sandwich in gf_sandwiches_in_state)
                if is_appropriate:
                    tray_place = tray_locations.get(tray)
                    if tray_place == waiting_place:
                        # Sandwich on tray at child's place: 1 action (serve)
                        min_cost_child = min(min_cost_child, 1)
                    elif tray_place is not None:
                        # Sandwich on tray elsewhere: 2 actions (move tray, serve)
                        min_cost_child = min(min_cost_child, 2)
                    # If tray_place is None, the tray is not located anywhere, which is an invalid state. Ignore this sandwich path.


            # b. Check sandwiches in kitchen
            for sandwich in sandwiches_in_kitchen:
                 # Check if sandwich is appropriate
                is_appropriate = (not is_allergic) or (sandwich in gf_sandwiches_in_state)
                if is_appropriate:
                    if tray_at_kitchen:
                        # Sandwich in kitchen, tray in kitchen: 3 actions (put, move, serve)
                        min_cost_child = min(min_cost_child, 3)
                    elif any_tray_exists:
                        # Sandwich in kitchen, tray elsewhere: 4 actions (move tray to kitchen, put, move, serve)
                        min_cost_child = min(min_cost_child, 4)
                    else:
                        # Sandwich in kitchen, no tray exists: Blocked path
                        min_cost_child = min(min_cost_child, BLOCKED_COST)


            # c. Check if new sandwich can be made if no existing one is suitable/reachable
            if min_cost_child == float('inf'):
                can_make_gf = gf_bread_available and gf_content_available
                can_make_reg = reg_bread_available and reg_content_available
                can_make_appropriate = can_make_gf if is_allergic else can_make_reg

                if can_make_appropriate:
                    if tray_at_kitchen:
                        # Can make sandwich, tray in kitchen: 4 actions (make, put, move, serve)
                        min_cost_child = min(min_cost_child, 4)
                    elif any_tray_exists:
                        # Can make sandwich, tray elsewhere: 5 actions (make, move tray to kitchen, put, move, serve)
                        min_cost_child = min(min_cost_child, 5)
                    else:
                        # Can make sandwich, no tray exists: Blocked path
                        min_cost_child = min(min_cost_child, BLOCKED_COST)
                else:
                    # Cannot make appropriate sandwich (ingredients missing): Blocked path
                    min_cost_child = min(min_cost_child, BLOCKED_COST)

            # If min_cost_child is still blocked, return BLOCKED_COST immediately
            if min_cost_child >= BLOCKED_COST:
                return BLOCKED_COST

            total_heuristic += min_cost_child

        # 7. Return total heuristic
        return total_heuristic
