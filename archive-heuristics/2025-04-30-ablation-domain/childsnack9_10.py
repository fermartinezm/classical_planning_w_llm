from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or not fact:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It calculates the minimum cost to serve each unserved child independently and sums these costs.

    # Assumptions
    - The goal is to achieve the `(served ?child)` predicate for all children specified in the task goals.
    - Children are either allergic or not allergic to gluten, specified in static facts.
    - Sandwiches are made from bread and content.
    - Gluten-free sandwiches require gluten-free bread and gluten-free content.
    - Allergic children must be served gluten-free sandwiches. Non-allergic children can be served any sandwich.
    - Sandwiches and trays can be moved between the kitchen and tables.
    - The heuristic calculates the minimum steps for each unserved child independently, ignoring potential resource conflicts or sharing (e.g., multiple children needing the same tray or ingredients).
    - Solvable problems have sufficient initial resources (bread, content, notexist sandwich objects, trays) to make all necessary sandwiches. A large penalty is applied if a child cannot be served with currently available resources in a state.

    # Heuristic Initialization
    - Extracts the set of children who are goals (need to be served).
    - Extracts static information: which children are allergic to gluten, which bread portions are gluten-free, and which content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Identify all children that are specified in the task goals.
    2.  Initialize the total heuristic cost to 0.
    3.  For each child identified in step 1:
        a.  Check if the child is already served in the current state. If yes, this child contributes 0 to the heuristic; proceed to the next child.
        b.  If the child is not served, find the table where the child is waiting.
        c.  Determine if the child is allergic to gluten based on the static facts.
        d.  Determine the type of sandwich required: gluten-free if the child is allergic, any sandwich otherwise.
        e.  Find the minimum number of actions required to get a suitable sandwich to this child and serve them, considering the current location and state of suitable sandwiches:
            i.  **Cost 1 (Serve):** Check if a suitable sandwich is already on a tray at the child's waiting table. If yes, the minimum cost for this child is 1 (the `serve` action).
            ii. **Cost 2 (Move Tray + Serve):** If not achievable with cost 1, check if a suitable sandwich is on a tray located in the kitchen. If yes, the minimum cost is 2 (1 for `move-tray` + 1 for `serve`).
            iii. **Cost 3 (Put on Tray + Move Tray + Serve):** If not achievable with cost 2, check if a suitable sandwich has been made and is currently in the kitchen (not yet on a tray). If yes, and if there is at least one tray available in the kitchen, the minimum cost is 3 (1 for `put-on-tray` + 1 for `move-tray` + 1 for `serve`).
            iv. **Cost 4 (Make + Put on Tray + Move Tray + Serve):** If not achievable with cost 3, check if a suitable sandwich can be made. This requires an available `notexist` sandwich object, suitable bread and content portions available in the kitchen (gluten-free if the child is allergic), and at least one tray available in the kitchen. If all these resources exist, the minimum cost is 4 (1 for `make-sandwich` + 1 for `put-on-tray` + 1 for `move-tray` + 1 for `serve`).
            v.  **Penalty:** If none of the above paths are possible with the currently available resources (e.g., no suitable ingredients left, no `notexist` objects, no trays in the kitchen when needed), assign a large penalty cost (e.g., 1000) to indicate a potentially difficult or impossible state for this child.
        f.  Add the calculated minimum cost for this child to the total heuristic cost.
    4.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal children and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal children from goal facts like (served child1)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract static information
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.ng_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.ng_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

    def _is_suitable_sandwich(self, sandwich, child, state_ng_sandwiches):
        """Check if a sandwich is suitable for a child based on allergy status."""
        child_is_allergic = child in self.allergic_children
        sandwich_is_ng = sandwich in state_ng_sandwiches

        if child_is_allergic:
            return sandwich_is_ng
        else:
            return True # Any sandwich is suitable for a non-allergic child

    def _can_make_suitable_sandwich(self, child, state_kitchen_bread, state_kitchen_content):
        """Check if ingredients exist in the kitchen to make a suitable sandwich for a child."""
        child_is_allergic = child in self.allergic_children

        if child_is_allergic:
            # Need GF bread and GF content in the kitchen
            has_ng_bread_in_kitchen = any(b in self.ng_bread for b in state_kitchen_bread)
            has_ng_content_in_kitchen = any(c in self.ng_content for c in state_kitchen_content)
            return has_ng_bread_in_kitchen and has_ng_content_in_kitchen
        else:
            # Need any bread and any content in the kitchen
            has_any_bread_in_kitchen = len(state_kitchen_bread) > 0
            has_any_content_in_kitchen = len(state_kitchen_content) > 0
            return has_any_bread_in_kitchen and has_any_content_in_kitchen

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Pre-process state facts for efficient lookup
        served_children_state = {get_parts(f)[1] for f in state if match(f, "served", "*")}
        waiting_children_state = {get_parts(f)[1]: get_parts(f)[2] for f in state if match(f, "waiting", "*", "*")}
        sandwiches_on_trays_state = {get_parts(f)[1]: get_parts(f)[2] for f in state if match(f, "ontray", "*", "*")} # sandwich -> tray
        # Filter for tray locations specifically
        tray_locations_state = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('tray'):
                 tray_locations_state[parts[1]] = parts[2]

        kitchen_made_sandwiches_state = {get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "*")}
        notexist_sandwiches_state = {get_parts(f)[1] for f in state if match(f, "notexist", "*")}
        kitchen_bread_state = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        kitchen_content_state = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}
        ng_sandwiches_state = {get_parts(f)[1] for f in state if match(f, "no_gluten_sandwich", "*")} # GF status of *made* sandwiches

        total_heuristic = 0
        penalty_cost = 1000 # Large cost if a child cannot be served with available resources

        # Check if any tray is in the kitchen (needed for costs 3 and 4)
        tray_in_kitchen_available = any(loc == 'kitchen' for loc in tray_locations_state.values())

        for child in self.goal_children:
            if child in served_children_state:
                continue # Child is already served

            child_table = waiting_children_state.get(child)
            if child_table is None:
                 # Child is a goal but not waiting? This state is likely problematic.
                 total_heuristic += penalty_cost
                 continue

            min_cost_for_child = float('inf')

            # Path 1: Sandwich on tray at child's table (Cost 1: Serve)
            # Iterate through sandwiches on trays
            for s, t in sandwiches_on_trays_state.items():
                # Check if the tray is at the child's table
                if tray_locations_state.get(t) == child_table:
                    # Check if the sandwich is suitable for the child
                    if self._is_suitable_sandwich(s, child, ng_sandwiches_state):
                        min_cost_for_child = 1
                        break # Found cheapest path for this child

            # Path 2: Sandwich on tray in kitchen (Cost 2: Move Tray + Serve)
            if min_cost_for_child > 1:
                # Iterate through sandwiches on trays
                for s, t in sandwiches_on_trays_state.items():
                    # Check if the tray is in the kitchen
                    if tray_locations_state.get(t) == 'kitchen':
                         # Check if the sandwich is suitable for the child
                        if self._is_suitable_sandwich(s, child, ng_sandwiches_state):
                            min_cost_for_child = 2
                            break # Found next cheapest path

            # Path 3: Made sandwich in kitchen (Cost 3: Put on Tray + Move Tray + Serve)
            if min_cost_for_child > 2 and tray_in_kitchen_available:
                 # Iterate through made sandwiches in the kitchen
                 for s in kitchen_made_sandwiches_state:
                     # Check if the sandwich is suitable for the child
                     if self._is_suitable_sandwich(s, child, ng_sandwiches_state):
                         min_cost_for_child = 3
                         break # Found next cheapest path

            # Path 4: Make new sandwich (Cost 4: Make + Put on Tray + Move Tray + Serve)
            if min_cost_for_child > 3 and tray_in_kitchen_available:
                # Check if there's any notexist sandwich object available
                has_notexist_sandwich_object = len(notexist_sandwiches_state) > 0
                # Check if suitable ingredients are available in the kitchen
                can_make_suitable = self._can_make_suitable_sandwich(child, kitchen_bread_state, kitchen_content_state)

                if has_notexist_sandwich_object and can_make_suitable:
                    min_cost_for_child = 4
                    # No break here, as we just checked possibility, not a specific sandwich object

            # Add cost for this child
            if min_cost_for_child == float('inf'):
                 # Cannot find any way to serve this child with current resources
                 total_heuristic += penalty_cost
            else:
                 total_heuristic += min_cost_for_child

        return total_heuristic
