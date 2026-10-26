from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    Estimates the number of actions required to serve all waiting children.
    It counts the necessary steps: making sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children.

    # Assumptions
    - Assumes sufficient bread, content, and 'notexist' sandwich instances are available to make needed sandwiches.
    - Assumes sufficient trays are available at the kitchen for 'put_on_tray' actions.
    - Assumes a tray can carry all sandwiches needed for children at a single location.
    - Assumes trays start at the kitchen or can easily reach it to pick up sandwiches.
    - The cost of moving a tray is counted once per destination location (excluding kitchen) that needs service and doesn't currently have a tray.

    # Heuristic Initialization
    - Identifies all children who need to be served (from goal facts).
    - Stores the initial waiting location and allergy status for each child (from static facts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of children who are in the goal state (served) but are not yet served in the current state. If this set is empty, the heuristic is 0.
    2. Count the number of gluten-free and regular sandwiches required based on the allergy status of the unserved children.
    3. Count the number of gluten-free and regular sandwiches currently available (either at the kitchen or already on trays).
    4. Estimate the number of 'make_sandwich' actions needed: This is the total number of required sandwiches minus the total number of available sandwiches, considering gluten-free requirements first.
    5. Estimate the number of 'put_on_tray' actions needed: This is the total number of required sandwiches minus the number of sandwiches already on trays. This counts the number of sandwiches that need to transition from being off-tray to on-tray.
    6. Estimate the number of 'move_tray' actions needed: Identify all distinct locations (excluding the kitchen) where unserved children are waiting (based on initial waiting facts). For each such location, if no tray is currently present at that location in the current state, increment the heuristic by 1 (representing the cost to move a tray there).
    7. Estimate the number of 'serve_sandwich' actions needed: This is simply the total number of unserved children, as each requires one serve action.
    8. The total heuristic value is the sum of the costs estimated in steps 4, 5, 6, and 7.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, initial waiting
        locations, and allergy status from static facts.
        """
        self.children_to_serve = set()
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "served" and len(args) == 1:
                self.children_to_serve.add(args[0])

        self.child_is_allergic = {}
        self.child_initial_location = {}

        # Static facts contain initial state info that doesn't change
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty/malformed facts

            if parts[0] == "allergic_gluten" and len(parts) == 2:
                self.child_is_allergic[parts[1]] = True
            elif parts[0] == "not_allergic_gluten" and len(parts) == 2:
                self.child_is_allergic[parts[1]] = False
            elif parts[0] == "waiting" and len(parts) == 3:
                 # Initial waiting locations are static
                 self.child_initial_location[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Step 1: Identify unserved children
        unserved_children = set()
        for child in self.children_to_serve:
            if f"(served {child})" not in state:
                unserved_children.add(child)

        if not unserved_children:
            return 0

        h = 0

        # Step 2: Count required sandwiches
        required_gf = sum(1 for c in unserved_children if self.child_is_allergic.get(c, False))
        required_any = sum(1 for c in unserved_children if not self.child_is_allergic.get(c, True)) # Default to not allergic if info missing

        # Step 3: Count available sandwiches
        available_gf_kitchen = 0
        available_reg_kitchen = 0
        available_gf_ontray = 0
        available_reg_ontray = 0

        # First, identify which sandwiches are gluten-free
        sandwich_is_gf = set()
        for fact in state:
             if match(fact, "no_gluten_sandwich", "*"):
                 sandwich_is_gf.add(get_parts(fact)[1])

        # Then, count based on location and GF status
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                if s in sandwich_is_gf:
                    available_gf_kitchen += 1
                else:
                    available_reg_kitchen += 1
            elif match(fact, "ontray", "*", "*"):
                s = get_parts(fact)[1]
                if s in sandwich_is_gf:
                    available_gf_ontray += 1
                else:
                    available_reg_ontray += 1

        # Step 4: Estimate sandwich making cost
        supply_gf = available_gf_kitchen + available_gf_ontray
        supply_reg = available_reg_kitchen + available_reg_ontray

        sandwiches_to_make_gf = max(0, required_gf - supply_gf)
        # Sandwiches needed for 'any' can be regular or surplus GF
        sandwiches_to_make_reg = max(0, required_any - supply_reg - max(0, supply_gf - required_gf))

        h += sandwiches_to_make_gf + sandwiches_to_make_reg

        # Step 5: Estimate put-on-tray cost
        total_needed = required_gf + required_any
        total_ontray = available_gf_ontray + available_reg_ontray
        sandwiches_to_put_on_tray = max(0, total_needed - total_ontray)

        # This counts the *number of sandwiches* that need the put_on_tray transition.
        # It assumes enough sandwiches are available at the kitchen (either initially or made).
        h += sandwiches_to_put_on_tray

        # Step 6: Estimate tray movement cost
        # Identify locations (excluding kitchen) where unserved children are waiting
        locations_with_unserved = set(self.child_initial_location.get(c) for c in unserved_children if self.child_initial_location.get(c) is not None)
        locations_needing_tray_trip = {p for p in locations_with_unserved if p != 'kitchen'}

        # Find current locations of all trays
        current_tray_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*") and match(fact, "*", "tray", None):
                 current_tray_locations.add(get_parts(fact)[2])

        # Count locations needing a tray trip that don't currently have one
        for loc in locations_needing_tray_trip:
            if loc not in current_tray_locations:
                h += 1

        # Step 7: Estimate serving cost
        h += len(unserved_children)

        return h
