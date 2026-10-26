from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the number of arguments in the pattern
    # This prevents matching "(at obj loc extra)" with pattern "at", "*", "*"
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of sandwiches that need to be made, the number of sandwiches
    at the kitchen that need to be put on trays, the number of locations with unserved
    children that need a tray moved there, and the number of children still needing
    to be served.

    # Assumptions
    - Ingredients (bread and content) are sufficient at the kitchen if needed to make sandwiches.
    - Trays are available if needed to put sandwiches on.
    - Any tray can be moved to any location.
    - The cost of each action type (make, put, move, serve) is 1.

    # Heuristic Initialization
    - Extract goal children (those who need to be served).
    - Extract static facts: which children are allergic and where each child is waiting.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are in the goal state (need to be served).
    2. From the current state, identify which of these goal children are *not* yet served.
    3. For each unserved child, determine if they need a gluten-free or regular sandwich based on static facts.
    4. Count the total number of unserved children. This contributes to the heuristic (representing the final 'serve' action for each).
    5. Count the number of gluten-free and regular sandwiches that have already been made (either at the kitchen or on a tray).
    6. Calculate how many *more* gluten-free and regular sandwiches are needed than are currently available. This contributes to the heuristic (representing the 'make_sandwich' actions).
    7. Count the number of sandwiches that are currently `at_kitchen_sandwich`. These need to be put on a tray. This contributes to the heuristic (representing the 'put_on_tray' actions).
    8. Identify all locations where unserved children are waiting.
    9. Identify all locations where trays are currently present.
    10. Count the number of locations from step 8 that are *not* in the set from step 9. These locations need a tray moved there. This contributes to the heuristic (representing the 'move_tray' actions).
    11. Sum up the counts from steps 4, 6, 7, and 10. This sum is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal children and static facts."""
        # Goal children are those mentioned in the (served ?c) goals
        self.goal_children = {get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")}
        self.static_facts = task.static

        # Map children to their waiting places from static facts
        self.child_waiting_place = {}
        for fact in self.static_facts:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_waiting_place[child] = place

        # Identify allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in self.static_facts
            if match(fact, "allergic_gluten", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their waiting places
        unserved_children_at_place = {}
        for child in self.goal_children:
            # Check if the child is served in the current state
            if '(served ' + child + ')' not in state:
                # Find the child's waiting place from the pre-calculated static map
                place = self.child_waiting_place.get(child)
                # Only consider children who are actually waiting somewhere (should always be true for goal children)
                if place:
                     unserved_children_at_place[child] = place

        # If all goal children are served, the heuristic is 0
        if not unserved_children_at_place:
            return 0

        h = 0

        # 4. Count unserved children (represents the final 'serve' action for each)
        # Each unserved child needs one 'serve' action.
        h += len(unserved_children_at_place)

        # 3. Determine children needing GF vs Regular sandwiches
        gf_needed_children = {c for c in unserved_children_at_place if c in self.allergic_children}
        reg_needed_children = set(unserved_children_at_place.keys()) - gf_needed_children

        # 5. Count available sandwiches (made, either at kitchen or on tray)
        # We need to know which available sandwiches are gluten-free.
        available_sandwiches = {s for s in state if '(at_kitchen_sandwich ' + s + ')' in state or any(match(fact, "ontray", s, "*") for fact in state)}
        available_gf_sandwiches = {s for s in available_sandwiches if '(no_gluten_sandwich ' + s + ')' in state}
        available_reg_sandwiches = available_sandwiches - available_gf_sandwiches

        # 6. Calculate sandwiches needed but not available (represents 'make_sandwich' actions)
        # We need to make enough GF sandwiches for all GF-allergic unserved children,
        # and enough regular sandwiches for all non-allergic unserved children.
        # We subtract the available ones of the correct type.
        needed_gf_to_make = max(0, len(gf_needed_children) - len(available_gf_sandwiches))
        needed_reg_to_make = max(0, len(reg_needed_children) - len(available_reg_sandwiches))
        h += needed_gf_to_make
        h += needed_reg_to_make

        # 7. Count sandwiches at kitchen (need to be put on tray - represents 'put_on_tray' actions)
        # Any sandwich currently at the kitchen needs a 'put_on_tray' action to become servable.
        sandwiches_at_kitchen = {s for s in state if '(at_kitchen_sandwich ' + s + ')' in state}
        h += len(sandwiches_at_kitchen)

        # 8. Identify locations with unserved children
        locations_with_unserved = set(unserved_children_at_place.values())

        # 9. Identify locations with trays
        # Find all 'at' facts where the first object is a tray.
        current_tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith('tray')}

        # 10. Count locations needing a tray (represents 'move_tray' actions)
        # A location needs a tray moved there if there are unserved children waiting there,
        # but no tray is currently at that location.
        locations_needing_tray = locations_with_unserved - current_tray_locations
        h += len(locations_needing_tray)

        # 11. Sum is already accumulated in h

        return h
