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
    # Ensure the number of parts matches the number of args unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children
    by counting the unsatisfied goal conditions and the items/tasks needed
    at different stages of the sandwich preparation and delivery pipeline.

    The stages are:
    1. Make sandwich (if not already existing)
    2. Put sandwich on tray (if not already on tray)
    3. Move tray to child's location (if tray not already there)
    4. Serve child

    The heuristic sums the estimated number of tasks needed at each stage,
    based on the total number of unserved children. It is non-admissible
    as it sums independent counts which might share resources or actions.

    # Heuristic Initialization
    - Identify all children who need to be served from the task goals.
    - Store static information about child allergies and waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are in the goal state but are not yet served
       in the current state. Let this count be `N_unserved`.
    2. The minimum number of `serve_sandwich` actions needed is `N_unserved`. Add `N_unserved` to the heuristic cost.
    3. Identify the unique locations where these unserved children are waiting.
    4. For each waiting location, check if there is *any* tray currently at that location.
    5. Count the number of waiting locations that do *not* have a tray. This is an estimate of the minimum number of `move_tray` actions needed to get trays to the required locations. Add this count to the heuristic cost.
    6. Count the total number of sandwiches that are currently on trays (`ontray`).
    7. The number of sandwiches that still need to be put on a tray is `max(0, N_unserved - N_ontray)`. Add this count to the heuristic cost (estimate for `put_on_tray` actions).
    8. Count the total number of sandwiches that currently exist (either `ontray` or `at_kitchen_sandwich`).
    9. The number of sandwiches that still need to be made is `max(0, N_unserved - N_existing)`. Add this count to the heuristic cost (estimate for `make_sandwich` actions).
    10. The total heuristic value is the sum of costs from steps 2, 5, 7, and 9.

    This heuristic simplifies resource constraints (like specific bread/content,
    notexist sandwich objects, or tray capacity/availability at kitchen)
    and gluten requirements beyond identifying unserved children, focusing
    on the number of items/tasks needed at each pipeline stage.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Identify all children that need to be served in the goal state.
        self.goal_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }

        # Map children to their allergy status.
        self.child_allergies = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergies[get_parts(fact)[1]] = True
            elif match(fact, "not_allergic_gluten", "*"):
                 self.child_allergies[get_parts(fact)[1]] = False

        # Map children to their waiting locations.
        self.child_waiting_location = {}
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_waiting_location[child] = place

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        h = 0  # Initialize heuristic value.

        # 1. Identify unserved children
        unserved_children = [
            child for child in self.goal_children
            if f"(served {child})" not in state
        ]
        N_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        # Add cost for the final 'serve' action for each unserved child.
        h += N_unserved

        # 3. Identify unique waiting locations for unserved children
        waiting_locations_needed = {
            self.child_waiting_location[child]
            for child in unserved_children
            if child in self.child_waiting_location # Should always be true in valid problems
        }

        # 4. & 5. Count locations needing a tray
        trays_at_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_type = get_parts(fact)[1].split('-')[0] # Simple type check based on naming convention
                if obj_type == 'tray':
                    tray, location = get_parts(fact)[1:]
                    trays_at_locations.setdefault(location, []).append(tray)

        N_locs_needing_tray = 0
        for location in waiting_locations_needed:
            if location not in trays_at_locations or not trays_at_locations[location]:
                 # Check if any tray is at this location
                 tray_found_at_loc = False
                 for fact in state:
                     if match(fact, "at", "*", location):
                         obj_type = get_parts(fact)[1].split('-')[0]
                         if obj_type == 'tray':
                             tray_found_at_loc = True
                             break
                 if not tray_found_at_loc:
                    N_locs_needing_tray += 1

        h += N_locs_needing_tray # Add cost for 'move_tray'

        # 6. Count sandwiches on trays
        N_ontray = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        # 7. Count sandwiches needing to be put on a tray
        # This is the number of needed sandwiches (N_unserved) minus those already on trays.
        N_needs_put_on_tray = max(0, N_unserved - N_ontray)
        h += N_needs_put_on_tray # Add cost for 'put_on_tray'

        # 8. Count sandwiches existing (on tray or in kitchen)
        N_kitchen_sandwich = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        N_existing = N_ontray + N_kitchen_sandwich

        # 9. Count sandwiches needing to be made
        # This is the number of needed sandwiches (N_unserved) minus those already existing.
        N_needs_make = max(0, N_unserved - N_existing)
        h += N_needs_make # Add cost for 'make_sandwich'

        return h

