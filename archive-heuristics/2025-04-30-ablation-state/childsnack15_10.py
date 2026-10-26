from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# The match function is provided in example heuristics but not strictly used
# in the final logic of this heuristic. Keeping it for consistency with examples.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It sums up the estimated costs for four main categories of actions:
    1. Serving each unserved child.
    2. Making any necessary sandwiches that don't currently exist.
    3. Putting sandwiches that are currently in the kitchen onto trays.
    4. Moving trays to locations where unserved children are waiting.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A suitable sandwich must be made (if not already), moved to the kitchen (if not already), put on a tray, and the tray moved to the child's location before serving.
    - The heuristic counts the *number of required actions* for these steps, largely ignoring strict resource limits (like the exact number of available ingredients, sandwich objects, or trays) beyond what's needed to count deficit/surplus of made sandwiches.
    - A single 'make_sandwich' action creates one sandwich which appears `at_kitchen_sandwich`.
    - A single 'put_on_tray' action moves one sandwich from `at_kitchen_sandwich` to `ontray`.
    - A single 'move_tray' action moves one tray between locations.
    - A single 'serve_sandwich' action serves one child.
    - A single tray move to a location can potentially serve multiple children waiting at that location. The heuristic counts the number of *locations* needing a tray, not the number of children at those locations.
    - Sandwiches needed by non-allergic children can be satisfied by either regular or gluten-free sandwiches, prioritizing surplus gluten-free ones if available after meeting allergic children's needs.

    # Heuristic Initialization
    - Extracts static information from the task: which children are allergic to gluten and the waiting location for each child.
    - Identifies the set of all children from the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Unserved Children:** Determine the set of children who are not yet marked as `served`. If this set is empty, the heuristic value is 0.
    2. **Initialize Heuristic:** Start with a heuristic value `h = 0`.
    3. **Serving Cost:** Add the number of unserved children to `h`. This represents the final `serve_sandwich` action required for each child.
    4. **Sandwich Preparation Cost (Making and Putting on Tray):**
       - Count the number of gluten-free and regular sandwiches required by the unserved children, considering allergy constraints and allowing non-allergic children to use surplus gluten-free sandwiches.
       - Count the number of gluten-free and regular sandwiches that have already been made (either `at_kitchen_sandwich` or `ontray`).
       - Calculate the deficit of needed sandwiches for both types, determining how many new gluten-free (`to_make_gf`) and regular (`to_make_reg`) sandwiches must be made.
       - Add `(to_make_gf + to_make_reg) * 2` to `h`. This accounts for the `make_sandwich` action and the subsequent `put_on_tray` action for each newly made sandwich.
       - Count the number of sandwiches that are currently `at_kitchen_sandwich` (that were not just accounted for in the 'to_make' step, i.e., they were already there at the start of the state). Add this count to `h`. This accounts for the `put_on_tray` action needed for these existing kitchen sandwiches.
    5. **Tray Movement Cost:**
       - Identify the set of unique locations where unserved children are waiting.
       - Identify the set of unique locations where trays are currently present.
       - Count the number of locations from the first set that are *not* in the second set. Add this count to `h`. This represents the estimated number of `move_tray` actions needed to get trays to all required locations.
    6. **Total Heuristic:** The final value of `h` is the estimated total number of actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if get_parts(fact)[0] == 'allergic_gluten'}
        self.waiting_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if get_parts(fact)[0] == 'waiting'}

        # Identify all children from the goal state (assuming all goals are served facts)
        self.all_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_children = self.all_children - served_children

        if not unserved_children:
            return 0 # Goal state reached

        h = 0

        # 3. Serving Cost
        # Each unserved child needs a final serve action.
        h += len(unserved_children)

        # Identify existing sandwiches and their types
        existing_sandwiches = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] in ['at_kitchen_sandwich', 'ontray']}
        no_gluten_sandwiches_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'no_gluten_sandwich'}

        avail_gf_made = {s for s in existing_sandwiches if s in no_gluten_sandwiches_in_state}
        avail_reg_made = existing_sandwiches - avail_gf_made # Sandwiches that exist but are not marked gluten-free

        # Count suitable sandwiches needed by unserved children
        gf_needed = sum(1 for c in unserved_children if c in self.allergic_children)
        reg_needed = sum(1 for c in unserved_children if c not in self.allergic_children)

        # 4. Sandwich Preparation Cost (Making and Putting on Tray)
        # Calculate how many new sandwiches need to be made.
        # Prioritize using available GF for allergic, then use surplus GF for non-allergic, then make regular for remaining non-allergic.
        to_make_gf = max(0, gf_needed - len(avail_gf_made))
        needed_for_reg = max(0, reg_needed - len(avail_reg_made))
        can_use_extra_gf = max(0, len(avail_gf_made) - gf_needed) # GF sandwiches not strictly needed by allergic kids
        to_make_reg = max(0, needed_for_reg - can_use_extra_gf) # Regular sandwiches needed after using surplus GF ones

        # Cost to make AND put on tray for new sandwiches
        # Each new sandwich requires 1 make action and 1 put_on_tray action.
        h += (to_make_gf + to_make_reg) * 2

        # Cost to put on tray for existing kitchen sandwiches
        # These are sandwiches that were already at the kitchen at the start of the state.
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'at_kitchen_sandwich'}
        h += len(kitchen_sandwiches) # Each put_on_tray action costs 1

        # 5. Tray Movement Cost
        # Identify locations of unserved children
        # Note: waiting_locations is populated from static facts, so it includes all children's waiting spots.
        # We only care about the spots of *unserved* children.
        waiting_places = {self.waiting_locations[c] for c in unserved_children if c in self.waiting_locations} # Ensure child is in static waiting facts

        # Identify locations of trays
        tray_locations = {get_parts(fact)[2] for fact in state if get_parts(fact)[0] == 'at' and get_parts(fact)[1].startswith('tray')}

        # Count unique waiting locations that do not currently have a tray
        places_needing_tray = waiting_places - tray_locations
        h += len(places_needing_tray) # Each move_tray action to a new location costs 1

        # 6. Total Heuristic
        return h
