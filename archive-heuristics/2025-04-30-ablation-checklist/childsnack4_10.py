from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    waiting children. It breaks down the process into necessary steps:
    making sandwiches, putting them on trays, moving trays to children's
    locations, and finally serving the children.

    # Assumptions
    - Sufficient bread, content, and 'notexist' sandwich objects are available
      initially to make all necessary sandwiches.
    - Sufficient trays are available initially.
    - The kitchen is the only place where sandwiches can be made and put on trays.

    # Heuristic Initialization
    - Extract child allergy information from static facts.
    - Identify all children that need to be served from the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums up the estimated costs for several key stages required
    to satisfy all unserved children:

    1.  **Cost to Serve:** Each unserved child requires a final 'serve' action.
        This is a base cost equal to the number of unserved children.
    2.  **Cost for Tray Movement:** Trays must be present at the locations
        where children are waiting. Estimate the number of tray movements
        needed by counting the number of unique locations where unserved
        children are waiting but no tray is currently present. Each such
        location requires at least one tray move.
    3.  **Cost to Make Sandwiches:** Sufficient sandwiches of the correct
        type (regular or gluten-free) must exist. Estimate the number of
        sandwiches that still need to be made by comparing the total number
        of sandwiches needed by unserved children (by type) against the
        number of suitable sandwiches already available (either in the kitchen
        or on trays).
    4.  **Cost to Put Sandwiches on Trays:** Sandwiches must be on trays
        before they can be moved or served. Estimate the number of sandwiches
        that need to be put on trays by comparing the total number of
        sandwiches needed by unserved children against the number of
        sandwiches already on trays. This counts the transition from
        'at_kitchen_sandwich' or 'notexist' to 'ontray'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child allergy information
        and the set of children that need to be served.
        """
        self.goals = task.goals
        self.static = task.static

        # Map child name to their allergy status ('gf' for allergic, 'reg' otherwise)
        self.child_allergy = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.child_allergy[parts[1]] = 'gf'
            elif parts[0] == 'not_allergic_gluten':
                 self.child_allergy[parts[1]] = 'reg'

        # Identify all children that are part of the goal (i.e., need to be served)
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.children_to_serve.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a state where all children in self.children_to_serve
        have the 'served' predicate true.
        """
        state = node.state

        # If all children are served, the heuristic is 0.
        if all(match(fact, "served", child) for child in self.children_to_serve):
             return 0

        # --- State Analysis ---
        unserved_children = set()
        children_waiting_at = {} # Map child -> place
        trays_at_place = {}      # Map place -> set of trays
        sandwiches_on_tray = {}  # Map tray -> set of sandwiches
        kitchen_sandwiches = set() # Set of sandwiches at_kitchen_sandwich
        gf_sandwiches = set()    # Set of sandwiches that are no_gluten_sandwich

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served' and parts[1] in self.children_to_serve:
                # This child is served, ignore for heuristic calculation
                pass
            elif parts[0] == 'waiting' and parts[1] in self.children_to_serve:
                 # This child is waiting and needs serving
                 unserved_children.add(parts[1])
                 children_waiting_at[parts[1]] = parts[2]
            elif parts[0] == 'at' and parts[1].startswith('tray'):
                 place = parts[2]
                 tray = parts[1]
                 trays_at_place.setdefault(place, set()).add(tray)
            elif parts[0] == 'ontray':
                 sandwich = parts[1]
                 tray = parts[2]
                 sandwiches_on_tray.setdefault(tray, set()).add(sandwich)
            elif parts[0] == 'at_kitchen_sandwich':
                 kitchen_sandwiches.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                 gf_sandwiches.add(parts[1])

        num_unserved = len(unserved_children)

        # If no children are unserved, heuristic is 0 (should be caught above, but double check)
        if num_unserved == 0:
            return 0

        # --- Heuristic Calculation ---
        h = 0

        # 1. Cost to Serve: Each unserved child needs a 'serve' action.
        h += num_unserved

        # 2. Cost for Tray Movement: Count locations needing a tray.
        locations_with_waiting_children = {children_waiting_at[c] for c in unserved_children}
        locations_with_trays = set(trays_at_place.keys())
        locations_needing_tray_move = locations_with_waiting_children - locations_with_trays
        h += len(locations_needing_tray_move)

        # 3. Cost to Make Sandwiches: Count needed sandwiches vs available ones.
        needed_reg = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'reg')
        needed_gf = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'gf')

        # Count available sandwiches anywhere (kitchen or on trays) by type
        avail_reg = 0
        avail_gf = 0
        all_made_sandwiches = kitchen_sandwiches.union(*sandwiches_on_tray.values()) # Union of all sets of sandwiches on trays
        for s in all_made_sandwiches:
            if s in gf_sandwiches:
                avail_gf += 1
            else:
                avail_reg += 1 # Assume non-GF is regular

        sandwiches_to_make_reg = max(0, needed_reg - avail_reg)
        sandwiches_to_make_gf = max(0, needed_gf - avail_gf)
        h += sandwiches_to_make_reg + sandwiches_to_make_gf

        # 4. Cost to Put Sandwiches on Trays: Count sandwiches needed on trays.
        # Total sandwiches needed on trays is the number of unserved children.
        # Count sandwiches already on trays.
        num_on_trays = sum(len(s_set) for s_set in sandwiches_on_tray.values())
        num_to_put_on_trays = max(0, num_unserved - num_on_trays)
        h += num_to_put_on_trays

        return h

