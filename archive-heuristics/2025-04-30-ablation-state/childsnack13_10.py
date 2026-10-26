from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It counts the necessary actions in four stages: making sandwiches,
    putting sandwiches on trays, moving trays to children's locations, and serving
    the children. It sums the counts for each stage, considering dependencies
    (e.g., sandwiches must be made before being put on trays). The heuristic is
    designed to be efficiently computable and informative for a greedy best-first
    search, aiming to minimize expanded nodes. It is not guaranteed to be admissible.

    # Assumptions
    - Each unserved child requires one sandwich.
    - Allergic children require gluten-free sandwiches; non-allergic children can
      accept any sandwich.
    - Enough bread, content, and sandwich objects exist in the initial problem
      to make all needed sandwiches if they are not already made.
    - Tray capacity is sufficient to hold all sandwiches needed at a location.
    - A tray needs to move to a location with unserved children only if no tray
      is currently at that location.

    # Heuristic Initialization
    - Extracts static information from the task: which children are allergic/not
      allergic and which bread/content items are gluten-free. This information
      is stored for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated by summing estimated costs for four main
    types of actions required to transition items/agents through the necessary
    stages to reach the goal of serving all children:

    1.  **Cost to Serve:** This is the most direct cost. Each child who is currently
        not served will eventually require one 'serve_sandwich' action. The cost
        is simply the total number of unserved children.

    2.  **Cost to Make Sandwiches:** Before a sandwich can be served, it must be made.
        We count how many sandwiches of each type (gluten-free and regular) are
        required to satisfy the needs of the unserved children, subtracting those
        sandwiches that are already made (either in the kitchen or on a tray).
        - Determine the number of gluten-free sandwiches strictly needed (equals
          the number of unserved allergic children).
        - Determine the number of regular sandwiches needed (equals the number
          of unserved non-allergic children).
        - Count the total number of made gluten-free and regular sandwiches
          currently present in the state.
        - Calculate the deficit of needed GF sandwiches vs. available made GF.
          This is the number of GF sandwiches that *must* be made.
        - Calculate the deficit of needed regular sandwiches vs. available made
          regular, noting that any excess made GF sandwiches can satisfy regular
          needs. This is the number of regular sandwiches that *can* be made (as
          regular or using excess GF resources).
        - The total 'make' cost is the sum of GF and regular sandwiches to make.

    3.  **Cost to Put on Trays:** After a sandwich is made, it appears in the kitchen
        and must be put on a tray before it can be moved to a child's location.
        This cost counts the number of sandwiches that need to transition from
        being in the kitchen to being on a tray. This includes sandwiches currently
        in the kitchen *plus* all sandwiches that were calculated in step 2 as
        needing to be made (since they will end up in the kitchen).

    4.  **Cost to Move Trays:** Once sandwiches are on trays, the trays must be moved
        from the kitchen (or their current location) to the locations where children
        are waiting. This cost estimates the minimum number of tray movements needed.
        It counts the number of distinct locations where unserved children are
        waiting, but where no tray is currently present. Each such location is
        assumed to require at least one 'move_tray' action to bring a tray there.

    The total heuristic value is the sum of the estimated costs from steps 1, 2, 3, and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts from the task.

        @param task: The planning task object containing initial state, goals, etc.
        """
        # Store static information about children's allergies and ingredient types
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gluten_free_bread = set() # Not directly used in heuristic calculation but good practice
        self.gluten_free_content = set() # Not directly used in heuristic calculation but good practice

        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif predicate == 'no_gluten_bread':
                self.gluten_free_bread.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.gluten_free_content.add(parts[1])

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: An integer estimate of the remaining cost to reach a goal state.
        """
        state = node.state

        # --- Parse the current state to extract relevant facts ---
        served_children = set()
        waiting_children_loc = {} # Maps child name to their waiting place
        sandwich_status = {} # Maps sandwich name to its status ('kitchen', 'ontray', 'notexist')
        sandwich_is_gf = set() # Set of sandwich names that are gluten-free
        tray_loc = {} # Maps tray name to its current place
        # ontray_map = {} # Maps sandwich name to the tray it's on (not strictly needed for this heuristic)

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'waiting':
                waiting_children_loc[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich':
                sandwich_status[parts[1]] = 'kitchen'
            elif predicate == 'ontray':
                sandwich_status[parts[1]] = 'ontray'
                # ontray_map[parts[1]] = parts[2]
            elif predicate == 'at':
                # In childsnacks domain, 'at' predicate is only used for trays
                tray_loc[parts[1]] = parts[2]
            elif predicate == 'notexist':
                sandwich_status[parts[1]] = 'notexist'
            elif predicate == 'no_gluten_sandwich':
                sandwich_is_gf.add(parts[1])

        # --- Calculate counts based on the parsed state ---

        # Identify unserved children and categorize by allergy
        unserved_children = {c for c in waiting_children_loc if c not in served_children}
        n_children_unserved = len(unserved_children)

        # If all children are served, the heuristic is 0 (goal state)
        if n_children_unserved == 0:
            return 0

        n_allergic_unserved = len({c for c in unserved_children if c in self.allergic_children})
        n_regular_unserved = len({c for c in unserved_children if c in self.not_allergic_children})

        # Count existing made sandwiches by location and type
        n_sandw_kitchen_gf = len([s for s, status in sandwich_status.items() if status == 'kitchen' and s in sandwich_is_gf])
        n_sandw_kitchen_reg = len([s for s, status in sandwich_status.items() if status == 'kitchen' and s not in sandwich_is_gf])
        n_sandw_ontray_gf = len([s for s, status in sandwich_status.items() if status == 'ontray' and s in sandwich_is_gf])
        n_sandw_ontray_reg = len([s for s, status in sandwich_status.items() if status == 'ontray' and s not in sandwich_is_gf])
        # n_sandw_notexist = len([s for s, status in sandwich_status.items() if status == 'notexist']) # Count of available sandwich objects

        # Identify locations with trays and locations with unserved children
        locations_with_trays = set(tray_loc.values())
        locations_with_unserved_children = {waiting_children_loc[c] for c in unserved_children}

        # --- Calculate heuristic components based on counts ---

        # 1. Cost to Serve: Each unserved child needs one final serve action.
        cost_serve = n_children_unserved

        # 2. Cost to Make Sandwiches: Estimate how many sandwiches still need to be created.
        needed_gf = n_allergic_unserved
        needed_reg = n_regular_unserved

        made_gf = n_sandw_kitchen_gf + n_sandw_ontray_gf
        made_reg = n_sandw_kitchen_reg + n_sandw_ontray_reg

        # Calculate how many GF sandwiches must be made
        to_make_gf = max(0, needed_gf - made_gf)

        # Calculate how many regular sandwiches must be made. Regular children can
        # use any sandwich. We use excess made GF sandwiches first.
        excess_made_gf = max(0, made_gf - needed_gf)
        to_make_reg = max(0, needed_reg - made_reg - excess_made_gf)

        cost_make = to_make_gf + to_make_reg

        # 3. Cost to Put on Trays: Estimate how many sandwiches need to be moved
        #    from the kitchen onto a tray. This includes sandwiches currently in
        #    the kitchen and those that will be made.
        in_kitchen = n_sandw_kitchen_gf + n_sandw_kitchen_reg
        cost_put = in_kitchen + cost_make

        # 4. Cost to Move Trays: Estimate how many locations with unserved children
        #    do not currently have a tray. Each such location likely needs a tray move.
        locations_needing_tray = locations_with_unserved_children - locations_with_trays
        cost_move = len(locations_needing_tray)

        # Total heuristic is the sum of estimated costs for each stage
        total_cost = cost_serve + cost_make + cost_put + cost_move

        return total_cost

