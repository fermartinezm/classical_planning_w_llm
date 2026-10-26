from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running within the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the minimum number of 'make_sandwich', 'put_on_tray',
    'move_tray', and 'serve_sandwich' actions needed based on the current state
    of children, sandwiches, ingredients, and trays.

    # Assumptions
    - The goal is to serve all children who are initially in a 'waiting' state.
    - Children's waiting locations and allergy statuses are static properties
      defined in the initial state.
    - Ingredient types (gluten-free or not) are static properties of the ingredients
      defined in the initial state.
    - The heuristic assumes the problem is solvable from the initial state;
      if the current state makes it unsolvable based on available/makable sandwiches,
      it returns infinity.
    - It assumes optimal grouping of sandwiches on trays and optimal tray movements
      to minimize actions of each type.

    # Heuristic Initialization
    - Identify all children who need to be served from the task goals.
    - Extract static information about each child (waiting location, allergy status)
      from the initial state.
    - Identify static properties of ingredients (gluten-free bread/content) from
      the initial state.
    - Identify all tray objects.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums the estimated minimum number of actions of four main types:
    make, put, move, and serve.

    1.  **Identify Unserved Children:** Determine which children from the initial
        waiting list (derived from goals and initial state) have not yet been
        served in the current state. Count the total number of unserved children
        (`N_unserved`) and categorize them by sandwich type needed (gluten-free
        or regular): `N_gf_unserved`, `N_reg_unserved`.
        If `N_unserved` is 0, the state is a goal state, and the heuristic is 0.

    2.  **Estimate Serve Actions:** Each unserved child requires one 'serve' action.
        Cost_serve = `N_unserved`.

    3.  **Estimate Tray Move Actions:** Identify the unique locations where unserved
        children are waiting. For each such location, check if any tray is currently
        present at that location in the current state. Count the number of unique
        locations with unserved children that *do not* have a tray. Each such location
        will require at least one 'move_tray' action to bring a tray there.
        Cost_move = Number of unique locations needing a tray visit that don't have one.

    4.  **Estimate Sandwich Actions (Make and Put):**
        Count the current number of gluten-free and regular sandwiches in the kitchen
        (`S_gf_kit`, `S_reg_kit`) and on trays (`S_gf_tray`, `S_reg_tray`) in the
        current state. Determine which sandwiches are gluten-free by checking the
        `(no_gluten_sandwich ?s)` predicate in the current state.
        Count the current number of gluten-free and regular bread and content
        portions in the kitchen (`B_gf_kit`, `B_reg_kit`, `C_gf_kit`, `C_reg_kit`)
        using the static ingredient type information.

        Calculate the number of sandwiches of each type that *can* be made from
        current kitchen ingredients: `Makable_gf = min(B_gf_kit, C_gf_kit)`,
        `Makable_reg = min(B_reg_kit, C_reg_kit)`.

        Determine the total number of gluten-free and regular sandwiches that need
        to be on trays to serve all unserved children (`N_gf_unserved`, `N_reg_unserved`).
        Subtract the sandwiches already on trays to find the deficit that must come
        from the kitchen:
        `Needed_on_tray_gf = max(0, N_gf_unserved - S_gf_tray)`
        `Needed_on_tray_reg = max(0, N_reg_unserved - S_reg_tray)`

        Check for unsolvability: If the total number of sandwiches available in the
        kitchen (`S_gf_kit`, `S_reg_kit`) plus those that can be made (`Makable_gf`,
        `Makable_reg`) is less than the deficit needed on trays for either type,
        the problem is unsolvable from this state. Return infinity.
        If `Needed_on_tray_gf > S_gf_kit + Makable_gf` or
           `Needed_on_tray_reg > S_reg_kit + Makable_reg`, return infinity.

        Estimate Put Actions: Each sandwich that needs to transition from the kitchen
        (either existing or newly made) onto a tray requires one 'put_on_tray' action.
        The number of sandwiches needing to be put is exactly the deficit needed on trays.
        Cost_put = `Needed_on_tray_gf + Needed_on_tray_reg`.

        Estimate Make Actions: Count how many of the sandwiches needed on trays
        (`Needed_on_tray_gf`, `Needed_on_tray_reg`) are *not* already present in the
        kitchen (`S_gf_kit`, `S_reg_kit`). These must be made.
        Cost_make_gf = `max(0, Needed_on_tray_gf - S_gf_kit)`
        Cost_make_reg = `max(0, Needed_on_tray_reg - S_reg_kit)`
        Cost_make = `Cost_make_gf + Cost_make_reg`.

    5.  **Total Heuristic:** Sum the costs from steps 2, 3, and 4.
        Heuristic = Cost_serve + Cost_move + Cost_put + Cost_make.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information and static facts.
        """
        self.goals = task.goals
        # Use initial state for static info like waiting locations and allergies
        self.initial_state = task.initial_state
        self.objects = task.objects # Get all objects

        # Identify children who need to be served (all initially waiting children)
        self.children_to_serve = set()
        # Store child info: {child_name: {'location': place, 'allergic': bool}}
        self.child_info = {}

        # Extract child info from initial state
        for fact in self.initial_state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.children_to_serve.add(child)
                self.child_info[child] = {'location': place}
            # Allergy info is also static
            elif match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                if child in self.child_info: # Ensure child was waiting
                     self.child_info[child]['allergic'] = True
            elif match(fact, "not_allergic_gluten", "*"):
                 child = get_parts(fact)[1]
                 if child in self.child_info: # Ensure child was waiting
                     self.child_info[child]['allergic'] = False

        # Identify static gluten-free ingredients from initial state
        self.gf_bread_types = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "no_gluten_bread", "*")}
        self.gf_content_types = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "no_gluten_content", "*")}

        # Identify all tray objects
        self.all_trays = {obj for obj, obj_type in self.objects.items() if obj_type == 'tray'}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = set()
        N_gf_unserved = 0
        N_reg_unserved = 0
        locations_needing_tray = set()

        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child in self.children_to_serve:
            if child not in served_children_in_state:
                unserved_children.add(child)
                locations_needing_tray.add(self.child_info[child]['location'])
                if self.child_info[child]['allergic']:
                    N_gf_unserved += 1
                else:
                    N_reg_unserved += 1

        N_unserved = len(unserved_children)

        # If all children are served, the heuristic is 0
        if N_unserved == 0:
            return 0

        # 2. Estimate Serve Actions
        Cost_serve = N_unserved

        # 3. Estimate Tray Move Actions
        N_locations_without_tray = 0
        for loc in locations_needing_tray:
            tray_at_location = False
            for tray in self.all_trays:
                if f'(at {tray} {loc})' in state:
                    tray_at_location = True
                    break
            if not tray_at_location:
                N_locations_without_tray += 1
        Cost_move = N_locations_without_tray

        # 4. Estimate Sandwich Actions (Make and Put)

        # Count current sandwiches and ingredients in the state
        S_gf_kit = 0
        S_reg_kit = 0
        S_gf_tray = 0
        S_reg_tray = 0
        B_gf_kit = 0
        B_reg_kit = 0
        C_gf_kit = 0
        C_reg_kit = 0

        # Determine which sandwiches are gluten-free in the current state
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                if s in gf_sandwiches_in_state:
                    S_gf_kit += 1
                else:
                    S_reg_kit += 1
            elif match(fact, "ontray", "*", "*"):
                s = get_parts(fact)[1]
                if s in gf_sandwiches_in_state:
                    S_gf_tray += 1
                else:
                    S_reg_tray += 1
            elif match(fact, "at_kitchen_bread", "*"):
                b = get_parts(fact)[1]
                if b in self.gf_bread_types:
                    B_gf_kit += 1
                else:
                    B_reg_kit += 1
            elif match(fact, "at_kitchen_content", "*"):
                c = get_parts(fact)[1]
                if c in self.gf_content_types:
                    C_gf_kit += 1
                else:
                    C_reg_kit += 1

        # Calculate makable sandwiches from kitchen ingredients
        Makable_gf = min(B_gf_kit, C_gf_kit)
        Makable_reg = min(B_reg_kit, C_reg_kit)

        # Calculate deficit of sandwiches needed on trays
        Needed_on_tray_gf = max(0, N_gf_unserved - S_gf_tray)
        Needed_on_tray_reg = max(0, N_reg_unserved - S_reg_tray)

        # Unsolvable check: Do we have enough sandwiches (existing + makable) to meet the deficit on trays?
        if Needed_on_tray_gf > S_gf_kit + Makable_gf or \
           Needed_on_tray_reg > S_reg_kit + Makable_reg:
            return float('inf') # Problem is unsolvable from this state

        # Estimate Put Actions: Number of sandwiches that need to move from kitchen to tray
        # This is the total number of sandwiches needed on trays that aren't already there.
        Cost_put = Needed_on_tray_gf + Needed_on_tray_reg

        # Estimate Make Actions: Number of sandwiches needed on trays that are not already in the kitchen
        # These are the ones that must be made before being put on a tray.
        Cost_make_gf = max(0, Needed_on_tray_gf - S_gf_kit)
        Cost_make_reg = max(0, Needed_on_tray_reg - S_reg_kit)
        Cost_make = Cost_make_gf + Cost_make_reg

        # 5. Total Heuristic
        total_heuristic = Cost_serve + Cost_move + Cost_put + Cost_make

        return total_heuristic
