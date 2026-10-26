from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def is_fact_in_state(state, predicate, *args):
    """Helper to check if a specific fact (predicate and args) is in the state."""
    # Construct the fact string and check for its presence in the state set
    fact_str = "(" + predicate + (" " + " ".join(args) if args else "") + ")"
    return fact_str in state

# Helper to extract objects with types (approximate based on predicates used in init/goal)
def extract_objects_from_fact(fact):
    """
    Extract object names and infer their types from a PDDL fact string.
    This relies on common predicate usage in the childsnacks domain.
    """
    parts = get_parts(fact)
    if not parts: return []
    predicate = parts[0]
    args = parts[1:]
    objects = []
    # Map predicates to argument types based on domain definition
    if predicate in ['at_kitchen_bread', 'no_gluten_bread'] and len(args) == 1:
        objects.append((args[0], 'bread-portion'))
    elif predicate in ['at_kitchen_content', 'no_gluten_content'] and len(args) == 1:
        objects.append((args[0], 'content-portion'))
    elif predicate in ['at_kitchen_sandwich', 'no_gluten_sandwich', 'notexist'] and len(args) == 1:
        objects.append((args[0], 'sandwich'))
    elif predicate in ['allergic_gluten', 'not_allergic_gluten', 'served'] and len(args) == 1:
        objects.append((args[0], 'child'))
    elif predicate == 'waiting' and len(args) == 2:
        objects.append((args[0], 'child'))
        objects.append((args[1], 'place'))
    elif predicate == 'ontray' and len(args) == 2:
        objects.append((args[0], 'sandwich'))
        objects.append((args[1], 'tray'))
    elif predicate == 'at' and len(args) == 2:
        objects.append((args[0], 'tray'))
        objects.append((args[1], 'place'))
    return objects


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all unserved
    children. It sums the estimated costs for the main stages of the process:
    making necessary sandwiches, putting sandwiches on trays, moving trays to
    children's locations, and finally serving the children.

    # Assumptions
    - The primary goal is to serve all children.
    - Each child requires one suitable sandwich (gluten-free for allergic children,
      any for non-allergic children).
    - Ingredients (bread, content) and 'notexist' sandwich slots are assumed
      sufficient if needed for making sandwiches.
    - Tray capacity is not a bottleneck; a single tray can serve all children
      at a specific location.
    - Enough trays exist in total to be moved to all necessary locations.

    # Heuristic Initialization
    The heuristic extracts and stores static information from the task:
    - Maps each child to their allergy status (allergic or not).
    - Maps each child to their waiting location.
    - Identifies all children, sandwiches, trays, and places in the problem instance
      by inspecting initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as the sum of four components:

    1.  **Cost for Serving (N_unserved):**
        Count the number of children who are not yet in the 'served' state.
        Each unserved child requires one 'serve' action. This gives a base cost.

    2.  **Cost for Making Sandwiches (N_make):**
        Determine the total number of suitable sandwiches required for all unserved
        children (gluten-free for allergic, any for non-allergic).
        Count the number of suitable sandwiches that already exist in the state
        (either 'at_kitchen_sandwich' or 'ontray').
        The number of sandwiches that need to be made is the total required minus
        the total available. Gluten-free sandwiches can satisfy non-allergic needs
        if there's an excess of gluten-free sandwiches after meeting all allergic needs.
        Each sandwich made costs 1 'make_sandwich' action.

    3.  **Cost for Putting on Tray (N_put_on_tray):**
        Sandwiches must be on a tray before they can be moved or served.
        Count the number of sandwiches that are currently 'at_kitchen_sandwich'.
        These need a 'put_on_tray' action.
        Additionally, any sandwiches that need to be newly made (from step 2)
        will also end up 'at_kitchen_sandwich' and require a 'put_on_tray' action.
        The total cost is the sum of existing 'at_kitchen_sandwich' count and the
        number of sandwiches to be made. Each costs 1 'put_on_tray' action.

    4.  **Cost for Moving Trays (N_move_tray):**
        Trays carrying sandwiches must be at the location where the children are waiting.
        Identify all distinct locations (excluding 'kitchen') where unserved children
        are waiting. These are the 'target locations'.
        Count the number of trays that are already present at these target locations.
        The number of tray movements needed is the number of target locations minus
        the number of trays already there. Each required movement costs 1 'move_tray' action.

    The total heuristic value is the sum of N_unserved + N_make + N_put_on_tray + N_move_tray.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        self.child_allergy = {}
        self.child_waiting_location = {}
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = set()

        # Collect all objects from initial state and goals to know all possible instances
        all_objects_with_types = set()
        for fact in task.initial_state | task.goals:
             all_objects_with_types.update(extract_objects_from_fact(fact))

        # Populate sets based on type
        for obj, obj_type in all_objects_with_types:
            if obj_type == 'child': self.all_children.add(obj)
            elif obj_type == 'sandwich': self.all_sandwiches.add(obj)
            elif obj_type == 'tray': self.all_trays.add(obj)
            elif obj_type == 'place': self.all_places.add(obj)

        # Extract static information from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]
            if predicate == 'allergic_gluten' and len(args) == 1:
                self.child_allergy[args[0]] = True
            elif predicate == 'not_allergic_gluten' and len(args) == 1:
                self.child_allergy[args[0]] = False
            elif predicate == 'waiting' and len(args) == 2:
                self.child_waiting_location[args[0]] = args[1]
            # Note: Other static facts like no_gluten_bread/content are not strictly
            # needed in __init__ for this heuristic's calculation logic, but could
            # be extracted here if a more complex heuristic required them.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Cost for Serving (N_unserved)
        # Count children whose 'served' goal fact is not in the current state
        unserved_children = {c for c in self.all_children if not is_fact_in_state(state, 'served', c)}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached, heuristic is 0

        # Group unserved children by allergy to determine sandwich needs
        unserved_allergic = {c for c in unserved_children if self.child_allergy.get(c, False)}
        unserved_non_allergic = unserved_children - unserved_allergic

        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # 2. Cost for Making Sandwiches (N_make)
        # Count available sandwiches by type, regardless of location (kitchen or ontray)
        ontray_gf = sum(1 for fact in state if match(fact, "ontray", "*", "*") and is_fact_in_state(state, "no_gluten_sandwich", get_parts(fact)[1]))
        ontray_reg = sum(1 for fact in state if match(fact, "ontray", "*", "*") and not is_fact_in_state(state, "no_gluten_sandwich", get_parts(fact)[1]))
        atk_gf = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and is_fact_in_state(state, "no_gluten_sandwich", get_parts(fact)[1]))
        atk_reg = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and not is_fact_in_state(state, "no_gluten_sandwich", get_parts(fact)[1]))

        total_avail_gf = ontray_gf + atk_gf
        total_avail_reg = ontray_reg + atk_reg

        needed_gf = N_allergic_unserved
        needed_reg = N_non_allergic_unserved

        # Calculate how many GF sandwiches need to be made
        make_gf_count = max(0, needed_gf - total_avail_gf)

        # Calculate how many regular sandwiches need to be made.
        # Excess available GF sandwiches can cover regular needs.
        excess_gf_avail = max(0, total_avail_gf - needed_gf)
        make_reg_count = max(0, needed_reg - total_avail_reg - excess_gf_avail)

        N_make = make_gf_count + make_reg_count

        # 3. Cost for Putting on Tray (N_put_on_tray)
        # Count sandwiches currently at the kitchen that need to be put on a tray
        num_atk_sandwiches = atk_gf + atk_reg
        # Add the sandwiches that will be made, as they will also start at the kitchen
        N_put_on_tray = num_atk_sandwiches + N_make

        # 4. Cost for Moving Trays (N_move_tray)
        # Identify locations where unserved children are waiting (excluding the kitchen)
        target_locations = set()
        for child in unserved_children:
            location = self.child_waiting_location.get(child)
            if location and location != 'kitchen':
                target_locations.add(location)

        # Count trays that are already present at any of the target locations
        trays_at_target = sum(1 for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[2] in target_locations)

        # The number of moves needed is the number of target locations that don't
        # currently have a tray. Assumes trays are available elsewhere (kitchen or non-target).
        N_move_tray = max(0, len(target_locations) - trays_at_target)

        # Total heuristic value is the sum of the estimated costs for each stage
        heuristic_value = N_unserved + N_make + N_put_on_tray + N_move_tray

        return heuristic_value
