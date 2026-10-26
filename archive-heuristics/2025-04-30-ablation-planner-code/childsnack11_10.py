from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(kitchen)' or '(notexist sandw1)' which might have varying lengths
    parts = fact[1:-1].split()
    return parts

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
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the necessary actions across four main stages for the
    remaining unserved children: making sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children.
    It attempts to account for shared resources like trays and the ability
    of one action (like moving a tray) to satisfy requirements for multiple
    children at the same location.

    # Assumptions
    - The goal is to serve all children specified in the initial goal list.
    - Children's allergies and waiting locations are static.
    - Enough bread, content, and 'notexist' sandwich objects exist initially
      to make all required sandwiches (the heuristic counts the *need* to make,
      but doesn't explicitly check resource availability beyond the count).
    - Enough trays exist to be moved to all locations where unserved children
      are waiting.
    - A tray is available somewhere to be moved to the kitchen if sandwiches
      need to be put on a tray and no tray is currently there.

    # Heuristic Initialization
    - Extracts the set of all children who need to be served from the goal.
    - Extracts which children are allergic or not allergic from static facts.
    - Extracts the waiting location for each child from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for four main
    types of actions needed to transition from the current state to a goal state
    where all target children are served:

    1.  **Cost to Make Sandwiches:**
        - Count the number of unserved allergic children (`num_allergic_unserved`). These require gluten-free sandwiches.
        - Count the total number of unserved children (`num_unserved`). These require any suitable sandwich (GF for allergic, any for non-allergic).
        - Count existing gluten-free sandwiches (in kitchen or on tray): `num_existing_gf`.
        - Count existing any sandwiches (in kitchen or on tray): `num_existing_any`.
        - Estimate the minimum number of `make_sandwich_no_gluten` actions needed: `needed_make_gf = max(0, num_allergic_unserved - num_existing_gf)`.
        - Estimate the minimum total number of `make_sandwich` (or `make_sandwich_no_gluten`) actions needed: `needed_make_any = max(0, num_unserved - num_existing_any)`.
        - The total cost for making sandwiches is the maximum of these two needs, as making a GF sandwich satisfies both a GF need and a general sandwich need: `make_cost = max(needed_make_gf, needed_make_any)`.

    2.  **Cost to Put Sandwiches on Trays:**
        - Count the number of sandwiches currently `at_kitchen_sandwich`. Each of these needs a `put_on_tray` action.
        - `put_cost = count of sandwiches at_kitchen_sandwich`.

    3.  **Cost to Move Trays:**
        - Identify all distinct locations where unserved children are waiting.
        - Identify all distinct locations where trays are currently located.
        - For each location where unserved children are waiting but no tray is present, add 1 to the `move_cost` (representing the need to move a tray there).
        - Additionally, if there are sandwiches `at_kitchen_sandwich` (requiring `put_on_tray`) but no tray is currently `at kitchen`, add 1 to the `move_cost` (representing the need to move a tray to the kitchen).

    4.  **Cost to Serve Children:**
        - Each unserved child requires one `serve_sandwich` or `serve_sandwich_no_gluten` action.
        - `serve_cost = num_unserved`.

    The total heuristic value is the sum of `make_cost + put_cost + move_cost + serve_cost`.
    If `num_unserved` is 0, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        and their waiting places and allergies.
        """
        super().__init__(task) # Call the base class constructor

        # Extract all children from the goal list
        self.all_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}

        # Extract static facts about children's allergies and waiting places
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {}

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                self.not_allergic_children.add(parts[1])
            elif parts[0] == 'waiting' and len(parts) == 3:
                self.child_waiting_place[parts[1]] = parts[2]

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all unserved children.
        """
        state = node.state

        # 1. Count unserved children
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}
        num_unserved = len(unserved_children)

        # If all children are served, the heuristic is 0 (goal state)
        if num_unserved == 0:
            return 0

        # Count unserved children by allergy type
        num_allergic_unserved = len({c for c in unserved_children if c in self.allergic_children})
        # num_non_allergic_unserved = num_unserved - num_allergic_unserved # Not explicitly needed for calculation below

        # 2. Count existing sandwiches (anywhere: kitchen or on tray)
        existing_sandwiches = set()
        existing_gf_sandwiches = set()

        # Pre-compute gluten-free status of sandwiches
        gf_sandwiches_in_state = {s for fact in state if match(fact, "no_gluten_sandwich", s)}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                s = parts[1]
                existing_sandwiches.add(s)
                if s in gf_sandwiches_in_state:
                    existing_gf_sandwiches.add(s)
            elif parts[0] == 'ontray' and len(parts) == 3:
                s = parts[1]
                existing_sandwiches.add(s)
                if s in gf_sandwiches_in_state:
                    existing_gf_sandwiches.add(s)

        num_existing_any = len(existing_sandwiches)
        num_existing_gf = len(existing_gf_sandwiches)

        # 3. Cost to Make Sandwiches
        # Need enough GF sandwiches for all allergic unserved children
        needed_make_gf = max(0, num_allergic_unserved - num_existing_gf)
        # Need enough total sandwiches for all unserved children
        needed_make_any = max(0, num_unserved - num_existing_any)
        # Total make actions is the max of these needs
        make_cost = max(needed_make_gf, needed_make_any)

        # 4. Cost to Put Sandwiches on Trays
        at_kitchen_sandwiches = {s for fact in state if match(fact, "at_kitchen_sandwich", s)}
        put_cost = len(at_kitchen_sandwiches)

        # 5. Cost to Move Trays
        move_cost = 0

        # Identify places where unserved children are waiting
        waiting_places = {self.child_waiting_place[c] for c in unserved_children if c in self.child_waiting_place}

        # Identify places where trays are located
        tray_locations = {p for fact in state if match(fact, "at", "*", p)}

        # Count places with waiting children that need a tray moved there
        places_needing_trays = {p for p in waiting_places if p != 'kitchen' and p not in tray_locations} # Children don't wait at kitchen
        move_cost += len(places_needing_trays)

        # Check if a tray needs to be moved to the kitchen for put_on_tray actions
        tray_at_kitchen = 'kitchen' in tray_locations
        if put_cost > 0 and not tray_at_kitchen:
             move_cost += 1 # Need one move action to bring a tray to the kitchen

        # 6. Cost to Serve Children
        serve_cost = num_unserved

        # Total heuristic is the sum of estimated costs for each stage
        total_cost = make_cost + put_cost + move_cost + serve_cost

        return total_cost

