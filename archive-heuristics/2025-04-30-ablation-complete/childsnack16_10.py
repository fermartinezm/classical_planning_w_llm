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
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for making necessary sandwiches, putting them
    on trays, moving trays to children's locations, and finally serving the children.

    # Assumptions
    - Assumes sufficient ingredients exist in the initial state to make any required sandwich.
    - Assumes sufficient trays exist to transport sandwiches.
    - Assumes trays can be moved directly between any two places.
    - Assumes a tray can hold multiple sandwiches.
    - Assumes a tray at a location with multiple children can serve them sequentially.

    # Heuristic Initialization
    - Extracts static information about children: their allergy status and waiting location.
    - Stores a list of all children in the problem and identifies all sandwich objects.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums up the estimated minimum number of actions required for different stages
    of the process for all unserved children:

    1.  **Count Unserved Children (Serve Actions):** Identify all children who are not yet
        in a `(served ?c)` state. The number of unserved children is a lower bound on the
        number of `serve_sandwich` actions needed. Let this be `N_unserved`.

    2.  **Count Sandwiches to Make (Make Actions):** Determine how many suitable sandwiches
        need to be created. Count unserved allergic children (`N_allergic_unserved`) and
        unserved non-allergic children (`N_non_allergic_unserved`). Count available
        gluten-free (`Avail_GF_anywhere`) and regular (`Avail_Reg_anywhere`) sandwiches
        (those not in `notexist` state).
        - Gluten-free sandwiches needed: `N_allergic_unserved`.
        - Regular sandwiches needed (can use regular or excess GF): `N_non_allergic_unserved`.
        - GF sandwiches to make: `max(0, N_allergic_unserved - Avail_GF_anywhere)`.
        - Regular sandwiches to make: `max(0, N_non_allergic_unserved - (Avail_Reg_anywhere + max(0, Avail_GF_anywhere - N_allergic_unserved)))`.
        - Total sandwiches to make (`N_make`) is the sum of GF and regular sandwiches to make.
        This is a lower bound on `make_sandwich` actions.

    3.  **Count Sandwiches Needing to be Put on Trays (Put Actions):** Sandwiches that are
        made or are currently `at_kitchen_sandwich` need to be put on a tray. Sandwiches
        already `ontray` do not. Count suitable sandwiches already on trays anywhere.
        - Count GF sandwiches on trays anywhere (`Avail_GF_ontray_anywhere`).
        - Count regular sandwiches on trays anywhere (`Avail_Reg_ontray_anywhere`).
        - Calculate how many unserved children can be matched with existing on-tray sandwiches (`Served_by_ontray`).
        - The number of children needing sandwiches that are *not* yet on trays is `max(0, N_unserved - Served_by_ontray)`. This is a lower bound on `put_on_tray` actions.

    4.  **Count Tray Movements Needed (Move Actions):** Trays need to be moved to locations
        where unserved children are waiting and do not have enough suitable sandwiches
        already available on trays at that location.
        - Group unserved children by location and allergy.
        - For each location `P` with unserved children, count unserved allergic (`N_allergic_at_P`)
          and non-allergic (`N_non_allergic_at_P`) children.
        - Count suitable sandwiches already on trays *at this specific place P* (`Avail_GF_ontray_at_P`, `Avail_Reg_ontray_at_P`).
        - Calculate the number of children at `P` who cannot be served by sandwiches already
          on trays at `P`.
        - If this number is greater than 0 for a location `P`, at least one tray move to `P`
          is needed. Count the number of such locations (`N_places_need_tray_move`). This is
        a lower bound on `move_tray` actions *to deliver sandwiches*.

    5.  **Total Heuristic Value:** The heuristic is the sum of the lower bounds for each
        action type: `N_unserved + N_make + max(0, N_unserved - Served_by_ontray) + N_places_need_tray_move`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        and identifying all potential sandwich objects.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.children = set()
        self.allergic_children = set()
        self.waiting_locations = {} # child -> place

        # Extract child info from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child = parts[1]
                self.children.add(child)
                self.allergic_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = parts[1]
                self.children.add(child)
            elif predicate == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_locations[child] = place
                # Ensure child is added even if allergy status isn't explicitly listed (shouldn't happen based on domain)
                self.children.add(child)

        # Identify all potential sandwich objects from the initial state's notexist facts
        self.all_sandwich_objects = {get_parts(fact)[1] for fact in initial_state if match(fact, "notexist", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # --- Step 1: Count Unserved Children ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.children - served_children
        N_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        # --- Step 2: Count Sandwiches to Make ---
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_non_allergic = unserved_children - unserved_allergic
        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # Count existing sandwiches (those not in notexist state)
        existing_sandwiches_in_state = {s for s in self.all_sandwich_objects if f'(notexist {s})' not in state}

        Avail_GF_anywhere = sum(1 for s in existing_sandwiches_in_state if f'(no_gluten_sandwich {s})' in state)
        Avail_Reg_anywhere = len(existing_sandwiches_in_state) - Avail_GF_anywhere

        Needed_GF_to_make = max(0, N_allergic_unserved - Avail_GF_anywhere)
        Needed_Reg_to_make = max(0, N_non_allergic_unserved - (Avail_Reg_anywhere + max(0, Avail_GF_anywhere - N_allergic_unserved)))
        N_make = Needed_GF_to_make + Needed_Reg_to_make

        # --- Step 3: Count Sandwiches Needing to be Put on Trays ---
        ontray_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        Avail_GF_ontray_anywhere = sum(1 for s in ontray_sandwiches if f'(no_gluten_sandwich {s})' in state)
        Avail_Reg_ontray_anywhere = len(ontray_sandwiches) - Avail_GF_ontray_anywhere

        # How many unserved children can be matched with existing on-tray sandwiches?
        # Prioritize GF for allergic, then use remaining GF and Reg for non-allergic.
        served_by_ontray_gf = min(N_allergic_unserved, Avail_GF_ontray_anywhere)
        remaining_ontray_gf = Avail_GF_ontray_anywhere - served_by_ontray_gf
        served_by_ontray_reg = min(N_non_allergic_unserved, Avail_Reg_ontray_anywhere + remaining_ontray_gf)
        Served_by_ontray = served_by_ontray_gf + served_by_ontray_reg

        N_put_on_tray_needed = max(0, N_unserved - Served_by_ontray)


        # --- Step 4: Count Tray Movements Needed ---
        unserved_children_by_place = {}
        for child in unserved_children:
            place = self.waiting_locations.get(child)
            if place: # Only consider children whose waiting location is known
                if place not in unserved_children_by_place:
                    unserved_children_by_place[place] = {'allergic': [], 'non_allergic': []}
                if child in self.allergic_children:
                    unserved_children_by_place[place]['allergic'].append(child)
                else:
                    unserved_children_by_place[place]['non_allergic'].append(child)

        N_places_need_tray_move = 0
        for place, children_at_place in unserved_children_by_place.items():
            N_allergic_at_P = len(children_at_place['allergic'])
            N_non_allergic_at_P = len(children_at_place['non_allergic'])

            # Count suitable sandwiches on trays *at this specific place P*
            Avail_GF_ontray_at_P = 0
            Avail_Reg_ontray_at_P = 0
            trays_at_P = {get_parts(fact)[1] for fact in state if match(fact, "at", "*", place) and get_parts(fact)[1].startswith('tray')}
            for tray_at_P in trays_at_P:
                sandwiches_on_this_tray = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", tray_at_P)}
                Avail_GF_on_this_tray = sum(1 for s in sandwiches_on_this_tray if f'(no_gluten_sandwich {s})' in state)
                Avail_Reg_on_this_tray = len(sandwiches_on_this_tray) - Avail_GF_on_this_tray
                Avail_GF_ontray_at_P += Avail_GF_on_this_tray
                Avail_Reg_ontray_at_P += Avail_Reg_on_this_tray

            # Calculate children at P who cannot be served by sandwiches already on trays at P
            served_at_P_by_ontray_gf = min(N_allergic_at_P, Avail_GF_ontray_at_P)
            remaining_ontray_at_P_gf = Avail_GF_ontray_at_P - served_at_P_by_ontray_gf
            served_at_P_by_ontray_reg = min(N_non_allergic_at_P, Avail_Reg_ontray_at_P + remaining_ontray_at_P_gf)
            Served_at_P_by_ontray = served_at_P_by_ontray_gf + served_at_P_by_ontray_reg

            Needed_at_P_from_elsewhere = max(0, N_allergic_at_P + N_non_allergic_at_P - Served_at_P_by_ontray)

            if Needed_at_P_from_elsewhere > 0:
                 N_places_need_tray_move += 1 # At least one move is needed to bring sandwiches here

        # --- Step 5: Total Heuristic Value ---
        # Sum of estimated actions for each stage transition needed for the unserved children
        # N_unserved: Cost for the final 'serve' action for each child.
        # N_make: Cost for the 'make_sandwich' action for each sandwich that needs creating.
        # N_put_on_tray_needed: Cost for the 'put_on_tray' action for sandwiches not yet on trays but needed.
        # N_places_need_tray_move: Cost for the 'move_tray' action to bring trays to locations needing deliveries.
        total_cost = N_unserved + N_make + N_put_on_tray_needed + N_places_need_tray_move

        return total_cost
