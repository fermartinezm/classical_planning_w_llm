from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact is at least the number of pattern arguments
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the number of children needing service and adds costs
    for the necessary steps: making sandwiches, putting them on trays, and
    moving trays to the children's locations.

    # Heuristic Calculation Strategy
    The heuristic sums the estimated costs for four main stages required to serve
    each unserved child:
    1.  **Serving:** Each unserved child requires a final 'serve' action. Cost = Number of unserved children.
    2.  **Making Sandwiches:** Count how many suitable sandwiches are needed in total for all unserved children. Subtract the number of suitable sandwiches already existing (either in the kitchen or on trays). The remainder must be made. Cost = Number of sandwiches that need to be made.
    3.  **Putting on Trays:** Count how many needed sandwiches are not already on trays. These must be put on trays. Cost = Number of needed sandwiches not already on trays.
    4.  **Moving Trays:** For each location where unserved children are waiting, check if the suitable sandwiches currently on trays at that location are sufficient. If not, a tray needs to be moved to that location. Cost = Number of distinct locations requiring a tray visit.

    This heuristic is non-admissible as it sums costs across stages and children,
    potentially overestimating, but aims to capture the main bottlenecks and
    dependencies in the domain.

    # Initialization
    The heuristic pre-processes static facts to determine which children are
    allergic and where each child is waiting. It also identifies all possible
    places and tray objects from the initial state and static facts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        places, and trays.
        """
        self.task = task # Store the task for access to goals, initial_state, static

        self.child_place = {}
        self.child_allergic = {}
        all_places_set = set()
        all_trays_set = set()

        # Get info from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.child_place[child] = place
                all_places_set.add(place)
            elif parts[0] == "allergic_gluten":
                self.child_allergic[parts[1]] = True
            elif parts[0] == "not_allergic_gluten":
                self.child_allergic[parts[1]] = False

        # Get places and trays from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 tray, place = parts[1], parts[2]
                 all_places_set.add(place)
                 all_trays_set.add(tray)
             elif parts[0] == "ontray": # Sandwiches on trays in initial state
                 s, t = parts[1], parts[2]
                 all_trays_set.add(t)

        # Add the constant kitchen place
        all_places_set.add("kitchen")

        self.all_places = list(all_places_set)
        self.all_trays = list(all_trays_set) # Store all tray objects found

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Estimates the cost based on the number of unserved children and the
        estimated actions needed to get suitable sandwiches to them.
        """
        state = node.state
        h = 0

        # --- Step 1: Identify unserved children and their needs ---
        unserved_children = []
        unserved_by_place = {p: [] for p in self.all_places}
        U_allergic = 0
        U_non_allergic = 0

        # Get the set of children who are currently served
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through all children known from static 'waiting' facts
        for child, place in self.child_place.items():
            if child not in served_children_in_state:
                unserved_children.append(child)
                unserved_by_place[place].append(child)
                # Check allergy status, default to False if not specified
                if self.child_allergic.get(child, False):
                    U_allergic += 1
                else:
                    U_non_allergic += 1

        N_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        # Base cost: Each unserved child needs a 'serve' action
        h += N_unserved

        # --- Step 2: Identify available sandwiches and trays in the current state ---
        S_kitchen_gf = set()
        S_kitchen_reg = set()
        S_ontray_gf = set()
        S_ontray_reg = set()
        Tray_loc = {}
        Tray_contents = {t: set() for t in self.all_trays} # Initialize with all known trays

        # Iterate through current state facts to populate sandwich and tray info
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at_kitchen_sandwich", "*"):
                s = parts[1]
                # Check if the specific fact (no_gluten_sandwich s) is in the state
                is_gf = f"(no_gluten_sandwich {s})" in state
                if is_gf:
                    S_kitchen_gf.add(s)
                else:
                    S_kitchen_reg.add(s)
            elif match(fact, "ontray", "*", "*"):
                s, t = parts[1], parts[2]
                if t not in Tray_contents:
                     # This tray wasn't in the initial state? Should not happen based on domain.
                     # Add it just in case, though it might indicate a problem in task parsing.
                     Tray_contents[t] = set()
                Tray_contents[t].add(s)
                # Check if the specific fact (no_gluten_sandwich s) is in the state
                is_gf = f"(no_gluten_sandwich {s})" in state
                if is_gf:
                    S_ontray_gf.add(s)
                else:
                    S_ontray_reg.add(s)
            elif match(fact, "at", "*", "*"):
                t, p = parts[1], parts[2]
                Tray_loc[t] = p

        Avail_S_kitchen_total = len(S_kitchen_gf) + len(S_kitchen_reg)
        Avail_S_ontray_total = len(S_ontray_gf) + len(S_ontray_reg)
        Avail_S_total = Avail_S_kitchen_total + Avail_S_ontray_total

        # --- Step 3: Calculate Cost for Making Sandwiches ---
        # We need N_unserved sandwiches in total. How many are missing?
        N_make = max(0, N_unserved - Avail_S_total)
        h += N_make # Each needed sandwich that doesn't exist costs 1 'make' action

        # --- Step 4: Calculate Cost for Putting on Tray ---
        # Needed sandwiches must end up on a tray. How many are not already on trays?
        N_put_on_tray = max(0, N_unserved - Avail_S_ontray_total)
        h += N_put_on_tray # Each needed sandwich not on a tray costs 1 'put_on_tray' action

        # --- Step 5: Calculate Cost for Moving Trays ---
        # Count locations with unserved children that need a tray visit.
        N_locations_needing_visit = 0
        for place, children_at_place in unserved_by_place.items():
            if not children_at_place:
                continue # No unserved children at this place

            # Count allergic and non-allergic children at this specific place
            U_P_allergic = sum(1 for c in children_at_place if self.child_allergic.get(c, False))
            U_P_non_allergic = len(children_at_place) - U_P_allergic

            # Count suitable sandwiches currently on trays at this place
            Avail_at_P_gf = 0
            Avail_at_P_reg = 0

            # Find trays currently located at this place
            trays_at_p = [t for t, p in Tray_loc.items() if p == place]

            for tray in trays_at_p:
                for sandwich in Tray_contents.get(tray, set()):
                     # Check if the specific fact (no_gluten_sandwich sandwich) is in the state
                     is_gf = f"(no_gluten_sandwich {sandwich})" in state
                     if is_gf:
                         Avail_at_P_gf += 1
                     else:
                         Avail_at_P_reg += 1

            # This location needs a tray visit if:
            # 1. There are unserved children here (already checked by 'if not children_at_place: continue').
            # 2. The available GF sandwiches at this location are insufficient for allergic children here.
            # 3. OR the total available sandwiches (GF + Reg) at this location are insufficient for all children here.
            if Avail_at_P_gf < U_P_allergic or (Avail_at_P_gf + Avail_at_P_reg) < (U_P_allergic + U_P_non_allergic):
                 N_locations_needing_visit += 1

        h += N_locations_needing_visit # Each location needing a visit costs at least 1 'move_tray' action

        return h

