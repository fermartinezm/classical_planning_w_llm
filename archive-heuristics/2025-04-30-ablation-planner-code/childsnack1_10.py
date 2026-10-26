from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It sums the estimated costs for making necessary sandwiches, putting them on trays,
    moving trays to the children's locations, and finally serving the children.

    # Assumptions
    - Each action (make, put, move, serve) has a cost of 1.
    - Sufficient bread, content, and 'notexist' sandwich objects are available in the kitchen
      to make any required sandwiches if they don't exist.
    - Sufficient trays are available to move sandwiches to different locations simultaneously
      if needed (though the heuristic only counts distinct locations needing a tray).
    - A tray can move directly from its current location to any other location in one 'move_tray' action.
    - The heuristic counts the *minimum* number of actions of each type needed across all
      unserved children, assuming actions can be shared (e.g., one tray move serves all
      children at that location, one made sandwich can serve any child needing that type).

    # Heuristic Initialization
    - Extracts the initial list of children who are waiting and their locations from the static facts.
    - Extracts the allergy status (allergic_gluten or not_allergic_gluten) for all children
      from the static facts. This is used to determine the type of sandwich needed for each child.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify Unserved Children: Determine which children from the initial waiting list
        are not yet marked as 'served' in the current state. Let U be the set of unserved children.
        If U is empty, the heuristic is 0 (goal state).
    2.  Count Serve Actions: Each unserved child requires a final 'serve' action. Add |U| to the total heuristic.
    3.  Categorize Needed Sandwiches: Based on the allergy status of children in U, count how many
        gluten-free sandwiches (N_gf_needed) and how many regular sandwiches (N_reg_needed) are required in total.
    4.  Count Available Sandwiches: Count the number of gluten-free and regular sandwiches
        that are currently 'at_kitchen_sandwich' or 'ontray' in the current state.
        Let these counts be Avail_gf_kitchen, Avail_reg_kitchen, Avail_gf_ontray, Avail_reg_ontray.
    5.  Calculate Sandwiches to Make: Determine how many sandwiches of each type still need
        to be made. This is the deficit between the total needed and the total available.
        Make_gf = max(0, N_gf_needed - (Avail_gf_kitchen + Avail_gf_ontray))
        Make_reg = max(0, N_reg_needed - (Avail_reg_kitchen + Avail_reg_ontray))
        Add Make_gf + Make_reg to the total heuristic.
    6.  Calculate Sandwiches to Put on Tray: Determine how many sandwiches need to be moved
        from the kitchen ('at_kitchen_sandwich') onto a tray ('ontray'). This includes
        sandwiches that were just made (Make_gf, Make_reg) and any existing
        'at_kitchen_sandwich' that are needed to meet the demand not already on trays.
        Needed_ontray_gf = max(0, N_gf_needed - Avail_gf_ontray)
        Needed_ontray_reg = max(0, N_reg_needed - Avail_reg_ontray)
        Available_at_kitchen_gf_total = Avail_gf_kitchen + Make_gf
        Available_at_kitchen_reg_total = Avail_reg_kitchen + Make_reg
        Put_gf = min(Available_at_kitchen_gf_total, Needed_ontray_gf)
        Put_reg = min(Available_at_kitchen_reg_total, Needed_ontray_reg)
        Add Put_gf + Put_reg to the total heuristic.
    7.  Calculate Tray Movements: Identify the distinct locations where unserved children
        are waiting. Identify the distinct locations where trays are currently present.
        Count the number of distinct waiting locations that do not currently have a tray.
        Add this count to the total heuristic.
    8.  Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Store goals to identify all children who need serving
        static_facts = task.static

        # Map child to their allergy status
        self.child_allergy = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'allergic_gluten':
                self.child_allergy[parts[1]] = 'gluten'
            elif parts and parts[0] == 'not_allergic_gluten':
                self.child_allergy[parts[1]] = 'not_gluten'

        # Map child to their initial waiting location
        self.child_waiting_location = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.child_waiting_location[child] = place

        # Identify all children who are initially waiting (and thus need to be served)
        self.children_to_serve = set(self.child_waiting_location.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.children_to_serve - served_children

        if not unserved_children:
            return 0 # Goal state

        # 2. Count Serve Actions
        total_cost = len(unserved_children)

        # 3. Categorize Needed Sandwiches
        N_gf_needed = 0
        N_reg_needed = 0
        unserved_waiting_locations = set()

        for child in unserved_children:
            if self.child_allergy.get(child) == 'gluten':
                N_gf_needed += 1
            else: # Assume not_allergic_gluten or unknown, treat as regular
                N_reg_needed += 1
            # Keep track of locations where unserved children are waiting
            # We get the location from the initial state mapping, assuming children don't move
            # (which is true in this domain - only trays move to children)
            if child in self.child_waiting_location:
                 unserved_waiting_locations.add(self.child_waiting_location[child])


        # 4. Count Available Sandwiches
        Avail_gf_kitchen = 0
        Avail_reg_kitchen = 0
        Avail_gf_ontray = 0
        Avail_reg_ontray = 0

        # Find all sandwiches mentioned in relevant state facts
        sandwiches_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] in ["at_kitchen_sandwich", "ontray", "no_gluten_sandwich"]:
                 if len(parts) > 1:
                     sandwiches_in_state.add(parts[1])

        gf_sandwiches_in_state = {s for s in sandwiches_in_state if f"(no_gluten_sandwich {s})" in state}

        for s in sandwiches_in_state:
            is_gf = s in gf_sandwiches_in_state
            is_at_kitchen = f"(at_kitchen_sandwich {s})" in state
            is_ontray = any(match(fact, "ontray", s, "*") for fact in state) # Check if on *any* tray

            if is_at_kitchen:
                if is_gf:
                    Avail_gf_kitchen += 1
                else:
                    Avail_reg_kitchen += 1
            elif is_ontray:
                 if is_gf:
                    Avail_gf_ontray += 1
                 else:
                    Avail_reg_ontray += 1


        # 5. Calculate Sandwiches to Make
        Make_gf = max(0, N_gf_needed - (Avail_gf_kitchen + Avail_gf_ontray))
        Make_reg = max(0, N_reg_needed - (Avail_reg_kitchen + Avail_reg_ontray))
        total_cost += Make_gf + Make_reg

        # 6. Calculate Sandwiches to Put on Tray
        # Sandwiches needed on trays are those required minus those already on trays
        Needed_ontray_gf = max(0, N_gf_needed - Avail_gf_ontray)
        Needed_ontray_reg = max(0, N_reg_needed - Avail_reg_ontray)

        # Sandwiches available at kitchen (either existing or newly made)
        Available_at_kitchen_gf_total = Avail_gf_kitchen + Make_gf
        Available_at_kitchen_reg_total = Avail_reg_kitchen + Make_reg

        # Number of put_on_tray actions needed is the minimum of what's needed on trays
        # from the kitchen and what's available at the kitchen (including newly made)
        Put_gf = min(Available_at_kitchen_gf_total, Needed_ontray_gf)
        Put_reg = min(Available_at_kitchen_reg_total, Needed_ontray_reg)
        total_cost += Put_gf + Put_reg

        # 7. Calculate Tray Movements
        # Find locations of all trays
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}

        # Count distinct waiting locations that do not have a tray
        locations_needing_tray_move = unserved_waiting_locations - tray_locations
        total_cost += len(locations_needing_tray_move)

        # 8. Return the total calculated heuristic value.
        return total_cost
