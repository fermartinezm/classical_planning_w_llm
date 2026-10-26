from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to reach a goal state by summing
    up the estimated costs for making sandwiches, putting them on trays,
    moving trays to children's locations, and serving the children.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by extracting static information from the task.

        Heuristic Initialization:
        - Stores the task goals.
        - Identifies children who are allergic or not allergic to gluten
          from static facts.
        - Identifies bread and content types that are gluten-free from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread_types = set()
        self.no_gluten_content_types = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_bread_types.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_content_types.add(parts[1])

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Summary:
        The heuristic estimates the remaining cost by summing the number of
        unserved children (representing the final 'serve' action for each),
        the number of sandwiches that still need to be made, the number of
        sandwiches at the kitchen that need to be put on trays, and the
        number of sandwich-tray units that need to be moved to a child's
        location.

        Assumptions:
        - Sufficient bread, content, and 'notexist' sandwich objects are
          available at the kitchen to make any required sandwiches.
        - Sufficient trays are available to perform 'put_on_tray' and
          'move_tray' actions as needed by the heuristic calculation.
          (This is a simplification; the heuristic does not explicitly model
           tray availability or contention).
        - The 'waiting' predicate for a child remains true until they are served.

        Step-By-Step Thinking for Computing Heuristic:
        1.  **Identify State Components:** Parse the current state to find:
            - Children who are served.
            - Children who are waiting and their locations.
            - Sandwiches at the kitchen (regular and gluten-free).
            - Sandwiches on trays (regular and gluten-free) and the trays they are on.
            - Tray locations.
            - Sandwiches that do not exist ('notexist').
            - Sandwiches that are gluten-free ('no_gluten_sandwich').

        2.  **Identify Unserved Children:** Determine which children are waiting but not yet served. Count the total number of unserved children (`N_unserved`), and specifically the number of allergic (`N_allergic_unserved`) and non-allergic (`N_regular_unserved`) unserved children. If `N_unserved` is 0, the state is a goal state, and the heuristic is 0.

        3.  **Estimate 'serve' Cost:** Each unserved child requires a final 'serve' action. Add `N_unserved` to the total heuristic cost.

        4.  **Estimate 'make_sandwich' Cost:** Calculate how many gluten-free and regular sandwiches still need to be made to satisfy the demand from unserved children, considering sandwiches already made (at kitchen or on trays). Add the total number of sandwiches to make (`Needed_GF_make + Needed_Reg_make`) to the heuristic cost.

        5.  **Estimate 'put_on_tray' Cost:** Calculate how many sandwiches currently at the kitchen (either initially or just made) are needed to satisfy the demand not met by sandwiches already on trays. These sandwiches need a 'put_on_tray' action. Add this count (`Total_put_on_tray`) to the heuristic cost.

        6.  **Estimate 'move_tray' Cost:** Each unserved child needs a suitable sandwich delivered on a tray to their location. Count how many unserved children *do not* currently have a suitable sandwich on a tray already at their location (`Children_with_sandwich_at_loc`). Each such child represents a "delivery" that requires a tray movement (or a sequence of actions including movement). Add the number of children needing delivery (`max(0, N_unserved - Children_with_sandwich_at_loc)`) to the heuristic cost. This is a simplified count of necessary tray movements.

        7.  **Sum Costs:** The total heuristic value is the sum of the costs estimated in steps 3, 4, 5, and 6.
        """
        state = node.state

        # 1. Identify State Components
        served_children = set()
        waiting_children_locations = {} # {child: place}
        sandwiches_at_kitchen = set()
        gf_sandwiches = set() # Dynamic predicate (no_gluten_sandwich)
        sandwiches_on_trays = {} # {sandwich: tray}
        tray_locations = {} # {tray: place}
        # notexist_sandwiches = set() # Not directly used in this heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'waiting':
                waiting_children_locations[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich':
                sandwiches_at_kitchen.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                gf_sandwiches.add(parts[1])
            elif predicate == 'ontray':
                sandwiches_on_trays[parts[1]] = parts[2]
            elif predicate == 'at':
                 # Handle (at tray kitchen) and (at tray place)
                 if len(parts) == 3:
                    tray_locations[parts[1]] = parts[2]
            # elif predicate == 'notexist':
            #     notexist_sandwiches.add(parts[1])
            # elif predicate in ('at_kitchen_bread', 'at_kitchen_content'):
            #     pass # Not directly used for counts in this heuristic

        # 2. Identify Unserved Children
        unserved_children_locations = {
            c: p for c, p in waiting_children_locations.items()
            if c not in served_children
        }
        N_unserved = len(unserved_children_locations)

        # Heuristic is 0 if all children are served
        if N_unserved == 0:
            return 0

        # 3. Estimate 'serve' Cost
        heuristic_cost = N_unserved

        N_allergic_unserved = sum(1 for c in unserved_children_locations if c in self.allergic_children)
        N_regular_unserved = N_unserved - N_allergic_unserved

        # Classify sandwiches currently made (at kitchen or on trays)
        GF_on_trays = sum(1 for s in sandwiches_on_trays if s in gf_sandwiches)
        Reg_on_trays = len(sandwiches_on_trays) - GF_on_trays
        GF_at_kitchen = sum(1 for s in sandwiches_at_kitchen if s in gf_sandwiches)
        Reg_at_kitchen = len(sandwiches_at_kitchen) - GF_at_kitchen

        GF_made = GF_on_trays + GF_at_kitchen
        Reg_made = Reg_on_trays + Reg_at_kitchen

        # 4. Estimate 'make_sandwich' Cost
        # How many more GF/Reg sandwiches are needed in total?
        Needed_GF_total = N_allergic_unserved
        Needed_Reg_total = N_regular_unserved

        # How many need to be made?
        Needed_GF_make = max(0, Needed_GF_total - GF_made)
        Needed_Reg_make = max(0, Needed_Reg_total - Reg_made)
        heuristic_cost += Needed_GF_make + Needed_Reg_make

        # 5. Estimate 'put_on_tray' Cost
        # Total sandwiches at kitchen after making the needed ones
        Total_GF_at_kitchen_after_make = GF_at_kitchen + Needed_GF_make
        Total_Reg_at_kitchen_after_make = Reg_at_kitchen + Needed_Reg_make

        # How many GF/Reg sandwiches are needed from kitchen/making pool
        # (i.e., not already on trays)?
        Needed_GF_from_kitchen_after_make = max(0, Needed_GF_total - GF_on_trays)
        Needed_Reg_from_kitchen_after_make = max(0, Needed_Reg_total - Reg_on_trays)

        # Number of sandwiches taken from the kitchen pool (initial + made)
        # that need to be put on trays. Limited by supply and demand.
        Put_on_tray_GF = min(Total_GF_at_kitchen_after_make, Needed_GF_from_kitchen_after_make)
        Put_on_tray_Reg = min(Total_Reg_at_kitchen_after_make, Needed_Reg_from_kitchen_after_make)
        Total_put_on_tray = Put_on_tray_GF + Put_on_tray_Reg
        heuristic_cost += Total_put_on_tray

        # 6. Estimate 'move_tray' Cost
        # Count how many unserved children already have a suitable sandwich
        # on a tray at their location.
        Children_with_sandwich_at_loc = 0
        # Group unserved children by location
        unserved_by_location = {}
        for child, loc in unserved_children_locations.items():
            unserved_by_location.setdefault(loc, []).append(child)

        # Group sandwiches on trays by location and type
        sandwiches_on_trays_by_location = {} # {loc: {'gf': [s, ...], 'reg': [s, ...]}, ...}
        for s, t in sandwiches_on_trays.items():
            if t in tray_locations:
                loc = tray_locations[t]
                sandwich_type = 'gf' if s in gf_sandwiches else 'reg'
                sandwiches_on_trays_by_location.setdefault(loc, {'gf': [], 'reg': []})[sandwich_type].append(s)

        # For each location with unserved children, find the maximum matching
        # between children and suitable sandwiches on trays at that location.
        for loc, children_at_loc in unserved_by_location.items():
            allergic_at_loc = [c for c in children_at_loc if c in self.allergic_children]
            regular_at_loc = [c for c in children_at_loc if c in self.not_allergic_children]

            gf_sandwiches_at_loc = sandwiches_on_trays_by_location.get(loc, {}).get('gf', [])
            reg_sandwiches_at_loc = sandwiches_on_trays_by_location.get(loc, {}).get('reg', [])

            # Match allergic children with GF sandwiches
            matched_gf = min(len(allergic_at_loc), len(gf_sandwiches_at_loc))
            remaining_gf_sandwiches = len(gf_sandwiches_at_loc) - matched_gf

            # Match regular children with regular sandwiches + surplus GF sandwiches
            matched_reg = min(len(regular_at_loc), len(reg_sandwiches_at_loc) + remaining_gf_sandwiches)

            Children_with_sandwich_at_loc += matched_gf + matched_reg

        # Number of sandwich-tray units that need to be moved to a child's location
        # This is the number of unserved children whose needs are not met by
        # sandwiches already at their location.
        Sandwiches_to_move = max(0, N_unserved - Children_with_sandwich_at_loc)
        heuristic_cost += Sandwiches_to_move

        # 7. Sum Costs (already accumulated in heuristic_cost)

        return heuristic_cost

