from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children.
    It sums up the estimated costs for four main stages for the remaining unserved children:
    1. Making necessary sandwiches (considering gluten requirements).
    2. Putting necessary sandwiches onto trays.
    3. Moving trays to the locations where children are waiting.
    4. Serving the children.

    # Assumptions
    - Sufficient bread and content are available in the kitchen to make any needed sandwiches.
    - Sufficient trays are available somewhere to be moved to locations needing service.
    - A single tray move to a location can satisfy the location requirement for all children waiting there.
    - Any sandwich on a tray can be moved to any location. Suitability (GF vs regular) is handled at the 'make' and 'serve' stages.
    - All children in the problem instance have their allergy status defined in the static facts.
    - The 'at' predicate in the state only applies to trays and places.

    # Heuristic Initialization
    - Extracts the list of all children involved in the task from the goals and static facts.
    - Stores the allergy status (allergic_gluten or not_allergic_gluten) for each child from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify which children are not yet served by comparing the state facts to the goal facts.
    2. Count the total number of unserved children (`num_unserved_total`). If 0, the heuristic is 0 (goal state).
    3. Separate unserved children into allergic (`num_unserved_allergic`) and non-allergic (`num_unserved_non_allergic`) based on pre-calculated allergy status.
    4. Identify the set of unique locations where unserved children are waiting by examining the `waiting` facts in the state. Store these in `locations_needing_service`.
    5. Count available sandwiches by examining the state facts:
       - `sandwiches_kitchen`: Set of sandwiches currently `at_kitchen_sandwich`.
       - `sandwiches_ontray`: Set of sandwiches currently `ontray`.
       - `gf_sandwiches_anywhere`: Set of sandwiches that are `no_gluten_sandwich`.
       - Calculate `Avail_Any_S_anywhere` (total existing sandwiches), `Avail_GF_S_anywhere` (total existing GF sandwiches), and `Avail_Reg_S_anywhere` (total existing regular sandwiches).
       - Calculate `On_Tray_S` (total sandwiches currently on trays).
    6. Count trays at each location by examining the `at` facts in the state for trays. Store counts in `Trays_at_Location_Count`.
    7. Calculate the 'make sandwich' cost (`make_cost`):
       - Estimate the number of GF sandwiches that still need to be made: `max(0, num_unserved_allergic - Avail_GF_S_anywhere)`.
       - Estimate the number of regular sandwiches that still need to be made for non-allergic children. This considers that non-allergic children can use any remaining GF sandwiches after allergic children's needs are accounted for, plus any available regular sandwiches. Calculate the deficit of sandwiches needed by non-allergic children compared to what's available for them.
       - Sum the estimated GF and regular sandwiches to make.
    8. Calculate the 'put on tray' cost (`put_cost`):
       - Estimate the number of sandwiches that need to be put on trays: `max(0, num_unserved_total - On_Tray_S)`. This is the total number of sandwiches required on trays minus those already there.
    9. Calculate the 'move tray' cost (`move_cost`):
       - Count the number of locations in `locations_needing_service` that currently have no tray present (`at ?t ?p` is false for all trays `?t` at that location `?p`).
    10. Calculate the 'serve' cost (`serve_cost`):
        - This is simply the number of unserved children (`num_unserved_total`), as each requires one serve action.
    11. The total heuristic value is the sum of the `make_cost`, `put_cost`, `move_cost`, and `serve_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The set of all children in the problem.
        - The allergy status for each child.
        """
        self.goals = task.goals
        self.static = task.static

        self.all_children = set()
        self.allergy_status = {} # child -> bool (True for allergic)

        # Extract all children from goals and static facts
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served" and len(parts) == 2:
                self.all_children.add(parts[1])

        for fact in self.static:
            parts = get_parts(fact)
            if parts and len(parts) == 2:
                predicate, child = parts
                if predicate == "allergic_gluten":
                    self.allergy_status[child] = True
                    self.all_children.add(child)
                elif predicate == "not_allergic_gluten":
                    self.allergy_status[child] = False
                    self.all_children.add(child)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = [c for c in self.all_children if c not in served_children]
        num_unserved_total = len(unserved_children)

        # 2. If all children are served, goal reached
        if num_unserved_total == 0:
            return 0

        # 3. Separate unserved children by allergy status
        num_unserved_allergic = sum(1 for c in unserved_children if self.allergy_status.get(c, False))
        num_unserved_non_allergic = num_unserved_total - num_unserved_allergic

        # 4. Identify locations needing service
        waiting_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*")}
        # Filter to include only locations for children who are still unserved
        locations_needing_service = {waiting_locations[c] for c in unserved_children if c in waiting_locations}

        # 5. Count available sandwiches and trays
        sandwiches_kitchen = set()
        sandwiches_ontray = set()
        gf_sandwiches_anywhere = set()
        tray_location_map = {} # tray -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwiches_kitchen.add(parts[1])
            elif predicate == "ontray" and len(parts) == 3:
                sandwiches_ontray.add(parts[1])
            elif predicate == "at" and len(parts) == 3:
                 # Assuming 'at' predicate only applies to trays and places based on domain.
                 tray_location_map[parts[1]] = parts[2]
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                gf_sandwiches_anywhere.add(parts[1])

        Avail_Any_S_anywhere = len(sandwiches_kitchen) + len(sandwiches_ontray)
        Avail_GF_S_anywhere = len([s for s in sandwiches_kitchen | sandwiches_ontray if s in gf_sandwiches_anywhere])
        Avail_Reg_S_anywhere = Avail_Any_S_anywhere - Avail_GF_S_anywhere # Regular sandwiches available

        On_Tray_S = len(sandwiches_ontray)

        Trays_at_Location_Count = {}
        for loc in tray_location_map.values():
            Trays_at_Location_Count[loc] = Trays_at_Location_Count.get(loc, 0) + 1

        # 6. Calculate heuristic components

        # Cost 1: Make sandwich
        # GF sandwiches needed for allergic children
        make_gf = max(0, num_unserved_allergic - Avail_GF_S_anywhere)
        # GF sandwiches available for non-allergic after serving allergic
        gf_available_for_non_allergic = max(0, Avail_GF_S_anywhere - num_unserved_allergic)
        # Total sandwiches available for non-allergic (remaining GF + regular)
        total_available_for_non_allergic = gf_available_for_non_allergic + Avail_Reg_S_anywhere
        # Regular sandwiches needed for non-allergic children
        make_reg = max(0, num_unserved_non_allergic - total_available_for_non_allergic)
        make_cost = make_gf + make_reg

        # Cost 2: Put on tray
        # Total sandwiches needed on trays is equal to the number of unserved children
        put_cost = max(0, num_unserved_total - On_Tray_S)

        # Cost 3: Move tray
        # Count locations needing a tray where none is present
        Num_Locations_Needing_Tray = sum(1 for p in locations_needing_service if Trays_at_Location_Count.get(p, 0) == 0)
        move_cost = Num_Locations_Needing_Tray

        # Cost 4: Serve
        # Each unserved child needs one serve action
        serve_cost = num_unserved_total

        # Total heuristic is the sum of costs for each stage
        total_cost = make_cost + put_cost + move_cost + serve_cost

        return total_cost
