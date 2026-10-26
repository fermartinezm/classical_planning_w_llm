from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or non-string inputs defensively
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove leading/trailing parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for four main stages, representing necessary transitions:
    1. Making necessary sandwiches (ingredients -> sandwich in kitchen).
    2. Putting necessary sandwiches onto trays (sandwich in kitchen -> sandwich on tray).
    3. Moving trays to the locations where children are waiting (tray at P1 -> tray at P2).
    4. Serving the children (sandwich on tray at P + waiting child -> served child).

    # Assumptions
    - Each unserved child requires one sandwich of the correct type (gluten-free for allergic, regular otherwise).
    - Sandwiches must be made in the kitchen, then put on a tray in the kitchen, then the tray must be moved to the child's location to serve.
    - Ingredients and 'notexist' sandwich objects are assumed sufficient to make any required sandwich type that is not already made. The heuristic counts *needed* makes based on the deficit of made sandwiches.
    - Trays can hold multiple sandwiches. Moving a tray counts as one action regardless of contents.
    - The heuristic counts the minimum number of actions of each type needed across the entire problem, summing them up. This is an additive heuristic.

    # Heuristic Initialization
    - Identify all children who need to be served (from task goals).
    - Store allergy status for each child (from static facts).
    - Store the waiting location for each child (from static facts).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are in the goal state but not yet served in the current state. Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (for the 'serve' actions).
    2. Categorize unserved children by allergy status to determine the total number of regular (`N_reg`) and gluten-free (`N_gf`) sandwiches required.
    3. Count the number of regular and gluten-free sandwiches currently available in the kitchen (`S_reg_kitchen`, `S_gf_kitchen`) and on trays anywhere (`S_reg_ontray`, `S_gf_ontray`).
    4. Calculate the number of sandwiches of each type that still need to be made: `To_make_reg = max(0, N_reg - (S_reg_kitchen + S_reg_ontray))` and `To_make_gf = max(0, N_gf - (S_gf_kitchen + S_gf_ontray))`. The cost for 'make' actions (`h_make`) is `To_make_reg + To_make_gf`.
    5. Calculate the number of sandwiches that need to be put on trays from the kitchen. This is the total number of sandwiches required (`N_reg + N_gf`) minus those already on trays (`S_reg_ontray + S_gf_ontray`), capped at zero. This is the *demand* for sandwiches to be put on trays: `Demand_put_on_tray = max(0, (N_reg + N_gf) - (S_reg_ontray + S_gf_ontray))`. The number of sandwiches available to be put on trays from the kitchen is the sum of those currently in the kitchen and those that will be made: `Supply_put_on_tray = S_reg_kitchen + S_gf_kitchen + To_make_reg + To_make_gf`. The cost for 'put_on_tray' actions (`h_put_on_tray`) is `min(Demand_put_on_tray, Supply_put_on_tray)`.
    6. Identify all places where unserved children are waiting (using static info and the list of unserved children).
    7. Identify all places where trays are currently located (from the state).
    8. Count the number of places identified in step 6 that do *not* have a tray present (identified in step 7). This is the number of 'move_tray' actions needed to get trays to delivery locations (`h_move_tray`).
    9. The total heuristic value is the sum of costs from steps 1, 4, 5, and 8: `h_serve + h_make + h_put_on_tray + h_move_tray`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal children (those who need to be served)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Store allergy status for children
        self.child_allergy = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["allergic_gluten", "not_allergic_gluten"]:
                child_name = parts[1]
                self.child_allergy[child_name] = parts[0] # Store "allergic_gluten" or "not_allergic_gluten"

        # Store waiting locations for children
        self.child_waiting_place = {}
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                child_name, place_name = get_parts(fact)[1:]
                self.child_waiting_place[child_name] = place_name

        # Store gluten-free status for bread and content (not strictly used in heuristic calculation logic, but kept for potential future refinement or understanding)
        # self.no_gluten_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        # self.no_gluten_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Count unserved children and categorize by allergy
        unserved_children = [c for c in self.goal_children if f"(served {c})" not in state]
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached

        N_reg = 0 # Children needing regular sandwiches
        N_gf = 0  # Children needing gluten-free sandwiches
        places_with_unserved_children = set()

        for child in unserved_children:
            allergy_status = self.child_allergy.get(child)
            if allergy_status == "allergic_gluten":
                N_gf += 1
            elif allergy_status == "not_allergic_gluten":
                N_reg += 1
            # Add the waiting place to the set
            place = self.child_waiting_place.get(child)
            if place: # Ensure place is known
                 places_with_unserved_children.add(place)


        # 2. Count available sandwiches
        S_reg_kitchen = 0
        S_gf_kitchen = 0
        S_reg_ontray = 0
        S_gf_ontray = 0

        # Track sandwiches that are GF
        gf_sandwiches_made = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s_name = get_parts(fact)[1]
                if s_name in gf_sandwiches_made:
                    S_gf_kitchen += 1
                else:
                    S_reg_kitchen += 1
            elif match(fact, "ontray", "*", "*"):
                 s_name = get_parts(fact)[1]
                 if s_name in gf_sandwiches_made:
                     S_gf_ontray += 1
                 else:
                     S_reg_ontray += 1

        S_reg_total = S_reg_kitchen + S_reg_ontray
        S_gf_total = S_gf_kitchen + S_gf_ontray

        # 3. Cost to make sandwiches
        # We need N_reg regular and N_gf gluten-free sandwiches in total.
        # Some might already exist (kitchen or ontray).
        To_make_reg = max(0, N_reg - S_reg_total)
        To_make_gf = max(0, N_gf - S_gf_total)
        h_make = To_make_reg + To_make_gf

        # 4. Cost to put sandwiches on trays
        # We need N_reg regular and N_gf gluten-free sandwiches on trays in total.
        # Some are already on trays (S_reg_ontray, S_gf_ontray).
        # The rest must be put on trays from the kitchen.
        Needed_on_trays_total = N_reg + N_gf
        Already_on_trays_total = S_reg_ontray + S_gf_ontray
        Demand_put_on_tray = max(0, Needed_on_trays_total - Already_on_trays_total)

        # Sandwiches available to be put on trays are those in the kitchen now
        # plus those we plan to make.
        Supply_put_on_tray = S_reg_kitchen + S_gf_kitchen + To_make_reg + To_make_gf

        # The number of put_on_tray actions is the minimum of the demand and the supply
        # of sandwiches that can reach the 'at_kitchen_sandwich' state.
        h_put_on_tray = min(Demand_put_on_tray, Supply_put_on_tray)


        # 5. Cost to move trays
        # Identify places where trays are currently located.
        places_with_trays = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, place_name = get_parts(fact)[1:]
                # Assuming objects starting with 'tray' are trays.
                if obj_name.startswith("tray"):
                     places_with_trays.add(place_name)

        # Count places with unserved children that don't have a tray
        Num_places_need_tray_move = 0
        for place in places_with_unserved_children:
            if place not in places_with_trays:
                Num_places_need_tray_move += 1

        h_move_tray = Num_places_need_tray_move

        # 6. Cost to serve children
        h_serve = N_unserved

        # Total heuristic is the sum of costs for each stage
        total_cost = h_make + h_put_on_tray + h_move_tray + h_serve

        return total_cost
