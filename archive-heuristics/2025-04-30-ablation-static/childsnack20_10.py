from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions required to serve all waiting children.
    It sums the estimated costs for:
    1. Making necessary sandwiches (considering allergy requirements and available sandwiches).
    2. Putting necessary sandwiches onto trays.
    3. Moving trays to locations where children are waiting.
    4. Serving the children.

    # Assumptions
    - Each unserved child requires one suitable sandwich to be served.
    - Each sandwich must be made (if not already), put on a tray, and the tray must be at the child's location for serving.
    - Trays can be reused and moved between locations.
    - Gluten-free sandwiches can satisfy the needs of both allergic and non-allergic children.
    - Regular sandwiches can only satisfy the needs of non-allergic children.
    - Resource availability (bread, content, notexist sandwich objects) is not strictly checked beyond counting needed 'make' actions; it's assumed solvable if a path exists.

    # Heuristic Initialization
    - Identify all children who need to be served from the goal state.
    - Store the allergy status (allergic_gluten or not_allergic_gluten) for each child from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are currently 'served'.
    2. Identify all children who are 'waiting' but not 'served'. Count the total number of unserved children (`N_waiting`).
    3. Separate unserved children into those who are 'allergic_gluten' (`N_waiting_allergic`) and those who are 'not_allergic_gluten' (`N_waiting_not_allergic`).
    4. Count the number of sandwiches currently 'ontray' (`N_ontray`).
    5. Count the number of sandwiches currently 'at_kitchen_sandwich' (`N_kitchen_sandwich`).
    6. Count the number of gluten-free sandwiches currently 'ontray' (`N_gf_ontray`).
    7. Count the number of gluten-free sandwiches currently 'at_kitchen_sandwich' (`N_gf_kitchen`).
    8. Calculate the total number of available gluten-free sandwiches (`N_gf_available = N_gf_ontray + N_gf_kitchen`).
    9. Calculate the total number of available regular sandwiches (`N_reg_available = (N_ontray - N_gf_ontray) + (N_kitchen_sandwich - N_gf_kitchen)`).
    10. Estimate the number of sandwiches that still need to be 'made':
        - Allergic children *must* have GF sandwiches. Calculate the deficit: `Needed_make_gf = max(0, N_waiting_allergic - N_gf_available)`.
        - Any surplus GF sandwiches (`Remaining_gf = max(0, N_gf_available - N_waiting_allergic)`) can be used for non-allergic children.
        - Non-allergic children need regular or surplus GF sandwiches. Calculate the deficit: `Needed_make_reg = max(0, N_waiting_not_allergic - (N_reg_available + Remaining_gf))`.
        - The total estimated 'make' actions is `Cost_make = Needed_make_gf + Needed_make_reg`.
    11. Estimate the number of sandwiches that need to be 'put_on_tray':
        - We need `N_waiting` sandwiches on trays eventually. `N_ontray` are already there.
        - The estimated 'put_on_tray' actions is `Cost_put_on_tray = max(0, N_waiting - N_ontray)`.
    12. Estimate the number of 'move_tray' actions:
        - Identify the set of places where unserved children are waiting (`Needed_places`).
        - Identify the set of places where trays are currently located (`Tray_places`).
        - The estimated 'move_tray' actions is the number of places in `Needed_places` that are not in `Tray_places`: `Cost_move_tray = len(needed_places - tray_places)`.
    13. Estimate the number of 'serve' actions:
        - Each unserved child needs one 'serve' action.
        - The estimated 'serve' actions is `Cost_serve = N_waiting`.
    14. The total heuristic value is the sum of the estimated costs for these four action types: `Cost_make + Cost_put_on_tray + Cost_move_tray + Cost_serve`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child allergy status.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all children from the goals
        self.children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served" and len(parts) == 2:
                self.children.add(parts[1])

        # Store allergy status for each child
        self.allergy_status = {} # child_name -> True if allergic, False otherwise
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and len(parts) == 2:
                predicate, obj = parts # obj is child for allergy predicates
                if predicate == "allergic_gluten":
                    self.allergy_status[obj] = True
                elif predicate == "not_allergic_gluten":
                    self.allergy_status[obj] = False

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        served_children = set()
        waiting_children_at_place = {} # {child: place}
        waiting_children_allergic = set()
        waiting_children_not_allergic = set()
        needed_places = set() # Places where unserved children are waiting

        ontray_sandwiches = set()
        at_kitchen_sandwiches = set()
        gf_sandwiches = set() # Sandwiches that are no_gluten_sandwich

        tray_locations = {} # {tray: place}

        # --- Step 1: Extract information from the current state ---
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "served" and len(parts) == 2:
                served_children.add(parts[1])
            elif predicate == "waiting" and len(parts) == 3:
                 child, place = parts[1], parts[2]
                 # Only consider children who are not yet served and are part of the goal
                 if child in self.children and child not in served_children:
                      waiting_children_at_place[child] = place
                      needed_places.add(place)
                      # Get allergy status from pre-calculated dictionary
                      if self.allergy_status.get(child, False): # Default to False if status unknown (shouldn't happen with valid PDDL)
                          waiting_children_allergic.add(child)
                      else:
                          waiting_children_not_allergic.add(child)
            elif predicate == "ontray" and len(parts) == 3:
                 sandwich, tray = parts[1], parts[2]
                 ontray_sandwiches.add(sandwich)
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                 sandwich = parts[1]
                 at_kitchen_sandwiches.add(sandwich)
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                 sandwich = parts[1]
                 gf_sandwiches.add(sandwich)
            elif predicate == "at" and len(parts) == 3:
                 obj, place = parts[1], parts[2]
                 # Check if the object is a tray (simple check based on name convention)
                 # A more robust check would use object types from task if available, but assuming 'tray' prefix is sufficient here.
                 if obj.startswith('tray'):
                     tray_locations[obj] = place

        # --- Step 2: Calculate heuristic components ---

        # Number of unserved children
        N_waiting = len(waiting_children_at_place)

        # If all children are served, the goal is reached.
        if N_waiting == 0:
            return 0

        N_waiting_allergic = len(waiting_children_allergic)
        N_waiting_not_allergic = len(waiting_children_not_allergic)

        N_ontray = len(ontray_sandwiches)
        N_kitchen_sandwich = len(at_kitchen_sandwiches)

        N_gf_ontray = len(ontray_sandwiches.intersection(gf_sandwiches))
        N_gf_kitchen = len(at_kitchen_sandwiches.intersection(gf_sandwiches))

        N_gf_available = N_gf_ontray + N_gf_kitchen
        N_reg_available = (N_ontray - N_gf_ontray) + (N_kitchen_sandwich - N_gf_kitchen)

        # Cost for making sandwiches
        # Need N_waiting_allergic GF sandwiches. Have N_gf_available.
        Needed_make_gf = max(0, N_waiting_allergic - N_gf_available)
        # Surplus GF sandwiches can serve non-allergic children
        Remaining_gf = max(0, N_gf_available - N_waiting_allergic)
        # Need N_waiting_not_allergic sandwiches (Reg or GF). Have N_reg_available + Remaining_gf.
        Needed_make_reg = max(0, N_waiting_not_allergic - (N_reg_available + Remaining_gf))
        Cost_make = Needed_make_gf + Needed_make_reg

        # Cost for putting sandwiches on trays
        # Need N_waiting sandwiches on trays eventually. N_ontray are already there.
        Cost_put_on_tray = max(0, N_waiting - N_ontray)

        # Cost for moving trays
        tray_places = set(tray_locations.values())
        Cost_move_tray = len(needed_places - tray_places)

        # Cost for serving children
        Cost_serve = N_waiting

        # Total heuristic value
        total_cost = Cost_make + Cost_put_on_tray + Cost_move_tray + Cost_serve

        return total_cost
