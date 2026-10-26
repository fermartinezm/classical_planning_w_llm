from fnmatch import fnmatch
# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle cases that are not valid PDDL fact strings
        return []
    # Use simple split as PDDL object names typically don't have spaces
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
    This heuristic estimates the number of actions required to serve all children
    by summing up the estimated costs for different stages of the process:
    serving children, making needed sandwiches, putting kitchen sandwiches on trays,
    and moving trays to child locations. It also checks for unsolvable states
    based on available ingredients and sandwich objects.

    # Assumptions
    - All children in the goal must be served.
    - Resources (bread, content, sandwich objects, trays) are initially sufficient
      in the problem instance to make all required sandwiches and deliveries,
      unless the current state explicitly shows a lack of resources in the kitchen
      or available sandwich objects. The heuristic checks the *current* state's
      available resources in the kitchen.
    - Tray capacity is sufficient for needed sandwiches at a location (or at least one tray move can bring enough).
    - Actions have a cost of 1.
    - The set of all sandwich objects is static and can be inferred from initial state facts.
    - The kitchen location is always named 'kitchen'.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the goal state.
    - Extracts static facts like gluten allergies.
    - Infers the total number of sandwich objects from the initial state facts.
    - Stores the name of the kitchen location.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for several steps:

    1.  **Serving Children:** Count the number of children who are not yet served (`N_unserved`). Each unserved child requires a 'serve' action. This contributes `N_unserved` to the heuristic.

    2.  **Making Sandwiches:** Determine the number of gluten-free (`make_gf`) and regular (`make_reg`) sandwiches that still need to be made to satisfy the needs of unserved allergic and non-allergic children, considering sandwiches already available (in kitchen or on trays). Check if enough ingredients (bread, content) and 'notexist' sandwich objects are available in the kitchen to make these. If not, the state is likely unsolvable, and the heuristic returns infinity. The number of sandwiches to make contributes `make_gf + make_reg` to the heuristic.

    3.  **Putting Kitchen Sandwiches on Trays:** Calculate the number of sandwiches currently in the kitchen that are needed (`put_kitchen_stock_count`). These, along with newly made sandwiches, must be put on trays. The total number of 'put' actions needed is `total_put_actions_needed = (make_gf + make_reg) + put_kitchen_stock_count`. If `total_put_actions_needed > 0` and no trays are currently in the kitchen, one action is added for moving a tray to the kitchen. The cost is `total_put_actions_needed` plus potentially 1 for a tray move to the kitchen. This contributes `put_cost` to the heuristic.

    4.  **Moving Trays:** For each location (excluding the kitchen) where unserved children are waiting, check if the number of sandwiches currently on trays at that location is less than the number of unserved children waiting there. If it is, at least one tray move is required to bring more sandwiches to that location. Count the number of such distinct locations (`move_locations_count`). This contributes `move_locations_count` to the heuristic.

    The total heuristic is the sum of these four components. If the state is a goal state (all children served), the heuristic is 0. If the state is determined to be unsolvable due to missing ingredients/objects, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract children who need to be served from the goal
        self.children_to_serve = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }

        # Extract children with gluten allergies from static facts
        self.allergic_children_static = {
            get_parts(fact)[1]
            for fact in self.static_facts
            if match(fact, "allergic_gluten", "*")
        }

        # Infer the total number of sandwich objects from initial state facts.
        # This assumes all sandwich objects are mentioned in the initial state facts
        # using predicates like at_kitchen_sandwich, ontray, no_gluten_sandwich, or notexist.
        # This is a pragmatic assumption based on typical PDDL structure and example files.
        initial_sandwich_objects = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1 and parts[0] in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist']:
                  # The first argument is a sandwich object in these predicates
                  sandwich_obj = parts[1]
                  initial_sandwich_objects.add(sandwich_obj)
        self.total_sandwich_objects = len(initial_sandwich_objects)

        self.kitchen_place = 'kitchen' # Assuming 'kitchen' is always the name for the kitchen place


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state facts ---
        served_children = set()
        waiting_children = {} # {location: set(children)}
        # allergic_children_state = set() # Allergic status is static, use self.allergic_children_static
        at_kitchen_bread = set()
        at_kitchen_content = set()
        at_kitchen_sandwiches = set()
        ontray_sandwiches = {} # {tray: set(sandwiches)}
        tray_location = {} # {tray: location}
        no_gluten_objects_state = set() # Objects currently marked as no_gluten
        notexist_sandwiches_state = set() # Sandwiches currently marked as notexist
        trays_in_state = set() # Keep track of all trays

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'served' and len(args) == 1:
                served_children.add(args[0])
            elif predicate == 'waiting' and len(args) == 2:
                child, place = args
                if place not in waiting_children:
                    waiting_children[place] = set()
                waiting_children[place].add(child)
            # elif predicate == 'allergic_gluten' and len(args) == 1:
            #      allergic_children_state.add(args[0]) # Allergic status is static
            elif predicate == 'at_kitchen_bread' and len(args) == 1:
                at_kitchen_bread.add(args[0])
            elif predicate == 'at_kitchen_content' and len(args) == 1:
                at_kitchen_content.add(args[0])
            elif predicate == 'at_kitchen_sandwich' and len(args) == 1:
                at_kitchen_sandwiches.add(args[0])
            elif predicate == 'ontray' and len(args) == 2:
                sandwich, tray = args
                if tray not in ontray_sandwiches:
                    ontray_sandwiches[tray] = set()
                ontray_sandwiches[tray].add(sandwich)
                trays_in_state.add(tray)
            elif predicate == 'at' and len(args) == 2:
                obj, place = args
                # Assuming only trays can be 'at' places other than kitchen initially
                # and the object is a tray. The domain confirms (at ?t - tray ?p - place).
                tray_location[obj] = place
                trays_in_state.add(obj)
            elif predicate == 'no_gluten_bread' and len(args) == 1:
                 no_gluten_objects_state.add(args[0])
            elif predicate == 'no_gluten_content' and len(args) == 1:
                 no_gluten_objects_state.add(args[0])
            elif predicate == 'no_gluten_sandwich' and len(args) == 1:
                 no_gluten_objects_state.add(args[0])
            elif predicate == 'notexist' and len(args) == 1:
                 notexist_sandwiches_state.add(args[0])


        # Identify unserved children from the goal list
        unserved_children = self.children_to_serve - served_children

        # If all children are served, the goal is reached
        if not unserved_children:
            return 0

        # --- Heuristic Calculation ---
        h = 0

        # 1. Cost for serve actions
        N_unserved = len(unserved_children)
        h += N_unserved

        # Count unserved children by allergy status
        unserved_allergic = unserved_children.intersection(self.allergic_children_static)
        unserved_not_allergic = unserved_children - unserved_allergic

        N_allergic_unserved = len(unserved_allergic)
        N_not_allergic_unserved = len(unserved_not_allergic)

        # Count needed sandwiches
        N_gf_needed = N_allergic_unserved
        N_reg_needed = N_not_allergic_unserved # These can be regular or GF

        # Count available sandwiches by type and location (kitchen vs ontray)
        N_gf_kitchen = len({s for s in at_kitchen_sandwiches if s in no_gluten_objects_state})
        N_reg_kitchen = len({s for s in at_kitchen_sandwiches if s not in no_gluten_objects_state})

        N_gf_ontray = len({s for tray, sandwiches in ontray_sandwiches.items() for s in sandwiches if s in no_gluten_objects_state})
        N_reg_ontray = len({s for tray, sandwiches in ontray_sandwiches.items() for s in sandwiches if s not in no_gluten_objects_state})

        N_ontray = N_gf_ontray + N_reg_ontray

        N_gf_avail = N_gf_kitchen + N_gf_ontray
        N_reg_avail = N_reg_kitchen + N_reg_ontray
        # N_total_avail = N_gf_avail + N_reg_avail # Not directly used in calculation below

        # Sandwiches that still need to be made
        make_gf = max(0, N_gf_needed - N_gf_avail)
        # Regular sandwiches needed after using available regular and excess available GF
        make_reg = max(0, N_reg_needed - (N_reg_avail + max(0, N_gf_avail - N_gf_needed)))

        # 2. Cost for make actions
        h += make_gf + make_reg

        # Check if making is possible (ingredients and notexist objects)
        N_gf_bread_avail = len({b for b in at_kitchen_bread if b in no_gluten_objects_state})
        N_reg_bread_avail = len({b for b in at_kitchen_bread if b not in no_gluten_objects_state})
        N_gf_content_avail = len({c for c in at_kitchen_content if c in no_gluten_objects_state})
        N_reg_content_avail = len({c for c in at_kitchen_content if c not in no_gluten_objects_state})
        N_notexist = len(notexist_sandwiches_state)

        # Check if enough GF ingredients/objects exist to make needed GF sandwiches
        if make_gf > N_gf_bread_avail or make_gf > N_gf_content_avail or make_gf > N_notexist:
             # Cannot make enough GF sandwiches
             return float('inf')

        # Check if enough ingredients/objects exist to make needed Regular sandwiches
        # Regular sandwiches can use regular ingredients or GF ingredients not needed for GF sandwiches
        remaining_notexist = N_notexist - make_gf
        remaining_gf_bread = N_gf_bread_avail - make_gf
        remaining_gf_content = N_gf_content_avail - make_gf

        total_bread_for_reg = N_reg_bread_avail + remaining_gf_bread
        total_content_for_reg = N_reg_content_avail + remaining_gf_content

        if make_reg > total_bread_for_reg or make_reg > total_content_for_reg or make_reg > remaining_notexist:
             # Cannot make enough Regular sandwiches
             return float('inf')


        # 3. Cost for put_on_tray actions (for sandwiches currently in kitchen or newly made)
        # Sandwiches that are made need a put action: make_gf + make_reg
        # Sandwiches initially in kitchen that are needed also need a put action.
        # Number of sandwiches initially in kitchen that must be used:
        # Total sandwiches needed = N_gf_needed + N_reg_needed
        # Sandwiches already on trays = N_ontray
        # Sandwiches that will be made = make_gf + make_reg
        # Sandwiches from kitchen stock that must be used = max(0, (N_gf_needed + N_reg_needed) - N_ontray - (make_gf + make_reg))
        put_kitchen_stock_count = max(0, (N_gf_needed + N_reg_needed) - N_ontray - (make_gf + make_reg))

        total_put_actions_needed = (make_gf + make_reg) + put_kitchen_stock_count

        put_cost = 0
        if total_put_actions_needed > 0:
             # To perform a put_on_tray action, a tray must be in the kitchen.
             # Count trays currently in the kitchen.
             N_trays_kitchen = len({t for t in trays_in_state if tray_location.get(t) == self.kitchen_place})

             put_cost = total_put_actions_needed
             if N_trays_kitchen == 0:
                 # If we need to put sandwiches on trays but no tray is in the kitchen,
                 # we need one action to move a tray to the kitchen first.
                 # This assumes there is at least one tray available somewhere to move.
                 # The problem should be solvable if there's at least one tray total.
                 # We don't check if a tray exists at all, assuming solvable problems.
                 put_cost += 1 # Cost for one tray move to kitchen

        h += put_cost


        # 4. Cost for move_tray actions to child locations
        # Count sandwiches on trays at each location
        sandwiches_on_trays_at = {} # {location: count}
        for tray, sandwiches in ontray_sandwiches.items():
            loc = tray_location.get(tray)
            if loc: # Tray must have a location
                if loc not in sandwiches_on_trays_at:
                    sandwiches_on_trays_at[loc] = 0
                sandwiches_on_trays_at[loc] += len(sandwiches)

        # Count unserved children at each location (excluding kitchen)
        unserved_at_loc = {} # {location: count}
        locations_with_unserved = set()
        for loc, children in waiting_children.items():
            if loc != self.kitchen_place:
                unserved_at_loc[loc] = len(children.intersection(unserved_children))
                if unserved_at_loc[loc] > 0:
                    locations_with_unserved.add(loc)

        # Count locations where more sandwiches are needed on trays than are currently there
        move_locations_count = 0
        for loc in locations_with_unserved:
            needed_at_loc = unserved_at_loc.get(loc, 0)
            available_at_loc = sandwiches_on_trays_at.get(loc, 0)
            if needed_at_loc > available_at_loc:
                move_locations_count += 1 # At least one tray move is needed to this location

        h += move_locations_count

        return h
