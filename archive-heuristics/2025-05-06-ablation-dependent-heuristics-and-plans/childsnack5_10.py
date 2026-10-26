from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

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
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the necessary actions for making sandwiches, putting them on trays,
    moving trays to children's locations, and serving the children. It aggregates
    these costs across all unserved children, considering shared resources like
    sandwich ingredients, sandwich objects, and trays.

    # Assumptions
    - Each unserved child requires one sandwich.
    - Allergic children require gluten-free sandwiches. Non-allergic children can take any sandwich.
    - Sandwiches are made in the kitchen.
    - Sandwiches must be put on a tray in the kitchen before being moved to other locations.
    - Trays can hold multiple sandwiches (infinite capacity assumed for heuristic).
    - A tray trip is needed from the kitchen to any location (outside the kitchen)
      that requires at least one sandwich delivery.
    - Actions have a cost of 1.
    - The problem is solvable (resource checks return infinity if clearly not).

    # Heuristic Initialization
    - Extracts static information: which children are allergic/not allergic,
      which bread/content items are gluten-free.
    - Identifies all children, trays, and places from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children and group them by their waiting location.
       Count total unserved children and total allergic unserved children.
    2. Count available resources in the current state: bread, content (including GF),
       available sandwich objects (`notexist`), sandwiches already made (in kitchen or on trays),
       and trays (especially those in the kitchen).
    3. Calculate the number of sandwiches that still need to be made (`make_any`).
       This is the total number of unserved children minus the number of sandwiches
       already made and available (in kitchen or on trays).
    4. Calculate the number of gluten-free sandwiches that still need to be made (`make_gf`).
       This is the number of allergic unserved children minus the number of GF sandwiches
       already made and available (in kitchen or on trays).
    5. Check if there are enough ingredients (bread, content, GF bread, GF content)
       and sandwich objects (`notexist`) to make the required sandwiches (`make_any` total, `make_gf` GF).
       If not, the state is likely a dead end, return infinity.
    6. The cost of making sandwiches is `make_any`.
    7. Calculate the number of sandwiches that need to be put on trays. This includes
       sandwiches currently `at_kitchen_sandwich` plus the `make_any` sandwiches just made.
       The cost is this total number.
    8. Calculate the cost of moving trays. Identify all locations (outside the kitchen)
       where unserved children are waiting and where there aren't already enough
       sandwiches on trays. Each such location requires at least one tray trip from the kitchen.
       The cost is the number of such distinct locations.
    9. Calculate the cost of serving. Each unserved child needs one `serve` action.
       The cost is the total number of unserved children.
    10. Calculate the cost of moving a tray to the kitchen if needed. If any sandwiches
        need to be put on trays (step 7) or any tray trips need to start from the kitchen (step 8),
        and there are no trays currently in the kitchen, one tray must be moved there first.
        Add 1 to the cost in this case.
    11. Sum the costs from steps 6, 7, 8, 9, and 10 to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and object lists."""
        self.goals = task.goals
        self.static = task.static

        # Extract object lists from task.facts
        self.all_children = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_sandwiches = set() # All potential sandwich objects

        # task.facts contains all possible ground facts, including type definitions
        for fact in task.facts:
            parts = get_parts(fact)
            if len(parts) == 2: # Facts like (type object)
                obj_type, obj_name = parts
                if obj_type == 'child':
                    self.all_children.add(obj_name)
                elif obj_type == 'tray':
                    self.all_trays.add(obj_name)
                elif obj_type == 'place':
                    self.all_places.add(obj_name)
                elif obj_type == 'bread-portion':
                    self.all_bread.add(obj_name)
                elif obj_type == 'content-portion':
                    self.all_content.add(obj_name)
                elif obj_type == 'sandwich':
                    self.all_sandwiches.add(obj_name)

        # Extract static properties
        self.allergic_children = {get_parts(f)[1] for f in self.static if match(f, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(f)[1] for f in self.static if match(f, "not_allergic_gluten", "*")}
        self.no_gluten_bread_objects = {get_parts(f)[1] for f in self.static if match(f, "no_gluten_bread", "*")}
        self.no_gluten_content_objects = {get_parts(f)[1] for f in self.static if match(f, "no_gluten_content", "*")}

        # Ensure kitchen is in places (it's a constant)
        self.all_places.add('kitchen')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs
        unserved_children = {c for c in self.all_children if f'(served {c})' not in state}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached

        unserved_at_place = {p: set() for p in self.all_places}
        allergic_unserved = {c for c in unserved_children if c in self.allergic_children}
        N_allergic_unserved = len(allergic_unserved)
        # N_non_allergic_unserved = N_unserved - N_allergic_unserved # Not strictly needed for counts below

        for c in unserved_children:
            # Find the place where the child is waiting
            waiting_fact = next((f for f in state if match(f, "waiting", c, "*")), None)
            if waiting_fact:
                 place = get_parts(waiting_fact)[2]
                 unserved_at_place[place].add(c)
            # else: child is not waiting? Assume valid problem state.

        N_unserved_at_p = {p: len(children) for p, children in unserved_at_place.items()}
        # N_allergic_unserved_at_p = {p: len({c for c in children if c in self.allergic_children}) for p, children in unserved_at_place.items()} # Not strictly needed for counts below


        # 2. Count available resources in current state
        at_kitchen_bread = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        at_kitchen_content = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}
        N_bread_kitchen = len(at_kitchen_bread)
        N_content_kitchen = len(at_kitchen_content)
        N_gf_bread_kitchen = len({b for b in at_kitchen_bread if b in self.no_gluten_bread_objects})
        N_gf_content_kitchen = len({c for c in at_kitchen_content if c in self.no_gluten_content_objects})
        N_sandwich_objects_available = sum(1 for f in state if match(f, "notexist", "*"))

        at_kitchen_sandwich = {get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "*")}
        N_sandwich_kitchen = len(at_kitchen_sandwich)
        N_gf_sandwich_kitchen = len({s for s in at_kitchen_sandwich if f'(no_gluten_sandwich {s})' in state})

        ontray_sandwiches = {get_parts(f)[1]: get_parts(f)[2] for f in state if match(f, "ontray", "*", "*")} # Map sandwich to tray
        N_ontray = len(ontray_sandwiches)
        N_gf_ontray = len({s for s in ontray_sandwiches if f'(no_gluten_sandwich {s})' in state})

        tray_locations = {get_parts(f)[1]: get_parts(f)[2] for f in state if match(f, "at", "*", "*") and get_parts(f)[1] in self.all_trays} # Map tray to place
        N_trays_kitchen = sum(1 for t, p in tray_locations.items() if p == 'kitchen')

        # Count sandwiches available on trays at each location
        ontray_at_place = {p: set() for p in self.all_places} # Set of sandwiches
        gf_ontray_at_place = {p: set() for p in self.all_places} # Set of GF sandwiches
        for s, t in ontray_sandwiches.items():
            p = tray_locations.get(t)
            if p: # Tray might not have a location if problem is malformed, or if tray is carried by robot (not in this domain)
                ontray_at_place[p].add(s)
                if f'(no_gluten_sandwich {s})' in state:
                    gf_ontray_at_place[p].add(s)

        N_ontray_at_p = {p: len(sandwiches) for p, sandwiches in ontray_at_place.items()}
        # N_gf_ontray_at_p = {p: len(sandwiches) for p, sandwiches in gf_ontray_at_place.items()} # Not strictly needed for counts below


        # 3. Calculate sandwich deficits and making cost
        Available_GF_sandwiches_total = N_gf_sandwich_kitchen + N_gf_ontray
        make_gf = max(0, N_allergic_unserved - Available_GF_sandwiches_total)

        Available_total_sandwiches = N_sandwich_kitchen + N_ontray
        make_any = max(0, N_unserved - Available_total_sandwiches)

        # Feasibility check for making:
        # Need enough GF ingredients for GF sandwiches
        if N_gf_bread_kitchen < make_gf or N_gf_content_kitchen < make_gf:
             return float('inf')
        # Need enough total ingredients for all sandwiches (GF ingredients can be used for regular)
        if N_bread_kitchen < make_any or N_content_kitchen < make_any:
             return float('inf')
        # Need enough sandwich objects
        if N_sandwich_objects_available < make_any:
             return float('inf')

        cost_make = make_any

        # 4. Calculate put-on-tray cost
        # Sandwiches currently at_kitchen_sandwich that need to be put on trays: N_sandwich_kitchen
        # Sandwiches just made that need to be put on trays: make_any
        # Total sandwiches needing put_on_tray action (if not already on tray):
        sandwiches_to_put_on_tray = N_sandwich_kitchen + make_any
        cost_put_on_tray = sandwiches_to_put_on_tray


        # 5. Calculate tray movement cost
        # Identify places p != kitchen that need sandwiches delivered from the kitchen.
        places_needing_delivery = set()
        for p in self.all_places:
            if p != 'kitchen':
                needed_at_p = N_unserved_at_p.get(p, 0)
                available_ontray_at_p = N_ontray_at_p.get(p, 0)
                # We need to deliver max(0, needed_at_p - available_ontray_at_p) sandwiches to place p.
                # If this number is > 0, we need at least one tray trip to p.
                if needed_at_p > available_ontray_at_p:
                     places_needing_delivery.add(p)

        num_trips = len(places_needing_delivery)
        cost_move_tray = num_trips

        # 6. Calculate serving cost
        cost_serve = N_unserved

        # 7. Calculate tray-to-kitchen move cost (if needed)
        # Is a tray needed in the kitchen for put_on_tray or move_tray actions?
        needs_put_on_tray_in_kitchen = (sandwiches_to_put_on_tray > 0)
        needs_move_from_kitchen = (num_trips > 0)
        cost_tray_to_kitchen = 1 if (needs_put_on_tray_in_kitchen or needs_move_from_kitchen) and N_trays_kitchen == 0 else 0

        # Total heuristic
        h = cost_make + cost_put_on_tray + cost_move_tray + cost_serve + cost_tray_to_kitchen

        return h
