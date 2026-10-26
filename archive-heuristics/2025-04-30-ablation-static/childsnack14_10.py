from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It does this by classifying unserved children based on the "stage"
    of readiness of a suitable sandwich/tray combination for them, and summing
    the estimated actions for each stage.

    The stages are:
    Stage 1: Suitable sandwich is already on a tray at the child's location (cost: 1 action - serve).
    Stage 2: Suitable sandwich is on a tray elsewhere (cost: 2 actions - move tray, serve).
    Stage 3: Suitable sandwich is at the kitchen (cost: 3 actions - put on tray, move tray, serve).
    Stage 4: Suitable sandwich needs to be made (cost: 4 actions - make, put on tray, move tray, serve).

    The heuristic greedily assigns available sandwiches/resources from lower stages
    to children, prioritizing allergic children for gluten-free sandwiches.

    # Assumptions
    - All children mentioned in the goal predicate `(served ?c)` need to be served.
    - A gluten-free sandwich can serve either an allergic or a non-allergic child.
    - A regular sandwich can only serve a non-allergic child.
    - Enough bread/content and trays are available in total to make all needed sandwiches and deliver them (this is a simplification for efficiency).
    - The 'kitchen' is a constant place object.
    - Sandwiches are either `notexist`, `at_kitchen_sandwich`, or `ontray`.

    # Heuristic Initialization
    - Extracts all objects of relevant types (child, sandwich, tray, place, bread, content) from the task's facts.
    - Extracts static facts, specifically child allergy information (`allergic_gluten`, `not_allergic_gluten`).
    - Identifies the set of children that are goals (`(served ?c)`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract dynamic state information: served children, waiting children and their locations, sandwich status (at kitchen, on tray, notexist), sandwich-tray mapping, tray locations, and which made sandwiches are gluten-free.
    2. Identify the set of children that need to be served (those in the goal `(served ?c)` predicate) and are not yet served.
    3. Count the total number of unserved children (`U`), unserved allergic children (`U_gf`), and unserved non-allergic children (`U_reg`). If `U` is 0, the heuristic is 0.
    4. Count the total number of made gluten-free (`Available_GF_anywhere`) and regular (`Available_Reg_anywhere`) sandwiches currently in the state (at kitchen or on a tray).
    5. Calculate the number of GF and regular sandwiches that still need to be made (`make_gf_potential`, `make_reg_potential`) to satisfy the unserved children's requirements, prioritizing GF for allergic children.
    6. Prepare pools of available suitable sandwiches by their current status/location:
       - Stage 1 Pool: Suitable sandwiches on trays at the specific location of an unserved child.
       - Stage 2 Pool: Suitable sandwiches on trays elsewhere (not at an unserved child's location).
       - Stage 3 Pool: Suitable sandwiches at the kitchen.
       - Stage 4 Pool: Potential sandwiches that can be made (`make_gf_potential`, `make_reg_potential`).
    7. Iterate through the list of unserved children. For each child, determine the cheapest stage (1 through 4) from which a suitable sandwich can be drawn from the available pools. Assign the child to this stage and decrement the corresponding pool count. Prioritize using GF sandwiches for allergic children first, then use remaining GF and regular sandwiches for non-allergic children.
    8. Calculate the total heuristic cost by summing the cost for each child based on their assigned stage (1 for Stage 1, 2 for Stage 2, 3 for Stage 3, 4 for Stage 4).
    9. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and object lists."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract all objects by type from task.facts
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant
        self.all_bread = set()
        self.all_content = set()

        # Helper to add objects from a fact string
        def add_objects_from_fact_str(fact_str):
            parts = get_parts(fact_str)
            if not parts: return # Handle empty fact string

            predicate = parts[0]
            args = parts[1:]

            # Based on domain predicates and action parameters
            if predicate in ['waiting', 'served', 'allergic_gluten', 'not_allergic_gluten']:
                if len(args) > 0: self.all_children.add(args[0])
                if predicate == 'waiting' and len(args) > 1: self.all_places.add(args[1])
            elif predicate in ['at_kitchen_bread', 'no_gluten_bread']:
                 if len(args) > 0: self.all_bread.add(args[0])
            elif predicate in ['at_kitchen_content', 'no_gluten_content']:
                 if len(args) > 0: self.all_content.add(args[0])
            elif predicate in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist']:
                 if len(args) > 0: self.all_sandwiches.add(args[0])
                 if predicate == 'ontray' and len(args) > 1: self.all_trays.add(args[1])
            elif predicate == 'at':
                 if len(args) > 0: self.all_trays.add(args[0])
                 if len(args) > 1: self.all_places.add(args[1])

        # Parse all possible facts to find all objects
        for fact_str in task.facts:
             add_objects_from_fact_str(fact_str)

        # Ensure kitchen is in places
        self.all_places.add('kitchen')

        # Extract child allergy information from static facts
        self.child_allergy_map = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                if len(parts) > 1: self.child_allergy_map[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                if len(parts) > 1: self.child_allergy_map[parts[1]] = False

        # Identify children that are goals
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served' and len(get_parts(g)) > 1}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract dynamic state information ---
        served_children = set()
        unserved_child_loc = {} # child -> place
        sandwich_status = {} # sandwich -> 'kitchen', 'ontray', 'notexist'
        sandwich_tray = {} # sandwich -> tray (if ontray)
        tray_loc = {} # tray -> place
        sandwich_is_gf_map = {} # sandwich -> True/False (only for made sandwiches)

        # Initialize sandwich status for all known sandwiches
        for s in self.all_sandwiches:
            sandwich_status[s] = 'unknown' # Default status

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'served':
                if len(parts) > 1: served_children.add(parts[1])
            elif predicate == 'waiting':
                 if len(parts) > 2:
                     # Only track waiting children that are in the goal and not yet served
                     child = parts[1]
                     if child in self.goal_children and child not in served_children:
                        unserved_child_loc[child] = parts[2]
            elif predicate == 'at_kitchen_sandwich':
                if len(parts) > 1: sandwich_status[parts[1]] = 'kitchen'
            elif predicate == 'ontray':
                if len(parts) > 2:
                    s, t = parts[1], parts[2]
                    sandwich_status[s] = 'ontray'
                    sandwich_tray[s] = t
            elif predicate == 'at':
                if len(parts) > 2:
                    t, p = parts[1], parts[2]
                    tray_loc[t] = p
            elif predicate == 'no_gluten_sandwich':
                if len(parts) > 1: sandwich_is_gf_map[parts[1]] = True
            elif predicate == 'notexist':
                 if len(parts) > 1:
                     # Only mark as notexist if it's one of the known sandwiches
                     if parts[1] in self.all_sandwiches:
                        sandwich_status[parts[1]] = 'notexist'

        unserved_children_list = list(unserved_child_loc.keys())
        U = len(unserved_children_list)

        if U == 0:
            return 0 # Goal reached

        U_gf = sum(1 for c in unserved_children_list if self.child_allergy_map.get(c, False))
        U_reg = U - U_gf

        # Count made sandwiches (at kitchen or ontray)
        made_sandwiches = {s for s, status in sandwich_status.items() if status in ['kitchen', 'ontray']}
        Available_GF_anywhere = {s for s in made_sandwiches if sandwich_is_gf_map.get(s, False)}
        Available_Reg_anywhere = made_sandwiches - Available_GF_anywhere

        # Sandwiches to make (potential count)
        make_gf_potential = max(0, U_gf - len(Available_GF_anywhere))
        remaining_available_gf = max(0, len(Available_GF_anywhere) - U_gf)
        make_reg_potential = max(0, U_reg - (len(Available_Reg_anywhere) + remaining_available_gf))

        # --- Prepare pools of available suitable sandwiches by location/status ---
        # Stage 1 Pool: on tray at child's location
        # Store as lists to allow popping (consuming)
        gf_at_loc_stage1_pool = {} # loc -> list of GF sandwiches
        reg_at_loc_stage1_pool = {} # loc -> list of Reg sandwiches

        # Stage 2 Pool: on tray elsewhere
        gf_on_tray_elsewhere_pool = [] # List of GF sandwiches
        reg_on_tray_elsewhere_pool = [] # List of Reg sandwiches

        # Stage 3 Pool: at kitchen
        gf_at_kitchen_pool = []
        reg_at_kitchen_pool = []

        unserved_locations = set(unserved_child_loc.values())

        for s in made_sandwiches:
            s_is_gf = sandwich_is_gf_map.get(s, False)
            s_status = sandwich_status[s]

            if s_status == 'ontray':
                t = sandwich_tray.get(s)
                p = tray_loc.get(t) if t else None # Handle case where tray is not at any location
                if p in unserved_locations:
                    # This sandwich is on a tray at a location where *some* unserved child is waiting.
                    # Add it to the pool for that specific location.
                    if s_is_gf:
                        gf_at_loc_stage1_pool.setdefault(p, []).append(s)
                    else:
                        reg_at_loc_stage1_pool.setdefault(p, []).append(s)
                else: # ontray elsewhere (not at any unserved location)
                    if s_is_gf:
                        gf_on_tray_elsewhere_pool.append(s)
                    else:
                        reg_on_tray_elsewhere_pool.append(s)
            elif s_status == 'kitchen':
                if s_is_gf:
                    gf_at_kitchen_pool.append(s)
                else:
                    reg_at_kitchen_pool.append(s)

        # --- Assign children to stages greedily and calculate cost ---
        h = 0

        # Process children
        for child in unserved_children_list:
            loc = unserved_child_loc[child]
            is_allergic = self.child_allergy_map.get(child, False)

            assigned_stage = None

            # Try Stage 1 (Cost 1: Serve)
            # Requires suitable sandwich on tray at location `loc`
            if is_allergic:
                if gf_at_loc_stage1_pool.get(loc, []):
                    gf_at_loc_stage1_pool[loc].pop() # Use one GF sandwich at this location
                    assigned_stage = 1
            else: # Not allergic
                # Prioritize GF if available at location
                if gf_at_loc_stage1_pool.get(loc, []):
                    gf_at_loc_stage1_pool[loc].pop() # Use one GF sandwich at this location
                    assigned_stage = 1
                elif reg_at_loc_stage1_pool.get(loc, []):
                    reg_at_loc_stage1_pool[loc].pop() # Use one Reg sandwich at this location
                    assigned_stage = 1

            if assigned_stage == 1:
                h += 1
                continue # Move to next child

            # Try Stage 2 (Cost 2: Move tray + Serve)
            # Requires suitable sandwich on tray elsewhere
            if is_allergic:
                if gf_on_tray_elsewhere_pool:
                    gf_on_tray_elsewhere_pool.pop() # Use one GF sandwich on tray elsewhere
                    assigned_stage = 2
            else: # Not allergic
                # Prioritize GF if available elsewhere on tray
                if gf_on_tray_elsewhere_pool:
                    gf_on_tray_elsewhere_pool.pop() # Use one GF sandwich on tray elsewhere
                    assigned_stage = 2
                elif reg_on_tray_elsewhere_pool:
                    reg_on_tray_elsewhere_pool.pop() # Use one Reg sandwich on tray elsewhere
                    assigned_stage = 2

            if assigned_stage == 2:
                h += 2
                continue # Move to next child

            # Try Stage 3 (Cost 3: Put on tray + Move tray + Serve)
            # Requires suitable sandwich at kitchen
            if is_allergic:
                if gf_at_kitchen_pool:
                    gf_at_kitchen_pool.pop() # Use one GF sandwich at kitchen
                    assigned_stage = 3
            else: # Not allergic
                # Prioritize GF if available at kitchen
                if gf_at_kitchen_pool:
                    gf_at_kitchen_pool.pop() # Use one GF sandwich at kitchen
                    assigned_stage = 3
                elif reg_at_kitchen_pool:
                    reg_at_kitchen_pool.pop() # Use one Reg sandwich at kitchen
                    assigned_stage = 3

            if assigned_stage == 3:
                h += 3
                continue # Move to next child

            # Try Stage 4 (Cost 4: Make + Put on tray + Move tray + Serve)
            # Requires potential to make a suitable sandwich
            if is_allergic:
                if make_gf_potential > 0:
                    make_gf_potential -= 1 # Use one potential GF make action
                    assigned_stage = 4
            else: # Not allergic
                # Prioritize making GF if needed and possible
                if make_gf_potential > 0:
                    make_gf_potential -= 1 # Use one potential GF make action
                    assigned_stage = 4
                elif make_reg_potential > 0:
                    make_reg_potential -= 1 # Use one potential Reg make action
                    assigned_stage = 4

            if assigned_stage == 4:
                h += 4
                continue # Move to next child

            # If we reach here, the child cannot be served with available resources
            # This indicates an unsolvable problem given the available objects and potential makes.
            # Return a large number.
            # print(f"Warning: Child {child} cannot be assigned a stage. Problem might be unsolvable.")
            return float('inf') # Or a large constant

        # If all children were assigned a stage, return the total cost
        return h
