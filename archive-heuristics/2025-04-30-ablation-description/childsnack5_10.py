# Assuming Heuristic base is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic # Need to import this

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handles facts like '(predicate arg1 arg2)'
    # Removes parentheses and splits by space.
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the remaining effort to serve all waiting children.
    It calculates a cost for each unserved child based on the most advanced stage
    of readiness of a suitable sandwich (correct gluten status) for them,
    and sums these costs. The stages are:
    1. Suitable sandwich is on a tray at the child's location (needs 1 action: serve).
    2. Suitable sandwich is on a tray elsewhere (needs 2 actions: move tray, serve).
    3. Suitable sandwich is in the kitchen (needs 3 actions: put on tray, move tray, serve).
    4. Suitable sandwich needs to be made (needs 4 actions: make, put on tray, move tray, serve).
    5. Ingredients are missing to make a suitable sandwich (penalty cost).
    Costs for stages 1 and 0 are adjusted if the child is waiting in the kitchen,
    as the 'move_tray' action to the child's location is not needed.

    # Assumptions
    - Each child needs exactly one sandwich.
    - Trays are available when needed for 'put_on_tray' or 'move_tray'.
    - Ingredients are sufficient if at least one pair of suitable bread/content is in the kitchen.
    - The cost of each base action (make, put, move, serve) is 1.
    - This heuristic is NOT admissible as it sums costs per child and may overcount shared resources (sandwiches, trays, ingredients). It is designed for greedy best-first search.

    # Heuristic Initialization
    - Extracts static information: child allergy status and child waiting locations.
    - Identifies all children who need to be served based on the goal state.
    - Identifies static gluten-free ingredients.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that are in the goal state but not in the current state (i.e., unserved children).
    2. For each unserved child:
       a. Determine the child's waiting location and allergy status from static facts.
       b. Determine the required gluten status for the sandwich (GF if child is allergic, any otherwise).
       c. Find the "best" (most advanced) stage a suitable sandwich is in:
          - Check if *any* suitable sandwich is on a tray at the child's location (Stage 3).
          - Else, check if *any* suitable sandwich is on a tray at a different location (Stage 2).
          - Else, check if *any* suitable sandwich is in the kitchen (Stage 1).
          - Else, check if ingredients are available in the kitchen to make a suitable sandwich (Stage 0).
          - Else, the child is in the "unmakeable" stage.
       d. Assign a cost to the child based on the determined stage, adjusting for kitchen location:
          - Stage 3 (Ontray at child_place): 1 (serve)
          - Stage 2 (Ontray elsewhere): 2 (move, serve)
          - Stage 1 (Kitchen): 1 (put) + (1 if child not in kitchen else 0) (move) + 1 (serve)
          - Stage 0 (Needs making): 1 (make) + 1 (put) + (1 if child not in kitchen else 0) (move) + 1 (serve)
          - Unmakeable: 100 (penalty)
       e. Add the calculated cost for this child to the total heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract static info: child allergy status and waiting locations
        self.child_allergy = {} # Map child name to boolean (True if allergic)
        self.child_waiting_place = {} # Map child name to place name

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.child_allergy[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.child_allergy[parts[1]] = False
            elif parts[0] == 'waiting':
                self.child_waiting_place[parts[1]] = parts[2]

        # Identify all children that need to be served (from goals)
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.children_to_serve.add(parts[1])

        # Identify static gluten-free ingredients (names)
        self.static_gf_bread_names = set()
        self.static_gf_content_names = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == 'no_gluten_bread':
                 self.static_gf_bread_names.add(parts[1])
             elif parts[0] == 'no_gluten_content':
                 self.static_gf_content_names.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract relevant state information ---
        served_children = set()
        sandwich_locations = {} # {sandwich_name: 'kitchen' or 'ontray_trayname'}
        tray_locations = {}     # {tray_name: place_name}
        sandwich_gluten_status = {} # {sandwich_name: boolean}
        kitchen_bread = set() # Names of bread portions in kitchen
        kitchen_content = set() # Names of content portions in kitchen
        existing_sandwiches = set() # Sandwiches that are not (notexist s)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_children.add(parts[1])
            elif parts[0] == 'at_kitchen_sandwich':
                sandwich_locations[parts[1]] = 'kitchen'
                existing_sandwiches.add(parts[1])
            elif parts[0] == 'ontray':
                sandwich_locations[parts[1]] = f'ontray_{parts[2]}' # Store tray name
                existing_sandwiches.add(parts[1])
            elif parts[0] == 'at':
                tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_sandwich':
                sandwich_gluten_status[parts[1]] = True
            elif parts[0] == 'at_kitchen_bread':
                kitchen_bread.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                kitchen_content.add(parts[1])
            # notexist facts are implicitly handled by checking existing_sandwiches

        # Default gluten status for sandwiches not explicitly marked GF (assume not GF)
        for s in existing_sandwiches:
             if s not in sandwich_gluten_status:
                 sandwich_gluten_status[s] = False

        # --- Calculate heuristic cost ---
        total_cost = 0

        # Find children who are in the goal but not yet served
        unserved_children = self.children_to_serve - served_children

        if not unserved_children:
            return 0 # Goal reached

        # Check available ingredients in kitchen for making *new* sandwiches
        # We just need to know *if* a suitable sandwich *can* be made, not how many times.
        can_make_regular = len(kitchen_bread) > 0 and len(kitchen_content) > 0
        can_make_gluten_free = any(b in kitchen_bread and b in self.static_gf_bread_names for b in self.static_gf_bread_names) and \
                               any(c in kitchen_content and c in self.static_gf_content_names for c in self.static_gf_content_names)


        # Iterate through unserved children and determine their stage cost
        for child in unserved_children:
            child_place = self.child_waiting_place.get(child)
            is_allergic = self.child_allergy.get(child, False) # Default to not allergic

            if child_place is None:
                 # Should not happen in valid problems, but handle defensively
                 total_cost += 100 # Penalty for child with unknown waiting place
                 continue

            required_gf_status = is_allergic

            # Find the best stage for this child
            best_stage_cost = 100 # Start with penalty (unmakeable)

            # Check Stage 3: Suitable sandwich on tray at child's place? (Cost 1)
            found_stage_3 = False
            for s in existing_sandwiches:
                is_gf = sandwich_gluten_status.get(s, False)
                # Check if sandwich is suitable
                if not required_gf_status or is_gf: # If allergic, must be GF. If not allergic, any is fine.
                    loc_info = sandwich_locations.get(s)
                    if loc_info and loc_info.startswith('ontray_'):
                        tray_name = loc_info.split('_')[1]
                        tray_loc = tray_locations.get(tray_name)
                        if tray_loc == child_place:
                            found_stage_3 = True
                            break # Found a suitable sandwich in the best stage

            if found_stage_3:
                best_stage_cost = 1 # Serve

            # If not Stage 3, check Stage 2: Suitable sandwich on tray elsewhere? (Cost 2)
            if best_stage_cost == 100:
                found_stage_2 = False
                for s in existing_sandwiches:
                    is_gf = sandwich_gluten_status.get(s, False)
                    if not required_gf_status or is_gf:
                        loc_info = sandwich_locations.get(s)
                        if loc_info and loc_info.startswith('ontray_'):
                            tray_name = loc_info.split('_')[1]
                            tray_loc = tray_locations.get(tray_name)
                            if tray_loc is not None and tray_loc != child_place:
                                found_stage_2 = True
                                break # Found a suitable sandwich in Stage 2

                if found_stage_2:
                    best_stage_cost = 2 # Move, Serve

            # If not Stage 3 or 2, check Stage 1: Suitable sandwich in kitchen? (Cost 3 normally)
            if best_stage_cost == 100:
                found_stage_1 = False
                for s in existing_sandwiches:
                    is_gf = sandwich_gluten_status.get(s, False)
                    if not required_gf_status or is_gf:
                        loc_info = sandwich_locations.get(s)
                        if loc_info == 'kitchen':
                            found_stage_1 = True
                            break # Found a suitable sandwich in Stage 1

                if found_stage_1:
                    # Cost: Put (1) + Move (1 if not kitchen else 0) + Serve (1)
                    move_cost = 1 if child_place != 'kitchen' else 0
                    best_stage_cost = 1 + move_cost + 1 # Put + Move + Serve

            # If not Stage 3, 2, or 1, check Stage 0: Can make suitable sandwich? (Cost 4 normally)
            if best_stage_cost == 100:
                 can_make_suitable = False
                 if required_gf_status:
                     can_make_suitable = can_make_gluten_free
                 else:
                     # Non-allergic can use regular or GF
                     can_make_suitable = can_make_regular or can_make_gluten_free

                 if can_make_suitable:
                     # Cost: Make (1) + Put (1) + Move (1 if not kitchen else 0) + Serve (1)
                     move_cost = 1 if child_place != 'kitchen' else 0
                     best_stage_cost = 1 + 1 + move_cost + 1 # Make + Put + Move + Serve
                 # Else: remains 100 (unmakeable)


            total_cost += best_stage_cost

        return total_cost
