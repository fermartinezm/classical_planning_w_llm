from fnmatch import fnmatch
from collections import defaultdict
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    parts = fact[1:-1].split()
    return parts

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

# Define a large number for infinity
INFINITY = float('inf')

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It calculates the cost based on the current state of sandwiches
    (location, type, existence) and ingredients, and the location of trays
    relative to unserved children. It sums the estimated costs for getting
    sandwiches into a servable state (on a tray at the child's location),
    moving trays to locations that need them, and performing the serve action.
    The costs are estimated based on the current state of the sandwich/ingredients.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Suitable sandwiches are non-gluten for allergic children and any sandwich
      for non-allergic children.
    - Sandwiches must be on a tray to be served.
    - The tray must be at the child's location to serve.
    - Ingredient pairs (bread + content) are consumed when making a sandwich.
    - Non-gluten ingredients can make non-gluten sandwiches. Any ingredient pair
      can make a regular sandwich.
    - Tray capacity is unlimited.
    - Enough trays exist in total to satisfy location needs.
    - The heuristic returns infinity if the current state's ingredients are
      insufficient to make the required sandwiches.

    # Heuristic Initialization
    - Extracts static information about children (allergy, waiting place) from task.static.
    - Extracts goal information (which children need to be served) from task.goals.
    - Extracts static gluten information for ingredients and sandwiches.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine which children need to be served based on the goal and current state. Count unserved allergic and non-allergic children, grouped by their waiting location.
    2.  **Count Available Sandwiches:** Count existing sandwiches (not `notexist`) by type (non-gluten/regular) and current state (`at_kitchen_sandwich`, `ontray kitchen`, `ontray location`).
    3.  **Count Available Ingredients:** Count available ingredient pieces (bread, content) in the kitchen by type (non-gluten/regular).
    4.  **Estimate Sandwich Costs:** Calculate the cost to get each needed sandwich into a servable state (on a tray at the required location), considering its current state and type:
        -   Already on tray at location (Cost 1: serve)
        -   On tray in kitchen (Cost 2: move_tray, serve)
        -   At kitchen (Cost 3: put_on_tray, move_tray, serve)
        -   Needs making (Cost 4: make, put_on_tray, move_tray, serve)
        Prioritize cheaper sources for both non-gluten and regular sandwiches. Account for ingredient availability (checking if enough NG pairs exist for needed NG sandwiches, and if enough total pairs exist for total needed sandwiches) and return infinity if insufficient.
    5.  **Estimate Tray Movement Cost:** Identify locations (excluding kitchen) where unserved children are waiting but no tray is present. Each such location requires at least one `move_tray` action to bring a tray there. This cost is added separately as one tray can serve multiple children/sandwiches at a location.
    6.  **Sum Costs:** The total heuristic value is the sum of the estimated costs for getting all sandwiches served (which includes make/put/move/serve actions for each) and the estimated cost for moving trays to locations that need them.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)

        # Store child information: {child_name: (waiting_place, is_allergic)}
        self.child_info = {}
        # Store static gluten info for ingredients/sandwiches
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.no_gluten_sandwiches_static = set() # Sandwiches made NG are always NG

        # First pass to gather static properties
        child_allergy = {} # {child_name: is_allergic}
        for fact in self.static:
             parts = get_parts(fact)
             if parts:
                 if parts[0] == 'allergic_gluten':
                     child_allergy[parts[1]] = True
                 elif parts[0] == 'not_allergic_gluten':
                     child_allergy[parts[1]] = False
                 elif parts[0] == 'no_gluten_bread':
                     self.no_gluten_breads.add(parts[1])
                 elif parts[0] == 'no_gluten_content':
                     self.no_gluten_contents.add(parts[1])
                 elif parts[0] == 'no_gluten_sandwich':
                     self.no_gluten_sandwiches_static.add(parts[1])

        # Second pass to link waiting places and allergy
        for fact in self.static:
             parts = get_parts(fact)
             if parts and parts[0] == 'waiting':
                 child, place = parts[1], parts[2]
                 is_allergic = child_allergy.get(child, False) # Default to not allergic if allergy not specified
                 self.child_info[child] = (place, is_allergic)


        # Store children that need to be served based on goal
        self.children_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = {c for c in self.children_to_serve if f"(served {c})" not in state}

        if not unserved_children:
            return 0 # Goal reached

        U_ng_p = defaultdict(int) # Unserved NG children per place
        U_reg_p = defaultdict(int) # Unserved Reg children per place
        locations_with_unserved = set()

        for child in unserved_children:
            if child not in self.child_info:
                 # Should not happen in valid problems where goal children are in static waiting facts
                 continue

            place, is_allergic = self.child_info[child]
            locations_with_unserved.add(place)
            if is_allergic:
                U_ng_p[place] += 1
            else:
                U_reg_p[place] += 1

        # Total needed sandwiches by type
        Needed_NG_total = sum(U_ng_p.values())
        Needed_Reg_total = sum(U_reg_p.values())


        # 2. Count Available Sandwiches (made) by state and type
        NG_kitchen = 0 # at_kitchen_sandwich
        Reg_kitchen = 0 # at_kitchen_sandwich
        NG_ontray_kitchen = 0
        Reg_ontray_kitchen = 0
        NG_ontray_location = 0 # Sum across all locations != kitchen
        Reg_ontray_location = 0 # Sum across all locations != kitchen

        # Find tray locations
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Assuming objects starting with 'tray' are trays
                if parts and len(parts) > 1 and parts[1].startswith('tray'):
                     tray_locations[parts[1]] = parts[2] # tray_name: place

        # Find all existing sandwiches (not notexist)
        existing_sandwiches = set()
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        # Infer all possible sandwich objects from state facts
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ["at_kitchen_sandwich", "ontray", "notexist", "no_gluten_sandwich"]:
                  if len(parts) > 1:
                       existing_sandwiches.add(parts[1])
        # Also get from static facts if any mention sandwiches (like no_gluten_sandwich)
        for fact in self.static:
             parts = get_parts(fact)
             if parts and parts[0] in ["no_gluten_sandwich"]:
                  if len(parts) > 1:
                       existing_sandwiches.add(parts[1])

        existing_sandwiches = {s for s in existing_sandwiches if s not in notexist_sandwiches}


        for s in existing_sandwiches:
            is_ng = (f"(no_gluten_sandwich {s})" in state) or (s in self.no_gluten_sandwiches_static)

            # Check location
            at_kitchen = f"(at_kitchen_sandwich {s})" in state
            ontray_fact = next((fact for fact in state if match(fact, "ontray", s, "*")), None)

            if at_kitchen:
                if is_ng:
                    NG_kitchen += 1
                else:
                    Reg_kitchen += 1
            elif ontray_fact:
                tray_name = get_parts(ontray_fact)[2]
                tray_place = tray_locations.get(tray_name)
                if tray_place == 'kitchen':
                    if is_ng:
                        NG_ontray_kitchen += 1
                    else:
                        Reg_ontray_kitchen += 1
                elif tray_place is not None: # It's on a tray at a non-kitchen location
                    if is_ng:
                        NG_ontray_location += 1
                    else:
                        Reg_ontray_location += 1
                # else: sandwich is on a tray whose location is unknown? Ignore for heuristic.


        # 3. Count Available Ingredients
        I_ng_bread = 0
        I_reg_bread = 0
        I_ng_content = 0
        I_reg_content = 0

        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                bread_name = get_parts(fact)[1]
                if bread_name in self.no_gluten_breads:
                    I_ng_bread += 1
                else:
                    I_reg_bread += 1
            elif match(fact, "at_kitchen_content", "*"):
                content_name = get_parts(fact)[1]
                if content_name in self.no_gluten_contents:
                    I_ng_content += 1
                else:
                    I_reg_content += 1

        Can_make_NG_pairs = min(I_ng_bread, I_ng_content)
        Can_make_Total_pairs = min(I_ng_bread + I_reg_bread, I_ng_content + I_reg_content)


        # 4. Estimate Sandwich Costs (Cost to get to 'served' state)

        # Cost for NG sandwiches:
        needed_ng = Needed_NG_total
        cost_ng = 0

        # Use State 4 (ontray loc, cost 1: serve)
        from_s4 = min(needed_ng, NG_ontray_location)
        cost_ng += from_s4 * 1
        needed_ng -= from_s4

        # Use State 3 (ontray kitchen, cost 2: move_tray, serve)
        from_s3 = min(needed_ng, NG_ontray_kitchen)
        cost_ng += from_s3 * 2
        needed_ng -= from_s3

        # Use State 2 (kitchen, cost 3: put_on_tray, move_tray, serve)
        from_s2 = min(needed_ng, NG_kitchen)
        cost_ng += from_s2 * 3
        needed_ng -= from_s2

        # Sandwiches that must be made NG
        Needed_NG_makable = needed_ng # These must come from making NG sandwiches
        if Needed_NG_makable > Can_make_NG_pairs:
             return INFINITY # Cannot make enough NG sandwiches

        cost_ng += Needed_NG_makable * 4 # make + put_on_tray + move_tray + serve
        needed_ng -= Needed_NG_makable # Should be 0 now


        # Cost for Reg sandwiches:
        needed_reg = Needed_Reg_total
        cost_reg = 0

        # Use State 4 (ontray loc, cost 1)
        from_s4 = min(needed_reg, Reg_ontray_location)
        cost_reg += from_s4 * 1
        needed_reg -= from_s4

        # Use State 3 (ontray kitchen, cost 2)
        from_s3 = min(needed_reg, Reg_ontray_kitchen)
        cost_reg += from_s3 * 2
        needed_reg -= from_s3

        # Use State 2 (kitchen, cost 3)
        from_s2 = min(needed_reg, Reg_kitchen)
        cost_reg += from_s2 * 3
        needed_reg -= from_s2

        # Sandwiches that must be made Reg
        Needed_Reg_makable = needed_reg # These must come from making Reg sandwiches

        # Check if total makable sandwiches (NG + Reg) exceed total ingredient pairs
        Total_sandwiches_to_make = Needed_NG_makable + Needed_Reg_makable
        if Total_sandwiches_to_make > Can_make_Total_pairs:
             return INFINITY # Cannot make enough total sandwiches

        cost_reg += Needed_Reg_makable * 4 # make + put_on_tray + move_tray + serve
        needed_reg -= Needed_Reg_makable # Should be 0 now


        Total_sandwich_cost = cost_ng + cost_reg

        # 5. Estimate Tray Movement Cost
        # Locations (excluding kitchen) where unserved children are waiting
        locations_with_unserved_loc = {p for p in locations_with_unserved if p != 'kitchen'}

        # Locations (excluding kitchen) that currently have a tray
        locations_with_tray_loc = {place for place in tray_locations.values() if place != 'kitchen'}

        # Locations needing a tray are those with unserved children but no tray
        needed_trays_at_locations = len(locations_with_unserved_loc - locations_with_tray_loc)

        Cost_move_trays = needed_trays_at_locations # Assumes trays are available to be moved

        # 6. Sum Costs
        # The Total_sandwich_cost includes the 'make', 'put_on_tray', 'move_tray', and 'serve' actions
        # for each sandwich needed. The Cost_move_trays is added separately because a single tray
        # move can satisfy the tray requirement for multiple children/sandwiches at a location.

        return Total_sandwich_cost + Cost_move_trays
