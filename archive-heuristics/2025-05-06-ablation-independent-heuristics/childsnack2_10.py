# Need to import Heuristic from the planner's heuristic base class
# Assuming the structure is heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment where the heuristic is run
# from task import Task # Not strictly needed for the code itself, but good for context

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the cost to reach a goal state by summing up the estimated minimum
    number of actions required to serve each unserved child. The cost for each
    child is determined by the current state of a suitable sandwich relative
    to the child's location (on tray at location, on tray elsewhere, in kitchen,
    or needs to be made) and the availability of a tray at the kitchen.

    Assumptions:
    - The heuristic assumes that for solvable problems, there are always enough
      sandwich objects (`notexist`) and ingredients available in the initial
      state to make all necessary sandwiches.
    - When a sandwich is in the kitchen or needs to be made, the heuristic
      estimates the cost assuming a tray is either available at the kitchen
      or can be moved there in a single action.
    - Resource sharing (multiple children needing the same sandwich/tray) is
      not explicitly modeled; the heuristic sums individual child costs, which
      might overestimate the total cost but serves as a greedy measure of
      progress.
    - The heuristic returns infinity if it detects that an unserved child
      cannot possibly be served based on the current state and available
      resources (no suitable sandwich exists anywhere and none can be made).

    Heuristic Initialization:
    The constructor pre-processes static facts from the task description:
    - Identifies all children, their waiting places, and allergy status.
    - Identifies all bread, content, and sandwich objects that are statically
      marked as gluten-free.
    - Identifies all tray objects present in the initial state or static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Parse the current state to identify:
       - Children who have been served.
       - Which sandwiches are on which trays.
       - The current location of each tray.
       - Which sandwiches, bread portions, and content portions are in the kitchen.
       - Which sandwich object names are available (`notexist`).
    3. Determine if any tray is currently located at the kitchen.
    4. Iterate through each child identified during initialization.
    5. If a child is already served in the current state, their contribution to
       the heuristic is 0; continue to the next child.
    6. For an unserved child waiting at a specific place:
       a. Determine if the child requires a gluten-free sandwich based on their
          allergy status (from static facts).
       b. Initialize the minimum estimated cost for this child (`child_h`) to infinity.
       c. Check if a suitable sandwich is already on a tray *at the child's waiting place*. If yes, update `child_h` to 1 (cost of `serve` action).
       d. If `child_h` is still greater than 1, check if a suitable sandwich is on a tray *not at the child's waiting place*. If yes, update `child_h` to 2 (cost of `move_tray` + `serve`).
       e. If `child_h` is still greater than 2, check if a suitable sandwich is *in the kitchen*. If yes, estimate the cost: 3 actions (`put_on_tray`, `move_tray`, `serve`) if a tray is available at the kitchen, or 4 actions (`move_tray` to kitchen, `put_on_tray`, `move_tray` to child, `serve`) if no tray is currently at the kitchen. Update `child_h` with this minimum cost.
       f. If `child_h` is still greater than the cost estimated in step (e), check if a suitable sandwich *can be made*. This requires an available `notexist` sandwich name and suitable ingredients in the kitchen (gluten-free ingredients for allergic children, any ingredients for non-allergic children). If yes, estimate the cost: 4 actions (`make_sandwich`, `put_on_tray`, `move_tray`, `serve`) if a tray is available at the kitchen, or 5 actions (`make_sandwich`, `move_tray` to kitchen, `put_on_tray`, `move_tray` to child, `serve`) if no tray is currently at the kitchen. Update `child_h` with this minimum cost.
       g. If, after checking all possibilities (a-f), `child_h` remains infinity, it indicates that this child cannot be served in any subsequent state reachable from the current one with the available resources. In this case, the total heuristic is set to infinity and returned immediately.
       h. Add the final `child_h` to the total heuristic `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self._parse_static_facts(task.static, task.initial_state)

    def _parse_static_facts(self, static_facts, initial_state_facts):
        """
        Parses static facts and initial state facts to extract information
        like child allergies, waiting places, gluten status of ingredients/sandwiches,
        and all object names (especially trays).
        """
        self.child_info = {}  # {child_name: [place, is_allergic]} (use list temporarily for mutation)
        self.no_gluten_bread_static = set()
        self.no_gluten_content_static = set()
        self.no_gluten_sandwich_static = set()
        self.all_trays = set()

        # Helper to parse fact string
        def parse_fact(fact_str):
             # Remove surrounding parentheses and split by space
            return fact_str.strip('()').split()

        # Parse static facts
        for fact_str in static_facts:
            parts = parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = [None, True]
                else:
                    self.child_info[child][1] = True # Set allergic status
            elif predicate == 'not_allergic_gluten':
                child = parts[1]
                if child not in self.child_info:
                     self.child_info[child] = [None, False]
                else:
                    self.child_info[child][1] = False # Set non-allergic status
            elif predicate == 'waiting':
                child, place = parts[1], parts[2]
                if child not in self.child_info:
                     self.child_info[child] = [place, None]
                else:
                    self.child_info[child][0] = place # Set waiting place
            elif predicate == 'no_gluten_bread':
                self.no_gluten_bread_static.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.no_gluten_content_static.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                self.no_gluten_sandwich_static.add(parts[1])
            elif predicate == 'at' and len(parts) == 3:
                 # Identify trays from initial locations
                 self.all_trays.add(parts[1])
            elif predicate == 'ontray' and len(parts) == 3:
                 # Identify trays from initial ontray facts
                 self.all_trays.add(parts[2])


        # Collect all tray objects from initial state facts as well
        for fact_str in initial_state_facts:
             parts = parse_fact(fact_str)
             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                 self.all_trays.add(parts[1])
             elif predicate == 'ontray' and len(parts) == 3:
                 self.all_trays.add(parts[2])

        # Ensure all children from goals are in child_info and have complete info
        # This handles cases where a child might only appear in the goal
        for goal_fact in self.task.goals:
             parts = parse_fact(goal_fact)
             if parts[0] == 'served':
                 child = parts[1]
                 if child not in self.child_info:
                     # Defaulting info for children only in goals (shouldn't happen in valid PDDL)
                     # Assume not allergic and waiting at kitchen if no info.
                     self.child_info[child] = ['kitchen', False] # Defaulting is risky, but needed if PDDL is incomplete
                 # If child is already in child_info but missing place or allergy,
                 # we could try to infer or default, but assuming valid PDDL,
                 # static facts provide complete info for all relevant objects.
                 # The current logic of updating the list handles cases where
                 # allergy and waiting facts are processed in any order.
                 if self.child_info[child][0] is None: self.child_info[child][0] = 'kitchen' # Default place
                 if self.child_info[child][1] is None: self.child_info[child][1] = False # Default allergy


        # Convert child_info list to tuple for immutability
        self.child_info = {c: tuple(info) for c, info in self.child_info.items()}


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Parse current state facts
        served_children = set()
        sandwich_on_tray = {} # {sandwich: tray}
        tray_location = {}    # {tray: place}
        kitchen_sandwiches = set()
        kitchen_bread = set()
        kitchen_content = set()
        notexist_sandwiches = set()

        # Helper to parse fact string
        def parse_fact(fact_str):
             # Remove surrounding parentheses and split by space
            return fact_str.strip('()').split()

        for fact_str in state:
            parts = parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'ontray':
                sandwich_on_tray[parts[1]] = parts[2]
            elif predicate == 'at':
                tray_location[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(parts[1])
            elif predicate == 'at_kitchen_bread':
                kitchen_bread.add(parts[1])
            elif predicate == 'at_kitchen_content':
                kitchen_content.add(parts[1])
            elif predicate == 'notexist':
                notexist_sandwiches.add(parts[1])

        # Check if any tray is at the kitchen
        tray_at_kitchen_available = any(tray_location.get(t) == 'kitchen' for t in self.all_trays)

        total_heuristic = 0

        # Iterate through all children that need to be served (from goals)
        # child_info contains all children mentioned in static facts,
        # which should cover all children in goals for valid problems.
        for child, (waiting_place, is_allergic) in self.child_info.items():
            # Skip if child is already served
            if child in served_children:
                continue

            # This child needs to be served at waiting_place
            # Determine required sandwich type
            needs_gf = is_allergic

            # Find minimum cost to serve this child
            child_h = float('inf')

            # --- Step 1: Check for suitable sandwiches on trays at the child's location ---
            # Cost: 1 (serve)
            for s, t in sandwich_on_tray.items():
                if tray_location.get(t) == waiting_place:
                    # Check if sandwich is suitable
                    s_is_gf = s in self.no_gluten_sandwich_static
                    if (needs_gf and s_is_gf) or (not needs_gf):
                        child_h = min(child_h, 1)
                        # Optimization: If we found the best case (cost 1), no need to check other sandwiches on trays at this location for *this* child.
                        # break # Can break here for cost 1, but min() handles it anyway.

            # --- Step 2: Check for suitable sandwiches on trays NOT at the child's location ---
            # Cost: 2 (move_tray + serve)
            if child_h > 1:
                for s, t in sandwich_on_tray.items():
                    if tray_location.get(t) != waiting_place:
                         # Check if sandwich is suitable
                        s_is_gf = s in self.no_gluten_sandwich_static
                        if (needs_gf and s_is_gf) or (not needs_gf):
                            child_h = min(child_h, 2)
                            # Optimization: If we found cost 2, no need to check other sandwiches on trays elsewhere for *this* child.
                            # break # Can break here for cost 2, but min() handles it anyway.


            # --- Step 3: Check for suitable sandwiches in the kitchen ---
            # Cost: 3 (put + move + serve) if tray at kitchen, 4 otherwise
            if child_h > 2:
                for s in kitchen_sandwiches:
                    # Check if sandwich is suitable
                    s_is_gf = s in self.no_gluten_sandwich_static
                    if (needs_gf and s_is_gf) or (not needs_gf):
                        cost = 3 if tray_at_kitchen_available else 4
                        child_h = min(child_h, cost)
                        # Optimization: If we found cost 3 or 4, no need to check other kitchen sandwiches for *this* child.
                        # break # Can break here, but min() handles it anyway.


            # --- Step 4: Check if a suitable sandwich can be made ---
            # Cost: 4 (make + put + move + serve) if tray at kitchen, 5 otherwise
            # Only consider making if current best cost is higher than making + subsequent steps
            min_cost_from_make = 4 if tray_at_kitchen_available else 5
            if child_h > min_cost_from_make - 1: # If current best is > cost of make + subsequent steps - 1 (i.e. >= cost of make + subsequent steps)
                 # Check if there's an available sandwich name
                 if len(notexist_sandwiches) > 0:
                     ingredients_available = False
                     if needs_gf:
                         # Need GF bread and GF content in kitchen
                         has_gf_bread = any(b in self.no_gluten_bread_static for b in kitchen_bread)
                         has_gf_content = any(c_p in self.no_gluten_content_static for c_p in kitchen_content)
                         ingredients_available = has_gf_bread and has_gf_content
                     else:
                         # Need any bread and any content in kitchen
                         has_any_bread = len(kitchen_bread) > 0
                         has_any_content = len(kitchen_content) > 0
                         ingredients_available = has_any_bread and has_any_content

                     if ingredients_available:
                         child_h = min(child_h, min_cost_from_make)
                         # Optimization: If we found cost 4 or 5, no need to check other makeable options (there's only one category anyway).
                         # break # Can break here, but min() handles it anyway.


            # If child_h is still infinity, this state is likely a dead end for this child
            if child_h == float('inf'):
                 return float('inf') # Return infinity immediately

            total_heuristic += child_h

        return total_heuristic
