from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children and sums the estimated number
    of required actions at different stages of the sandwich delivery pipeline:
    making sandwiches, putting them on trays, moving trays to locations,
    and finally serving the children.

    Assumptions:
    - Each action (make_sandwich, put_on_tray, move_tray, serve_sandwich) costs 1.
    - Sandwiches are consumed when a child is served (removing the ontray fact).
    - The total number of sandwiches that can ever be made is limited by the
      initial count of 'notexist' sandwich objects.
    - Ingredient counts (bread/content) in the kitchen limit the number of
      sandwiches that can be made from ingredients.
    - Tray availability and location are considered for move_tray costs.
    - The heuristic assumes a simplified, parallelizable flow where the cost
      is the sum of necessary actions across different resources/stages.
    - Solvable problems are assumed after an initial check based on the total
      number of children vs the total number of sandwiches that can ever be made.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic extracts static information
    from the task, including which children are allergic, their initial
    waiting places, the gluten-free bread/content types, and the initial
    count of 'notexist' sandwich objects. It performs a basic solvability
    check: if the initial number of waiting children exceeds the total number
    of sandwiches that can ever be made (initial 'notexist' count), the problem
    is deemed unsolvable, and a flag is set.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify served children, available ingredients
        in the kitchen, sandwiches in the kitchen, sandwiches on trays, tray locations,
        'notexist' sandwich objects, and sandwiches marked as gluten-free.
    2.  Identify the set of unserved children from the initial waiting children
        who are not yet marked as 'served' in the current state.
    3.  If there are no unserved children, the goal is reached, return 0.
    4.  If the problem was marked as unsolvable during initialization, return infinity.
    5.  Count the total number of gluten-free (GF) and regular (REG) sandwiches
        needed, which corresponds to the number of unserved allergic and non-allergic children, respectively.
    6.  Initialize the heuristic value `h` to 0.
    7.  **Serve Cost**: Add the number of unserved children to `h`. This is the minimum number of 'serve' actions required.
    8.  **Make Cost**: Calculate the deficit of made sandwiches (those in the kitchen or on trays) compared to the total needed sandwiches (GF and REG). This deficit represents the minimum number of 'make_sandwich' actions required. Add this count to `h`.
    9.  **Put_on_tray Cost**: Calculate the deficit of sandwiches that are currently on trays compared to the total needed sandwiches. This deficit represents the minimum number of 'put_on_tray' actions required. However, these actions can only be performed on sandwiches available in the kitchen (either initially or newly made). Calculate the potential number of sandwiches available in the kitchen (initial kitchen stock + estimated 'must_make' sandwiches). The number of 'put_on_tray' actions contributing to the heuristic is the minimum of the 'needed on tray' count and the 'available in kitchen potential' count for both GF and REG types. Add this count to `h`.
    10. **Move_tray Cost**: Identify the distinct non-kitchen locations where unserved children are waiting. Count how many of these locations currently do not have a tray. Each such location requires at least one 'move_tray' action to bring a tray there. Add this count to `h`.
    11. Return the total calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract static information
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static_facts if get_parts(fact)[0] == 'allergic_gluten'
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in self.static_facts if get_parts(fact)[0] == 'not_allergic_gluten'
        }
        self.no_gluten_breads = {
            get_parts(fact)[1] for fact in self.static_facts if get_parts(fact)[0] == 'no_gluten_bread'
        }
        self.no_gluten_contents = {
            get_parts(fact)[1] for fact in self.static_facts if get_parts(fact)[0] == 'no_gluten_content'
        }

        # Extract initial waiting children and their places
        self.initial_waiting_children = set()
        self.waiting_places = {} # child -> place
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.initial_waiting_children.add(child)
                self.waiting_places[child] = place

        # Count initial notexist sandwiches - this is the total number of
        # sandwiches that can ever be made throughout the plan.
        self.initial_notexist_count = sum(1 for fact in self.initial_state if get_parts(fact)[0] == 'notexist')

        # Check if the problem is potentially unsolvable based on sandwich objects
        # If the number of children needing service initially is more than
        # the total number of sandwiches that can ever be made, it's unsolvable.
        if len(self.initial_waiting_children) > self.initial_notexist_count:
             self._unsolvable = True
        else:
             self._unsolvable = False

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated cost to reach the goal, or float('inf') if unsolvable.
        """
        state = node.state

        # If marked as unsolvable in init, return infinity immediately
        if hasattr(self, '_unsolvable') and self._unsolvable:
             return float('inf')

        # Parse dynamic state information
        served_children = set()
        at_kitchen_breads = set()
        at_kitchen_contents = set()
        at_kitchen_sandwiches = set()
        ontray_sandwiches_dict = {} # sandwich -> tray
        trays_at_place = {} # tray -> place
        notexist_sandwiches = set()
        no_gluten_sandwiches_state = set() # Sandwiches marked as no_gluten in this state

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'served':
                served_children.add(parts[1])
            elif predicate == 'at_kitchen_bread':
                at_kitchen_breads.add(parts[1])
            elif predicate == 'at_kitchen_content':
                at_kitchen_contents.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(parts[1])
            elif predicate == 'ontray':
                ontray_sandwiches_dict[parts[1]] = parts[2] # sandwich -> tray
            elif predicate == 'at':
                 # Check if it's a tray at a place
                 if parts[1].startswith('tray'): # Assuming tray objects start with 'tray'
                    trays_at_place[parts[1]] = parts[2] # tray -> place
            elif predicate == 'notexist':
                notexist_sandwiches.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                no_gluten_sandwiches_state.add(parts[1])


        # Identify unserved children
        unserved = {c for c in self.initial_waiting_children if c not in served_children}

        # Goal reached if no unserved children
        if not unserved:
            return 0

        # Count needed GF and REG sandwiches among unserved children
        needed_gf = sum(1 for c in unserved if c in self.allergic_children)
        needed_reg = len(unserved) - needed_gf

        # --- Heuristic Calculation (Sum of actions needed) ---
        h = 0

        # 1. Cost for Serve actions
        # Each unserved child needs one serve action.
        h += len(unserved)

        # 2. Cost for Make actions
        # Count sandwiches already made (in kitchen or on tray)
        avail_made_sandwiches = at_kitchen_sandwiches | set(ontray_sandwiches_dict.keys())
        avail_gf_made = len({s for s in avail_made_sandwiches if s in no_gluten_sandwiches_state})
        avail_reg_made = len({s for s in avail_made_sandwiches if s not in no_gluten_sandwiches_state})

        # Number of sandwiches that *must* be made to meet the total need
        # This is the deficit of made sandwiches vs the total needed.
        must_make_gf = max(0, needed_gf - avail_gf_made)
        must_make_reg = max(0, needed_reg - avail_reg_made)

        # Add the cost for making these sandwiches.
        # Note: This doesn't check if ingredients/notexist are available,
        # relying on the initial solvability check and the assumption that
        # necessary ingredients/objects will become available in a solvable problem.
        h += must_make_gf + must_make_reg

        # 3. Cost for Put_on_tray actions
        # Count sandwiches already on trays
        avail_ontray_sandwiches = set(ontray_sandwiches_dict.keys())
        avail_gf_ontray = len({s for s in avail_ontray_sandwiches if s in no_gluten_sandwiches_state})
        avail_reg_ontray = len({s for s in avail_ontray_sandwiches if s not in no_gluten_sandwiches_state})

        # Number of sandwiches that *need* to be put on trays (deficit vs total needed on trays)
        needed_ontray_gf = max(0, needed_gf - avail_gf_ontray)
        needed_ontray_reg = max(0, needed_reg - avail_reg_ontray)

        # These must come from kitchen stock (initial or newly made)
        avail_gf_kitchen_initial = len({s for s in at_kitchen_sandwiches if s in no_gluten_sandwiches_state})
        avail_reg_kitchen_initial = len({s for s in at_kitchen_sandwiches if s not in no_gluten_sandwiches_state})

        # Total available in kitchen (initial + newly made) to be put on tray
        # We assume the 'must_make' sandwiches will end up in the kitchen.
        avail_gf_kitchen_potential = avail_gf_kitchen_initial + must_make_gf
        avail_reg_kitchen_potential = avail_reg_kitchen_initial + must_make_reg

        # Number of put_on_tray actions possible/needed is limited by what's needed
        # and what's potentially available in the kitchen.
        can_put_on_tray_gf = min(needed_ontray_gf, avail_gf_kitchen_potential)
        can_put_on_tray_reg = min(needed_ontray_reg, avail_reg_kitchen_potential)

        h += can_put_on_tray_gf + can_put_on_tray_reg

        # 4. Cost for Move_tray actions
        # Count distinct non-kitchen locations where unserved children are waiting
        # and there is currently no tray.
        needed_locations = {self.waiting_places[c] for c in unserved if self.waiting_places[c] != 'kitchen'}
        locations_with_tray = set(trays_at_place.values())
        locations_needing_tray = {p for p in needed_locations if p != 'kitchen' and p not in locations_with_tray} # Exclude kitchen as a needing location

        # Each location needing a tray requires at least one move_tray action
        # (assuming a tray is available somewhere).
        h += len(locations_needing_tray)

        return h
