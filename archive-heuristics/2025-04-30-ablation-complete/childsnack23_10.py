from collections import defaultdict
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define get_parts function locally as it's used within the heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Assume Heuristic base class is defined elsewhere and imported
# If running standalone for testing, you might need a mock class:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the number of 'serve' actions needed, plus the actions needed to get
    the required sandwiches (make, put on tray) to the children's locations (move tray).
    It aggregates the needs across all unserved children and available resources.

    # Assumptions
    - Each unserved child requires one suitable sandwich (gluten status matches allergy).
    - Sandwiches must be made in the kitchen, put on a tray, moved to the child's location, and then served.
    - Trays can hold multiple sandwiches.
    - Resources (bread, content, sandwich objects) are sufficient to make all needed sandwiches (for solvable problems).
    - The cost of each relevant action (make, put-on-tray, move-tray, serve) is 1.

    # Heuristic Initialization
    - Extracts static information about child allergies and the gluten status of bread and content types.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are in the goal state but are not yet marked as 'served' in the current state.
    2. For each unserved child, determine their waiting location and allergy status using facts from the current state and static facts.
    3. Count the number of unserved children needing a gluten-free sandwich and those needing a regular sandwich at each specific waiting location.
    4. Count the number of suitable sandwiches (GF or Regular) that are currently on trays at each specific waiting location.
    5. Calculate the deficit of suitable sandwiches on trays at each location (needed count - available count, minimum 0). These are the sandwiches that must be brought to the location.
    6. Count the number of suitable sandwiches (GF or Regular) that are currently in the kitchen ('at_kitchen_sandwich').
    7. Calculate the number of suitable sandwiches that *must* be made. This is the total deficit across all locations minus the suitable sandwiches already available in the kitchen (minimum 0).
    8. Identify locations where children are waiting, there is a deficit of sandwiches on trays, and there are currently no trays present. These locations will require a tray to be moved there.
    9. The heuristic value is the sum of:
       - The total number of unserved children (representing the 'serve' actions).
       - The total number of sandwiches needed from the kitchen or creation to satisfy the location deficits (representing the 'put_on_tray' actions).
       - The total number of sandwiches that must be made (representing the 'make_sandwich' actions).
       - The number of unique locations that require a tray to be moved there (representing the 'move_tray' actions).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract static allergy information
        self.child_is_allergic = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.child_is_allergic[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.child_is_allergic[parts[1]] = False

        # Extract static gluten status for bread and content types
        self.bread_is_gf = set()
        self.content_is_gf = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'no_gluten_bread':
                self.bread_is_gf.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.content_is_gf.add(parts[1])

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # --- Step 1 & 2: Identify unserved children and their needs/locations ---
        served_children = set()
        child_place = {} # Map child -> current waiting place
        sandwich_is_gf_in_state = set() # Gluten status of sandwiches already made

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_children.add(parts[1])
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                child_place[child] = place
            elif parts[0] == 'no_gluten_sandwich':
                sandwich_is_gf_in_state.add(parts[1])

        # Identify unserved children based on goals and state
        # Goals are represented as a frozenset of facts, e.g., frozenset({'(served child1)', '(served child2)'})
        unserved_children = set()
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts[0] == 'served':
                 child = parts[1]
                 if child not in served_children:
                     unserved_children.add(child)


        # If all children are served, the goal is reached, heuristic is 0.
        if not unserved_children:
            return 0

        # Count needed sandwiches by type and location
        children_at_p_gf = defaultdict(int)
        children_at_p_reg = defaultdict(int)
        waiting_places = set()

        for child in unserved_children:
            place = child_place.get(child)
            if place: # Ensure child is actually waiting somewhere
                waiting_places.add(place)
                if self.child_is_allergic.get(child, False): # Default to False if allergy status unknown
                    children_at_p_gf[place] += 1
                else:
                    children_at_p_reg[place] += 1

        # --- Step 3 & 4: Count available sandwiches and trays ---
        sandwich_ontray = {} # Map sandwich -> tray
        tray_at = {} # Map tray -> place
        sandwich_kitchen = set()
        sandwich_notexist = set()
        bread_kitchen = set()
        content_kitchen = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                sandwich_ontray[parts[1]] = parts[2]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('tray'): # Check type is tray
                 tray_at[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich':
                sandwich_kitchen.add(parts[1])
            elif parts[0] == 'notexist':
                sandwich_notexist.add(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                bread_kitchen.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                content_kitchen.add(parts[1])


        # Count sandwiches on trays at locations
        sandwiches_ontray_at_p_gf = defaultdict(int)
        sandwiches_ontray_at_p_reg = defaultdict(int)
        for sandwich, tray in sandwich_ontray.items():
            place = tray_at.get(tray)
            if place: # Ensure tray is located
                if sandwich in sandwich_is_gf_in_state:
                    sandwiches_ontray_at_p_gf[place] += 1
                else:
                    sandwiches_ontray_at_p_reg[place] += 1

        # Count sandwiches in kitchen
        sandwiches_kitchen_gf = sum(1 for s in sandwich_kitchen if s in sandwich_is_gf_in_state)
        sandwiches_kitchen_reg = len(sandwich_kitchen) - sandwiches_kitchen_gf

        # Count resources for making sandwiches (needed for 'must_make' calculation)
        # num_notexist = len(sandwich_notexist)
        # num_bread_gf = sum(1 for b in bread_kitchen if b in self.bread_is_gf)
        # num_content_gf = sum(1 for c in content_kitchen if c in self.content_is_gf)
        # num_bread_reg = len(bread_kitchen) - num_bread_gf
        # num_content_reg = len(content_kitchen) - num_content_gf

        # Count trays at locations
        trays_at_p = defaultdict(int)
        for place in tray_at.values():
            trays_at_p[place] += 1

        # --- Step 5, 6, 7: Calculate deficits and sandwiches to make ---
        total_needed_from_kitchen_gf = 0
        total_needed_from_kitchen_reg = 0
        locations_needing_tray_move = set()

        for place in waiting_places:
            needed_gf_p = children_at_p_gf[place]
            needed_reg_p = children_at_p_reg[place]
            available_ontray_gf_p = sandwiches_ontray_at_p_gf[place]
            available_ontray_reg_p = sandwiches_ontray_at_p_reg[place]

            deficit_gf_p = max(0, needed_gf_p - available_ontray_gf_p)
            deficit_reg_p = max(0, needed_reg_p - available_ontray_reg_p)

            total_needed_from_kitchen_gf += deficit_gf_p
            total_needed_from_kitchen_reg += deficit_reg_p

            # Step 8: Identify locations needing a tray move
            # A location needs a tray move if there's a deficit of sandwiches
            # that are *not* already on trays at that location, AND there are no trays there.
            if (deficit_gf_p + deficit_reg_p > 0) and (trays_at_p[place] == 0):
                 locations_needing_tray_move.add(place)

        # Calculate sandwiches that must be made (needed from kitchen/creation minus those already in kitchen)
        must_make_gf = max(0, total_needed_from_kitchen_gf - sandwiches_kitchen_gf)
        must_make_reg = max(0, total_needed_from_kitchen_reg - sandwiches_kitchen_reg)

        # --- Step 9: Calculate heuristic components ---
        h = 0

        # Cost for 'serve' actions: 1 per unserved child
        h += len(unserved_children)

        # Cost for 'put_on_tray' actions: 1 per sandwich that needs to move from kitchen/creation to a tray at a location
        h += total_needed_from_kitchen_gf + total_needed_from_kitchen_reg

        # Cost for 'make_sandwich' actions: 1 per sandwich that must be made
        h += must_make_gf + must_make_reg

        # Cost for 'move_tray' actions: 1 per location needing a tray move
        h += len(locations_needing_tray_move)

        return h
