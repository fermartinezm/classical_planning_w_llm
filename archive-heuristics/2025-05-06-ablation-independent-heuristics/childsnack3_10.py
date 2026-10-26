# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, [args])."""
    # Remove surrounding brackets and split by space
    parts = fact_string.strip('()').split()
    predicate = parts[0]
    args = parts[1:]
    return (predicate, args)

# Assuming the following classes are available from the planner environment
from heuristics.heuristic_base import Heuristic
from task import Operator, Task

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to serve all unserved
    children. It decomposes the problem into subgoals for each unserved child
    and sums up the estimated actions for the necessary steps:
    1. Making a suitable sandwich if none is available.
    2. Putting a kitchen sandwich onto a tray if needed.
    3. Moving a tray to the kitchen if needed for step 2.
    4. Moving a tray to a child's location if no tray is present there.
    5. Serving the sandwich to the child.

    Assumptions:
    - The problem instance is solvable. This implies sufficient total resources
      (bread, content, sandwich objects, trays) exist across the initial state
      to satisfy the goal, even if they are not currently in the required
      locations or forms.
    - The heuristic does not attempt to match specific sandwiches to specific
      children or trays, nor does it plan optimal tray movements involving
      multiple stops. It provides a relaxed estimate based on counts of
      needed items and locations.
    - The heuristic is designed for greedy best-first search and does not need
      to be admissible.

    Heuristic Initialization:
    The constructor extracts static information from the task definition:
    - The set of all children that need to be served (from the goal).
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - The waiting place for each child.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify the set of children who are not yet served by comparing the
        current state's 'served' facts with the goal children identified during
        initialization. If all children are served, the heuristic is 0.
    2.  The number of 'serve' actions needed is equal to the number of unserved
        children. Add this to the total heuristic.
    3.  Categorize unserved children into those needing gluten-free sandwiches
        (allergic) and those needing regular sandwiches (not allergic).
    4.  Count the available sandwiches in the current state, distinguishing
        between gluten-free and regular, and whether they are in the kitchen
        ('at_kitchen_sandwich') or already on a tray ('ontray').
    5.  Calculate the number of *new* gluten-free sandwiches that must be made:
        This is the number of unserved allergic children minus the total number
        of available gluten-free sandwiches (in kitchen or on trays), minimum 0.
        Add this count to the total heuristic (each make action costs 1).
    6.  Calculate the number of *new* regular sandwiches that must be made:
        This is the number of unserved non-allergic children minus the total
        number of available regular sandwiches (in kitchen or on trays), minimum 0.
        Add this count to the total heuristic (each make action costs 1).
    7.  Calculate the number of gluten-free sandwiches currently in the kitchen
        ('at_kitchen_sandwich') that are needed to serve unserved allergic
        children. This is the minimum of the available GF kitchen sandwiches
        and the number of unserved allergic children who still need a GF
        sandwich after accounting for those already on trays and those that
        need to be made. Add this count to the total heuristic (each put_on_tray
        action costs 1).
    8.  Calculate the number of regular sandwiches currently in the kitchen
        ('at_kitchen_sandwich') that are needed to serve unserved non-allergic
        children. This is the minimum of the available Reg kitchen sandwiches
        and the number of unserved non-allergic children who still need a Reg
        sandwich after accounting for those already on trays and those that
        need to be made. Add this count to the total heuristic (each put_on_tray
        action costs 1).
    9.  Check if any 'put_on_tray' actions are needed (i.e., if the sum from
        steps 7 and 8 is greater than 0). If so, check if there is any tray
        currently located in the 'kitchen'. If not, one 'move_tray' action is
        needed to bring a tray to the kitchen. Add 1 to the heuristic if this
        move is necessary.
    10. Identify the set of distinct places where unserved children are waiting.
    11. Identify the set of distinct places where trays are currently located.
    12. Calculate the number of distinct places from step 10 that are *not*
        in the set from step 11. This is the number of places that need a tray
        moved to them. Add this count to the total heuristic (each move_tray
        action costs 1).
    13. The total heuristic value is the sum of costs from steps 2, 5, 6, 7, 8, 9, and 12.
    """

    def __init__(self, task):
        super().__init__(task) # Call parent constructor
        self.task = task
        self.goal_children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children = {} # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        # Extract goal children
        for goal_fact_string in task.goals:
            pred, args = parse_fact(goal_fact_string)
            if pred == 'served' and len(args) == 1:
                self.goal_children.add(args[0])

        # Extract static information
        for static_fact_string in task.static:
            pred, args = parse_fact(static_fact_string)
            if pred == 'allergic_gluten' and len(args) == 1:
                self.allergic_children.add(args[0])
            elif pred == 'not_allergic_gluten' and len(args) == 1:
                self.not_allergic_children.add(args[0])
            elif pred == 'waiting' and len(args) == 2:
                self.waiting_children[args[0]] = args[1] # child -> place
            elif pred == 'no_gluten_bread' and len(args) == 1:
                self.no_gluten_bread.add(args[0])
            elif pred == 'no_gluten_content' and len(args) == 1:
                self.no_gluten_content.add(args[0])

    def __call__(self, node):
        state = node.state

        # Parse current state facts
        current_served_children = set()
        current_at_kitchen_bread = set()
        current_at_kitchen_content = set()
        current_at_kitchen_sandwich = set()
        current_ontray = {} # sandwich -> tray
        current_at_tray = {} # tray -> place
        current_no_gluten_sandwich = set()
        current_notexist_sandwich = set()

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'served' and len(args) == 1:
                current_served_children.add(args[0])
            elif pred == 'at_kitchen_bread' and len(args) == 1:
                current_at_kitchen_bread.add(args[0])
            elif pred == 'at_kitchen_content' and len(args) == 1:
                current_at_kitchen_content.add(args[0])
            elif pred == 'at_kitchen_sandwich' and len(args) == 1:
                current_at_kitchen_sandwich.add(args[0])
            elif pred == 'ontray' and len(args) == 2:
                current_ontray[args[0]] = args[1] # sandwich -> tray
            elif pred == 'at' and len(args) == 2:
                current_at_tray[args[0]] = args[1] # tray -> place
            elif pred == 'no_gluten_sandwich' and len(args) == 1:
                current_no_gluten_sandwich.add(args[0])
            elif pred == 'notexist' and len(args) == 1:
                current_notexist_sandwich.add(args[0])

        # 1. Identify unserved children
        unserved_children = self.goal_children - current_served_children

        # Goal reached
        if not unserved_children:
            return 0

        # Initialize heuristic components
        h_make = 0
        h_put_on_tray = 0
        h_tray_to_kitchen = 0
        h_move_tray = 0
        h_serve = 0

        # 2. Cost for serving
        h_serve = len(unserved_children)

        # 3. Categorize unserved children
        unserved_allergic = unserved_children & self.allergic_children
        unserved_non_allergic = unserved_children & self.not_allergic_children

        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # 4. Count available sandwiches by type and location
        S_gf_kitchen = {s for s in current_at_kitchen_sandwich if s in current_no_gluten_sandwich}
        S_reg_kitchen = {s for s in current_at_kitchen_sandwich if s not in current_no_gluten_sandwich}
        S_gf_ontray = {s for s in current_ontray.keys() if s in current_no_gluten_sandwich}
        S_reg_ontray = {s for s in current_ontray.keys() if s not in current_no_gluten_sandwich}

        S_gf_avail = S_gf_kitchen | S_gf_ontray
        S_reg_avail = S_reg_kitchen | S_reg_ontray

        # 5 & 6. Cost for making sandwiches
        # Number of *additional* GF/Reg sandwiches needed beyond what's available
        Make_gf = max(0, N_allergic_unserved - len(S_gf_avail))
        Make_reg = max(0, N_non_allergic_unserved - len(S_reg_avail))
        h_make = Make_gf + Make_reg

        # 7 & 8. Cost for putting kitchen sandwiches on tray
        # These are sandwiches currently in the kitchen that are needed.
        # The number needed from kitchen is the total needed minus those already on trays and those we have to make.
        Needed_from_kitchen_gf = max(0, N_allergic_unserved - len(S_gf_ontray) - Make_gf)
        Needed_from_kitchen_reg = max(0, N_non_allergic_unserved - len(S_reg_ontray) - Make_reg)

        Use_kitchen_gf = min(len(S_gf_kitchen), Needed_from_kitchen_gf)
        Use_kitchen_reg = min(len(S_reg_kitchen), Needed_from_kitchen_reg)

        h_put_on_tray = Use_kitchen_gf + Use_kitchen_reg

        # 9. Cost for moving a tray to the kitchen if needed for put_on_tray
        if h_put_on_tray > 0:
            tray_at_kitchen = any(place == 'kitchen' for place in current_at_tray.values())
            if not tray_at_kitchen:
                h_tray_to_kitchen = 1

        # 10, 11, 12. Cost for moving trays to children's locations
        # Identify places where unserved children are waiting
        places_waiting = {self.waiting_children[c] for c in unserved_children if c in self.waiting_children}
        # Identify places where trays are currently located
        places_with_trays = set(current_at_tray.values())
        # Count distinct places needing a tray where none is present
        places_needing_tray_moved = places_waiting - places_with_trays
        h_move_tray = len(places_needing_tray_moved)

        # 13. Total heuristic
        total_heuristic = h_make + h_put_on_tray + h_tray_to_kitchen + h_move_tray + h_serve

        return total_heuristic
