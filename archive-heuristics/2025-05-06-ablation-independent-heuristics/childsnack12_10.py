import re
from collections import defaultdict

# Assuming Heuristic base class and Task class are available from the planner environment
from heuristics.heuristic_base import Heuristic
from task import Task

# Helper function to parse a PDDL fact string
def parse_fact(fact_str):
    """Parses a PDDL fact string into predicate and objects."""
    # Removes leading/trailing parentheses and splits by spaces
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal state (all children served)
    by summing the estimated costs for each unserved child. The cost for serving
     a child is estimated based on the "readiness" of a suitable sandwich.
    Sandwiches are categorized into readiness levels based on their current state
    and location:
    Level 4: On a tray at a location where children are waiting (cost 1: serve).
    Level 3: On a tray at a location where no children are waiting (cost 2: move_tray + serve).
    Level 2: At the kitchen (cost 3: put_on_tray + move_tray + serve).
    Level 1: Does not exist yet (cost 4: make_sandwich + put_on_tray + move_tray + serve).
    The heuristic prioritizes using sandwiches from higher readiness levels and
    prioritizes serving allergic children with gluten-free sandwiches.

    Assumptions:
    - The problem instance is solvable (enough total resources exist).
    - Tray capacity is effectively infinite or sufficient.
    - Ingredients at the kitchen are sufficient to make needed sandwiches if sandwich objects exist.
    - The heuristic is not admissible; it's designed for greedy best-first search.

    Heuristic Initialization:
    The constructor parses the static facts from the task to identify allergic/non-allergic
    children and their waiting locations. This information is stored for use in
    the heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify:
        - Which children are already served.
        - The location of each tray.
        - The state of each sandwich (at_kitchen_sandwich, ontray).
        - Whether a sandwich is gluten-free.
        - Available sandwich objects (`notexist`).
    2.  Identify unserved children and group them by location and allergy status
        (allergic/non-allergic).
    3.  Identify places where unserved children are waiting.
    4.  Count available sandwiches by type (GF/Reg) and readiness level:
        - Level 4: On tray at a place with unserved children.
        - Level 3: On tray at a place without unserved children (includes kitchen).
        - Level 2: At the kitchen.
        - Level 1: Non-existent sandwich objects (`notexist`).
    5.  Calculate the total number of GF sandwiches needed (for unserved allergic children).
    6.  Calculate the total number of Any sandwiches needed (for unserved non-allergic children).
    7.  Allocate available GF sandwiches to satisfy the GF need, starting from Level 4 down to Level 1.
        - For each GF sandwich used from Level k, add k to the total heuristic cost.
        - Keep track of remaining GF sandwiches at each level.
        - If GF need cannot be met, return infinity.
    8.  Allocate available Any sandwiches (remaining GF + available Reg) to satisfy the Any need, starting from Level 4 down to Level 1.
        - For each Any sandwich used from Level k, add k to the total heuristic cost.
        - Keep track of remaining sandwiches at each level.
        - If Any need cannot be met, return infinity.
    9.  The total heuristic value is the sum of costs accumulated in steps 7 and 8.
    10. If the goal is already reached (all children served), the heuristic is 0.
    """

    def __init__(self, task):
        super().__init__(task)
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.child_location = {}

        # Parse static facts once during initialization
        for fact_str in task.static:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'allergic_gluten':
                self.allergic_children.add(objects[0])
            elif predicate == 'not_allergic_gluten':
                self.non_allergic_children.add(objects[0])
            elif predicate == 'waiting':
                self.child_location[objects[0]] = objects[1]

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # --- Step 1: Parse State ---
        served_children = set()
        tray_location = {} # {tray_name: place_name}
        sandwich_state = {} # {sandwich_name: 'kitchen' or 'ontray'}
        sandwich_ontray_tray = {} # {sandwich_name: tray_name} if state is 'ontray'
        gf_sandwich_names_in_state = set()
        num_notexist = 0

        for fact_str in state:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'served':
                served_children.add(objects[0])
            elif predicate == 'at': # (at ?t ?p)
                tray_location[objects[0]] = objects[1]
            elif predicate == 'at_kitchen_sandwich': # (at_kitchen_sandwich ?s)
                sandwich_name = objects[0]
                sandwich_state[sandwich_name] = 'kitchen'
            elif predicate == 'ontray': # (ontray ?s ?t)
                sandwich_name = objects[0]
                tray_name = objects[1]
                sandwich_state[sandwich_name] = 'ontray'
                sandwich_ontray_tray[sandwich_name] = tray_name
            elif predicate == 'no_gluten_sandwich': # (no_gluten_sandwich ?s)
                 gf_sandwich_names_in_state.add(objects[0])
            elif predicate == 'notexist': # (notexist ?s)
                 num_notexist += 1

        # --- Step 2 & 3: Identify Unserved Children and Places ---
        unserved_children = {c for c in self.child_location.keys() if c not in served_children}
        places_with_unserved = {self.child_location[c] for c in unserved_children}

        # --- Step 4: Count Available Sandwiches by Type and Readiness Level ---
        Avail_GF_L4 = 0 # On tray at a place with unserved children
        Avail_Reg_L4 = 0 # On tray at a place with unserved children
        Avail_GF_L3 = 0 # On tray elsewhere (place without unserved children, including kitchen)
        Avail_Reg_L3 = 0 # On tray elsewhere (place without unserved children, including kitchen)
        Avail_GF_L2 = 0 # At the kitchen
        Avail_Reg_L2 = 0 # At the kitchen
        Avail_L1 = num_notexist # Non-existent sandwich objects

        for s, state_type in sandwich_state.items():
            is_gf = s in gf_sandwich_names_in_state

            if state_type == 'kitchen':
                if is_gf: Avail_GF_L2 += 1
                else: Avail_Reg_L2 += 1
            elif state_type == 'ontray':
                tray_name = sandwich_ontray_tray.get(s)
                if tray_name and tray_name in tray_location:
                    s_place = tray_location[tray_name]
                    if s_place in places_with_unserved:
                        # On tray at a place with unserved children (Level 4)
                        if is_gf: Avail_GF_L4 += 1
                        else: Avail_Reg_L4 += 1
                    else:
                        # On tray elsewhere (Level 3) - includes kitchen if tray is there
                        if is_gf: Avail_GF_L3 += 1
                        else: Avail_Reg_L3 += 1
                # else: sandwich is on tray but tray location is unknown/not a named place.
                # These sandwiches are not counted in L3/L4. This is acceptable for a heuristic.


        # --- Step 5 & 6: Calculate Needs ---
        N_allergic_unserved = len([c for c in unserved_children if c in self.allergic_children])
        N_non_allergic_unserved = len([c for c in unserved_children if c in self.non_allergic_children])

        total_heuristic_cost = 0

        # --- Step 7: Allocate GF sandwiches for Allergic Needs ---
        Needed_GF = N_allergic_unserved

        # Use L4 GF
        use = min(Needed_GF, Avail_GF_L4)
        total_heuristic_cost += use * 1
        Needed_GF -= use
        Avail_GF_L4 -= use # Consume the available resource

        # Use L3 GF
        use = min(Needed_GF, Avail_GF_L3)
        total_heuristic_cost += use * 2
        Needed_GF -= use
        Avail_GF_L3 -= use # Consume the available resource

        # Use L2 GF
        use = min(Needed_GF, Avail_GF_L2)
        total_heuristic_cost += use * 3
        Needed_GF -= use
        Avail_GF_L2 -= use # Consume the available resource

        # Use L1 GF (make new)
        use = min(Needed_GF, Avail_L1) # Limited by notexist objects
        total_heuristic_cost += use * 4
        Needed_GF -= use
        Avail_L1 -= use # Consume the available resource

        # If GF need cannot be met, return infinity
        if Needed_GF > 0:
            return float('inf')

        # --- Step 8: Allocate Any sandwiches for Non-Allergic Needs ---
        Needed_Any = N_non_allergic_unserved

        # Available Any at each level (remaining GF + available Reg)
        # Note: Avail_GF_L4, Avail_GF_L3, Avail_GF_L2 have been reduced by GF needs
        Avail_Any_L4 = Avail_GF_L4 + Avail_Reg_L4
        Avail_Any_L3 = Avail_GF_L3 + Avail_Reg_L3
        Avail_Any_L2 = Avail_GF_L2 + Avail_Reg_L2
        Avail_Any_L1 = Avail_L1 # Avail_L1 has been reduced by GF needs

        # Use L4 Any
        use = min(Needed_Any, Avail_Any_L4)
        total_heuristic_cost += use * 1
        Needed_Any -= use
        # No need to track consumption of GF/Reg separately here, as they are combined

        # Use L3 Any
        use = min(Needed_Any, Avail_Any_L3)
        total_heuristic_cost += use * 2
        Needed_Any -= use

        # Use L2 Any
        use = min(Needed_Any, Avail_Any_L2)
        total_heuristic_cost += use * 3
        Needed_Any -= use

        # Use L1 Any (make new)
        use = min(Needed_Any, Avail_Any_L1)
        total_heuristic_cost += use * 4
        Needed_Any -= use

        # If Any need cannot be met, return infinity
        if Needed_Any > 0:
            return float('inf')

        # --- Step 9: Return Total Heuristic Cost ---
        return total_heuristic_cost
