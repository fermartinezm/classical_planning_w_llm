"""
Domain-dependent heuristic for the childsnacks domain.

This heuristic estimates the number of actions required to reach a goal state
by summing up the estimated costs of several sub-problems:
1. Serving unserved children.
2. Making enough sandwiches of the correct type.
3. Putting sandwiches onto trays.
4. Moving trays to the locations where children are waiting.

The heuristic is non-admissible and designed to guide a greedy best-first search.
"""

from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Helper function to parse a PDDL fact string into a list of parts.
    Removes surrounding parentheses and splits by space.
    e.g., '(at tray1 kitchen)' -> ['at', 'tray1', 'kitchen']
    """
    # Handle potential empty facts or malformed strings gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.
    Estimates the remaining actions based on unserved children,
    sandwich availability, and tray locations.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by extracting static information from the task.

        Heuristic Initialization:
        - Stores the task goals.
        - Processes static facts and initial state to identify:
            - All children and their allergy status (allergic_gluten or not).
            - All children and their initial waiting place.
            - All objects of types tray, place, sandwich, bread-portion, content-portion
              present in the initial state or static facts.
        - This information is stored in instance variables for quick access
          during heuristic computation.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.child_allergy = {} # child -> 'gluten' or 'none'
        self.child_initial_place = {} # child -> place
        self.all_children = set()
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant place
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()

        # Collect all objects and static information from initial state and static facts
        all_facts = initial_state | static_facts
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts if any
             pred = parts[0]

             if pred == 'allergic_gluten':
                 if len(parts) > 1:
                     child = parts[1]
                     self.child_allergy[child] = 'gluten'
                     self.all_children.add(child)
             elif pred == 'not_allergic_gluten':
                 if len(parts) > 1:
                     child = parts[1]
                     self.child_allergy[child] = 'none'
                     self.all_children.add(child)
             elif pred == 'waiting':
                 if len(parts) > 2: # Ensure enough parts for child and place
                     child, place = parts[1], parts[2]
                     self.child_initial_place[child] = place
                     self.all_children.add(child)
                     self.all_places.add(place)
             elif pred == 'at':
                 if len(parts) > 2: # Ensure enough parts for obj and place
                     obj1, obj2 = parts[1], parts[2]
                     if obj1.startswith('tray'): self.all_trays.add(obj1)
                     if obj2.startswith('table') or obj2 == 'kitchen': self.all_places.add(obj2)
             elif pred == 'at_kitchen_bread' or pred == 'no_gluten_bread':
                 if len(parts) > 1: self.all_bread.add(parts[1])
             elif pred == 'at_kitchen_content' or pred == 'no_gluten_content':
                 if len(parts) > 1: self.all_content.add(parts[1])
             elif pred in ['at_kitchen_sandwich', 'ontray', 'notexist', 'no_gluten_sandwich']:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])

        # Ensure all children mentioned in goals are included
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served' and len(parts) > 1:
                self.all_children.add(parts[1])

        # Ensure all places mentioned in initial tray locations or child initial places are included
        # This is mostly covered by the loop over all_facts, but explicitly adding places
        # from initial_state 'at' facts and child_initial_place values ensures completeness.
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) > 2 and parts[1] in self.all_trays:
                 self.all_places.add(parts[2])
        for place in self.child_initial_place.values():
             self.all_places.add(place)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify Unserved Children: Determine which children were initially
           waiting but have not yet been served in the current state. The count
           of these children represents the minimum number of 'serve' actions
           needed. Add this count to the heuristic. If this count is zero,
           the goal is reached, and the heuristic is 0.
        2. Count Required Sandwiches (Make Actions):
           - Determine how many unserved children are allergic (requiring GF
             sandwiches) and how many are not allergic.
           - Count the number of available GF sandwiches and regular sandwiches
             in the current state (either in the kitchen or on trays).
           - Calculate the number of additional GF sandwiches that need to be
             made to satisfy allergic children.
           - Calculate the number of additional regular sandwiches that need to
             be made to satisfy non-allergic children, considering that surplus
             GF sandwiches can be used for non-allergic children.
           - The sum of these two counts is the estimated number of 'make_sandwich'
             actions needed. Add this to the heuristic.
           Assumptions: This step assumes sufficient bread and content portions
           are available to make the required sandwiches.
        3. Count Required Put-on-Tray Actions:
           - Determine the total number of sandwiches needed on trays (equal to
             the number of unserved children).
           - Count the number of sandwiches already on trays in the current state.
           - The difference (if positive) is the estimated number of 'put_on_tray'
             actions needed to move sandwiches from the kitchen onto trays. Add
             this difference to the heuristic.
           Assumptions: This step assumes trays are available in the kitchen
           when needed for putting sandwiches on them.
        4. Count Required Move-Tray Actions:
           - Identify all distinct places where unserved children are waiting.
           - Identify all distinct places where trays are currently located.
           - For each place where unserved children are waiting (excluding the
             kitchen, as serving from the kitchen is not possible), if no tray
             is currently located at that place, a 'move_tray' action is needed
             to bring a tray there.
           - Count the number of such places. This count is the estimated number
             of 'move_tray' actions needed. Add this count to the heuristic.
           Assumptions: This step assumes a tray is available to be moved to
           each required location.

        The total heuristic value is the sum of the counts from steps 1, 2, 3, and 4.
        """
        state = node.state
        heuristic = 0

        # 1. Identify unserved waiting children
        unserved_children = {} # child -> place
        for child in self.all_children:
            # Check if the child was initially waiting
            if child in self.child_initial_place:
                # Check if the child is not yet served
                if '(served {})'.format(child) not in state:
                    unserved_children[child] = self.child_initial_place[child]

        N_unserved = len(unserved_children)

        # If no children are unserved, goal is reached
        if N_unserved == 0:
            return 0

        # Base heuristic: one action per child to serve them
        heuristic += N_unserved

        # Categorize unserved children by allergy
        allergic_unserved = {c: p for c, p in unserved_children.items() if self.child_allergy.get(c) == 'gluten'}
        # Children not in child_allergy map are assumed not allergic
        not_allergic_unserved = {c: p for c, p in unserved_children.items() if self.child_allergy.get(c) != 'gluten'}

        N_allergic_unserved = len(allergic_unserved)
        N_not_allergic_unserved = len(not_allergic_unserved)

        # 2. Count required sandwiches (make actions)
        available_sandwiches = {s for s in self.all_sandwiches if '(at_kitchen_sandwich {})'.format(s) in state or any('(ontray {} {})'.format(s, t) in state for t in self.all_trays)}
        available_gf_sandwiches = {s for s in available_sandwiches if '(no_gluten_sandwich {})'.format(s) in state}
        available_reg_sandwiches = available_sandwiches - available_gf_sandwiches

        N_available_gf = len(available_gf_sandwiches)
        N_available_reg = len(available_reg_sandwiches)

        # GF sandwiches needed for allergic children
        N_gf_to_make = max(0, N_allergic_unserved - N_available_gf)

        # Sandwiches needed for non-allergic children (can be reg or surplus GF)
        N_needed_for_non_allergic = N_not_allergic_unserved
        N_available_for_non_allergic = N_available_reg + max(0, N_available_gf - N_allergic_unserved)
        N_reg_to_make = max(0, N_needed_for_non_allergic - N_available_for_non_allergic)

        heuristic += N_gf_to_make + N_reg_to_make # Add make actions

        # 3. Count required put_on_tray actions
        N_sandwiches_ontray = sum(1 for s in self.all_sandwiches if any('(ontray {} {})'.format(s, t) in state for t in self.all_trays))

        # Need N_unserved sandwiches on trays.
        # We need to put max(0, N_unserved - N_sandwiches_ontray) more sandwiches on trays.
        # This assumes we have enough sandwiches available (which is covered by the make actions heuristic).
        heuristic += max(0, N_unserved - N_sandwiches_ontray) # Add put_on_tray actions

        # 4. Count required move_tray actions
        places_with_unserved = set(unserved_children.values())
        tray_locations = {p for t in self.all_trays for p in self.all_places if '(at {} {})'.format(t, p) in state}

        # Count places (excluding kitchen) where unserved children are waiting but no tray is present
        # A tray at the kitchen doesn't help serve children waiting elsewhere.
        N_move_tray_needed = sum(1 for place in places_with_unserved if place != 'kitchen' and place not in tray_locations)

        heuristic += N_move_tray_needed # Add move_tray actions

        return heuristic
