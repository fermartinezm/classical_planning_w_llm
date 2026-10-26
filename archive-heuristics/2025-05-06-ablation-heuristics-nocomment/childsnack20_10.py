# Assuming heuristics.heuristic_base provides the Heuristic class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the cost to reach the goal state (all children served) by summing
    the estimated costs for the remaining steps for each unserved child. It
    considers the need to make sandwiches, put them on trays, move trays to
    the children's locations, and finally serve the children.

    Assumptions:
    - The heuristic is not admissible; it may overestimate the cost.
    - Sufficient bread, content, and 'notexist' sandwich objects are available
      in the kitchen to make any required new sandwiches.
    - Sufficient trays are available to be moved to locations needing them.
    - The predicate '(at ?t ?p)' is only used for trays.

    Heuristic Initialization:
    In the __init__ method, the heuristic extracts and stores static information
    from the task:
    - The set of children that need to be served (from the goal).
    - The allergy status (allergic_gluten or not_allergic_gluten) for each child.
    - The waiting place for each child.
    - The set of gluten-free bread and content portions (though these are not
      directly used in the current heuristic calculation, they are part of
      the domain's static info and could be used in more complex heuristics).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (node.state):
    1. Identify the set of children who are in the goal but not yet served
       in the current state. These are the 'unserved_children'.
    2. If the set of unserved children is empty, the goal is reached, and the
       heuristic value is 0.
    3. Initialize the heuristic value with the number of unserved children.
       This represents the minimum number of 'serve' actions required (1 per child).
    4. Categorize the unserved children into those needing gluten-free
       sandwiches ('N_unserved_gf') and those needing regular sandwiches
       ('N_unserved_reg'), using the static allergy information.
    5. Identify the set of gluten-free sandwiches that exist in the current state
       (those with the property '(no_gluten_sandwich s)').
    6. Count suitable sandwiches available anywhere (on trays or in the kitchen):
       - Count GF sandwiches on trays ('N_gf_on_trays').
       - Count regular sandwiches on trays ('N_reg_on_trays').
       - Count GF sandwiches in the kitchen ('N_gf_in_kitchen').
       - Count regular sandwiches in the kitchen ('N_reg_in_kitchen').
       Total available GF = N_gf_on_trays + N_gf_in_kitchen.
       Total available Reg = N_reg_on_trays + N_reg_in_kitchen.
    7. Calculate the number of *new* gluten-free sandwiches that still need
       to be made: `num_make_gf = max(0, N_unserved_gf - Total available GF)`.
    8. Calculate the number of *new* regular sandwiches that still need
       to be made: `num_make_reg = max(0, N_unserved_reg - Total available Reg)`.
    9. Sum the counts from steps 7 and 8 to get the total number of sandwiches
       that need to be made ('num_to_make'). Add 'num_to_make * 1' to the
       heuristic value (cost of 'make_sandwich' action).
    10. Calculate the number of GF sandwiches needed that are *not* already on trays:
        `num_gf_not_on_trays = max(0, N_unserved_gf - N_gf_on_trays)`.
    11. Calculate the number of regular sandwiches needed that are *not* already on trays:
        `num_reg_not_on_trays = max(0, N_unserved_reg - N_reg_on_trays)`.
    12. Sum the counts from steps 10 and 11 to get the total number of sandwiches
        that need to be put on trays ('num_to_put'). Add 'num_to_put * 1' to the
        heuristic value (cost of 'put_on_tray' action).
    13. Identify the set of distinct places where unserved children are waiting,
        using the static waiting place information.
    14. Identify the set of distinct places where trays are currently located
        in the state by checking '(at t p)' facts for trays.
    15. Count the number of places from step 13 that are *not* in the set
        from step 14. These are places where a tray needs to be moved.
    16. Add this count to the heuristic value. This estimates the cost of the
        'move_tray' actions required (1 per location needing a tray).
    17. The final sum is the heuristic estimate for the current state.
    """
    def __init__(self, task):
        super().__init__(task)
        # Extract static information
        self.goal_children = set()
        self.child_allergy = {} # child -> True if allergic, False otherwise
        self.child_waiting_place = {} # child -> place
        # Although extracted, gf_bread and gf_content are not used in this specific heuristic logic
        self.gf_bread = set()
        self.gf_content = set()

        # Assuming task.goals is a frozenset of goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

        # Assuming task.static is a frozenset of static facts
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'allergic_gluten':
                self.child_allergy[parts[1]] = True
            elif pred == 'not_allergic_gluten':
                self.child_allergy[parts[1]] = False
            elif pred == 'waiting':
                self.child_waiting_place[parts[1]] = parts[2]
            elif pred == 'no_gluten_bread':
                self.gf_bread.add(parts[1])
            elif pred == 'no_gluten_content':
                self.gf_content.add(parts[1])

    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_children = self.goal_children - served_children

        # 2. If all goal children are served, the heuristic is 0.
        num_unserved_children = len(unserved_children)
        if num_unserved_children == 0:
            return 0

        # 3. Initialize heuristic with base cost (serve action)
        h = num_unserved_children # 1 action per child

        # 4. Categorize unserved children by allergy
        N_unserved_gf = sum(1 for c in unserved_children if self.child_allergy.get(c, False))
        N_unserved_reg = num_unserved_children - N_unserved_gf

        # 5. Identify GF sandwiches that exist in the state
        no_gluten_sandwich_names_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'no_gluten_sandwich'}

        # 6. Count suitable sandwiches available anywhere
        ontray_facts = {fact for fact in state if get_parts(fact)[0] == 'ontray'}
        at_kitchen_sandwich_facts = {fact for fact in state if get_parts(fact)[0] == 'at_kitchen_sandwich'}

        N_gf_on_trays = sum(1 for fact in ontray_facts if get_parts(fact)[1] in no_gluten_sandwich_names_in_state)
        N_reg_on_trays = sum(1 for fact in ontray_facts if get_parts(fact)[1] not in no_gluten_sandwich_names_in_state)

        N_gf_in_kitchen = sum(1 for fact in at_kitchen_sandwich_facts if get_parts(fact)[1] in no_gluten_sandwich_names_in_state)
        N_reg_in_kitchen = sum(1 for fact in at_kitchen_sandwich_facts if get_parts(fact)[1] not in no_gluten_sandwich_names_in_state)

        N_gf_available = N_gf_on_trays + N_gf_in_kitchen
        N_reg_available = N_reg_on_trays + N_reg_in_kitchen

        # 7 & 8. Calculate new sandwiches to make
        num_make_gf = max(0, N_unserved_gf - N_gf_available)
        num_make_reg = max(0, N_unserved_reg - N_reg_available)
        num_to_make = num_make_gf + num_make_reg

        # 9. Add cost for making new sandwiches
        h += num_to_make * 1 # 1 action for make_sandwich

        # 10 & 11. Calculate sandwiches needed that are not on trays
        # These need to be put on trays.
        num_gf_not_on_trays = max(0, N_unserved_gf - N_gf_on_trays)
        num_reg_not_on_trays = max(0, N_unserved_reg - N_reg_on_trays)
        num_to_put = num_gf_not_on_trays + num_reg_not_on_trays

        # 12. Add cost for putting sandwiches on trays
        h += num_to_put * 1 # 1 action for put_on_tray

        # 13. Identify places with waiting children
        places_with_unserved = {self.child_waiting_place[c] for c in unserved_children if c in self.child_waiting_place}

        # 14. Identify places with trays
        places_with_trays = {get_parts(fact)[2] for fact in state if get_parts(fact)[0] == 'at' and get_parts(fact)[1].startswith('tray')}

        # 15. Count places needing a tray
        num_places_with_waiting_children_but_no_tray = len(places_with_unserved - places_with_trays)

        # 16. Add cost for moving trays
        h += num_places_with_waiting_children_but_no_tray * 1 # 1 action for move_tray

        # 17. Return total heuristic value
        return h
