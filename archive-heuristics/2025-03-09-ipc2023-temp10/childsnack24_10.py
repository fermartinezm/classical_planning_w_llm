from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class childsnack24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children.
    It considers the number of children waiting, the number of sandwiches that need to be made,
    and the number of trays that need to be moved.

    # Assumptions:
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten, depending on the child's allergies.
    - Trays are initially at the kitchen and may need to be moved to the serving place.

    # Heuristic Initialization
    - Identify children, their allergies, and their waiting places.
    - Identify bread and content portions available at the kitchen.
    - Identify trays and their initial locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of unserved children.
    2. For each unserved child, determine if they are allergic to gluten.
    3. Estimate the number of sandwiches that need to be made:
       - If there are more children allergic to gluten than gluten-free ingredients, estimate based on available ingredients.
       - If there are more children not allergic to gluten than regular ingredients, estimate based on available ingredients.
    4. Estimate the number of 'make_sandwich' actions needed.
    5. Estimate the number of 'put_on_tray' actions needed.
    6. Estimate the number of 'serve_sandwich' actions needed, considering gluten allergies.
    7. Estimate the number of 'move_tray' actions needed, if the tray is not at the waiting place.
    8. Sum up all the estimated action counts to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children = {}  # child: place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.bread = set()
        self.content = set()
        self.trays = set()
        self.tray_locations = {} # tray: place

        for fact in self.static:
            fact_parts = self._extract_objects_from_fact(fact)
            predicate = fact_parts[0]

            if predicate == 'allergic_gluten':
                self.allergic_children.add(fact_parts[1])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(fact_parts[1])
            elif predicate == 'waiting':
                self.waiting_children[fact_parts[1]] = fact_parts[2]
            elif predicate == 'no_gluten_bread':
                self.no_gluten_bread.add(fact_parts[1])
            elif predicate == 'no_gluten_content':
                self.no_gluten_content.add(fact_parts[1])

        for fact in task.initial_state:
            fact_parts = self._extract_objects_from_fact(fact)
            predicate = fact_parts[0]

            if predicate == 'at_kitchen_bread':
                self.bread.add(fact_parts[1])
            elif predicate == 'at_kitchen_content':
                self.content.add(fact_parts[1])
            elif predicate == 'at':
                self.trays.add(fact_parts[1])
                self.tray_locations[fact_parts[1]] = fact_parts[2]


    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state

        served_children = set()
        for fact in state:
            fact_parts = self._extract_objects_from_fact(fact)
            predicate = fact_parts[0]
            if predicate == 'served':
                served_children.add(fact_parts[1])

        unserved_children = set(self.allergic_children) | set(self.not_allergic_children) - served_children
        num_unserved_children = len(unserved_children)

        if num_unserved_children == 0 and self.goal_reached(state):
            return 0

        num_gluten_free_sandwiches_needed = 0
        num_regular_sandwiches_needed = 0

        for child in unserved_children:
            if child in self.allergic_children:
                num_gluten_free_sandwiches_needed += 1
            else:
                num_regular_sandwiches_needed += 1

        num_make_sandwich_actions = 0
        num_put_on_tray_actions = 0
        num_serve_sandwich_actions = 0
        num_move_tray_actions = 0

        # Estimate the number of sandwiches to make
        num_make_sandwich_actions = num_gluten_free_sandwiches_needed + num_regular_sandwiches_needed

        # Estimate the number of put_on_tray actions
        num_put_on_tray_actions = num_make_sandwich_actions

        # Estimate the number of serve_sandwich actions
        num_serve_sandwich_actions = num_unserved_children

        # Estimate the number of move_tray actions
        for tray, location in self.tray_locations.items():
            for child, place in self.waiting_children.items():
                if child not in served_children and location != place:
                    num_move_tray_actions += 1
                    break # only count one move per tray


        heuristic_value = (
            num_make_sandwich_actions +
            num_put_on_tray_actions +
            num_serve_sandwich_actions +
            num_move_tray_actions
        )

        return heuristic_value

    def _extract_objects_from_fact(self, fact):
        """Extract objects from a PDDL fact string."""
        return fact[1:-1].split()

    def goal_reached(self, state):
        """Check if the goal has been reached."""
        return self.goals <= state
