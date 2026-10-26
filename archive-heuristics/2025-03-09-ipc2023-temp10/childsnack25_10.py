from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class childsnack25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the actions of making sandwiches, putting them on trays, moving trays to the children, and serving the sandwiches.

    # Assumptions:
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten, depending on the child's allergies.
    - Trays can be moved between places.
    - We prioritize making sandwiches for allergic children first.

    # Heuristic Initialization
    - Extract information about children's allergies and their waiting locations from the static facts.
    - Identify available bread and content portions at the kitchen.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of children waiting to be served.
    2. For each child, determine if they are allergic to gluten.
    3. Count the number of gluten-free and regular sandwiches needed.
    4. Check the availability of bread and content portions at the kitchen.
    5. Estimate the number of 'make_sandwich' actions required based on the number of sandwiches needed and available ingredients.
    6. Estimate the number of 'put_on_tray' actions required based on the number of sandwiches made and available trays at the kitchen.
    7. Estimate the number of 'move_tray' actions required to bring the trays to the waiting children.
    8. Estimate the number of 'serve_sandwich' actions required to serve each child with the appropriate sandwich.
    9. Sum up the estimated costs of all actions to get the heuristic value.
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
        self.bread_at_kitchen = set()
        self.content_at_kitchen = set()

        for fact in self.static:
            fact = fact[1:-1]
            parts = fact.split()
            predicate = parts[0]

            if predicate == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            elif predicate == 'waiting':
                self.waiting_children[parts[1]] = parts[2]
            elif predicate == 'no_gluten_bread':
                self.no_gluten_bread.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.no_gluten_content.add(parts[1])

        for fact in task.initial_state:
            fact = fact[1:-1]
            parts = fact.split()
            predicate = parts[0]

            if predicate == 'at_kitchen_bread':
                self.bread_at_kitchen.add(parts[1])
            elif predicate == 'at_kitchen_content':
                self.content_at_kitchen.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if the current state is the goal state
        if self.goal_reached(state):
            return 0

        waiting_children_count = 0
        served_children_count = 0
        for fact in state:
            fact = fact[1:-1]
            parts = fact.split()
            predicate = parts[0]
            if predicate == 'waiting':
                waiting_children_count += 1
            elif predicate == 'served':
                served_children_count += 1

        num_waiting_children = len(self.waiting_children) - served_children_count

        if num_waiting_children <= 0:
            return 0

        num_allergic_waiting = 0
        num_not_allergic_waiting = 0

        for child in self.waiting_children:
            served = False
            for fact in state:
                fact = fact[1:-1]
                parts = fact.split()
                predicate = parts[0]
                if predicate == 'served' and parts[1] == child:
                    served = True
                    break
            if not served:
                if child in self.allergic_children:
                    num_allergic_waiting += 1
                else:
                    num_not_allergic_waiting += 1

        make_sandwich_actions = 0
        put_on_tray_actions = 0
        serve_sandwich_actions = num_waiting_children
        move_tray_actions = 0

        # Estimate make_sandwich actions
        num_sandwiches_needed = num_allergic_waiting + num_not_allergic_waiting
        make_sandwich_actions = num_sandwiches_needed

        # Estimate put_on_tray actions
        num_trays_needed = num_sandwiches_needed
        put_on_tray_actions = num_trays_needed

        # Estimate move_tray actions (simplified: one move per tray)
        move_tray_actions = 0
        trays_at_kitchen = 0
        trays_at_tables = 0
        for fact in state:
            fact = fact[1:-1]
            parts = fact.split()
            predicate = parts[0]
            if predicate == 'at' and parts[2] == 'kitchen' and parts[1].startswith('tray'):
                trays_at_kitchen += 1
            elif predicate == 'at' and parts[2] != 'kitchen' and parts[1].startswith('tray'):
                trays_at_tables += 1

        move_tray_actions = num_waiting_children

        return make_sandwich_actions + put_on_tray_actions + serve_sandwich_actions + move_tray_actions

    def goal_reached(self, state):
        """Check if the goal is reached in the given state."""
        return self.goals <= state
