from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class childsnack9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children.
    It considers the number of children waiting, the number of sandwiches that need to be made,
    and the number of trays that need to be moved.

    # Assumptions:
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten.
    - Trays are initially at the kitchen and may need to be moved to serve children.

    # Heuristic Initialization
    - Count the number of children allergic and not allergic to gluten.
    - Identify the available bread and content portions, distinguishing between gluten-free and regular options.
    - Store the locations of trays.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served, distinguishing between those with and without gluten allergies.
    2. Count the number of sandwiches that need to be made, considering the available ingredients.
    3. Estimate the number of 'make_sandwich' actions required based on the number of children and available ingredients.
    4. Estimate the number of 'put_on_tray' actions required for each sandwich.
    5. Estimate the number of 'serve_sandwich' actions required for each child.
    6. Estimate the number of 'move_tray' actions required to bring trays to the children.
    7. Sum up all the estimated actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.trays = set()

        for fact in static_facts:
            if match(fact, "(allergic_gluten", "*"):
                self.allergic_children.add(fact[1:-1].split()[1])
            elif match(fact, "(not_allergic_gluten", "*"):
                self.not_allergic_children.add(fact[1:-1].split()[1])
            elif match(fact, "(no_gluten_bread", "*"):
                self.no_gluten_bread.add(fact[1:-1].split()[1])
            elif match(fact, "(no_gluten_content", "*"):
                self.no_gluten_content.add(fact[1:-1].split()[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Check if the current state is the goal state
        if self.goals <= state:
            return 0

        waiting_allergic = 0
        waiting_not_allergic = 0
        for fact in state:
            if match(fact, "(waiting", "*", "*"):
                child = fact[1:-1].split()[1]
                if child in self.allergic_children:
                    waiting_allergic += 1
                elif child in self.not_allergic_children:
                    waiting_not_allergic += 1

        num_sandwiches_needed = waiting_allergic + waiting_not_allergic

        available_bread = 0
        available_content = 0
        available_no_gluten_bread = 0
        available_no_gluten_content = 0

        for fact in state:
            if match(fact, "(at_kitchen_bread", "*"):
                available_bread += 1
                bread = fact[1:-1].split()[1]
                if bread in self.no_gluten_bread:
                    available_no_gluten_bread += 1
            elif match(fact, "(at_kitchen_content", "*"):
                available_content += 1
                content = fact[1:-1].split()[1]
                if content in self.no_gluten_content:
                    available_no_gluten_content += 1

        num_make_sandwich_actions = 0
        if num_sandwiches_needed > 0:
            num_make_sandwich_actions = min(available_bread, available_content, num_sandwiches_needed)

        num_put_on_tray_actions = 0
        for fact in state:
            if match(fact, "(at_kitchen_sandwich", "*"):
                num_put_on_tray_actions += 1

        num_serve_sandwich_actions = waiting_allergic + waiting_not_allergic

        num_move_tray_actions = 0
        trays_at_kitchen = 0
        for fact in state:
            if match(fact, "(at", "*", "kitchen)"):
                trays_at_kitchen += 1
        
        num_move_tray_actions = 0
        waiting_places = set()
        for fact in state:
            if match(fact, "(waiting", "*", "*"):
                waiting_places.add(fact[1:-1].split()[2])
        
        trays_at_waiting_places = 0
        for fact in state:
            if match(fact, "(at", "*", "*"):
                place = fact[1:-1].split()[2]
                if place != "kitchen" and place in waiting_places:
                    trays_at_waiting_places += 1
        
        num_move_tray_actions = max(0, len(waiting_places) - trays_at_waiting_places)

        heuristic_value = (
            num_make_sandwich_actions
            + num_put_on_tray_actions
            + num_serve_sandwich_actions
            + num_move_tray_actions
        )

        return heuristic_value
