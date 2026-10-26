from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children their sandwiches. It considers the need for gluten-free options for allergic children and the movements of trays and sandwiches.

    # Assumptions:
    - Each child needs exactly one sandwich.
    - A sandwich can be made if the required bread and content portions are available.
    - Trays must be in the kitchen to make sandwiches.
    - A sandwich must be on a tray before it can be served.
    - Serving a sandwich may require moving the tray to the child's location.

    # Heuristic Initialization
    - Extracts static facts about children's allergies, gluten-free ingredients, and initial tray positions.
    - Uses these facts to compute the minimum number of actions needed.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served.
    2. Determine how many sandwiches are already made and how many more are needed.
    3. For each needed sandwich:
       a. Check if the required bread and content portions are available.
       b. If not, calculate the actions needed to obtain them.
       c. Ensure the tray is in the kitchen before making the sandwich.
    4. Calculate the actions needed to move the tray to the child's location if necessary.
    5. Sum all the actions needed for making and serving the sandwiches.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts about the domain."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.tray_positions = dict()

        for fact in static_facts:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten '):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(not_allergic_gluten '):
                self.not_allergic_children.add(parts[1])
            elif fact.startswith('(no_gluten_bread '):
                self.gluten_free_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content '):
                self.gluten_free_contents.add(parts[1])
            elif fact.startswith('(at tray'):
                tray = parts[1]
                place = parts[3]
                self.tray_positions[tray] = place

    def __call__(self, node):
        """Estimate the minimum number of actions needed to reach the goal."""
        state = node.state

        # Function to extract components from a fact string
        def get_parts(fact):
            return fact[1:-1].split()

        # Count the number of children waiting
        waiting_children = 0
        for fact in state:
            if fact.startswith('(waiting '):
                parts = get_parts(fact)
                child = parts[1]
                waiting_children += 1

        # Number of sandwiches needed
        sandwiches_needed = waiting_children

        # Count existing sandwiches
        existing_sandwiches = 0
        for fact in state:
            if fact.startswith('(at_kitchen_sandwich '):
                existing_sandwiches += 1

        # Calculate sandwiches still needed
        sandwiches_to_make = max(0, sandwiches_needed - existing_sandwiches)

        # Actions for making each sandwich
        actions = 0

        # For each sandwich to make
        for _ in range(sandwiches_to_make):
            # Check if a tray is available in the kitchen
            tray_in_kitchen = any(fact.startswith('(at tray') and get_parts(fact)[3] == 'kitchen' for fact in state)
            if not tray_in_kitchen:
                actions += 1  # Move tray to kitchen

            # Check if there are available bread and content portions
            available_breads = [fact for fact in state if fact.startswith('(at_kitchen_bread ') and fact not in state]
            available_contents = [fact for fact in state if fact.startswith('(at_kitchen_content ') and fact not in state]

            # For allergic children, need gluten-free ingredients
            # Assume one allergic child per sandwich for simplicity
            is_allergic = len(self.allergic_children) > 0
            if is_allergic:
                # Check if there are gluten-free bread and content
                has_gluten_free_bread = any(bread in self.gluten_free_breads for bread in available_breads)
                has_gluten_free_content = any(content in self.gluten_free_contents for content in available_contents)
                if not has_gluten_free_bread or not has_gluten_free_content:
                    actions += 2  # Need to get gluten-free ingredients

            # Make the sandwich
            actions += 1  # make_sandwich or make_sandwich_no_gluten

            # Put on tray
            actions += 1  # put_on_tray

        # Actions for serving each child
        for child in self.allergic_children.union(self.not_allergic_children):
            # Check if the child is waiting
            if f'(waiting {child} *)' in state:
                # Check if the sandwich is already on a tray at the child's place
                on_tray = any(fact.startswith('(ontray sandwich{0} tray'.format(child)) for fact in state)
                if not on_tray:
                    # Move tray to child's place
                    actions += 1  # move_tray
                # Serve the sandwich
                actions += 1  # serve_sandwich or serve_sandwich_no_gluten

        return actions
