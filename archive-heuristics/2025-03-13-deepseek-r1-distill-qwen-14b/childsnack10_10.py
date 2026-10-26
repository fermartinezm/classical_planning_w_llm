from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - Sandwiches can be made with or without gluten depending on the child's allergy.
    - Trays must be moved to the child's location before serving.

    # Heuristic Initialization
    - Extract static facts about gluten-free ingredients and children's allergies.
    - Track the initial locations of trays and available ingredients.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting for their sandwiches.
    2. Determine how many sandwiches have already been made and served.
    3. Calculate the number of sandwiches still needed.
    4. For each needed sandwich:
       a. Check if the required ingredients (bread and content) are available.
       b. Determine if a tray is needed to move the sandwich to the child.
       c. Calculate the actions required to make, move, and serve the sandwich.
    5. Sum the actions required for all needed sandwiches to get the total heuristic value.
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
        self.waiting_locations = dict()
        self.trays = set()

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
            elif fact.startswith('(waiting '):
                child = parts[1]
                location = parts[3]
                self.waiting_locations[child] = location
            elif fact.startswith('(at tray'):
                self.trays.add(parts[1])

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all children."""
        state = node.state

        # Extract current state information
        current_trays = set()
        current_sandwiches = set()
        current_breads = set()
        current_contents = set()
        current_served = set()
        current_allergic = set()
        current_not_allergic = set()

        for fact in state:
            if fact.startswith('(ontray '):
                _, sandwich, tray = fact[1:-1].split()
                current_sandwiches.add(sandwich)
                current_trays.add(tray)
            elif fact.startswith('(at_kitchen_bread '):
                bread = fact[1:-1].split()[1]
                current_breads.add(bread)
            elif fact.startswith('(at_kitchen_content '):
                content = fact[1:-1].split()[1]
                current_contents.add(content)
            elif fact.startswith('(served '):
                child = fact[1:-1].split()[1]
                current_served.add(child)
            elif fact.startswith('(allergic_gluten '):
                child = fact[1:-1].split()[1]
                current_allergic.add(child)
            elif fact.startswith('(not_allergic_gluten '):
                child = fact[1:-1].split()[1]
                current_not_allergic.add(child)

        # Count children who still need to be served
        total_children = self.allergic_children.union(self.not_allergic_children)
        unserved_children = total_children - current_served
        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0

        # Calculate needed resources
        needed_sandwiches = num_unserved
        available_trays = self.trays - current_trays

        # Estimate actions
        total_actions = 0

        # Actions for making sandwiches
        total_actions += needed_sandwiches * 2  # Make each sandwich requires 2 actions (bread and content)

        # Actions for moving trays
        trays_needed = max(0, needed_sandwiches - len(available_trays))
        total_actions += trays_needed * 2  # Move each tray requires 2 actions (pick up and drop off)

        # Actions for serving
        total_actions += needed_sandwiches  # Each serve action is 1 step

        return total_actions
