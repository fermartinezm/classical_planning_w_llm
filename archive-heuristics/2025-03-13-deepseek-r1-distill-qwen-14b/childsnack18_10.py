from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A child with gluten allergy requires a gluten-free sandwich.
    - Each sandwich requires one bread and one content portion.
    - Each sandwich must be placed on a tray before serving.
    - Each tray can be moved to the child's location if not already there.

    # Heuristic Initialization
    - Extract static facts about children's allergies, gluten status of ingredients, and tray positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If yes, no actions are needed.
    2. For unserved children:
       a. Determine if a gluten-free sandwich is required based on the child's allergy.
       b. Check if the necessary ingredients (bread and content) are available.
       c. If the sandwich doesn't exist, add an action to make it.
       d. If the sandwich isn't on a tray, add an action to put it on a tray.
       e. If the tray isn't at the child's location, add an action to move the tray.
       f. Add an action to serve the sandwich.
    3. Sum the actions for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into data structures
        self.allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.tray_positions = dict()

        for fact in static_facts:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten'):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(no_gluten_bread'):
                self.no_gluten_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content'):
                self.no_gluten_contents.add(parts[1])
            elif fact.startswith('(at tray'):
                tray = parts[1]
                place = parts[3]
                self.tray_positions[tray] = place

    def __call__(self, node):
        """Compute the estimated number of actions to serve all children."""
        state = node.state
        current_served = set()
        current_sandwiches = set()
        current_trays = dict()

        for fact in state:
            if fact.startswith('(served '):
                child = fact.split()[1]
                current_served.add(child)
            elif fact.startswith('(at_kitchen_sandwich '):
                sandwich = fact.split()[1]
                current_sandwiches.add(sandwich)
            elif fact.startswith('(ontray '):
                sandwich = fact.split()[1]
                tray = fact.split()[2]
                current_trays[sandwich] = tray
            elif fact.startswith('(at tray'):
                tray = fact.split()[1]
                place = fact.split()[3]
                self.tray_positions[tray] = place

        total_actions = 0

        for goal in self.goals:
            if goal.startswith('(served '):
                child = goal.split()[1]
                if child in current_served:
                    continue

                # Determine if the child is allergic
                if child in self.allergic_children:
                    sandwich_type = 'no_gluten'
                else:
                    sandwich_type = 'regular'

                # Check if the sandwich exists
                if child not in current_sandwiches:
                    total_actions += 1  # Make sandwich action

                # Check if the sandwich is on a tray
                if child not in current_trays:
                    total_actions += 1  # Put on tray action
                else:
                    tray = current_trays[child]

                # Check if the tray is at the child's place
                if tray not in self.tray_positions or self.tray_positions[tray] != 'table' + child[-1]:
                    total_actions += 1  # Move tray action

                # Serve the sandwich
                total_actions += 1  # Serve action

        return total_actions
