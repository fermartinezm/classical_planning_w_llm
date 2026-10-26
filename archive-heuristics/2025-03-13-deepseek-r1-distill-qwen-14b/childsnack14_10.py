from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must receive exactly one sandwich.
    - A sandwich can be either regular or no-gluten, depending on the child's allergy status.
    - A sandwich must be placed on a tray before it can be served.
    - If a child is waiting at a different location than the tray, the tray must be moved.

    # Heuristic Initialization
    - Extract static facts about no-gluten bread, no-gluten content, and children's allergies.
    - Track the current state of each child, sandwich, and tray.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If yes, no actions are needed.
    2. For each unserved child:
       a. Check if a suitable sandwich exists.
       b. If the sandwich is not made, estimate 2 actions (make sandwich + put on tray).
       c. If the sandwich is made but not on the correct tray, estimate 1 action (move tray).
       d. If the sandwich is on the tray but not served, estimate 1 action (serve).
    3. Sum the estimated actions for all children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about the domain."""
        self.goals = task.goals
        static_facts = task.static

        # Extract no-gluten bread portions
        self.no_gluten_bread = set()
        for fact in static_facts:
            if fnmatch(fact, '(no_gluten_bread *)'):
                bread = fact.split()[2]
                self.no_gluten_bread.add(bread)

        # Extract no-gluten content portions
        self.no_gluten_content = set()
        for fact in static_facts:
            if fnmatch(fact, '(no_gluten_content *)'):
                content = fact.split()[2]
                self.no_gluten_content.add(content)

        # Extract allergic children
        self.allergic_children = set()
        for fact in static_facts:
            if fnmatch(fact, '(allergic_gluten *)'):
                child = fact.split()[2]
                self.allergic_children.add(child)

        # Extract waiting locations for each child
        self.waiting_location = {}
        for fact in static_facts:
            if fnmatch(fact, '(waiting * * *)'):
                parts = fact.split()
                child = parts[1]
                location = parts[3]
                self.waiting_location[child] = location

    def __call__(self, node):
        """Estimate the minimum number of actions needed to serve all children."""
        state = node.state
        total_actions = 0

        # Track which children have been served
        served_children = set()
        for fact in state:
            if fnmatch(fact, '(served *)'):
                served_children.add(fact.split()[1])

        # Track which sandwiches exist
        sandwiches = set()
        for fact in state:
            if fnmatch(fact, '(notexist *)'):
                sandwiches.add(fact.split()[1])

        # Track which trays are at the kitchen
        trays_at_kitchen = set()
        for fact in state:
            if fnmatch(fact, '(at tray* kitchen)'):
                trays_at_kitchen.add(fact.split()[1])

        # Track which trays are on the kitchen
        trays_on_kitchen = set()
        for fact in state:
            if fnmatch(fact, '(ontray * tray*)'):
                tray = fact.split()[2]
                trays_on_kitchen.add(tray)

        # For each child, determine the required actions
        for fact in state:
            if fnmatch(fact, '(waiting * * *)'):
                parts = fact.split()
                child = parts[1]
                if child in served_children:
                    continue  # Already served, skip

                # Check if the child is allergic
                if child in self.allergic_children:
                    # Need a no-gluten sandwich
                    sandwich_needed = f'sandw{child}'
                else:
                    # Regular sandwich is sufficient
                    sandwich_needed = next(iter(sandwiches)) if sandwiches else None

                # Check if the sandwich is already made
                sandwich_made = any(fnmatch(f, f'(at_kitchen_sandwich {sandwich_needed})') for f in state)
                if not sandwich_made:
                    # Need to make the sandwich: 2 actions (make + put on tray)
                    total_actions += 2
                else:
                    # Check if the sandwich is on the correct tray
                    tray_on_kitchen = None
                    for f in state:
                        if fnmatch(f, f'(ontray {sandwich_needed} tray*)'):
                            tray_on_kitchen = f.split()[2]
                            break
                    if tray_on_kitchen is None:
                        # Sandwich is not on any tray, need to put on tray: 1 action
                        total_actions += 1
                    else:
                        # Check if the tray is already at the child's location
                        if tray_on_kitchen in trays_at_kitchen:
                            # Tray is at kitchen, need to move it: 1 action
                            total_actions += 1
                        else:
                            # Tray is already at the child's location, just serve: 1 action
                            total_actions += 1

        return total_actions
