from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their appropriate sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - Allergic children require no-gluten sandwiches.
    - Non-allergic children can be served either type of sandwich.
    - A sandwich must be made before it can be put on a tray.
    - A tray must be at the serving location before a child can be served.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available gluten-free ingredients, and tray positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of no-gluten and regular sandwiches needed based on children's allergy status.
    2. Count how many no-gluten and regular sandwiches have already been made.
    3. Calculate the number of make actions needed for each sandwich type.
    4. For each sandwich, check if it's on a tray. If not, add an action to put it on a tray.
    5. For each tray, determine if it needs to be moved to the serving location. If so, add actions for moving.
    6. For each child, if they haven't been served, add an action to serve them.
    7. Sum all these actions to get the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract information from static facts
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.tray_positions = dict()

        for fact in static_facts:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten'):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(not_allergic_gluten'):
                self.non_allergic_children.add(parts[1])
            elif fact.startswith('(no_gluten_bread'):
                self.gluten_free_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content'):
                self.gluten_free_contents.add(parts[1])
            elif fact.startswith('(at tray'):
                tray = parts[1]
                position = parts[2]
                self.tray_positions[tray] = position

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all children."""
        state = node.state

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Check if all children are already served
        if all(match(fact, 'served', '*') for fact in state):
            return 0

        # Count needed sandwich types
        num_no_gluten_needed = len(self.allergic_children)
        num_regular_needed = len(self.non_allergic_children)

        # Count current available sandwiches
        current_no_gluten = 0
        current_regular = 0
        for fact in state:
            if match(fact, 'at_kitchen_sandwich', '*'):
                parts = fact[1:-1].split()
                if match(fact, 'no_gluten_sandwich', '*'):
                    current_no_gluten += 1
                else:
                    current_regular += 1

        # Calculate needed makes
        no_gluten_to_make = max(num_no_gluten_needed - current_no_gluten, 0)
        regular_to_make = max(num_regular_needed - current_regular, 0)
        make_cost = no_gluten_to_make + regular_to_make

        # Calculate tray and serving actions
        tray_serving_cost = 0
        for child in self.allergic_children.union(self.non_allergic_children):
            # Check if child is served
            served = any(match(fact, 'served', child) for fact in state)
            if served:
                continue

            # Determine required sandwich type
            if child in self.allergic_children:
                sandwich_type = 'no_gluten'
            else:
                sandwich_type = 'regular'

            # Check if sandwich is on a tray
            sandwich_made = False
            for fact in state:
                if match(fact, 'at_kitchen_sandwich', '*') and match(fact, 'no_gluten_sandwich' if sandwich_type == 'no_gluten' else '', '*'):
                    sandwich_made = True
                    break
            if not sandwich_made:
                make_cost += 1  # Need to make the sandwich first

            # Check if tray is at serving location
            tray_used = None
            for fact in state:
                if match(fact, 'ontray', '*', '*'):
                    tray = fact[1:-1].split()[1]
                    tray_used = tray
                    break
            if tray_used:
                tray_pos = self.tray_positions.get(tray_used, 'kitchen')
                serving_pos = 'table' + child[-1]  # Assuming child1 waits at table1, etc.
                if tray_pos != serving_pos:
                    tray_serving_cost += 2  # Move tray to serving location and back
            else:
                tray_serving_cost += 2  # Move tray to serving location and back

            # Serve the child
            tray_serving_cost += 1

        total_cost = make_cost + tray_serving_cost

        return total_cost
