from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - Sandwiches can be either regular or no-gluten, depending on the child's allergy status.
    - The kitchen is the central location for preparing sandwiches.
    - Trays must be moved to the child's location after being loaded with a sandwich.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.
    - Maps each child to their required sandwich type and goal location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who have not yet been served.
    2. For each unserved child:
       a. Check if their sandwich exists and is properly prepared.
       b. Determine if the sandwich is on a tray.
       c. Check if the tray is already at the child's location.
       d. If not, calculate the actions needed to move the tray.
    3. Sum the actions needed for all unserved children to estimate the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.waiting_locations = dict()
        self.goal_locations = dict()

        # Parse static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten '):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(no_gluten_bread '):
                self.no_gluten_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content '):
                self.no_gluten_contents.add(parts[1])
            elif fact.startswith('(waiting '):
                child, loc = parts[1], parts[2]
                self.waiting_locations[child] = loc
            elif fact.startswith('(served '):
                pass  # Goal facts will be handled separately

        # Parse goal conditions to map children to their goal locations
        for goal in self.goals:
            parts = goal[1:-1].split()
            if goal.startswith('(served '):
                child = parts[1]
                self.goal_locations[child] = self.waiting_locations[child]

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all children."""
        state = node.state
        total_actions = 0

        # Check which children are not yet served
        unserved_children = []
        for child in self.goal_locations:
            if f'(served {child})' not in state:
                unserved_children.append(child)

        # For each unserved child, determine the required actions
        for child in unserved_children:
            # Check if the child is allergic and requires a no-gluten sandwich
            if child in self.allergic_children:
                sandwich_type = 'no_gluten'
            else:
                sandwich_type = 'regular'

            # Check if the sandwich already exists
            sandwich_exists = any(fact.startswith(f'(at_kitchen_sandwich {child})') for fact in state)
            if not sandwich_exists:
                # Need to make the sandwich
                total_actions += 1  # make_sandwich or make_sandwich_no_gluten

            # Check if the sandwich is on a tray
            tray_fact = None
            for fact in state:
                if fact.startswith(f'(ontray {child} '):
                    tray_fact = fact
                    break
            if not tray_fact:
                # Need to put the sandwich on a tray
                total_actions += 1  # put_on_tray

            # Determine the current tray location
            tray_location = None
            for fact in state:
                if fact.startswith('(at tray'):
                    tray_location = fact.split()[-1]
                    break

            # Get the child's waiting location
            child_location = self.waiting_locations[child]

            # If tray is not already at the child's location, move it
            if tray_location != child_location:
                total_actions += 1  # move_tray

        return total_actions
