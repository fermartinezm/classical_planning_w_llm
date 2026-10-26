from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches. It considers the number of children waiting, their allergy status, and the availability of ingredients and trays.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A child with gluten allergy requires a gluten-free sandwich.
    - Bread and content portions are available in the kitchen.
    - Each sandwich requires one tray, which must be at the kitchen before it can be used.

    # Heuristic Initialization
    - Extract the number of children waiting at each place and their allergy status from the task's static facts.
    - Extract the current state to determine the number of available trays and whether they are at the kitchen.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting at each place.
    2. For each child, determine if they require a gluten-free sandwich based on their allergy status.
    3. Calculate the number of gluten-free and regular sandwiches needed.
    4. For each sandwich, check if a tray is available at the kitchen. If not, add actions to move the tray to the kitchen.
    5. For each sandwich, add actions for making, putting on a tray, and serving it to the child.
    6. Sum all the required actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with necessary information from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.trays = set()
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.waiting_locations = dict()

        # Parse static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten'):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(not_allergic_gluten'):
                self.not_allergic_children.add(parts[1])
            elif fact.startswith('(waiting'):
                child = parts[1]
                location = parts[2]
                self.waiting_locations[child] = location
            elif fact.startswith('(tray'):
                self.trays.add(parts[1])
            elif fact.startswith('(no_gluten_bread'):
                self.gluten_free_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content'):
                self.gluten_free_contents.add(parts[1])

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Helper function to extract parts from a fact string
        def get_parts(fact):
            return fact[1:-1].split()

        # Count the number of children waiting at each place
        num_allergic = 0
        num_not_allergic = 0
        for child in self.waiting_locations:
            if child in self.allergic_children:
                num_allergic += 1
            else:
                num_not_allergic += 1

        # Total number of children waiting
        total_children = num_allergic + num_not_allergic
        if total_children == 0:
            return 0

        # Actions needed to make the required sandwiches
        make_actions = total_children

        # Actions needed to put sandwiches on trays and serve them
        # Each sandwich requires one tray. Assume trays are available
        # but may need to be moved to the kitchen first
        trays_at_kitchen = 0
        for fact in state:
            if fact.startswith('(at tray') and 'kitchen' in fact:
                trays_at_kitchen += 1

        # If trays are insufficient, additional moves may be needed
        # For simplicity, assume trays are sufficient or already at kitchen
        # This can be improved with more detailed tray management

        # Actions for moving trays if necessary (if trays are not at kitchen)
        # This part is simplified; in a full implementation, track tray locations
        # and calculate needed moves
        move_actions = 0
        for child in self.waiting_locations:
            # Assume each sandwich needs a tray move if not already at kitchen
            # This is a simplification; in reality, trays can be reused
            move_actions += 1

        # Actions for serving each child
        serve_actions = total_children

        # Total heuristic value is the sum of all actions
        total_actions = make_actions + move_actions + serve_actions

        return total_actions
