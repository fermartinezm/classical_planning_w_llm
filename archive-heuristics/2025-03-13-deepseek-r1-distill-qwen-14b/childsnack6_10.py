from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches by:
    1. Counting the number of children waiting for each type of sandwich.
    2. Determining how many sandwiches of each type need to be made.
    3. Calculating the actions required to make, transport, and serve the sandwiches.

    # Assumptions:
    - Each sandwich requires one bread and one content portion.
    - A child waiting at a place must be served from a sandwich on a tray at that place.
    - If a child is allergic, they must receive a no-gluten sandwich.
    - Each sandwich can be transported and served in a fixed number of actions.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.
    - Maps children to their required sandwich type based on allergy status.
    - Counts available bread and content portions for each type.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of children waiting for each type of sandwich (allergic and non-allergic).
    2. Determine how many no-gluten and regular sandwiches need to be made based on availability.
    3. For each required sandwich type, calculate the number of make, transport, and serve actions needed.
    4. Sum the actions for all required sandwiches to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.allergic_children = set()
        self.no_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.trays = set()

        # Parse static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten '):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(not_allergic_gluten '):
                self.no_allergic_children.add(parts[1])
            elif fact.startswith('(no_gluten_bread '):
                self.no_gluten_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content '):
                self.no_gluten_contents.add(parts[1])
            elif fact.startswith('(tray '):
                self.trays.add(parts[1])

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state

        # Helper function to extract components from a fact string
        def get_parts(fact):
            return fact[1:-1].split()

        # Count children waiting at each place
        waiting_children = {}
        for fact in state:
            if fact.startswith('(waiting '):
                parts = get_parts(fact)
                child, place = parts[1], parts[2]
                if place not in waiting_children:
                    waiting_children[place] = []
                waiting_children[place].append(child)

        # Count available bread and content portions
        available_breads = {'regular': 0, 'no_gluten': 0}
        available_contents = {'regular': 0, 'no_gluten': 0}
        for fact in state:
            if fact.startswith('(at_kitchen_bread '):
                b = get_parts(fact)[1]
                if b in self.no_gluten_breads:
                    available_breads['no_gluten'] += 1
                else:
                    available_breads['regular'] += 1
            elif fact.startswith('(at_kitchen_content '):
                c = get_parts(fact)[1]
                if c in self.no_gluten_contents:
                    available_contents['no_gluten'] += 1
                else:
                    available_contents['regular'] += 1

        # Count required sandwiches
        required_sandwiches = {'no_gluten': 0, 'regular': 0}
        for place, children in waiting_children.items():
            for child in children:
                if child in self.allergic_children:
                    required_sandwiches['no_gluten'] += 1
                else:
                    required_sandwiches['regular'] += 1

        # Calculate possible sandwiches based on available resources
        possible_no_gluten = min(
            required_sandwiches['no_gluten'],
            available_breads['no_gluten'],
            available_contents['no_gluten']
        )
        possible_regular = min(
            required_sandwiches['regular'],
            available_breads['regular'],
            available_contents['regular']
        )

        # Calculate remaining sandwiches needed
        remaining_no_gluten = max(
            required_sandwiches['no_gluten'] - possible_no_gluten,
            0
        )
        remaining_regular = max(
            required_sandwiches['regular'] - possible_regular,
            0
        )

        # Calculate actions needed for each type
        actions = 0
        # Making sandwiches
        actions += remaining_no_gluten + remaining_regular
        # Putting on trays
        actions += remaining_no_gluten + remaining_regular
        # Serving
        actions += remaining_no_gluten + remaining_regular

        return actions
