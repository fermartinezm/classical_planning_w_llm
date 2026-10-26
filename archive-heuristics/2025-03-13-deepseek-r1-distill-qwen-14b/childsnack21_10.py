from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A sandwich can be served only if it is on a tray and the child is waiting.
    - Allergic children require no-gluten sandwiches, which must be made with no-gluten bread and content.
    - Non-allergic children can be served regular sandwiches.

    # Heuristic Initialization
    - Extracts goal conditions (all children must be served).
    - Extracts static facts about no-gluten ingredients and children's allergies.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served.
    2. For unserved children:
       a. If allergic, check if a no-gluten sandwich is available.
       b. If not allergic, check if a regular sandwich is available.
       c. If sandwich is already made and on a tray, only serving is needed (1 action).
       d. If sandwich is not made, both making and serving are needed (2 actions).
    3. Sum the required actions for all children.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal and static information."""
        self.goals = task.goals  # All children must be served
        static_facts = task.static

        # Extract no-gluten bread and content
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        for fact in static_facts:
            if match(fact, "no_gluten_bread", "*"):
                self.no_gluten_bread.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.no_gluten_content.add(get_parts(fact)[1])

        # Extract allergic children
        self.allergic_children = set()
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Track served children
        served = set()
        for fact in state:
            if match(fact, "served", "*"):
                served.add(get_parts(fact)[1])

        # Track children waiting at places
        waiting = {}
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                waiting[child] = place

        # Track sandwiches on trays
        sandwiches_on_tray = set()
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                sandwiches_on_tray.add(sandwich)

        # Track no-gluten sandwiches
        no_gluten_sandwiches = set()
        for fact in state:
            if match(fact, "no_gluten_sandwich", "*"):
                no_gluten_sandwiches.add(get_parts(fact)[1])

        # Track available bread and content
        available_bread = set()
        available_content = set()
        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                available_bread.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "*"):
                available_content.add(get_parts(fact)[1])

        total_actions = 0

        # Process each child
        for goal in self.goals:
            if not match(goal, "served", "*"):
                continue  # Not a serving goal
            child = get_parts(goal)[1]

            if child in served:
                continue  # Already served

            # Check if child is allergic
            if child in self.allergic_children:
                # Allergic children need no-gluten sandwiches
                # Check if a no-gluten sandwich is available
                if any(sandwich.startswith(f"{child}_") for sandwich in no_gluten_sandwiches):
                    # Sandwich is already made, just needs to be served
                    total_actions += 1
                else:
                    # Need to make and serve the sandwich
                    total_actions += 2
            else:
                # Non-allergic children can have regular sandwiches
                # Check if any regular sandwich is available
                if any(sandwich.startswith(f"{child}_") for sandwich in sandwiches_on_tray):
                    # Sandwich is already made, just needs to be served
                    total_actions += 1
                else:
                    # Need to make and serve the sandwich
                    total_actions += 2

        return total_actions
