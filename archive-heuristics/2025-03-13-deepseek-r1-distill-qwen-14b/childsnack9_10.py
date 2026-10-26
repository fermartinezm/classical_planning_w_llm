from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children by:
    1. Checking if each child has their required sandwich type (gluten-free or regular).
    2. Ensuring the necessary tray is in the kitchen.
    3. Calculating the actions needed to make, place, and serve each sandwich.

    # Assumptions:
    - Each child waiting at a place needs exactly one sandwich.
    - A sandwich can be served only if it is on a tray at the child's location.
    - If a child is allergic to gluten, they require a gluten-free sandwich.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.
    - Maps each child to their required sandwich type.
    - Tracks the current location of each tray.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are waiting and need a sandwich.
    2. For each child, determine if their required sandwich is already made and on a tray.
    3. If the tray isn't in the kitchen, add the cost to move it.
    4. For each child without a served sandwich, add the cost to make, place, and serve it.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Map each child to their required sandwich type (None if not specified)
        self.child_sandwich = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                parts = get_parts(fact)
                child = parts[1]
                self.child_sandwich[child] = "no_gluten"
            elif match(fact, "not_allergic_gluten", "*"):
                parts = get_parts(fact)
                child = parts[1]
                self.child_sandwich[child] = "regular"

        # Track the location of each tray
        self.tray_locations = {}
        for fact in static_facts:
            if match(fact, "at", "*", "kitchen"):
                parts = get_parts(fact)
                tray = parts[1]
                self.tray_locations[tray] = "kitchen"

    def __call__(self, node):
        """Compute the estimated number of actions to reach the goal state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Check if all children are already served
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                served_children.add(parts[1])

        if served_children == set(self.child_sandwich.keys()):
            return 0  # All children are served

        total_cost = 0

        # Check each child who needs a sandwich
        for child, sandwich_type in self.child_sandwich.items():
            if child in served_children:
                continue

            # Check if the sandwich is already made and on a tray
            sandwich_made = False
            tray_on_kitchen = False
            for fact in state:
                if match(fact, "no_gluten_sandwich", "*") or match(fact, "at_kitchen_sandwich", "*"):
                    parts = get_parts(fact)
                    if sandwich_type == "no_gluten" and match(fact, "no_gluten_sandwich", "*"):
                        sandwich_made = True
                    elif sandwich_type == "regular" and match(fact, "at_kitchen_sandwich", "*"):
                        sandwich_made = True
                    if match(fact, "at_kitchen_sandwich", "*"):
                        tray = get_parts(fact)[2]
                        if tray in self.tray_locations:
                            tray_location = self.tray_locations[tray]
                            if tray_location == "kitchen":
                                tray_on_kitchen = True

            if not sandwich_made:
                # Need to make the sandwich
                total_cost += 1  # make_sandwich or make_sandwich_no_gluten

            if not tray_on_kitchen:
                # Need to move tray to kitchen
                total_cost += 1  # move_tray action

            # Put sandwich on tray and serve
            total_cost += 2  # put_on_tray and serve_sandwich*

        return total_cost
