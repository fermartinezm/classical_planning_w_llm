from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by:
    1. Making the required sandwiches.
    2. Placing each sandwich on a tray.
    3. Moving the tray to the child's location if necessary.
    4. Serving the sandwich to the child.

    # Assumptions:
    - Each child corresponds to exactly one sandwich.
    - A sandwich must be made before it can be placed on a tray.
    - A tray must be at the child's location before the sandwich can be served.

    # Heuristic Initialization
    - Extracts static facts about children's allergies and the availability of gluten-free ingredients.
    - Tracks the current state of each sandwich and tray.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If not, proceed.
    2. For each unserved child, determine if their sandwich has been made. If not, add 1 action.
    3. Check if the sandwich is placed on a tray. If not, add 1 action.
    4. Determine the current location of the tray holding the sandwich. If it's not at the child's location, add 1 action.
    5. Add 1 action for serving the sandwich to the child.
    6. Sum all the actions for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal conditions."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.children = set()
        self.allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.trays = set()
        self.locations = set()

        for fact in self.static:
            if match(fact, "* child"):
                self.children.add(fact.split()[1])
            elif match(fact, "allergic_gluten *"):
                self.allergic_children.add(fact.split()[1])
            elif match(fact, "no_gluten_bread *"):
                self.no_gluten_breads.add(fact.split()[1])
            elif match(fact, "no_gluten_content *"):
                self.no_gluten_contents.add(fact.split()[1])
            elif match(fact, "tray*"):
                self.trays.add(fact.split()[1])
            elif match(fact, "* city"):
                self.locations.add(fact.split()[1])

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        heuristic = 0

        # Track served children
        served = set()
        for fact in state:
            if match(fact, "(served *"):
                served.add(fact.split()[1])

        # For each child not served
        for child in self.children:
            if child in served:
                continue

            # Check if the sandwich is made
            sandwich_made = False
            for fact in state:
                if match(fact, "(at_kitchen_sandwich *") and \
                   (fact.split()[1] == f"{child}"):
                    sandwich_made = True
                    break
            if not sandwich_made:
                heuristic += 1

            # Check if the sandwich is on a tray
            sandwich_on_tray = False
            tray = None
            for fact in state:
                if match(fact, "(ontray * *") and \
                   (fact.split()[1] == f"{child}"):
                    sandwich_on_tray = True
                    tray = fact.split()[2]
                    break
            if not sandwich_on_tray:
                heuristic += 1
                continue

            # Determine the tray's current location
            tray_location = None
            for fact in state:
                if match(fact, "(at * *") and \
                   (fact.split()[1] == tray):
                    tray_location = fact.split()[2]
                    break

            # Determine the child's current location
            child_location = None
            for fact in state:
                if match(fact, "(waiting * *") and \
                   (fact.split()[1] == child):
                    child_location = fact.split()[2]
                    break

            # If the tray is not at the child's location, add move action
            if tray_location != child_location:
                heuristic += 1

            # Add serve action
            heuristic += 1

        return heuristic
