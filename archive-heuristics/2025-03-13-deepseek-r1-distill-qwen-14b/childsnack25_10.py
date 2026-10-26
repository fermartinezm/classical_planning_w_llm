from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by calculating four actions per unserved child: making the sandwich, putting it on a tray, moving the tray to the child's place, and serving the sandwich.

    # Assumptions:
    - Each child requires exactly one sandwich.
    - Each sandwich requires one action to make, one to put on a tray, one to move the tray, and one to serve.
    - Resources (bread and content) are assumed to be sufficient.

    # Heuristic Initialization
    - Extract the list of children that need to be served from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that need to be served by checking the goal conditions.
    2. For each child, check if they are already served in the current state.
    3. Count the number of unserved children.
    4. Multiply the count by 4 to estimate the total number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

        # Extract required children from goals
        self.required_children = set()
        for goal in self.goals:
            if goal.startswith('(served ') and goal.endswith(')'):
                child = goal[len('(served '):-1]
                self.required_children.add(child)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all unserved children."""
        state = node.state
        count = 0

        for child in self.required_children:
            served_fact = f'(served {child})'
            if served_fact not in state:
                count += 1

        return 4 * count
