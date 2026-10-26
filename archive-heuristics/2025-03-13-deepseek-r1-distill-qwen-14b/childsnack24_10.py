from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches. Each sandwich requires three actions: making, putting on a tray, and serving. If a tray is already at the child's place, it saves one action per sandwich.

    # Assumptions:
    - Each child needs exactly one sandwich.
    - A tray can hold multiple sandwiches.
    - If a tray is already at a child's place, it saves one action per sandwich served from that tray.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children waiting to be served and group them by their place.
    2. For each group, check if any tray is already at the child's place.
    3. For each group:
       - If a tray is present, each sandwich requires 3 actions.
       - If no tray is present, each sandwich requires 4 actions (including moving the tray once per group).
    4. Sum the actions for all groups to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state

        # Group children by their waiting place
        groups = {}
        for fact in state:
            if fact.startswith('(waiting '):
                parts = fact[1:-1].split()
                child = parts[1]
                place = parts[2]
                if place not in groups:
                    groups[place] = 0
                groups[place] += 1

        # Check if any tray is present at each place
        trays_at_place = set()
        for fact in state:
            if fact.startswith('(at tray'):
                parts = fact[1:-1].split()
                place = parts[2]
                trays_at_place.add(place)

        # Calculate the heuristic value
        heuristic = 0
        for place, count in groups.items():
            if place in trays_at_place:
                heuristic += 3 * count
            else:
                heuristic += 4 * count

        return heuristic
