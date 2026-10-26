from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by making sandwiches and moving them to the children's locations.

    # Assumptions:
    - Each child requires one sandwich.
    - A sandwich can be made, placed on a tray, moved if necessary, and served.
    - If a sandwich is already on a tray at the child's location, only the serving action is needed.
    - If a sandwich is on a tray in the kitchen, it needs to be moved and then served.
    - If no sandwich exists, it must be made, placed on a tray, moved if necessary, and served.

    # Heuristic Initialization
    - Extract the goal conditions and static facts from the task.
    - Identify the waiting place for each child from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are not yet served.
    2. For each unserved child, check if there is a sandwich on a tray at their waiting place.
    3. Count the number of unserved children and those with available sandwiches at their place.
    4. Calculate the total actions needed: 3 actions per child if a sandwich is available at their place, otherwise 4 actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        self.waiting_place = {}
        for fact in static_facts:
            if fact.startswith('(waiting'):
                parts = fact[1:-1].split()
                child = parts[1]
                place = parts[2]
                self.waiting_place[child] = place

    def __call__(self, node):
        """Estimate the minimum cost to serve all children."""
        state = node.state
        unserved_children = []
        for child in self.waiting_place:
            served = False
            for fact in state:
                if fact == f'(served {child})':
                    served = True
                    break
            if not served:
                unserved_children.append(child)
        n = len(unserved_children)
        if n == 0:
            return 0
        t = 0
        for c in unserved_children:
            p = self.waiting_place[c]
            has_sandwich = False
            for fact in state:
                if fact.startswith('(ontray'):
                    parts = fact[1:-1].split()
                    s = parts[1]
                    t_tray = parts[2]
                    if f'(at {t_tray} {p})' in state:
                        has_sandwich = True
                        break
            if has_sandwich:
                t += 1
        return 3 * n + max(n - t, 0)
