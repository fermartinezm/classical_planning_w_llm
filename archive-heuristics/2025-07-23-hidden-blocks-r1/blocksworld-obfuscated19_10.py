from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (r9oudocu).

    Summary:
        This heuristic estimates the number of actions required to achieve the goal by counting:
            - The number of missing goal edges (x272hpoe)
            - The number of missing goal tokens (dhjk1eyy)
            - The number of missing goal properties (xk87au5b)
        Each missing item is assumed to require 2 actions, and the heuristic returns twice the total number of missing items.

    Assumptions:
        - The goal condition only consists of atoms of the form (dhjk1eyy ?x), (xk87au5b ?x), and (x272hpoe ?x ?y).
        - Extra edges in the state (not in the goal) are allowed and do not incur cost.
        - The state of an object being in q71n8mtz is not part of the goal and is ignored.
        - The global flag (hxtpdpff) is not part of the goal and is ignored.

    Heuristic Initialization:
        - The goal conditions are parsed to extract:
            self.goal_dhjk: set of objects that must have dhjk1eyy.
            self.goal_xk87: set of objects that must have xk87au5b.
            self.goal_edges: set of tuples (A, B) for which (x272hpoe A B) must hold.

    Step-By-Step Thinking for Computing Heuristic:
        1. For the current state, extract:
            current_dhjk: set of objects with dhjk1eyy.
            current_xk87: set of objects with xk87au5b.
            current_edges: set of tuples (A, B) with x272hpoe.
        2. Compute the sets of missing items:
            missing_dhjk = goal_dhjk - current_dhjk
            missing_xk87 = goal_xk87 - current_xk87
            missing_edges = goal_edges - current_edges
        3. Let total_missing = |missing_dhjk| + |missing_xk87| + |missing_edges|
        4. Heuristic value = 2 * total_missing
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_dhjk = set()
        self.goal_xk87 = set()
        self.goal_edges = set()

        for goal in task.goals:
            if goal.startswith('(') and goal.endswith(')'):
                content = goal[1:-1].split()
                if not content:
                    continue
                pred = content[0]
                if pred == 'dhjk1eyy' and len(content) == 2:
                    self.goal_dhjk.add(content[1])
                elif pred == 'xk87au5b' and len(content) == 2:
                    self.goal_xk87.add(content[1])
                elif pred == 'x272hpoe' and len(content) == 3:
                    self.goal_edges.add((content[1], content[2]))

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        current_dhjk = set()
        current_xk87 = set()
        current_edges = set()

        for fact in state:
            if fact.startswith('(') and fact.endswith(')'):
                content = fact[1:-1].split()
                if not content:
                    continue
                pred = content[0]
                if pred == 'dhjk1eyy' and len(content) == 2:
                    current_dhjk.add(content[1])
                elif pred == 'xk87au5b' and len(content) == 2:
                    current_xk87.add(content[1])
                elif pred == 'x272hpoe' and len(content) == 3:
                    current_edges.add((content[1], content[2]))

        missing_dhjk = self.goal_dhjk - current_dhjk
        missing_xk87 = self.goal_xk87 - current_xk87
        missing_edges = self.goal_edges - current_edges

        total_missing = len(missing_edges) + len(missing_dhjk) + len(missing_xk87)
        return 2 * total_missing
