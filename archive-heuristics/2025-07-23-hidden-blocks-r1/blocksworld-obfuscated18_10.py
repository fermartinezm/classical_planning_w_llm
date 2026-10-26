from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal by:
    1. Counting missing and extra connections (x272hpoe) in the current state compared to the goal.
    2. Counting objects that are not ready (xk87au5b) but must be ready in the goal.
    3. Counting objects that are not held (dhjk1eyy) but must be held in the goal.
    Each discrepancy contributes a cost of 2, as each typically requires at least one or two actions to resolve.

    Assumptions:
    - The goal conditions only include (dhjk1eyy ?x), (xk87au5b ?x), and (x272hpoe ?x ?y) predicates.
    - Object names in state facts do not contain spaces.

    Heuristic Initialization:
    - Extracts goal conditions for connections (x272hpoe), readiness (xk87au5b), and held objects (dhjk1eyy).

    Step-By-Step Thinking for Computing Heuristic:
    1. For connections (x272hpoe):
        - Missing connections in the current state: cost += 2 per missing connection.
        - Extra connections not in the goal: cost += 2 per extra connection.
    2. For readiness (xk87au5b):
        - Objects that must be ready in the goal but are not ready in the current state: cost += 2 per object.
    3. For held objects (dhjk1eyy):
        - Objects that must be held in the goal but are not held in the current state: cost += 2 per object.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_links = set()
        self.goal_ready = set()
        self.goal_held = set()

        for goal in task.goals:
            if goal.startswith('(x272hpoe '):
                parts = goal[10:-1].split()
                if len(parts) >= 2:
                    obj1 = parts[0]
                    obj2 = parts[1]
                    self.goal_links.add((obj1, obj2))
            elif goal.startswith('(xk87au5b '):
                obj = goal[10:-1]
                self.goal_ready.add(obj)
            elif goal.startswith('(dhjk1eyy '):
                obj = goal[10:-1]
                self.goal_held.add(obj)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_links = set()
        current_ready = set()
        current_held = set()

        for fact in state:
            if fact.startswith('(x272hpoe '):
                parts = fact[10:-1].split()
                if len(parts) >= 2:
                    obj1 = parts[0]
                    obj2 = parts[1]
                    current_links.add((obj1, obj2))
            elif fact.startswith('(xk87au5b '):
                obj = fact[10:-1]
                current_ready.add(obj)
            elif fact.startswith('(dhjk1eyy '):
                obj = fact[10:-1]
                current_held.add(obj)

        missing_links = self.goal_links - current_links
        extra_links = current_links - self.goal_links
        missing_ready = self.goal_ready - current_ready
        missing_held = self.goal_held - current_held

        cost_missing_links = 2 * len(missing_links)
        cost_extra_links = 2 * len(extra_links)
        cost_missing_ready = 2 * len(missing_ready)
        cost_missing_held = 2 * len(missing_held)

        total_cost = cost_missing_links + cost_extra_links + cost_missing_ready + cost_missing_held
        return total_cost
