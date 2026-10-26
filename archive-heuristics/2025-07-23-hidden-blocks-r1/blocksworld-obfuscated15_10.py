from heuristics.heuristic_base import Heuristic

class SecretDomainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal state by considering:
    1. The number of missing and extra 'x272hpoe' links.
    2. The difference in counts of 'dhjk1eyy' and 'xk87au5b' tokens between the current state and the goal.

    Assumptions:
    - The goal does not specify the state of the global flag 'hxtpdpff'.
    - The net change in 'dhjk1eyy' and 'xk87au5b' tokens must be equal for the state to be solvable.

    Heuristic Initialization:
    - Extracts goal conditions for 'x272hpoe', 'dhjk1eyy', and 'xk87au5b' from the task.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify existing 'x272hpoe' links, 'dhjk1eyy', and 'xk87au5b' tokens.
    2. Compare with the goal conditions to determine:
        a. Missing links (goal links not present in the current state).
        b. Extra links (current links not required in the goal).
        c. Difference in counts of 'dhjk1eyy' and 'xk87au5b' tokens between current state and goal.
    3. If the differences in token counts are not equal, return a high cost (indicating an unsolvable state).
    4. Otherwise, the heuristic cost is the sum of missing and extra links plus the absolute value of the token difference.
    """

    def __init__(self, task):
        self.goal_links = set()
        self.goal_dhjk1eyy = set()
        self.goal_xk87au5b = set()
        
        for goal in task.goals:
            if goal.startswith('(') and goal.endswith(')'):
                content = goal[1:-1].strip()
                parts = content.split()
                if not parts:
                    continue
                pred = parts[0]
                if pred == 'x272hpoe' and len(parts) == 3:
                    self.goal_links.add((parts[1], parts[2]))
                elif pred == 'dhjk1eyy' and len(parts) == 2:
                    self.goal_dhjk1eyy.add(parts[1])
                elif pred == 'xk87au5b' and len(parts) == 2:
                    self.goal_xk87au5b.add(parts[1])

    def __call__(self, node):
        state = node.state
        current_links = set()
        current_dhjk1eyy = set()
        current_xk87au5b = set()
        
        for fact in state:
            if fact == '(hxtpdpff)':
                continue
            if fact.startswith('(') and fact.endswith(')'):
                content = fact[1:-1].strip()
                parts = content.split()
                if not parts:
                    continue
                pred = parts[0]
                if pred == 'x272hpoe' and len(parts) == 3:
                    current_links.add((parts[1], parts[2]))
                elif pred == 'dhjk1eyy' and len(parts) == 2:
                    current_dhjk1eyy.add(parts[1])
                elif pred == 'xk87au5b' and len(parts) == 2:
                    current_xk87au5b.add(parts[1])
        
        missing_links = self.goal_links - current_links
        extra_links = current_links - self.goal_links
        cost_links = len(missing_links) + len(extra_links)
        
        D_dhjk = len(self.goal_dhjk1eyy) - len(current_dhjk1eyy)
        D_xk87 = len(self.goal_xk87au5b) - len(current_xk87au5b)
        
        if D_dhjk != D_xk87:
            return 1000000
        
        cost_tokens = abs(D_dhjk)
        return cost_links + cost_tokens
