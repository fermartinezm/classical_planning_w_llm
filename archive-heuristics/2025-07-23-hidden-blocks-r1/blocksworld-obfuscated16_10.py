from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    Summary:
        This heuristic estimates the number of actions required to reach the goal state in the secret_domain.
        It calculates the heuristic value by considering three components:
          1. Missing links (x272hpoe) in the goal configuration
          2. Objects missing the 'dhjk1eyy' (A) property required by the goal
          3. Objects missing the 'xk87au5b' (B) property required by the goal
        
        The heuristic formula is: 
          h = |missing_links| + |missing_A| + 2 * |missing_B|
        
        This accounts for:
          - Each missing link requires at least one action (m82i1lr3) to create
          - Each missing A property requires at least one action (transfer or creation)
          - Each missing B property requires at least two actions (create C then convert to A+B)

    Assumptions:
        - The goal condition only involves x272hpoe, dhjk1eyy, and xk87au5b predicates
        - The flag (hxtpdpff) state doesn't affect goal satisfaction
        - Extra links/properties beyond goal requirements don't require removal
        - Creating B requires two actions (get C via fpk054nz then dgdzvgpm)
        - Movement of A between objects and link creation are independent

    Heuristic Initialization:
        - Parse goal conditions into:
            goal_links: set of (x,y) tuples for required links
            goal_A: set of objects requiring dhjk1eyy
            goal_B: set of objects requiring xk87au5b
        - Static facts are unused in this domain

    Step-By-Step Thinking for Computing Heuristic:
        1. Extract current state components:
            - current_links: existing (x,y) from x272hpoe facts
            - current_A: objects with dhjk1eyy
            - current_B: objects with xk87au5b
        2. Compute differences:
            missing_links = goal_links - current_links
            missing_A = goal_A - current_A
            missing_B = goal_B - current_B
        3. Calculate heuristic:
            h = len(missing_links) + len(missing_A) + 2 * len(missing_B)
        4. Return heuristic value
    """

    def __init__(self, task):
        self.goal_links = set()
        self.goal_A = set()
        self.goal_B = set()
        
        for goal in task.goals:
            parts = goal[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'x272hpoe' and len(parts) == 3:
                self.goal_links.add((parts[1], parts[2]))
            elif parts[0] == 'dhjk1eyy' and len(parts) == 2:
                self.goal_A.add(parts[1])
            elif parts[0] == 'xk87au5b' and len(parts) == 2:
                self.goal_B.add(parts[1])

    def __call__(self, node):
        state = node.state
        current_links = set()
        current_A = set()
        current_B = set()
        
        for fact in state:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'x272hpoe' and len(parts) == 3:
                current_links.add((parts[1], parts[2]))
            elif parts[0] == 'dhjk1eyy' and len(parts) == 2:
                current_A.add(parts[1])
            elif parts[0] == 'xk87au5b' and len(parts) == 2:
                current_B.add(parts[1])
        
        missing_links = self.goal_links - current_links
        missing_A = self.goal_A - current_A
        missing_B = self.goal_B - current_B
        
        return len(missing_links) + len(missing_A) + 2 * len(missing_B)
