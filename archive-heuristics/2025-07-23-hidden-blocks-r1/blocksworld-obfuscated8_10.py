from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal state by counting:
    - The number of missing relations (x272hpoe) in the current state compared to the goal.
    - The number of objects that are required to be active (dhjk1eyy) in the goal but are not active in the current state.
    - The number of objects that are required to be marked (xk87au5b) in the goal but are not marked in the current state.
    Each of these counts is multiplied by a factor (2) to estimate the total actions needed.

    Assumptions:
    - Each missing relation requires approximately 2 actions to establish (e.g., creating a q71n8mtz object and applying m82i1lr3).
    - Each missing active object requires approximately 2 actions to activate (e.g., using dgdzvgpm or m82i1lr3).
    - Each missing marked object requires approximately 2 actions to mark (e.g., using dgdzvgpm after ensuring the object is q71n8mtz).

    Heuristic Initialization:
    - Extracts goal relations (x272hpoe), active objects (dhjk1eyy), and marked objects (xk87au5b) from the task goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal condition, categorize it into relations, active objects, or marked objects.
    2. In the current state, collect existing relations, active objects, and marked objects.
    3. Count how many goal relations are missing in the current state.
    4. Count how many goal active objects are missing in the current state.
    5. Count how many goal marked objects are missing in the current state.
    6. The heuristic value is 2 times the sum of these counts, representing an estimate of the required actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_relations = set()
        self.goal_active_objects = set()
        self.goal_marked_objects = set()
        
        for goal in task.goals:
            if goal.startswith('(x272hpoe ') and goal.endswith(')'):
                parts = goal[1:-1].split()
                if len(parts) == 3 and parts[0] == 'x272hpoe':
                    self.goal_relations.add((parts[1], parts[2]))
            elif goal.startswith('(dhjk1eyy ') and goal.endswith(')'):
                parts = goal[1:-1].split()
                if len(parts) == 2 and parts[0] == 'dhjk1eyy':
                    self.goal_active_objects.add(parts[1])
            elif goal.startswith('(xk87au5b ') and goal.endswith(')'):
                parts = goal[1:-1].split()
                if len(parts) == 2 and parts[0] == 'xk87au5b':
                    self.goal_marked_objects.add(parts[1])
    
    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        missing_relations = 0
        missing_active = 0
        missing_marked = 0
        
        current_relations = set()
        for fact in state:
            if fact.startswith('(x272hpoe ') and fact.endswith(')'):
                parts = fact[1:-1].split()
                if len(parts) == 3 and parts[0] == 'x272hpoe':
                    current_relations.add((parts[1], parts[2]))
        missing_relations = len(self.goal_relations - current_relations)
        
        for obj in self.goal_active_objects:
            fact_str = '(dhjk1eyy {})'.format(obj)
            if fact_str not in state:
                missing_active += 1
        
        for obj in self.goal_marked_objects:
            fact_str = '(xk87au5b {})'.format(obj)
            if fact_str not in state:
                missing_marked += 1
        
        return 2 * (missing_relations + missing_active + missing_marked)
