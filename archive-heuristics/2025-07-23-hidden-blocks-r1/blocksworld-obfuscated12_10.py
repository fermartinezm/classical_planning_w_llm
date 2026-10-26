from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    def __init__(self, task):
        self.task = task
        self.goal_links = set()
        self.goal_A = set()
        self.goal_B = set()
        
        for goal in task.goals:
            if goal.startswith('(') and goal.endswith(')'):
                content = goal[1:-1].strip()
                parts = content.split()
                if not parts:
                    continue
                pred = parts[0]
                if pred == "x272hpoe" and len(parts) == 3:
                    self.goal_links.add((parts[1], parts[2]))
                elif pred == "dhjk1eyy" and len(parts) == 2:
                    self.goal_A.add(parts[1])
                elif pred == "xk87au5b" and len(parts) == 2:
                    self.goal_B.add(parts[1])
    
    def __call__(self, node):
        state = node.state
        
        current_links = set()
        current_A = set()
        current_B = set()
        current_C = set()
        
        for fact in state:
            if fact.startswith('(') and fact.endswith(')'):
                content = fact[1:-1].strip()
                parts = content.split()
                if not parts:
                    continue
                pred = parts[0]
                if pred == "x272hpoe" and len(parts) == 3:
                    current_links.add((parts[1], parts[2]))
                elif pred == "dhjk1eyy" and len(parts) == 2:
                    current_A.add(parts[1])
                elif pred == "xk87au5b" and len(parts) == 2:
                    current_B.add(parts[1])
                elif pred == "q71n8mtz" and len(parts) == 2:
                    current_C.add(parts[1])
        
        missing_links = self.goal_links - current_links
        H = len(missing_links)
        
        for (X, Y) in missing_links:
            if X not in current_C:
                H += 1
            if Y not in current_A:
                H += 1
        
        first_args = set(X for (X, Y) in self.goal_links)
        missing_A = self.goal_A - current_A
        missing_A_not_in_first_args = missing_A - first_args
        H += len(missing_A_not_in_first_args)
        
        missing_B = self.goal_B - current_B
        H += len(missing_B)
        
        return H
