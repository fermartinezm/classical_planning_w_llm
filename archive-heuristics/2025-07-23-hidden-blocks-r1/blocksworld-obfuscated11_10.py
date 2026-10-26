from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal state by considering four components:
    1. Missing dhjk1eyy predicates in D_goal objects
    2. Missing xk87au5b predicates in X_goal objects
    3. Missing x272hpoe relations in R_goal
    4. Sources of missing relations not in state B (q71n8mtz)
    
    The heuristic sums these components to estimate total remaining actions.

    # Assumptions
    - Goal relations form a set of chains where each object can be a source at most once
    - Objects required to have xk87au5b in the goal cannot be used as relation sources
    - The flag hxtpdpff's state doesn't require explicit accounting in the heuristic
    - Goal conditions only involve dhjk1eyy, xk87au5b, and x272hpoe predicates

    # Heuristic Initialization
    - Extract goal conditions into three sets during initialization:
        D_goal: Objects requiring dhjk1eyy
        X_goal: Objects requiring xk87au5b
        R_goal: Required x272hpoe relations (as (source, target) tuples)

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse current state to extract sets of:
        - Objects with dhjk1eyy
        - Objects with xk87au5b
        - Objects with q71n8mtz (state B)
        - Existing x272hpoe relations
    2. Compute:
        D_missing: D_goal objects missing dhjk1eyy
        X_missing: X_goal objects missing xk87au5b
        R_missing: R_goal relations not present
    3. For each missing relation, if its source isn't in state B (q71n8mtz), 
       add it to set S (counted once per source object)
    4. Heuristic value = D_missing + X_missing + R_missing + |S|
    """

    def __init__(self, task):
        self.D_goal = set()
        self.X_goal = set()
        self.R_goal = set()
        
        # Extract goal conditions from task
        for goal in task.goals:
            if goal.startswith('(') and goal.endswith(')'):
                parts = goal[1:-1].split()
                if not parts:
                    continue
                pred = parts[0]
                args = parts[1:]
                if pred == 'dhjk1eyy' and len(args) == 1:
                    self.D_goal.add(args[0])
                elif pred == 'xk87au5b' and len(args) == 1:
                    self.X_goal.add(args[0])
                elif pred == 'x272hpoe' and len(args) == 2:
                    self.R_goal.add((args[0], args[1]))

    def __call__(self, node):
        state = node.state
        dhjk1eyy_set = set()
        xk87au5b_set = set()
        q71n8mtz_set = set()
        x272hpoe_set = set()
        
        # Parse current state facts
        for fact in state:
            if fact.startswith('(') and fact.endswith(')'):
                parts = fact[1:-1].split()
                if not parts:
                    continue
                pred = parts[0]
                args = parts[1:]
                if pred == 'dhjk1eyy' and len(args) == 1:
                    dhjk1eyy_set.add(args[0])
                elif pred == 'xk87au5b' and len(args) == 1:
                    xk87au5b_set.add(args[0])
                elif pred == 'q71n8mtz' and len(args) == 1:
                    q71n8mtz_set.add(args[0])
                elif pred == 'x272hpoe' and len(args) == 2:
                    x272hpoe_set.add((args[0], args[1]))
        
        # Compute missing components
        D_missing = sum(1 for obj in self.D_goal if obj not in dhjk1eyy_set)
        X_missing = sum(1 for obj in self.X_goal if obj not in xk87au5b_set)
        R_missing = 0
        S = set()
        
        for rel in self.R_goal:
            if rel not in x272hpoe_set:
                R_missing += 1
                src, _ = rel
                # Source not in state B (q71n8mtz)
                if src not in q71n8mtz_set:
                    S.add(src)
        
        return D_missing + X_missing + R_missing + len(S)
