class SecretDomainHeuristic:
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    This heuristic estimates the number of actions required to achieve the goal state by:
      - Counting each missing 'x272hpoe' relation (from the goal) in the current state as 2 actions.
      - Counting each missing 'xk87au5b' predicate (from the goal) in the current state as 2 actions.

    Assumptions
    - The goal conditions for 'x272hpoe' and 'xk87au5b' are independent of other predicates.
    - Each missing relation or predicate requires a fixed number of actions (2) to achieve.

    Heuristic Initialization
    - Extract goal conditions for 'x272hpoe' relations and 'xk87au5b' predicates during initialization.

    Step-By-Step Thinking for Computing Heuristic
    1. For the current state, identify all 'x272hpoe' relations and 'xk87au5b' predicates.
    2. Compare these with the goal conditions to find missing relations and predicates.
    3. For each missing 'x272hpoe' relation, add 2 to the heuristic value.
    4. For each missing 'xk87au5b' predicate, add 2 to the heuristic value.
    5. Sum these values to get the total heuristic estimate.
    """

    def __init__(self, task):
        self.goal_links = set()
        self.goal_xk87au5b = set()
        
        for goal in task.goals:
            parts = goal.strip('()').split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'x272hpoe' and len(parts) >= 3:
                obj1, obj2 = parts[1], parts[2]
                self.goal_links.add((obj1, obj2))
            elif predicate == 'xk87au5b' and len(parts) >= 2:
                obj = parts[1]
                self.goal_xk87au5b.add(obj)
    
    def __call__(self, node):
        state = node.state
        missing_links = 0
        for link in self.goal_links:
            obj1, obj2 = link
            fact_str = f"(x272hpoe {obj1} {obj2})"
            if fact_str not in state:
                missing_links += 1
        
        current_xk87au5b = set()
        for fact in state:
            parts = fact.strip('()').split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'xk87au5b' and len(parts) >= 2:
                obj = parts[1]
                current_xk87au5b.add(obj)
        
        missing_xk87au5b = self.goal_xk87au5b - current_xk87au5b
        
        return 2 * missing_links + 2 * len(missing_xk87au5b)
