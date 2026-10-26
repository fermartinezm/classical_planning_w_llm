class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by counting the missing conditions for:
      - Objects that must be in state1 (only hasA: dhjk1eyy and not hasB: xk87au5b and not hasC: q71n8mtz)
      - Objects that must be in state2 (only hasB and not hasA and not hasC)
      - Required directed links (x272hpoe)
    The heuristic value is 2 * max(missing_state1, missing_state2, missing_links).

    # Assumptions
    - The goal conditions only involve state1, state2, and links.
    - Each missing condition requires one `m82i1lr3` action, which in turn requires one `oecve6pr` action (total 2 actions per condition).
    - The initial state has sufficient state0 objects to be converted into state3 for producing required state1 and state2 objects.

    # Heuristic Initialization
    - Extract the goal conditions to identify objects that must be in state1 or state2 and the required links.
    - Store these in sets for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each object required to be in state1, check if it is currently in state1 (hasA true, hasB false, hasC false). If not, count it as missing.
    2. Similarly, for each object required to be in state2, check if it is currently in state2 (hasB true, hasA false, hasC false). If not, count it as missing.
    3. For each required link, check if it exists in the current state. If not, count it as missing.
    4. Let N be the maximum of the counts of missing state1, state2, and link conditions.
    5. The heuristic value is 2 * N, accounting for the two actions (oecve6pr and m82i1lr3) needed per condition.
    """

    def __init__(self, task):
        self.goal_state1 = set()
        self.goal_state2 = set()
        self.goal_links = set()
        
        for goal in task.goals:
            parts = goal.strip().replace('(', '').replace(')', '').split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'dhjk1eyy' and len(parts) == 2:
                self.goal_state1.add(parts[1])
            elif predicate == 'xk87au5b' and len(parts) == 2:
                self.goal_state2.add(parts[1])
            elif predicate == 'x272hpoe' and len(parts) == 3:
                self.goal_links.add((parts[1], parts[2]))
    
    def __call__(self, node):
        state = node.state
        missing_state1 = 0
        for obj in self.goal_state1:
            hasA = f'(dhjk1eyy {obj})' in state
            hasB = f'(xk87au5b {obj})' in state
            hasC = f'(q71n8mtz {obj})' in state
            if not (hasA and not hasB and not hasC):
                missing_state1 += 1
        
        missing_state2 = 0
        for obj in self.goal_state2:
            hasA = f'(dhjk1eyy {obj})' in state
            hasB = f'(xk87au5b {obj})' in state
            hasC = f'(q71n8mtz {obj})' in state
            if not (not hasA and hasB and not hasC):
                missing_state2 += 1
        
        missing_links = 0
        for (obj1, obj2) in self.goal_links:
            if f'(x272hpoe {obj1} {obj2})' not in state:
                missing_links += 1
        
        N = max(missing_state1, missing_state2, missing_links)
        return 2 * N
