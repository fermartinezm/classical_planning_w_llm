from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by:
      1. Counting the number of missing goal links, each requiring one action3.
      2. For each receiver in missing links: 
            - If currently AB, add 1 (for action1 to convert to C).
            - If not C and not AB, add a large penalty.
      3. For each donor in missing links:
            - If currently C, add 1 (for action2 to convert to AB).
            - If not A and not AB and not C, add a large penalty.
      4. For objects in A_goal or B_goal not involved in links:
            - If not in desired state, add 2 actions per object.

    # Assumptions
    - The state of objects (A, B, AB, C) is derived from current state predicates.
    - Links created by action3 will set receivers to A and donors to B (if initially AB) or nothing (if initially A).
    - Objects not in links can achieve desired states in at most 2 actions.

    # Heuristic Initialization
    - Extract goal links and sets of objects required to be A or B.
    - Initialize data structures to track object states and goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine object states (A, B, AB, C) and existing links.
    2. Identify missing goal links.
    3. For each missing link:
        a. Receiver must be C (cost 1 if AB, else penalty).
        b. Donor must be A or AB (cost 1 if C, else penalty).
        c. Add 1 for the link creation (action3).
    4. For objects in A_goal or B_goal not involved in links:
        a. If not in desired state, add 2 actions per object.
    5. Sum all costs for the heuristic value.
    """

    def __init__(self, task):
        self.task = task
        self.goal_links = set()
        self.A_goal = set()
        self.B_goal = set()
        
        for goal in task.goals:
            parts = goal.strip('()').split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'x272hpoe':
                if len(parts) >= 3:
                    self.goal_links.add((parts[1], parts[2]))
            elif predicate == 'dhjk1eyy':
                if len(parts) >= 2:
                    self.A_goal.add(parts[1])
            elif predicate == 'xk87au5b':
                if len(parts) >= 2:
                    self.B_goal.add(parts[1])
                    
    def __call__(self, node):
        state = node.state
        current_links = set()
        obj_state = {}
        free = False
        
        for fact in state:
            parts = fact.strip('()').split()
            if not parts:
                continue
            pred = parts[0]
            if pred == 'x272hpoe' and len(parts) >= 3:
                current_links.add((parts[1], parts[2]))
            elif pred == 'hxtpdpff':
                free = True
            elif pred == 'dhjk1eyy' and len(parts) >= 2:
                obj = parts[1]
                if obj in obj_state:
                    if obj_state[obj] == 'B':
                        obj_state[obj] = 'AB'
                else:
                    obj_state[obj] = 'A'
            elif pred == 'xk87au5b' and len(parts) >= 2:
                obj = parts[1]
                if obj in obj_state:
                    if obj_state[obj] == 'A':
                        obj_state[obj] = 'AB'
                else:
                    obj_state[obj] = 'B'
            elif pred == 'q71n8mtz' and len(parts) >= 2:
                obj = parts[1]
                obj_state[obj] = 'C'
        
        missing_links = self.goal_links - current_links
        receivers = set(x for (x, y) in missing_links)
        donors = set(y for (x, y) in missing_links)
        
        BIG_PENALTY = 1000000
        cost = 0
        
        for x in receivers:
            if x in obj_state:
                if obj_state[x] == 'C':
                    cost_x = 0
                elif obj_state[x] == 'AB':
                    cost_x = 1
                else:
                    cost_x = BIG_PENALTY
            else:
                cost_x = BIG_PENALTY
            cost += cost_x
        
        for y in donors:
            if y in obj_state:
                if obj_state[y] in ['A', 'AB']:
                    cost_y = 0
                elif obj_state[y] == 'C':
                    cost_y = 1
                else:
                    cost_y = BIG_PENALTY
            else:
                cost_y = BIG_PENALTY
            cost += cost_y
        
        cost += len(missing_links)
        
        for x in self.A_goal:
            if x in receivers:
                continue
            if x in obj_state and obj_state[x] == 'A':
                continue
            cost += 2
        
        for x in self.B_goal:
            if x in donors:
                if x in obj_state and obj_state[x] == 'B':
                    continue
                elif x in obj_state and obj_state[x] in ['AB', 'C']:
                    continue
                else:
                    cost += 2
            else:
                if x in obj_state and obj_state[x] == 'B':
                    continue
                cost += 2
        
        return cost
