from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal by:
      1. Checking unary goal conditions (xk87au5b and dhjk1eyy) and counting actions needed to satisfy them.
      2. Counting missing binary goal conditions (x272hpoe edges) at a fixed cost per edge.

    Assumptions:
      - An object in mode B (q71n8mtz) can be converted to mode A (dhjk1eyy and xk87au5b) with one action (dgdzvgpm).
      - Objects not in mode A or B (inactive) cannot be activated if they require xk87au5b in the goal.
      - Each missing x272hpoe edge requires one action to create.

    Heuristic Initialization:
      - Extracts goal conditions for unary predicates (xk87au5b, dhjk1eyy) and binary predicate (x272hpoe).
      - Initializes data structures to store these goals.

    Step-By-Step Thinking for Computing Heuristic:
      1. For each object requiring xk87au5b in the goal:
            - If already satisfied, skip.
            - If in mode B, add 1 action (dgdzvgpm).
            - If in mode A or inactive, add a high penalty (dead end).
      2. For each object requiring dhjk1eyy in the goal and not fixed in step 1:
            - If already satisfied, skip.
            - If in mode B, add 1 action (dgdzvgpm).
            - If inactive, add 2 actions (activation via m82i1lr3 and fpk054nz).
      3. For each missing x272hpoe edge, add 1 action.
      4. Sum the counts from steps 1-3 for the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal conditions
        self.goal_xk87au5b = set()
        self.goal_dhjk1eyy = set()
        self.goal_edges = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "xk87au5b" and len(parts) == 2:
                self.goal_xk87au5b.add(parts[1])
            elif predicate == "dhjk1eyy" and len(parts) == 2:
                self.goal_dhjk1eyy.add(parts[1])
            elif predicate == "x272hpoe" and len(parts) == 3:
                self.goal_edges.add((parts[1], parts[2]))
    
    def __call__(self, node):
        state = node.state
        h = 0
        fixed_dhjk = set()
        
        # Process xk87au5b goals
        for obj in self.goal_xk87au5b:
            xk_fact = f'(xk87au5b {obj})'
            if xk_fact in state:
                continue
            q71_fact = f'(q71n8mtz {obj})'
            dhjk_fact = f'(dhjk1eyy {obj})'
            if q71_fact in state:
                h += 1
                fixed_dhjk.add(obj)
            elif dhjk_fact in state:
                h += 10000  # Dead end: in mode A without xk87au5b
            else:
                h += 10000  # Dead end: inactive and cannot activate with xk87au5b
        
        # Process dhjk1eyy goals
        for obj in self.goal_dhjk1eyy:
            if obj in fixed_dhjk:
                continue
            dhjk_fact = f'(dhjk1eyy {obj})'
            if dhjk_fact in state:
                continue
            q71_fact = f'(q71n8mtz {obj})'
            if q71_fact in state:
                h += 1
            else:
                # Inactive: requires activation via edge operations (2 actions)
                h += 2
        
        # Process x272hpoe goals
        for (a, b) in self.goal_edges:
            edge_fact = f'(x272hpoe {a} {b})'
            if edge_fact not in state:
                h += 1
        
        return h
