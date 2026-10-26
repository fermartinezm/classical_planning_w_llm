from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal by:
      - For each missing goal link: base cost 1, plus 1 for each unmet precondition (source object not in state C or target object not in state A).
      - For each goal condition requiring an object to have A that is unmet: cost 1.
      - For each goal condition requiring an object to have B that is unmet: cost 1.

    Assumptions
    -----------
    - The goal is a conjunction of atoms of the form (x272hpoe X Y), (dhjk1eyy X), or (xk87au5b X).
    - The state representation uses strings exactly matching the PDDL facts.

    Heuristic Initialization
    ------------------------
    Extracts the goal conditions into three sets:
        - goal_links: set of tuples (X, Y) for each goal link (x272hpoe X Y).
        - goal_A: set of strings of the form '(dhjk1eyy X)' for each goal condition.
        - goal_B: set of strings of the form '(xk87au5b X)' for each goal condition.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. For each goal link (X, Y):
        - If the link is missing in the current state, add 1.
        - If the source object X is not in state C (i.e., (q71n8mtz X) is missing), add 1.
        - If the target object Y is not in state A (i.e., (dhjk1eyy Y) is missing), add 1.
    2. For each goal condition requiring an object to have A:
        - If the condition is missing in the current state, add 1.
    3. For each goal condition requiring an object to have B:
        - If the condition is missing in the current state, add 1.
    4. Sum all the costs to get the heuristic value.
    """

    def __init__(self, task):
        self.goal_links = set()
        self.goal_A = set()
        self.goal_B = set()
        
        for fact in task.goals:
            parts = fact[1:-1].split()
            predicate = parts[0]
            if predicate == "x272hpoe":
                obj1 = parts[1]
                obj2 = parts[2]
                self.goal_links.add((obj1, obj2))
            elif predicate == "dhjk1eyy":
                self.goal_A.add(fact)
            elif predicate == "xk87au5b":
                self.goal_B.add(fact)
    
    def __call__(self, node):
        state = node.state
        h = 0
        
        for (X, Y) in self.goal_links:
            link_fact = f"(x272hpoe {X} {Y})"
            if link_fact not in state:
                cost = 1
                if f"(q71n8mtz {X})" not in state:
                    cost += 1
                if f"(dhjk1eyy {Y})" not in state:
                    cost += 1
                h += cost
        
        for a_goal in self.goal_A:
            if a_goal not in state:
                h += 1
        
        for b_goal in self.goal_B:
            if b_goal not in state:
                h += 1
        
        return h
