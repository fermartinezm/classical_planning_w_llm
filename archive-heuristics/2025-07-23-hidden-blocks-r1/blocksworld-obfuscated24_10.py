from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal by considering:
    - The number of missing relations (x272hpoe) in the current state compared to the goal.
    - The difference in the count of dhjk1eyy tokens between the current state and the goal.
    - The number of missing xk87au5b conditions for objects specified in the goal.
    - The number of objects missing dhjk1eyy that are required in the goal.

    Assumptions:
    - The goal conditions only include the predicates: x272hpoe, xk87au5b, and dhjk1eyy.
    - The global flag (hxtpdpff) is not part of the goal conditions.
    - The heuristic does not need to be admissible.

    Heuristic Initialization:
    - Extracts the goal conditions into sets for relations, xk87au5b objects, and dhjk1eyy objects.
    - The static facts are not used as the domain does not have static information affecting the heuristic.

    Step-By-Step Thinking for Computing Heuristic:
    1. For the current state, extract:
        - Existing relations (x272hpoe) as tuples of objects.
        - Objects having xk87au5b.
        - Objects having dhjk1eyy.
    2. Compute the cost components:
        a. Number of relations in the goal not present in the current state.
        b. Absolute difference between the number of dhjk1eyy tokens in the goal and current state.
        c. Number of objects required to have xk87au5b in the goal but missing in the current state.
        d. Number of objects required to have dhjk1eyy in the goal but missing in the current state.
    3. Sum these components to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_relations = set()
        self.goal_xk87au5b = set()
        self.goal_dhjk1eyy = set()
        
        for fact in task.goals:
            parts = fact[1:-1].split()
            predicate = parts[0]
            if predicate == "x272hpoe":
                obj1, obj2 = parts[1], parts[2]
                self.goal_relations.add((obj1, obj2))
            elif predicate == "xk87au5b":
                obj = parts[1]
                self.goal_xk87au5b.add(obj)
            elif predicate == "dhjk1eyy":
                obj = parts[1]
                self.goal_dhjk1eyy.add(obj)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        
        current_relations = set()
        current_xk87au5b = set()
        current_dhjk1eyy = set()
        
        for fact in state:
            parts = fact[1:-1].split()
            predicate = parts[0]
            if predicate == "x272hpoe":
                obj1, obj2 = parts[1], parts[2]
                current_relations.add((obj1, obj2))
            elif predicate == "xk87au5b":
                obj = parts[1]
                current_xk87au5b.add(obj)
            elif predicate == "dhjk1eyy":
                obj = parts[1]
                current_dhjk1eyy.add(obj)
        
        cost = 0
        missing_relations = self.goal_relations - current_relations
        cost += len(missing_relations)
        
        current_count = len(current_dhjk1eyy)
        goal_count = len(self.goal_dhjk1eyy)
        cost += abs(goal_count - current_count)
        
        missing_xk87au5b = [obj for obj in self.goal_xk87au5b if obj not in current_xk87au5b]
        cost += len(missing_xk87au5b)
        
        missing_dhjk1eyy = self.goal_dhjk1eyy - current_dhjk1eyy
        cost += len(missing_dhjk1eyy)
        
        return cost
