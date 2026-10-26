from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal state by:
      1. Counting the number of goal edges not present in the initial state (each contributes 2 actions).
      2. For objects required to be in state A1 (dhjk1eyy) in the goal but not initially, 
         and not set by an edge action, adding 1 or 2 actions based on their initial state.
      3. For objects required to be in state A2 (xk87au5b) in the goal but not initially, 
         adding 1 or 2 actions based on their initial state.

    Assumptions:
      - The global flag 'hxtpdpff' is ignored as it does not affect the goal.
      - The goal edges must be built, requiring preparation of source objects.
      - Objects not initially in the required states (A1 or A2) require additional actions.

    Heuristic Initialization:
      - Extracts goal conditions: edges, A1 states, and A2 states.
      - Identifies initial edges, A1 states, A2 states, and C states.

    Step-By-Step Thinking for Computing Heuristic:
      1. Identify goal edges and initial edges. Missing edges contribute 2 actions each.
      2. Identify goal A1 objects not initially in A1 and not set by an edge. 
         - If in state C, add 1 action (dgdzvgpm).
         - Otherwise, add 2 actions (estimate).
      3. Identify goal A2 objects not initially in A2.
         - If in state C, add 1 action (dgdzvgpm).
         - Otherwise, add 2 actions (estimate).
      4. Sum the contributions from missing edges and state mismatches.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and initial state information."""
        self.task = task
        self.goals = task.goals
        self.state = task.initial_state

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract initial edges, A1, A2, and C states
        initial_edges = set()
        initial_A1 = set()
        initial_A2 = set()
        initial_C = set()
        
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 0:
                continue
            predicate = parts[0]
            if predicate == "x272hpoe" and len(parts) == 3:
                initial_edges.add((parts[1], parts[2]))
            elif predicate == "dhjk1eyy" and len(parts) == 2:
                initial_A1.add(parts[1])
            elif predicate == "xk87au5b" and len(parts) == 2:
                initial_A2.add(parts[1])
            elif predicate == "q71n8mtz" and len(parts) == 2:
                initial_C.add(parts[1])
        
        # Extract goal edges, A1, and A2
        goal_edges = set()
        goal_A1 = set()
        goal_A2 = set()
        
        for fact in self.goals:
            parts = get_parts(fact)
            if len(parts) == 0:
                continue
            predicate = parts[0]
            if predicate == "x272hpoe" and len(parts) == 3:
                goal_edges.add((parts[1], parts[2]))
            elif predicate == "dhjk1eyy" and len(parts) == 2:
                goal_A1.add(parts[1])
            elif predicate == "xk87au5b" and len(parts) == 2:
                goal_A2.add(parts[1])
        
        # Identify sources in goal edges
        sources = {u for (u, v) in goal_edges}
        
        heuristic_value = 0
        
        # Cost for missing edges: 2 per edge
        missing_edges = goal_edges - initial_edges
        heuristic_value += len(missing_edges) * 2
        
        # Cost for goal_A1 objects
        for obj in goal_A1:
            if obj in sources:
                continue
            if obj in initial_A1:
                continue
            if obj in initial_C:
                heuristic_value += 1
            else:
                heuristic_value += 2
        
        # Cost for goal_A2 objects
        for obj in goal_A2:
            if obj in initial_A2:
                continue
            if obj in initial_C:
                heuristic_value += 1
            else:
                heuristic_value += 2
                
        return heuristic_value
