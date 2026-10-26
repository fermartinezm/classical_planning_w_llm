from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal by counting:
      1. The number of goal facts not satisfied in the current state.
      2. An additional penalty if there are extra links (x272hpoe facts present in the state but not in the goal) 
         and the global flag (hxtpdpff) is not set, as breaking these links requires the flag to be set.

    Assumptions:
    - The goal consists of positive literals only; extra facts in the state are acceptable.
    - The global flag (hxtpdpff) is represented as the string '(hxtpdpff)' when true.
    - Extra links must be broken using the action fpk054nz, which requires the flag to be set.

    Heuristic Initialization:
    - The goals are stored as a set for quick lookup.
    - Static facts are not used in this heuristic.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal fact, check if it is present in the current state. If not, increment the count.
    2. Identify any extra links (x272hpoe facts in the state not in the goal).
    3. If there are extra links and the global flag (hxtpdpff) is not set, add 1 to the count. This accounts for 
       the action needed to set the flag (via dgdzvgpm or m82i1lr3) before breaking the extra links.
    4. Return the total count as the heuristic value.
    """

    def __init__(self, task):
        """Store the goal conditions from the task."""
        self.goals = task.goals
        # We don't use static facts in this heuristic.

    def __call__(self, node):
        """Compute the heuristic value for the given node's state."""
        state = node.state
        count = 0
        
        # Count the number of goal facts not present in the current state.
        for goal in self.goals:
            if goal not in state:
                count += 1
        
        # Check for extra links (x272hpoe facts in the state that are not in the goals)
        extra_links = []
        for fact in state:
            if fact.startswith('(x272hpoe ') and fact not in self.goals:
                extra_links.append(fact)
        
        # If there are extra links and the flag is not set, add a penalty of 1.
        if extra_links and '(hxtpdpff)' not in state:
            count += 1
        
        return count
