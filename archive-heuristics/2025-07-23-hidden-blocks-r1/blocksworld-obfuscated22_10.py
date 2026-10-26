from heuristics.heuristic_base import Heuristic

class SecretDomainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.
    
    # Summary
    This heuristic estimates the number of actions required to reach the goal 
    by counting the number of unsatisfied goal conditions in the current state. 
    Each goal condition is represented as a string (PDDL fact), and the heuristic 
    value is simply the count of these goal conditions that are not present in 
    the current state. This approach is efficient and serves as a direct measure 
    of progress toward the goal.
    
    # Assumptions
    - The goal conditions are independent of each other.
    - Each unsatisfied goal condition will require at least one action to achieve.
    - Static facts (always true) are already accounted for and not included in 
      the relevant goal conditions.
      
    # Heuristic Initialization
    During initialization, the heuristic:
    1. Stores the set of goal conditions from the task.
    2. Computes the relevant dynamic goals by excluding static facts (which are 
       always true and do not need to be achieved).
    
    # Step-By-Step Thinking for Computing Heuristic
    1. For a given state, the heuristic checks each relevant goal condition.
    2. If a goal condition is not present in the current state, it is counted.
    3. The total count of such missing goal conditions is returned as the 
       heuristic value.
    4. Goal states have a heuristic value of 0.
    5. Non-goal states have a positive value equal to the number of 
       unsatisfied goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with task information.
        
        Args:
            task: The planning task containing goals and static facts.
        """
        # Store the set of goal conditions
        self.goals = task.goals
        # Exclude static facts from relevant goals (they're always true)
        self.relevant_goals = self.goals - task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given node.
        
        Args:
            node: The node representing the current state.
            
        Returns:
            The number of unsatisfied goal conditions in the current state.
        """
        state = node.state
        # Count relevant goals not present in the current state
        return sum(1 for goal in self.relevant_goals if goal not in state)
