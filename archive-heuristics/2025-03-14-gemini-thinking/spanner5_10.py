from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It counts the number of nuts that are currently 'loose' in the given state.
    Each loose nut is assumed to require at least one 'tighten_nut' action to be tightened.

    # Assumptions:
    - For each loose nut, it is assumed that the man is at the nut's location,
      is carrying a usable spanner, and the spanner is usable.
    - The heuristic simplifies the problem by only counting the number of loose nuts,
      ignoring the cost of actions like 'walk' and 'pickup_spanner'.
    - This heuristic is admissible under the assumption that tightening each nut requires at least one action,
      and we are underestimating the cost by not considering other necessary actions.
      However, for greedy best-first search, admissibility is not required, and the focus is on efficiency
      and guiding the search effectively.

    # Heuristic Initialization
    - No specific initialization is needed for this heuristic beyond the base class.
      It directly uses the goal conditions and static facts provided by the task,
      although static facts are not explicitly used in this simple heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for the heuristic value to 0.
    2. Iterate through the goal conditions.
    3. For each goal condition that is of the form '(tightened ?n)', check if the corresponding nut '?n' is NOT tightened in the current state.
    4. To check if a nut is NOT tightened, look for a fact in the current state that matches '(loose ?n)'.
    5. For every nut that is found to be 'loose' in the current state (and is expected to be 'tightened' in the goal), increment the heuristic counter by 1.
    6. Return the final count as the estimated number of actions required.

    This heuristic essentially counts the number of 'loose' nuts in the current state that need to be 'tightened' to achieve the goal.
    It is a simplification and may underestimate the actual number of actions, but it is efficient to compute.
    """

    def __init__(self, task):
        """Initialize the spanner heuristic."""
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        loose_nuts_in_state = set()
        for fact in state:
            if fact.startswith('(loose '):
                nut = fact[7:-1] # Extract nut name from '(loose nut1)'
                loose_nuts_in_state.add(nut)

        tightened_nuts_in_goal = set()
        for goal_fact in self.goals:
            if goal_fact.startswith('(tightened '):
                nut = goal_fact[11:-1] # Extract nut name from '(tightened nut1)'
                tightened_nuts_in_goal.add(nut)

        for nut in tightened_nuts_in_goal:
            loose_fact = f'(loose {nut})'
            tightened_fact = f'(tightened {nut})'
            if loose_fact in state:
                heuristic_value += 1
            elif tightened_fact in state:
                pass # Nut already tightened, no cost
            else:
                # Should not happen if initial state and goal are well-defined, but for robustness consider it as needing tightening
                heuristic_value += 1

        return heuristic_value
