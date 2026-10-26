from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts 4 actions for each passenger who is not yet boarded and served,
    and 2 actions for each passenger who is boarded but not yet served.
    This is a simplification that assumes each passenger requires a fixed number of actions
    regardless of the current lift position or other passengers.

    # Assumptions:
    - Each unboarded and unserved passenger requires 4 actions: move to origin, board, move to destination, depart.
    - Each boarded and unserved passenger requires 2 actions: move to destination, depart.
    - This heuristic does not consider the efficiency of serving multiple passengers with a single lift movement.

    # Heuristic Initialization
    - No initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each passenger in the problem instance.
    3. For each passenger, check if the predicate '(served ?p)' is true in the current state.
    4. If '(served ?p)' is not true, check if '(boarded ?p)' is true.
    5. If '(boarded ?p)' is true, add 2 to the heuristic value (estimated actions: move to destination, depart).
    6. If '(boarded ?p)' is false, add 4 to the heuristic value (estimated actions: move to origin, board, move to destination, depart).
    7. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        super().__init__(task)
        self.passengers = set()
        for fact in task.initial_state | task.goals | task.static:
            parts = self._get_parts(fact)
            if parts and parts[0] in ['origin', 'destin']:
                self.passengers.add(parts[1])

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        if not isinstance(fact, str):
            return None
        if not fact.startswith('(') or not fact.endswith(')'):
            return None
        return fact[1:-1].split()

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0
        for passenger in self.passengers:
            served_predicate = f'(served {passenger})'
            boarded_predicate = f'(boarded {passenger})'
            if served_predicate not in state:
                if boarded_predicate in state:
                    heuristic_value += 2
                else:
                    heuristic_value += 4
        return heuristic_value
