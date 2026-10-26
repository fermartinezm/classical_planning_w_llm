from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: '(on r2 r4)' -> ['on', 'r2', 'r4']
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This class provides a domain-dependent heuristic for the rod-rings domain. It estimates
    how many actions are needed to rearrange the rings to satisfy the goal configuration
    defined by "on" and "top"/"bot" relationships.

    Assumptions
    -----------
    1. Each ring may appear in the goal state in one or more facts (e.g., (top r2 s1), (on r2 r4)).
    2. Each ring can appear in multiple goal facts, and each ring must satisfy all those goal facts
       to be considered "correctly placed."
    3. This heuristic treats each ring independently and checks whether all its relevant
       goal facts are present in the current state. If not, it adds a fixed cost for that ring.

    Heuristic Initialization
    ------------------------
    During initialization:
    - We parse all goal facts from the task (task.goals).
    - For each ring mentioned in each goal fact, we add that goal fact to a set associated
      with that ring. The result is a dictionary mapping ring -> set of goal facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies the full goal (task.goal_reached), return 0.
    2. Otherwise, for each ring that appears in any goal fact, retrieve the set of goal facts
       for that ring.
    3. Check if all those facts are currently present in the state. If yes, the ring is in
       its correct state and contributes 0 to the heuristic.
    4. If at least one fact for a ring is missing, we assume we need roughly one or two "put"
       actions to fix that ring. This heuristic adds a cost of 2 for that ring.
    5. Sum these costs across all rings to obtain the final heuristic value.

    This approach is fast to compute and typically reduces node expansions by focusing on
    which rings are out of place.
    """

    def __init__(self, task):
        """
        Parse the goal conditions and gather which goal facts refer to which rings.
        """
        self.task = task
        self.ring_goals = {}  # Dictionary: ring -> set of goal facts referencing that ring

        for goal_fact in task.goals:
            parts = get_parts(goal_fact)  # e.g., ['on', 'r2', 'r4']
            # Identify all rings in the goal fact (any argument named 'r?')
            for arg in parts[1:]:  # skip the predicate name at parts[0]
                # A rudimentary check to see if it looks like a ring name (domain-specific).
                # Alternatively, we might extract ring objects from the domain, but that
                # is not strictly required here.
                if fnmatch(arg, "r*"):  # simplistic test: ring objects often named 'r1', 'r2', etc.
                    if arg not in self.ring_goals:
                        self.ring_goals[arg] = set()
                    self.ring_goals[arg].add(goal_fact)

    def __call__(self, node):
        """
        Return an estimate of the number of actions needed to achieve all goal conditions.
        """
        state = node.state

        # If the current state is already a goal, heuristic is 0.
        if self.task.goal_reached(state):
            return 0

        cost = 0
        # For each ring that appears in at least one goal fact...
        for ring, required_facts in self.ring_goals.items():
            # Check if all required facts for this ring are present in the current state.
            if not required_facts.issubset(state):
                # If at least one is missing, add a fixed cost (e.g., 2).
                # This reflects that we likely need at least one or two "put" actions for this ring.
                cost += 2

        return cost