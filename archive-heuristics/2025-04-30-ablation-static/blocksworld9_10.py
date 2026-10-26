# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic check for expected format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
         # Return empty list or handle error as appropriate for the environment
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number
    of goal conditions related to block positions (`on` and `on-table` facts)
    that are not satisfied in the current state. Each unsatisfied goal position
    fact contributes 1 to the heuristic value.

    # Assumptions
    - The goal is primarily defined by the desired stacking configuration of blocks
      using `on` and `on-table` predicates.
    - `clear` goal conditions are typically satisfied if the blocks below are
      correctly placed and nothing is on top. This heuristic focuses on the
      placement of blocks relative to what should be immediately below them.
    - The heuristic value is 0 if and only if all goal `on` and `on-table`
      conditions are met. For typical Blocksworld problems, this implies
      the goal state has been reached (assuming `clear` goals are implicitly
      satisfied by the correct stack structure).

    # Heuristic Initialization
    - Extracts the set of goal facts that are of the form `(on ?x ?y)` or
      `(on-table ?x)`. These represent the target positions of blocks within
      the goal stacks and are the only goal conditions considered by this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Retrieve the set of goal facts related to block positions (`on` and `on-table`)
       that was pre-computed during the heuristic's initialization (`self.goal_positions`).
    3. For each goal fact in the `self.goal_positions` set:
       - Check if this specific goal fact string is present in the current state's
         set of facts (`node.state`).
       - If the goal fact string is NOT found within the current state, it means
         this particular goal condition is not yet satisfied. Increment `h` by 1.
    4. After checking all goal position facts, the final value of `h` represents
       the total count of unsatisfied goal `on` and `on-table` conditions. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions related to
        block positions (`on` and `on-table`).
        """
        super().__init__(task)
        # Store goal facts that define block positions
        self.goal_positions = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Consider only 'on' and 'on-table' goal facts
            if parts and (parts[0] == 'on' or parts[0] == 'on-table'):
                self.goal_positions.add(goal)

        # Static facts are not used in this specific heuristic calculation,
        # but are available via self.static if needed for other heuristics.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        The heuristic is the count of unsatisfied goal `on` and `on-table` facts.
        """
        state = node.state # state is a frozenset of fact strings

        h = 0
        # Count how many goal position facts are not in the current state
        for goal_fact in self.goal_positions:
            if goal_fact not in state:
                h += 1

        return h
