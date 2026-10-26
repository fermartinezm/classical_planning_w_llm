# Assuming Heuristic base class is available from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Depending on expected input robustness, could log a warning or raise error
         return [] # Return empty list for malformed facts
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting
    the number of blocks that are not on their correct support (either another
    block or the table) as specified in the goal, plus the number of goal
    predicates of the form (clear ?x) that are not satisfied.

    # Assumptions
    - The goal state is defined by a set of (on ?x ?y), (on-table ?x), and
      (clear ?x) predicates, forming one or more stacks on the table.
    - Blocks not mentioned in the goal structure can be anywhere. This heuristic
      primarily focuses on blocks whose position is specified in the goal.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired support (block or table)
      for each block and the set of blocks that must be clear in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Parse the current state to determine the current support for each block
       (on another block, on the table, or being held) and the set of blocks
       that are currently clear.
    3. Iterate through each block whose desired support is specified in the goal:
       - Determine the block's current support from the state facts. A block
         being held has 'holding' as its support for this check. If a block
         is not found in the current state's support mapping (which shouldn't
         happen in a valid Blocksworld state where every block is somewhere),
         it's also considered misplaced.
       - If the current support does not match the desired goal support for this
         block, increment the heuristic value.
    4. Iterate through each block that must be clear according to the goal:
       - Check if the block is currently clear in the state.
       - If it is not clear, increment the heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support and clear requirements.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Map block to its desired support (another block or 'table')
        self.goal_support = {}
        # Set of blocks that must be clear in the goal
        self.goal_clear = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore other goal predicates if any

        # Blocksworld domain has no static facts defined in the provided definition.
        # If it did, they would be in task.static and processed here if needed.
        # static_facts = task.static
        # No static facts relevant to this heuristic needed from the provided domain.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the cost to reach the goal.
        """
        state = node.state

        # Determine current support for each block and which blocks are clear
        current_support = {}
        current_clear = set()
        # We don't strictly need 'currently_holding' as a separate variable,
        # we can just check if the block's support is 'holding' in current_support.

        for state_fact in state:
            parts = get_parts(state_fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                # A block being held has 'holding' as its support for this heuristic's logic
                current_support[block] = 'holding'
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                current_clear.add(block)
            # Ignore other state predicates like 'arm-empty'

        heuristic_value = 0

        # 1. Count blocks not on their correct support
        for block, desired_support in self.goal_support.items():
            # Get the block's current support. If the block isn't in current_support
            # (e.g., due to parsing error or unexpected state fact), treat it as misplaced.
            # In a valid BW state, a block is always on/on-table/holding, so it should be in current_support.
            current_sup = current_support.get(block)

            if current_sup != desired_support:
                 # Block is either held, or on a physical support that is wrong,
                 # or its location is unknown (shouldn't happen in valid states).
                 heuristic_value += 1

        # 2. Count goal clear predicates that are not satisfied
        for block in self.goal_clear:
            # Check if the goal requires block to be clear, but it isn't in the current state
            if block not in current_clear:
                heuristic_value += 1

        return heuristic_value
