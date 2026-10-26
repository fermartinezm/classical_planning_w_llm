# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import defaultdict

# Define helper functions if they are not in the base class
def _get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential leading/trailing whitespace and parentheses
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
         return fact[1:-1].split()
    return fact.split() # Should not happen for valid PDDL facts

def _match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = _get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by summing two components for each block that is part of the goal configuration:
    1. A cost of 1 if the block is not currently on the correct block or table as specified in the goal (or if it's being held).
    2. The number of blocks currently stacked directly on top of it.

    This heuristic is non-admissible and aims to prioritize states where blocks are closer to their goal positions and have fewer blocks obstructing them.

    # Assumptions
    - The goal is a conjunction of `(on ?x ?y)` and `(on-table ?z)` predicates defining the desired final stack configuration.
    - All blocks mentioned in the goal are present in the initial state (and thus in all reachable states, unless held).
    - Action costs are uniform (e.g., 1).

    # Heuristic Initialization
    - The goal configuration (`goal_below`) is extracted from the task's goal predicates, mapping each block to the block or 'table' it should be directly on top of. Only blocks explicitly mentioned in `on` or `on-table` goal predicates are considered.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the block immediately below each block (`current_below`) and which blocks are directly on top of others (`on_top`). Also identify if any block is currently being held (`holding_block`).
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block `X` that is part of the goal configuration (i.e., present as a key in the `goal_below` map).
    4. For each block `X`:
       a. Determine if `X` is in its correct position relative to the block/table below it according to the goal:
          - If `X` is currently being held (`holding_block == X`), it is not on a base, so it's considered misplaced relative to its goal base. Add 1 to `h`.
          - If `X` is not being held, check its current base (`current_below.get(X)`). If `X` is not found in `current_below` (shouldn't happen in valid states for goal blocks unless held), or if `current_below.get(X)` is different from `goal_below[X]`, it is in the wrong position relative to its goal base. Add 1 to `h`.
          - If `X` is not held and `current_below.get(X) == goal_below[X]`, it is in the correct position relative to its goal base. Add 0 for this component.
       b. Count the number of blocks currently stacked directly on top of `X` in the current state. Add this count (`len(on_top[X])`) to `h`. These blocks must be moved before `X` can be moved or stacked upon.
    5. The total value of `h` after iterating through all goal blocks is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration.
        """
        # Call the base class constructor. Assume it handles self.goals and self.static.
        super().__init__(task)

        # Extract goal configuration: block -> block_below or 'table'
        self.goal_below = {}
        # Assuming goals are in the format (predicate arg1 arg2 ...)
        for goal in self.goals:
            parts = _get_parts(goal) # Use the helper function
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_below = parts[2]
                self.goal_below[block_on] = block_below
            elif predicate == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                self.goal_below[block_on_table] = 'table'
            # Ignore other goal predicates like (clear X) or (arm-empty) for this heuristic

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse the current state
        current_below = {}
        on_top = defaultdict(set)
        holding_block = None

        for fact in state:
            parts = _get_parts(fact) # Use the helper function
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_below = parts[2]
                current_below[block_on] = block_below
                on_top[block_below].add(block_on)
            elif predicate == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                current_below[block_on_table] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                holding_block = parts[1]
            # Ignore clear, arm-empty, object facts for this heuristic

        # 2. Initialize heuristic value
        h = 0

        # 3. Iterate through each block that is part of the goal configuration
        for block in self.goal_below:
            # 4a. Check if the block is in the correct position relative to below
            is_misplaced_relative_to_below = False
            if holding_block == block:
                 # If holding, it's not on the correct base
                 is_misplaced_relative_to_below = True
            elif block in current_below:
                 if current_below[block] != self.goal_below[block]:
                     is_misplaced_relative_to_below = True
            else:
                 # Block is not held and not in current_below.
                 # This implies the block is not on anything or the table according to the state facts.
                 # In a valid blocksworld state, a block must be either on something, on the table, or held.
                 # If a block from the goal is not found in state facts (neither held nor in current_below),
                 # it indicates an invalid state representation or the block somehow disappeared.
                 # Assuming valid states, this 'else' block should ideally not be reached for blocks in goal_below.
                 # However, if it is reached, the block is certainly not in its goal position relative to below.
                 is_misplaced_relative_to_below = True


            if is_misplaced_relative_to_below:
                h += 1

            # 4b. Count blocks on top
            # Blocks on top need to be moved regardless of whether this block is in the right place below
            # A block being held cannot have anything on top.
            # We only add the count if the block is actually in the state (not held) and can have things on it.
            if holding_block != block and block in on_top:
                 h += len(on_top[block])
            # Note: If a block is not in current_below and not holding, it also cannot have blocks on top.
            # The 'else' case above for is_misplaced_relative_to_below implicitly handles this.

        # Heuristic is 0 only if for all goal blocks X:
        # 1. X is not held AND (X is in current_below AND current_below[X] == goal_below[X])
        # 2. X is not held AND X has no blocks on top (len(on_top[X]) == 0)
        # This means all goal blocks are on their correct base and are clear.
        # This matches the required properties of goal blocks in the goal state.
        # The heuristic does not explicitly check for (arm-empty) or (clear X) for non-goal blocks,
        # but these are often derived or less critical for the core block stacking problem.
        # The heuristic being 0 implies the goal stack configuration is achieved and the blocks involved are clear.

        return h
