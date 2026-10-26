from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# If running as a standalone file for testing, you might need a mock Heuristic class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a mock Heuristic class for standalone testing if needed
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential empty facts or malformed input
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle error or return empty list depending on desired robustness
        # For PDDL facts from a parser, this basic split should be sufficient
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting the number of blocks
    that are not in their correct position within the goal stacks, relative to a
    correctly placed base, plus a penalty if the arm needs to be empty.

    # Assumptions
    - The goal specifies the desired configuration of certain blocks into stacks
      or on the table using `on` and `on-table` predicates.
    - Blocks not mentioned in `on` or `on-table` goal facts are not considered
      by the structural part of the heuristic.
    - `(clear ?x)` goals are implicitly handled by the need to build stacks,
      except for the top block which is handled by the structural check.
    - `(arm-empty)` goal is handled separately.

    # Heuristic Initialization
    - Parses the goal facts to build the target configuration (`goal_config`)
      mapping each block to the block it should be on, or 'table'.
    - Identifies the set of blocks (`goal_blocks`) that are part of the
      specified goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If yes, return 0.
    2. Build the current configuration (`current_config`) mapping each block
       to what it is currently on ('table', another block, or 'holding').
       This map is only built for blocks present in the current state.
    3. Initialize a status tracker (`correctly_placed_status`) for all blocks
       in `goal_blocks` to False.
    4. Identify blocks that are correctly placed at the base of goal stacks:
       These are blocks `b` from `goal_blocks` where the goal is `(on-table b)`
       AND the current state has `(on-table b)`. Mark these as correctly placed
       and add them to a queue for further processing.
    5. Iteratively propagate the "correctly placed" status: While the queue
       is not empty, take a block `base` from the queue. Find all blocks `b`
       from `goal_blocks` whose goal is to be directly on `base` (`goal_config[b] == base`).
       If such a block `b` is currently on `base` (`current_config.get(b) == base`)
       and is not already marked correctly placed, mark `b` as correctly placed
       and add it to the queue. Using `.get(b)` handles cases where a goal block
       might be missing from the current state (e.g., being held).
    6. Count the number of blocks in `goal_blocks` that are *not* marked
       as correctly placed. This is the base structural heuristic value.
    7. Check if `(arm-empty)` is a goal and if the arm is currently holding
       a block. If both are true, add 1 to the heuristic value (representing
       the cost to put down or stack the held block).
    8. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals
        self.goal_config = {}  # Maps block -> block_below or 'table'
        self.goal_blocks = set() # Blocks explicitly mentioned in on/on-table goals

        for goal_fact in self.goals:
            predicate, *args = get_parts(goal_fact)
            if predicate == "on":
                if len(args) == 2:
                    block, base = args
                    self.goal_config[block] = base
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(base)
            elif predicate == "on-table":
                if len(args) == 1:
                    block = args[0]
                    self.goal_config[block] = 'table'
                    self.goal_blocks.add(block)
            # Ignore (clear ?x) and (arm-empty) goals for structural part initialization

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Build current configuration
        current_config = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                if len(args) == 2:
                    block, base = args
                    current_config[block] = base
            elif predicate == "on-table":
                if len(args) == 1:
                    block = args[0]
                    current_config[block] = 'table'
            elif predicate == "holding":
                if len(args) == 1:
                    block = args[0]
                    current_config[block] = 'holding' # Special state

        # 3. Initialize correctly placed status for goal blocks
        correctly_placed_status = {b: False for b in self.goal_blocks}

        # 4. Identify base cases for correctness (on the table)
        queue = []
        for block in self.goal_blocks:
            # Check if block has a goal position defined and it's 'table'
            if block in self.goal_config and self.goal_config[block] == 'table':
                # Check if block is currently on the table
                # Use .get() in case the block is 'holding' or not in state for some reason
                if current_config.get(block) == 'table':
                    if not correctly_placed_status[block]:
                        correctly_placed_status[block] = True
                        queue.append(block)

        # 5. Iteratively propagate correctness up the stacks
        # Use a list as a queue
        q_index = 0
        while q_index < len(queue):
            current_base = queue[q_index]
            q_index += 1

            # Find blocks from goal_blocks whose goal is to be on current_base
            for block in self.goal_blocks:
                if block in self.goal_config and self.goal_config[block] == current_base:
                    # Check if block is currently on current_base
                    # Use .get() in case the block is 'holding' or not in state
                    if current_config.get(block) == current_base:
                        # If block is not already marked correctly placed
                        if not correctly_placed_status[block]:
                            correctly_placed_status[block] = True
                            queue.append(block) # Add this block as a potential base

        # 6. Count blocks not correctly placed structurally
        misplaced_structural = sum(1 for block in self.goal_blocks if not correctly_placed_status[block])

        # 7. Add penalty for arm state if arm-empty is a goal
        arm_penalty = 0
        arm_goal_empty = "(arm-empty)" in self.goals
        arm_is_empty = "(arm-empty)" in state

        if arm_goal_empty and not arm_is_empty:
             arm_penalty = 1 # Need at least one action (putdown/stack) to empty the arm

        # 8. Return total heuristic value
        return misplaced_structural + arm_penalty
