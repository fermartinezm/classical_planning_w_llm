# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing or if the base class
# is not provided in the execution environment. Replace with actual import if needed.
class Heuristic:
    """
    Dummy base class for heuristic functions.
    In a real planning system, this would likely be an abstract base class
    with methods like __init__(self, task) and __call__(self, node).
    """
    def __init__(self, task):
        """
        Initializes the heuristic with the planning task.
        The task object typically contains initial state, goals, operators, etc.
        """
        pass # Placeholder

    def __call__(self, node):
        """
        Computes the heuristic value for a given state node.
        Args:
            node: A state node object (assumed to have a 'state' attribute
                  which is a frozenset of PDDL fact strings).
        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        raise NotImplementedError("Heuristic subclass must implement __call__")


# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    # Example: "(arm-empty)" -> ["arm-empty"]
    return fact[1:-1].split()

# Helper function to parse state facts
def parse_state_facts(state):
    """
    Parses the state facts to extract block positions, stack structure, and arm state.

    Args:
        state: A frozenset of PDDL fact strings representing the current state.

    Returns:
        A tuple containing:
        - current_base_map: Dict mapping block -> block_below or 'table'.
        - current_top_map: Dict mapping block_below -> block_on_top (direct relationship).
        - is_holding: The block being held, or None if arm is empty.
        - arm_empty: Boolean indicating if the arm is empty.
        - state_blocks: Set of all block names mentioned in the state facts.
    """
    current_base = {}
    current_top = {}
    is_holding = None
    arm_empty = False
    state_blocks = set() # Blocks mentioned in state facts

    for fact in state:
        parts = get_parts(fact)
        predicate = parts[0]
        if predicate == 'on':
            b_on, b_under = parts[1], parts[2]
            current_base[b_on] = b_under
            current_top[b_under] = b_on
            state_blocks.add(b_on)
            state_blocks.add(b_under)
        elif predicate == 'on-table':
            b = parts[1]
            current_base[b] = 'table'
            state_blocks.add(b)
        elif predicate == 'holding':
            is_holding = parts[1]
            state_blocks.add(is_holding)
        elif predicate == 'arm-empty':
            arm_empty = True
        elif predicate == 'clear':
             state_blocks.add(parts[1]) # Add block to set even if only clear

    return current_base, current_top, is_holding, arm_empty, state_blocks

# Helper function to parse goal facts
def parse_goal_facts(goals):
    """
    Parses the goal facts to extract the desired base for each block.

    Args:
        goals: A frozenset of PDDL fact strings representing the goal state.

    Returns:
        A tuple containing:
        - goal_base_map: Dict mapping block -> desired_block_below or 'table'.
        - goal_blocks: Set of all block names mentioned in the goal facts.
    """
    goal_base_map = {}
    goal_blocks = set()
    for goal in goals:
        parts = get_parts(goal)
        predicate = parts[0]
        if predicate == 'on':
            block = parts[1]
            base = parts[2]
            goal_base_map[block] = base
            goal_blocks.add(block)
            goal_blocks.add(base)
        elif predicate == 'on-table':
            block = parts[1]
            goal_base_map[block] = 'table'
            goal_blocks.add(block)
        # Ignore (clear X) and (arm-empty) goals for this heuristic
    return goal_base_map, goal_blocks

# Helper function to count blocks on top of a given block
def count_blocks_on_top(block, current_top_map):
    """
    Counts the number of blocks currently stacked directly or indirectly on top of 'block'.

    Args:
        block: The name of the block to count on top of.
        current_top_map: Dict mapping block_below -> block_on_top.

    Returns:
        The total count of blocks on top.
    """
    count = 0
    current = block
    # Traverse upwards from the block using the current_top_map
    while current in current_top_map:
        count += 1
        current = current_top_map[current]
    return count

# Helper function to check if a block is in its correct stack segment recursively
def is_correctly_placed_recursive(block, state, goal_base_map, memo):
    """
    Checks if a block is in its correct position relative to its goal base,
    and if that base is also correctly placed, recursively down to the table.

    Args:
        block: The name of the block to check.
        state: The current state (frozenset of facts).
        goal_base_map: Dict mapping block -> desired_base.
        memo: Dictionary for memoization {block: boolean_result}.

    Returns:
        True if the block is in its correct stack segment, False otherwise.
    """
    # Memoization check
    if block in memo:
        return memo[block]

    # Find current base for the block
    current_base = None
    for fact in state:
        parts = get_parts(fact)
        if parts[0] == 'on' and parts[1] == block:
            current_base = parts[2]
            break
        elif parts[0] == 'on-table' and parts[1] == block:
            current_base = 'table'
            break

    # If block is held or not on a base/table, it's not correctly placed on a base/table
    if current_base is None:
         memo[block] = False
         return False

    # Find goal base for the block
    goal_base = goal_base_map.get(block)

    # If block has no goal position specified, it's not part of the goal structure we care about
    # Assume blocks without goal positions don't contribute to heuristic cost (they are "correctly placed" in that they don't need moving to a specific spot)
    if goal_base is None:
         memo[block] = True
         return True

    # Check if current base matches goal base
    if current_base != goal_base:
        memo[block] = False
        return False

    # Current base matches goal base. Check recursively if the base is correct.
    if goal_base == 'table':
        # Block is on table and goal is on table. Correctly placed.
        memo[block] = True
        return True
    else:
        # Block is on goal_base (which is a block Y). Check if Y is correctly placed.
        # Y must also have a goal position defined (unless Y is 'table', handled above).
        if goal_base not in goal_base_map:
             # This indicates an issue with the goal definition itself.
             # A block Y is a goal base for X, but Y itself has no goal base?
             # Treat as not correctly placed, as the stack below is ill-defined.
             memo[block] = False
             return False

        is_base_correct = is_correctly_placed_recursive(goal_base, state, goal_base_map, memo)
        memo[block] = is_base_correct
        return is_base_correct


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to place blocks
    into their correct positions within the goal stacks, prioritizing
    clearing blocks that are obstructing misplaced blocks. It counts blocks
    that are not part of a correctly built stack segment from the bottom up
    and adds the cost to move them and clear blocks on top.

    # Heuristic Initialization
    - Parses the goal predicates to determine the desired base for each block.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine block positions, stack structure,
       and arm state.
    2. Identify all blocks that have a specific goal position (on another block
       or on the table) based on the goal state.
    3. For each block with a goal position, recursively check if it is currently
       in its correct stack segment (i.e., on its goal base, and that base is
       also correctly placed, down to the table). Memoize results to avoid redundant computation.
    4. Initialize the heuristic value to 0.
    5. Iterate through all blocks that have a goal position:
       - If a block is NOT in its correct stack segment (meaning it's misplaced or
         the stack below it is incorrect):
         - If the block is currently held by the arm:
           - Add 1 (estimated cost to put down the held block).
           - If its goal base is not the table, add 2 (estimated cost to pick up
             and stack it later from the table).
         - If the block is on the table or another block (i.e., not held):
           - Add 2 (estimated cost to pick up/unstack and put down/stack the block).
           - Count the number of blocks currently stacked directly or indirectly on top of this block.
           - Add 2 times this count (estimated cost to unstack and put down each block on top).
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base positions for blocks.
        """
        # Parse goal facts once during initialization
        self.goal_base_map, self.goal_blocks = parse_goal_facts(task.goals)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Parse the current state facts
        current_base_map, current_top_map, is_holding, arm_empty, state_blocks = parse_state_facts(state)

        # Consider all blocks that have a goal position specified.
        # Blocks not in the goal structure don't contribute to the heuristic cost.
        all_blocks_to_consider = self.goal_blocks

        # Initialize heuristic value
        h = 0

        # Memoization dictionary for the recursive correctness check.
        # This prevents redundant computation for blocks that are bases of multiple stacks.
        memo = {}

        # Iterate through blocks that need to be in a specific goal position
        for block in all_blocks_to_consider:
            # Check if the block is in its correct stack segment recursively.
            # The recursive helper handles finding the current base internally.
            # A held block will result in is_correctly_placed_recursive returning False
            # because its current_base is None.
            if not is_correctly_placed_recursive(block, state, self.goal_base_map, memo):
                # This block is not in its correct stack segment

                # If the block is currently held by the arm
                if block == is_holding:
                    # It costs 1 action (putdown or stack) to free the arm.
                    # If the block is also misplaced relative to its goal base,
                    # it will need further actions after being put down.
                    h += 1 # Cost to put down the held block

                    # If its goal base is not the table, it will need to be picked up
                    # and stacked later. Add estimated cost for this.
                    goal_base = self.goal_base_map.get(block)
                    # Check if goal_base exists (it should for blocks in all_blocks_to_consider)
                    # and if it's not the table.
                    if goal_base is not None and goal_base != 'table':
                         # Estimated cost: pickup (1) + stack (1) = 2 actions later
                         h += 2

                # If the block is on the table or another block (i.e., not held)
                else:
                    # This block needs to be moved. Estimated cost is 2 actions
                    # (pickup/unstack + putdown/stack).
                    h += 2

                    # Additionally, any blocks on top of this block must be moved first.
                    # Estimated cost to clear each block on top is 2 actions (unstack + putdown).
                    num_on_top = count_blocks_on_top(block, current_top_map)
                    h += 2 * num_on_top # Estimated cost to clear blocks on top

        return h

