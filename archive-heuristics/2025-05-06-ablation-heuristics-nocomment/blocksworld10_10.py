from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to parse a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    This heuristic estimates the cost to reach the goal state by combining
    several factors related to the block positions and the state of the arm.
    It counts the number of blocks that are not in their final position within
    the goal stacks (relative to the table), penalizes blocks that are currently
    on top of any other block (as they need to be moved), and adds a penalty
    if the arm is holding a block. The heuristic is designed for greedy best-first
    search and is not admissible.

    Assumptions:
    - The PDDL domain is Blocksworld with standard predicates (on, on-table,
      clear, holding, arm-empty) and actions (pickup, putdown, stack, unstack).
    - Goal states are conjunctions of on, on-table, and clear predicates.
    - The arm should typically be empty in the goal state unless a specific block
      is required to be held (which is rare in standard Blocksworld goals).
      This heuristic implicitly assumes arm-empty is desired if no holding fact
      is present in the goal.
    - The heuristic is non-admissible and aims to minimize expanded nodes.

    Heuristic Initialization:
    The constructor processes the goal state from the task definition. It builds
    data structures to represent the desired final configuration of blocks:
    - goal_on_map: A dictionary where keys are blocks and values are the blocks
                     they should be directly on top of, based on goal (on ?x ?y) facts.
    - goal_on_table_set: A set of blocks that should be on the table according
                         to goal (on-table ?x) facts.
    - goal_clear_set: A set of blocks that should be clear according to goal
                      (clear ?x) facts.
    - goal_holding: The block that should be held in the goal, or None if no
                    (holding ?x) fact is in the goal (implying arm-empty is desired).

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state provided to the __call__ method. Build data
       structures representing the current configuration:
       - current_on_map: Maps a block to the block it is currently on.
       - current_on_table_set: Set of blocks currently on the table.
       - current_clear_set: Set of blocks currently clear.
       - current_holding: The block currently held, or None if arm-empty.
       - current_top_map: Maps a block to the block currently on top of it
                          (assuming only one block can be directly on top).

    2. Identify all unique blocks mentioned in either the current state or the goals.

    3. Calculate the first component of the heuristic: the number of blocks that
       are not in their final position relative to the table within the goal stacks.
       This is determined using a recursive helper function `is_correctly_placed_in_stack`.
       This function checks if a block is on its correct goal base (another block
       or the table) AND if the block below it is also correctly placed, recursively
       down to the table. Blocks not part of goal 'on' or 'on-table' facts are
       considered correctly placed for this specific stack-building component.
       The count of blocks for which `is_correctly_placed_in_stack` returns False
       (among those involved in goal stacks) is calculated. This count is multiplied by 2.
       This component estimates the cost of getting blocks into their final stack positions.

    4. Calculate the second component: the number of blocks that are currently on
       top of any other block. These blocks need to be moved (unstacked) to clear
       the block below. This count is added to the heuristic. This component
       estimates the cost of clearing operations.

    5. Calculate the third component: add 1 to the heuristic if the arm is currently
       holding a block. This block needs to be placed (stacked or putdown) to free
       the arm. This component estimates the cost of the final placement action.

    6. The total heuristic value is the sum of the three components. A heuristic
       value of 0 is returned if and only if the state is a goal state (all goal
       conditions are met, including implicit arm-empty if applicable).
    """
    def __init__(self, task):
        self.goals = task.goals
        # Parse goals
        self.goal_on_map = {}
        self.goal_on_table_set = set()
        self.goal_clear_set = set()
        self.goal_holding = None # Assuming only one block can be held in goal, or arm-empty

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                # Ensure parts has enough elements before accessing index 2
                if len(parts) > 2:
                    self.goal_on_map[parts[1]] = parts[2]
            elif predicate == 'on-table':
                 if len(parts) > 1:
                    self.goal_on_table_set.add(parts[1])
            elif predicate == 'clear':
                 if len(parts) > 1:
                    self.goal_clear_set.add(parts[1])
            elif predicate == 'holding':
                 if len(parts) > 1:
                    self.goal_holding = parts[1]
            # Note: arm-empty is implicitly a goal if no holding fact is in the goal

    def is_correctly_placed_in_stack(self, block, state_on_map, state_on_table_set):
        """
        Recursively checks if a block is in its correct position relative to the
        table within the goal stack structure.
        """
        # If block is not part of any goal stack, consider it correctly placed for stack purposes
        if block not in self.goal_on_map and block not in self.goal_on_table_set:
            return True

        # Check if block is on the table in the goal
        if block in self.goal_on_table_set:
            # Check if block is on the table in the state
            return block in state_on_table_set

        # Check if block is on another block in the goal
        if block in self.goal_on_map:
            goal_below = self.goal_on_map[block]
            # Check if block is on the correct block in the state
            if state_on_map.get(block) != goal_below:
                return False
            # Recursively check the block below
            return self.is_correctly_placed_in_stack(goal_below, state_on_map, state_on_table_set)

        # Should not reach here if block is in goal_on_map or goal_on_table_set
        # This case might occur if a block is mentioned in goal_clear_set but not on/on-table
        # We handle this by the initial check.
        return True # Default to True if not explicitly in goal stacks

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Parse state
        current_on_map = {} # block_on_top -> block_below
        current_on_table_set = set()
        current_clear_set = set()
        current_holding = None
        current_top_map = {} # block_below -> block_on_top (assuming max 1 block on top)

        # Collect all blocks mentioned in the state for iteration
        all_blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) > 2:
                    block_on_top, block_below = parts[1], parts[2]
                    current_on_map[block_on_top] = block_below
                    current_top_map[block_below] = block_on_top # Assuming only one block on top
                    all_blocks_in_state.add(block_on_top)
                    all_blocks_in_state.add(block_below)
            elif predicate == 'on-table':
                 if len(parts) > 1:
                    current_on_table_set.add(parts[1])
                    all_blocks_in_state.add(parts[1])
            elif predicate == 'clear':
                 if len(parts) > 1:
                    current_clear_set.add(parts[1])
                    all_blocks_in_state.add(parts[1])
            elif predicate == 'holding':
                 if len(parts) > 1:
                    current_holding = parts[1]
                    all_blocks_in_state.add(parts[1])
            elif predicate == 'arm-empty':
                 pass # current_holding remains None
            # Add any other objects mentioned in facts (e.g., types if present, though not in blocksworld)
            # For blocksworld, objects are just blocks, already covered.

        h = 0

        # Component 1: Blocks not correctly placed in stack relative to table (H15) * 2
        misplaced_in_stack_count = 0
        # Iterate over all blocks that are part of the goal stacks
        blocks_in_goal_stacks = set(self.goal_on_map.keys()) | self.goal_on_table_set

        for block in blocks_in_goal_stacks:
             if not self.is_correctly_placed_in_stack(block, current_on_map, current_on_table_set):
                 misplaced_in_stack_count += 1

        h += 2 * misplaced_in_stack_count

        # Component 2: Number of blocks that are on top of *any* block
        # These blocks need to be unstacked.
        h += len(current_on_map)

        # Component 3: 1 if arm is holding something
        # The held block needs to be placed.
        if current_holding is not None:
            h += 1

        return h

