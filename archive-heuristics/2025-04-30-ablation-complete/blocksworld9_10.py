from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to get all blocks on top of a block (directly or indirectly)
# Requires a map from block_below -> block_above derived from the state's 'on' facts.
def get_blocks_on_top_recursive(block, state_on_map_below_to_above, memo):
    if block in memo:
        return memo[block]

    on_top = set()
    block_above = state_on_map_below_to_above.get(block)
    if block_above:
        on_top.add(block_above)
        on_top.update(get_blocks_on_top_recursive(block_above, state_on_map_below_to_above, memo))

    memo[block] = on_top
    return on_top


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are either misplaced
    within a goal stack or are obstructing a required 'clear' condition.
    It counts the total number of unique blocks that need to be moved out
    of the way to achieve the goal configuration, including those stacked
    on top of them.

    # Assumptions
    - The goal defines a specific configuration of blocks stacked on each other
      or placed on the table.
    - All blocks mentioned in goal 'on' or 'on-table' facts are considered
      part of the goal configuration.
    - The heuristic counts the number of blocks that need to be moved out
      of the way. Each such block and anything on top of it must be moved
      at least once.

    # Heuristic Initialization
    - Parses the goal facts to build a map of required support for each block
      (`goal_support_map`) and a set of blocks that must be clear in the goal
      (`goal_clear_set`). It also creates a map from a block to the block
      that should be directly on top of it in the goal (`goal_block_above_map`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse Goal: In the constructor, create `goal_support_map` (mapping block
       to the block or 'table' it should be on), `goal_block_above_map` (mapping
       block_below to block_above in goal stacks), and `goal_clear_set` (set of
       blocks that must be clear). Identify all blocks mentioned in goal 'on'
       or 'on-table' facts (`all_blocks_in_goal`).

    2. Parse State: In the `__call__` method, parse the current state to quickly
       find the current support for each block (`current_support_map`), build
       a map representing the current stack structure (`state_on_map_below_to_above`),
       and identify currently clear blocks (`current_clear_set`).

    3. Identify Correctly Placed Blocks: Determine which blocks are currently
       part of a stack segment that matches the goal configuration, starting
       from the table up. This is done iteratively:
       - Initialize an empty set `correctly_placed_blocks`.
       - Initialize a queue `q`.
       - For each block `X` in `all_blocks_in_goal`, if its goal support is
         'table' AND its current support is 'table', add `X` to
         `correctly_placed_blocks` and `q`.
       - While `q` is not empty, dequeue a `supported_block`. Find the block
         that should be directly on top of `supported_block` in the goal
         (`block_above_in_goal`) using `goal_block_above_map`. If such a block
         exists AND its current support is `supported_block`, AND it is not
         already in `correctly_placed_blocks`, add it to `correctly_placed_blocks`
         and enqueue it.

    4. Identify Blocks to Move: Initialize an empty set `blocks_to_move`.
       Initialize a memoization dictionary `on_top_memo` for the recursive helper.
       - Condition 1 (Misplaced in Goal Stack): Iterate through all blocks `X`
         in `all_blocks_in_goal`. If `X` is NOT in `correctly_placed_blocks`:
         Add `X` to `blocks_to_move`. Recursively find all blocks currently
         stacked on top of `X` in the state (using `state_on_map_below_to_above`
         and `on_top_memo`) and add them to `blocks_to_move`.
       - Condition 2 (Obstructing Clear Goal): Iterate through all blocks `X`
         in `goal_clear_set`. If `(clear X)` is NOT in `current_clear_set`:
         Find the block `Y` currently directly on top of `X` using
         `state_on_map_below_to_above`. If `Y` exists: Add `Y` to `blocks_to_move`.
         Recursively find all blocks currently stacked on top of `Y` in the
         state (using `state_on_map_below_to_above` and `on_top_memo`) and add
         them to `blocks_to_move`.

    5. Compute Heuristic Value: The heuristic value is the total number of
       unique blocks in the `blocks_to_move` set. This represents the minimum
       number of blocks that must be moved at least once to potentially clear
       the way or place blocks correctly.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Parse goal facts to build goal_support_map, goal_block_above_map, and goal_clear_set
        self.goal_support_map = {} # block -> support_block_or_table
        self.goal_block_above_map = {} # support_block -> block_above
        self.goal_clear_set = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block_above, block_below = parts[1], parts[2]
                    self.goal_support_map[block_above] = block_below
                    self.goal_block_above_map[block_below] = block_above
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_support_map[block] = 'table'
            elif predicate == 'clear':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_clear_set.add(block)

        # Collect all blocks mentioned in goal on/on-table facts
        self.all_blocks_in_goal = set(self.goal_support_map.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 2. Parse current state for quick lookups
        current_support_map = {} # block -> support_block_or_table_or_hand
        state_on_map_below_to_above = {} # block_below -> block_above
        current_clear_set = set()
        # current_holding = None # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block_above, block_below = parts[1], parts[2]
                    current_support_map[block_above] = block_below
                    state_on_map_below_to_above[block_below] = block_above
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    current_support_map[block] = 'table'
            elif predicate == 'clear':
                 if len(parts) == 2:
                    block = parts[1]
                    current_clear_set.add(block)
            elif predicate == 'holding':
                 if len(parts) == 2:
                    block = parts[1]
                    current_support_map[block] = 'hand'
                    # current_holding = block # Store the block being held

        # 3. Identify Correctly Placed Blocks (iterative bottom-up)
        correctly_placed_blocks = set()
        q = deque()

        # Add blocks that should be on the table and are on the table
        for block in self.all_blocks_in_goal:
            goal_support = self.goal_support_map.get(block)
            if goal_support == 'table':
                if current_support_map.get(block) == 'table':
                    correctly_placed_blocks.add(block)
                    q.append(block)

        # Propagate correctness upwards
        while q:
            supported_block = q.popleft()
            # Find blocks that should be on top of supported_block in the goal
            block_above_in_goal = self.goal_block_above_map.get(supported_block)
            if block_above_in_goal:
                # Check if block_above_in_goal is currently on supported_block
                if current_support_map.get(block_above_in_goal) == supported_block:
                    # If it's not already marked, mark it and add to queue
                    if block_above_in_goal not in correctly_placed_blocks:
                         correctly_placed_blocks.add(block_above_in_goal)
                         q.append(block_above_in_goal)

        # 4. Identify Blocks to Move
        blocks_to_move = set()
        on_top_memo = {} # Memoization for get_blocks_on_top_recursive

        # Condition 1: Blocks not in correct stack segment
        for block in self.all_blocks_in_goal:
            if block not in correctly_placed_blocks:
                blocks_to_move.add(block)
                blocks_on_top = get_blocks_on_top_recursive(block, state_on_map_below_to_above, on_top_memo)
                blocks_to_move.update(blocks_on_top)

        # Condition 2: Blocks obstructing clear goals
        for block_to_be_clear in self.goal_clear_set:
            # Check if it's actually clear in the state
            if block_to_be_clear not in current_clear_set:
                # Find the block directly on top of block_to_be_clear
                obstructing_block = state_on_map_below_to_above.get(block_to_be_clear)
                # If there is an obstructing block (should be, if not clear and not holding)
                if obstructing_block:
                     blocks_to_move.add(obstructing_block)
                     blocks_on_top = get_blocks_on_top_recursive(obstructing_block, state_on_map_below_to_above, on_top_memo)
                     blocks_to_move.update(blocks_on_top)

        # 5. Compute Heuristic Value
        # The heuristic is the number of unique blocks identified to be moved.
        return len(blocks_to_move)

