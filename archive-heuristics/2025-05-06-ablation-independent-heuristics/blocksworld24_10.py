import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task and Operator are in task.py

# Configure logging for potential warnings
logging.basicConfig(level=logging.INFO)

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing two components:
        1. For each block that is not in its correct goal position relative to its support
           (i.e., on the wrong block, on the table when it should be on a block,
           on a block when it should be on the table, or being held), it adds 1.
        2. For each block identified in component 1, it adds the number of blocks
           currently stacked directly on top of it. This accounts for the immediate
           cost of clearing the misplaced block before it can be moved.

    Assumptions:
        - The input state is a frozenset of PDDL fact strings.
        - The task object contains a list of all objects in task.objects.
        - The goal is defined primarily by (on ?x ?y) and (on-table ?x) predicates.
        - Blocks not mentioned in goal (on ?x ?y) or (on-table ?x) predicates
          are implicitly desired to be on the table.
        - States are valid blocksworld states (every block is either on something,
          on the table, or holding).

    Heuristic Initialization:
        The constructor pre-processes the goal state to determine the desired
        support for each block. It creates a dictionary `goal_support` where
        `goal_support[b]` is the block `b` should be on, or the string 'table'
        if `b` should be on the table. Blocks not explicitly mentioned in goal
        (on ?x ?y) or (on-table ?x) predicates are added to `goal_support`
        with a value of 'table'.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to determine the current support for each block
           and which block is currently on top of which. This involves iterating
           through the state facts:
           - Initialize `current_support` and `current_on_top` dictionaries.
           - For each fact `(on b x)`: `current_support[b] = x`, `current_on_top[x] = b`.
           - For each fact `(on-table b)`: `current_support[b] = 'table'`.
           - For fact `(holding b)`: `current_support[b] = 'holding'`.
        2. Initialize the heuristic value `h` to 0.
        3. Iterate through all blocks defined in `task.objects`.
        4. For each block `b`:
           a. Determine its goal support (`goal_pos`) using the pre-calculated `goal_support` map.
              If `b` is not in the map, its goal support is 'table'.
           b. Determine its current support (`current_pos`) using the parsed state information.
              If `b` is not found in `current_support` after parsing (which shouldn't happen in valid states),
              it indicates an invalid state, and the heuristic returns infinity. Otherwise,
              `current_pos` will be 'table', 'holding', or a block name.
           c. If `current_pos` is different from `goal_pos`:
              i. Increment `h` by 1 (cost for moving the block `b`).
              ii. Find the block currently on top of `b` using the `current_on_top` map.
              iii. While there is a block `temp_on_top` directly on top of the current block `temp_block` (starting with `temp_block = b`):
                   Increment `h` by 1 (cost for moving the block `temp_on_top`).
                   Move up the stack: `temp_block = temp_on_top`.
        5. Return the final heuristic value `h`.
        6. The heuristic is 0 if and only if all blocks are in their correct goal position
           relative to their support and no block is being held. This corresponds to
           the goal state where all required `(on b x)` and `(on-table b)` predicates
           are true, and `(arm-empty)` is true (no block is holding).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_support = self._parse_goal(task.goals, task.objects)

    def _parse_goal(self, goals, objects):
        """
        Parses the goal facts to determine the desired support for each block.
        """
        goal_support = {}
        # Explicit goal positions
        for goal in goals:
            # Goal facts are strings like '(on b1 b2)' or '(on-table b1)'
            parts = goal.strip('()').split()
            if not parts: # Skip empty strings from malformed facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block = parts[1]
                support = parts[2]
                goal_support[block] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                goal_support[block] = 'table'
            # Ignore other goal predicates like (clear ?) for this heuristic's support logic

        # Implicit goal position for blocks not mentioned in 'on' or 'on-table'
        # is on the table.
        for obj in objects:
            if obj not in goal_support:
                goal_support[obj] = 'table'

        return goal_support

    def __call__(self, node):
        """
        Computes the blocksworld heuristic for the given state.
        """
        state = node.state

        # 1. Parse the current state structure
        current_support = {}
        current_on_top = {} # Map: block_below -> block_on_top

        # Initialize current_support for all objects. This helps detect invalid states
        # where a block's position is not defined.
        for obj in self.task.objects:
             current_support[obj] = None

        for fact in state:
            parts = fact.strip('()').split()
            if not parts: # Skip empty strings from malformed facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block = parts[1]
                support = parts[2]
                current_support[block] = support
                current_on_top[support] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'holding'
            # Ignore other state predicates like (clear ?) or (arm-empty) for this heuristic's support logic

        # 2. Initialize heuristic value
        h = 0

        # 3. Iterate through all blocks
        for block in self.task.objects:
            # 4a. Determine goal support
            # Default to 'table' if block is not explicitly in goal_support (handles blocks not in goal)
            goal_pos = self.goal_support.get(block, 'table')

            # 4b. Determine current support
            current_pos = current_support.get(block)

            # Check for invalid state: if a block's position is not defined in the state facts
            if current_pos is None:
                 # This block's position is not defined in the state. Invalid state.
                 # logging.warning(f"Block {block} position not found in state: {state}") # Optional: log warning
                 return float('inf') # Indicate unsolvable/invalid state

            # 4c. If current position is different from goal position
            if current_pos != goal_pos:
                h += 1 # Cost for the block itself being misplaced

                # Add cost for blocks on top of this misplaced block
                temp_block = block
                # Find the block directly on top of temp_block
                block_on_top = current_on_top.get(temp_block)
                while block_on_top is not None:
                    h += 1 # Cost for the block on top
                    temp_block = block_on_top # Move up the stack
                    block_on_top = current_on_top.get(temp_block) # Find next block on top

        # 6. Return heuristic value
        # The heuristic is 0 if and only if all blocks are in their correct goal position
        # relative to their support and no block is being held. This corresponds to
        # the goal state where all required (on b x) and (on-table b) predicates
        # are true, and (arm-empty) is true (no block is holding).
        # The heuristic is finite for valid states.
        # The heuristic is efficiently computable (linear in number of blocks and stack heights).

        return h
