# Required imports based on the provided code structure
from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming these are available in the environment

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are not in their final position within the goal stacks,
    adding 2 for each such block (estimating the cost of moving it).
    It also counts the number of blocks that are currently on top of other blocks that are not clear,
    adding 1 for each block that needs to be moved to clear something below it.
    Finally, it adds 1 if the arm is not empty.

    Assumptions:
    - Standard Blocksworld domain with predicates (on ?x ?y), (on-table ?x),
      (clear ?x), (holding ?x), (arm-empty).
    - Goal is a conjunction of these predicates.
    - State is represented as a frozenset of predicate strings.
    - Blocks are identified by strings (e.g., 'b1', 'b2').
    - The state representation is consistent (e.g., a block is not on two things,
      not on itself, stacks are trees/forests).

    Heuristic Initialization:
    In the constructor, the goal predicates from the task are parsed to build
    data structures representing the goal configuration:
    - self.goal_on_map: A dictionary mapping a block X to block Y if (on X Y) is a goal.
    - self.goal_on_table_set: A set of blocks X if (on-table X) is a goal.
    - self.goal_clear_set: A set of blocks X if (clear X) is a goal.
    - self.goal_blocks: A set of all blocks mentioned in goal (on X Y) or (on-table X) predicates.
    Static facts are ignored as they are not relevant to state changes in Blocksworld.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state predicates to build data structures representing
       the current configuration:
       - state_on_map: A dictionary mapping a block X to block Y if (on X Y) is true.
       - state_on_table_set: A set of blocks X if (on-table X) is true.
       - state_clear_set: A set of blocks X if (clear X) is true.
       - state_holding: The block being held, or None if arm is empty.
       - all_blocks_in_state: A set of all blocks present in the state.
    2. Initialize a memoization dictionary `_final_pos_memo` to store results
       of the recursive `_is_in_final_position` checks for the current state.
    3. Initialize the heuristic value `h` to 0.
    4. Identify blocks that are part of the goal structure (`self.goal_blocks`)
       but are not in their final position. A block X is in its final position
       if:
       - (on-table X) is a goal AND (on-table X) is true in the state.
       - OR (on X Y) is a goal AND (on X Y) is true in the state AND Y is
         in its final position.
       This check is performed by the recursive helper function `_is_in_final_position`.
    5. For each such "misplaced" goal block, add 2 to `h`. This estimates the
       cost of moving the block itself from its current incorrect location
       to its correct final location (e.g., involves a pickup/unstack and a stack/putdown action).
    6. Add cost for clearing blocks that are currently not clear in the state.
       For each block X in the state such that (clear X) is false:
       Find the stack of blocks currently on top of X using `_get_block_on_top`.
       Add the height of this stack to `h`. Each block on top needs to be moved
       (unstack + putdown/stack elsewhere), contributing to the cost of clearing X.
    7. If the arm is not empty (`state_holding` is not None), add 1 to `h`.
       This estimates the cost of freeing the arm (putdown or stack).
    8. Return the total `h` value.
    The heuristic returns 0 if and only if the state is a goal state (all goal
    predicates are true, which implies all goal blocks are in final position,
    all required blocks are clear, and arm is empty if (arm-empty) is a goal).
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals # Keep goals for easy access
        self.goal_on_map = {}
        self.goal_on_table_set = set()
        self.goal_clear_set = set()
        self.goal_blocks = set()

        # Parse goal predicates
        for goal in self.goals:
            parsed = self._parse_predicate(goal)
            if parsed[0] == 'on':
                if len(parsed) == 3: # Ensure correct number of arguments
                    self.goal_on_map[parsed[1]] = parsed[2]
                    self.goal_blocks.add(parsed[1])
                    self.goal_blocks.add(parsed[2])
            elif parsed[0] == 'on-table':
                 if len(parsed) == 2: # Ensure correct number of arguments
                    self.goal_on_table_set.add(parsed[1])
                    self.goal_blocks.add(parsed[1])
            elif parsed[0] == 'clear':
                 if len(parsed) == 2: # Ensure correct number of arguments
                    self.goal_clear_set.add(parsed[1])
                    # Blocks in clear goals might not be in on/on-table goals,
                    # but they don't define the stack structure for _is_in_final_position.
                    # We don't add them to goal_blocks for the final position check.
            # Ignore arm-empty and holding goals for goal_blocks set

        # Memoization dictionary for _is_in_final_position
        # This needs to be re-initialized for each state evaluation
        self._final_pos_memo = {} # Will be cleared in __call__


    def _parse_predicate(self, predicate_str):
        """Parses a predicate string into a tuple (name, arg1, arg2, ...)."""
        # Remove surrounding brackets and split by space
        parts = predicate_str.strip('()').split()
        return tuple(parts)

    def _get_current_position(self, block, state_on_map, state_on_table_set, state_holding):
        """Returns the current position of a block: ('on', below_block), ('on-table',), ('holding',), or None."""
        if state_holding == block:
            return ('holding',)
        if block in state_on_table_set:
            return ('on-table',)
        if block in state_on_map:
            return ('on', state_on_map[block])
        return None # Should not happen for a block known to be in the state

    def _get_block_on_top(self, block, state_on_map):
        """Returns the block directly on top of the given block in the state, or None."""
        # state_on_map maps block -> block_below. We need block_below -> block_on_top.
        # We need to iterate through state_on_map values to find the key.
        for b_on_top, b_below in state_on_map.items():
            if b_below == block:
                return b_on_top
        return None

    def _is_in_final_position(self, block, state_on_map, state_on_table_set, goal_on_map, goal_on_table_set, memo):
        """
        Recursively checks if a block is in its final position within the goal structure.
        Uses memoization to avoid redundant calculations.
        """
        if block in memo:
            return memo[block]

        # A block not in goal_blocks cannot be in a final position *within the goal structure*.
        if block not in self.goal_blocks:
             result = False
             memo[block] = result
             return result

        # Base case 1: Block should be on the table in the goal
        if block in goal_on_table_set:
            result = block in state_on_table_set
            memo[block] = result
            return result

        # Base case 2: Block should be on another block in the goal
        if block in goal_on_map:
            below_goal = goal_on_map[block]
            # Check if block is currently on the correct block AND the block below is in its final position
            result = (block in state_on_map and state_on_map[block] == below_goal and
                      self._is_in_final_position(below_goal, state_on_map, state_on_table_set, goal_on_map, goal_on_table_set, memo))
            memo[block] = result
            return result

        # Should not reach here if block is in goal_blocks but not in goal_on_table_set or goal_on_map.
        # This would indicate an inconsistent goal definition (a block in goal_blocks must be in one of these).
        # Return False defensively.
        result = False
        memo[block] = result
        return result


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # 1. Parse current state
        state_on_map = {} # Maps block -> block_below
        state_on_table_set = set()
        state_clear_set = set()
        state_holding = None
        all_blocks_in_state = set()

        for fact_str in state:
            parsed = self._parse_predicate(fact_str)
            if parsed[0] == 'on':
                if len(parsed) == 3:
                    state_on_map[parsed[1]] = parsed[2]
                    all_blocks_in_state.add(parsed[1])
                    all_blocks_in_state.add(parsed[2])
            elif parsed[0] == 'on-table':
                if len(parsed) == 2:
                    state_on_table_set.add(parsed[1])
                    all_blocks_in_state.add(parsed[1])
            elif parsed[0] == 'clear':
                if len(parsed) == 2:
                    state_clear_set.add(parsed[1])
                    all_blocks_in_state.add(parsed[1]) # Add block to set of all blocks
            elif parsed[0] == 'holding':
                if len(parsed) == 2:
                    state_holding = parsed[1]
                    all_blocks_in_state.add(parsed[1])
            elif parsed[0] == 'arm-empty':
                 pass # No data structure needed for this fact

        # 2. Initialize memoization for this state
        self._final_pos_memo = {}

        # 3. Initialize heuristic value
        h = 0

        # 4. & 5. Count misplaced goal blocks and add cost (2 per block)
        misplaced_goal_blocks = set()
        for block in self.goal_blocks:
             # Check if the goal block is in its final position within the goal structure
             # _is_in_final_position handles cases where the block might be held or not on table/another block
            if not self._is_in_final_position(block, state_on_map, state_on_table_set, self.goal_on_map, self.goal_on_table_set, self._final_pos_memo):
                 misplaced_goal_blocks.add(block)

        h += 2 * len(misplaced_goal_blocks)

        # 6. Add cost for clearing blocks that are not clear
        # Iterate through all blocks present in the state
        for block in all_blocks_in_state:
            if block not in state_clear_set: # If block is not clear
                # Find block on top of X in state
                # Note: A block that is 'holding' is not on top of anything,
                # and is also not in state_clear_set. We handle 'holding' separately.
                if state_holding != block: # Don't count clearing cost for the block being held
                    current = self._get_block_on_top(block, state_on_map)
                    if current is not None: # There is something on block
                        # Count the stack height above block
                        stack_height = 0
                        temp = current
                        while temp is not None:
                            stack_height += 1
                            temp = self._get_block_on_top(temp, state_on_map)
                        h += stack_height # Each block on top needs to be moved

        # 7. Add cost if arm is not empty
        if state_holding is not None:
            h += 1

        return h
