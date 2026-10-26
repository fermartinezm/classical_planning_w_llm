from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting structural differences between the current state and the goal state.
    It counts for each block mentioned in the goal location facts:
    1. If the block is not on its correct goal support (another block or the table).
    2. If the block immediately above it is not the correct block according to the goal
       (or if the block should be clear in the goal but isn't).
    Additionally, it adds 1 if the arm should be empty in the goal but is not.

    # Assumptions
    - The goal state is defined by a set of `on` and `on-table` facts, and optionally `arm-empty`.
    - The heuristic does not consider the specific sequence of actions, only the number of required changes.
    - The heuristic is not admissible but aims to guide greedy best-first search effectively.

    # Heuristic Initialization
    - Extracts the desired support (`goal_under`) and the desired block on top (`goal_above`)
      for each block involved in the goal location facts.
    - Identifies the set of all blocks involved in the goal location facts.
    - Checks if `(arm-empty)` is a goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the current support (`current_under`)
       and the current block on top (`current_above`) for each block.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block `X` that is part of the goal location configuration:
       a. Determine the block/table `goal_pos` that `X` should be on in the goal
          using the precomputed `goal_under` map.
       b. Determine the block/table `current_pos` that `X` is currently on
          using the `current_under` map derived from the state. Handle the case where the block is held.
       c. If `current_pos` is different from `goal_pos`, increment `h` by 1.
       d. Determine the block `goal_block_above` that should be directly on top of `X`
          in the goal (or `None` if `X` should be clear) using the precomputed `goal_above` map.
       e. Determine the block `current_block_above` that is currently directly on top of `X`
          (or `None` if `X` is clear) using the `current_above` map derived from the state.
       f. If `current_block_above` is different from `goal_block_above`, increment `h` by 1.
    4. Check if `(arm-empty)` is a goal fact and if the arm is not empty in the current state.
       If both are true, increment `h` by 1.
    5. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are empty in Blocksworld.
        # static_facts = task.static

        self.goal_under = {}
        self.goal_above = {}
        self.goal_blocks = set()
        self.goal_arm_empty = False

        # Extract goal location facts and build goal_under/goal_above maps
        goal_supports = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, support = parts[1], parts[2]
                self.goal_under[block] = support
                self.goal_above[support] = block
                self.goal_blocks.add(block)
                self.goal_blocks.add(support) # Supports can also be blocks
                goal_supports.add(support)
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_under[block] = 'table'
                self.goal_blocks.add(block)
                goal_supports.add('table')
            elif parts[0] == "arm-empty":
                self.goal_arm_empty = True

        # For supports in the goal that should be clear, add None to goal_above
        for support in goal_supports:
             if support not in self.goal_above:
                 self.goal_above[support] = None

        # Any block in goal_blocks that is not a support and not in goal_above means it should be clear
        # This handles the top block of any goal stack.
        # Iterate over all blocks that appear anywhere in goal location facts
        all_blocks_in_goal_config = set(self.goal_under.keys()) | set(self.goal_above.keys()) - {'table'}
        for block in all_blocks_in_goal_config:
             if block not in self.goal_above:
                 self.goal_above[block] = None

        # The set of blocks we care about are those explicitly mentioned as being 'on' something or 'on-table' in the goal
        self.goal_blocks = set(self.goal_under.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        current_under = {}
        current_above = {}
        arm_is_empty = False
        held_block = None

        # Parse current state to build current_under/current_above maps and arm state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, support = parts[1], parts[2]
                current_under[block] = support
                current_above[support] = block
            elif parts[0] == "on-table":
                block = parts[1]
                current_under[block] = 'table'
            elif parts[0] == "arm-empty":
                arm_is_empty = True
            elif parts[0] == "holding":
                held_block = parts[1]
                # A held block is 'under' the arm
                current_under[held_block] = 'arm'
            # 'clear' facts are implicitly handled by checking if a block is a key in current_above


        total_cost = 0  # Initialize action cost counter.

        # Calculate cost based on block positions and blocks above
        for block in self.goal_blocks:
            goal_pos = self.goal_under.get(block) # Should always exist for blocks in goal_blocks

            # Find current position. Default to None if block isn't on anything, table, or held.
            # This case shouldn't happen in valid states, but None is safer than KeyError.
            current_pos = current_under.get(block)

            # If goal_pos is None, it means the block is a support in the goal but not
            # explicitly placed on something or table. This shouldn't be in goal_blocks
            # based on the current definition of goal_blocks.
            # assert goal_pos is not None, f"Block {block} in goal_blocks has no goal_under entry"


            if current_pos != goal_pos:
                total_cost += 1 # Block is on the wrong support (or held when it shouldn't be, etc.)

            # Check the block above X.
            # We only care about blocks that are supports in the goal or are the top of a goal stack.
            # The goal_above map is built for these.
            goal_block_above = self.goal_above.get(block) # None if should be clear

            # Find the block currently above this block.
            current_block_above = current_above.get(block) # None if is clear

            if current_block_above != goal_block_above:
                 total_cost += 1 # The block above is wrong (wrong block or wrong clear state)


        # Add cost for arm state if necessary
        if self.goal_arm_empty and not arm_is_empty:
            total_cost += 1 # Need one action (putdown or stack) to empty the arm

        return total_cost
