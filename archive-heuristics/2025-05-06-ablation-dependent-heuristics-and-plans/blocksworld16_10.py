from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting:
    1. Blocks that are not in their correct position within the goal stacks,
       considering stacks must be built from the bottom up.
    2. Blocks that are obstructing blocks required to be clear in the goal.
    3. A penalty if the arm is holding a block.

    # Assumptions
    - The goal state defines a set of desired stacks (using `on` and `on-table` predicates)
      and a set of blocks that must be clear (`clear` predicates).
    - Blocks not mentioned in the goal predicates can be anywhere, provided they don't
      violate the goal conditions (e.g., blocking a block that needs to be clear).
    - The arm must be empty in the goal state (implicitly handled if no block is held).

    # Heuristic Initialization
    - Parses the goal conditions to identify:
        - `goal_config`: A mapping from a block to the block it should be directly on top of,
                         or 'table' if it should be on the table.
        - `goal_clear`: A set of blocks that must be clear in the goal state.
    - Identifies the set of all blocks involved in the goal configuration (`goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - `current_on`: Mapping from a block to the block it is currently on.
       - `current_on_table`: Set of blocks currently on the table.
       - `current_holding`: Set containing the block currently held by the arm (if any).
       - `current_is_base`: Inverse mapping of `current_on` (from base block to the block on top).

    2. Calculate Cost Component 1 (Stack Configuration):
       - Identify blocks that are "correctly stacked" relative to the goal configuration.
       - A block `B` is correctly stacked if:
         - Its goal is `(on-table B)` AND `B` is currently on the table. OR
         - Its goal is `(on B Y)` AND `B` is currently on `Y` AND `Y` is correctly stacked.
       - This is computed iteratively: start with blocks correctly on the table, then blocks correctly on those, and so on, until no new blocks can be added.
       - The cost is the total number of blocks involved in the goal stacks (`goal_blocks`) minus the number of blocks found to be `correctly_stacked`. This counts blocks that are not in their correct relative position within the goal stacks, starting from the bottom.

    3. Calculate Cost Component 2 (Blocking Clear Goals):
       - For each block `C` that must be clear in the goal (`C` in `goal_clear`):
         - Check if `C` is currently blocked (i.e., some block `Y` is `(on Y C)`).
         - If `C` is blocked, count the number of blocks in the stack currently sitting on top of `C` (including the first block directly on `C`). Each of these blocks needs to be moved. Add this count to the heuristic.

    4. Calculate Cost Component 3 (Arm Status):
       - If the arm is currently holding a block, add a small penalty (e.g., 1) as this block needs to be placed somewhere.

    5. The total heuristic value is the sum of Cost 1, Cost 2, and Cost 3.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Parse goal conditions to build goal_config and goal_clear
        self.goal_config = {} # block -> base_block or 'table'
        self.goal_clear = set() # set of blocks that must be clear
        self.goal_blocks = set() # set of blocks that are keys in goal_config

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base_block = parts[1], parts[2]
                self.goal_config[block] = base_block
                self.goal_blocks.add(block)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_config[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == 'clear' and len(parts) == 2:
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore other goal predicates like arm-empty, holding, etc.

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse the current state
        current_on = {} # block -> base_block
        current_on_table = set() # set of blocks on table
        current_holding = set() # set containing the held block (at most one)
        current_is_base = {} # base_block -> block_on_top (inverse of current_on)
        arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base_block = parts[1], parts[2]
                current_on[block] = base_block
                current_is_base[base_block] = block # Store inverse mapping
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on_table.add(block)
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_holding.add(block)
            elif predicate == 'arm-empty':
                arm_empty = True # Not directly used in heuristic calculation, but good to parse

        heuristic_value = 0

        # 2. Calculate Cost Component 1 (Stack Configuration)
        correctly_stacked = set()
        changed = True
        while changed:
            changed = False
            # Iterate over blocks that are supposed to be part of a goal stack
            for block in self.goal_blocks:
                if block not in correctly_stacked:
                    goal_base = self.goal_config.get(block)

                    if goal_base == 'table':
                        # Goal: block should be on the table
                        if block in current_on_table:
                            correctly_stacked.add(block)
                            changed = True
                    elif goal_base is not None:
                        # Goal: block should be on goal_base
                        # Check if block is currently on goal_base AND goal_base is correctly stacked
                        if current_on.get(block) == goal_base and goal_base in correctly_stacked:
                            correctly_stacked.add(block)
                            changed = True
                    # Note: Blocks that are goal bases but not themselves on anything in the goal
                    # (i.e., not keys in goal_config) are not added to correctly_stacked here.
                    # This is correct as they don't contribute to building a stack *upwards*
                    # towards a goal_block.

        # Cost is the number of goal blocks not yet correctly stacked
        heuristic_value += len(self.goal_blocks) - len(correctly_stacked)

        # 3. Calculate Cost Component 2 (Blocking Clear Goals)
        blocking_cost = 0
        for clear_block in self.goal_clear:
            # Find the block currently directly on top of clear_block
            block_on_C = current_is_base.get(clear_block)
            if block_on_C is not None:
                # clear_block is not clear. Count the height of the stack above it.
                stack_height = 0
                current = block_on_C
                while current is not None:
                    stack_height += 1
                    # Find the block on top of 'current'
                    current = current_is_base.get(current)
                blocking_cost += stack_height # Add 1 for each block in the blocking stack

        heuristic_value += blocking_cost

        # 4. Calculate Cost Component 3 (Arm Status)
        # If the arm is holding a block, it needs at least one action (putdown or stack)
        if any(current_holding):
             heuristic_value += 1 # Add a small penalty

        # Ensure heuristic is 0 only for goal states.
        # If heuristic_value is 0 based on the calculation above, it means:
        # - All goal_blocks are correctly stacked.
        # - All goal_clear blocks are clear.
        # - The arm is empty.
        # This implies all goal predicates (on, on-table, clear, arm-empty if it was a goal) are met.
        # The Task.goal_reached method checks if task.goals <= state.
        # Our heuristic being 0 should align with this.
        # Let's double check if arm-empty is *always* implicitly a goal if clear goals are met.
        # In blocksworld, if all blocks are in their final positions and required clear conditions met,
        # the arm must be empty to have achieved the final placements. So arm-empty is usually
        # a necessary condition for the goal state, even if not explicitly listed in goals.
        # Our heuristic implicitly handles this by adding a cost if the arm is not empty.
        # If the state is the goal state, our calculation should yield 0.
        # If state is goal:
        # - Cost 1: All goal_blocks are correctly stacked -> 0.
        # - Cost 2: All goal_clear blocks are clear -> 0.
        # - Cost 3: Arm is empty -> 0.
        # So, h=0 for goal states.
        # If h=0, does it mean it's a goal state?
        # h=0 implies Cost1=0, Cost2=0, Cost3=0.
        # Cost1=0 means all goal_blocks are correctly stacked relative to goal_config. This implies all (on X Y) and (on-table Z) goal facts are true.
        # Cost2=0 means all goal_clear blocks are clear.
        # Cost3=0 means arm is empty.
        # If the goal is *exactly* the set of (on X Y), (on-table Z), (clear C), and (arm-empty) facts, then h=0 iff state is goal.
        # If the goal contains other facts (unlikely in standard blocksworld), or if arm-empty is not in the goal but is required,
        # the heuristic might be 0 for a state that is not technically the goal state according to task.goals,
        # but is very close or effectively solved from a block arrangement perspective.
        # However, the requirement is that h=0 *only* for goal states.
        # The simplest way to guarantee this is to explicitly check if the state is the goal state.
        # But heuristic computation should be fast. Checking goal_reached is O(|goals|).
        # Our calculation is roughly O(|goal_blocks| + |state| + |goal_clear| * stack_height).
        # Let's rely on the calculation being correct and assume the goal definition is standard.
        # The current calculation should be 0 iff all goal on/on-table facts are true AND all goal clear facts are true AND arm is empty.
        # If arm-empty is not in task.goals but is required, our h=0 state might not be task.goal_reached.
        # If arm-empty *is* in task.goals, our h=0 state implies arm-empty is true.
        # Let's assume standard blocksworld goals where arm-empty is implicitly required or explicitly listed.
        # The current calculation seems robust for standard blocksworld goals.

        return heuristic_value

