from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting blocks
    that are either not in their correct position relative to the block below them
    (or the table) or have the wrong block placed directly on top of them
    (or should be clear but are not). Each such "unhappy" block contributes 1
    to the heuristic value.

    # Assumptions
    - The goal state defines a specific configuration of blocks in stacks or on the table.
    - All blocks mentioned in the initial or goal state are relevant to the heuristic calculation.
    - The heuristic focuses on the immediate adjacency relationships between blocks
      and the table.

    # Heuristic Initialization
    - Extract all block names involved in the problem from the initial and goal states.
    - Parse the goal facts to build two maps representing the desired configuration:
        - `goal_pos_map`: Maps each block to the block it should be immediately on,
                          or 'table' if it should be on the table.
        - `goal_block_above_map`: Maps each block to the block that should be
                                  immediately on top of it, or None if it should be clear.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Collect all block names from the task's initial state and goal state.
    2. Parse the goal facts (`task.goals`) to create `goal_pos_map` and `goal_block_above_map`.
       - For each `(on X Y)` goal fact, set `goal_pos_map[X] = Y` and `goal_block_above_map[Y] = X`.
       - For each `(on-table X)` goal fact, set `goal_pos_map[X] = 'table'`.
       - After processing all `on` goal facts, for any block `B` that is not a value in `goal_block_above_map` (meaning no block should be on it in the goal), set `goal_block_above_map[B] = None`.
    3. Parse the current state facts (`node.state`) to create `current_pos_map` and `current_block_above_map` using the same logic as step 2, but considering `(holding X)` facts which set `current_pos_map[X] = 'holding'`. After processing `on` facts, for any block `B` not a value in `current_block_above_map`, set `current_block_above_map[B] = None`.
    4. Initialize the heuristic value `h = 0`.
    5. Iterate through each block `B` identified in step 1.
    6. For block `B`, get its desired position below (`goal_below_B = goal_pos_map.get(B)`) and its current position below (`current_below_B = current_pos_map.get(B)`).
    7. Check the first condition for "unhappiness": If `B` has a specified goal position below it (`goal_below_B is not None`) AND its current position below is different (`current_below_B != goal_below_B`), increment `h`. This block is misplaced relative to what's below it.
    8. If the first condition is NOT met (i.e., `B` is in its correct position below, or it doesn't have a specified position below in the goal), check the second condition:
       - Get the block that should be on top of `B` in the goal (`goal_above_B = goal_block_above_map.get(B)`).
       - Get the block that is currently on top of `B` (`current_above_B = current_block_above_map.get(B)`).
       - If the block currently on top is different from the block that should be on top (`current_above_B != goal_above_B`), increment `h`. This covers cases where `B` should be clear but isn't, or has the wrong block on top.
    9. After iterating through all blocks, return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration and block names."""
        self.task = task # Store task for access to goals and initial state

        # 1. Collect all block names
        self.all_blocks = set()
        # Collect from initial state
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] in ['on', 'on-table', 'holding', 'clear']:
                 # Add all objects mentioned in the fact (skip predicate)
                for obj in parts[1:]:
                    self.all_blocks.add(obj)
        # Collect from goal state
        for fact in task.goals:
            parts = get_parts(fact)
            if parts and parts[0] in ['on', 'on-table', 'clear']:
                # Add all objects mentioned in the fact (skip predicate)
                for obj in parts[1:]:
                    self.all_blocks.add(obj)

        # 2. Parse goal facts to build goal configuration maps
        self.goal_pos_map, self.goal_block_above_map = self._build_config_maps(task.goals, self.all_blocks)

    def _build_config_maps(self, facts_set, all_blocks):
        """Helper to build position and block-above maps from a set of facts."""
        pos_map = {}
        block_above_map = {}

        for fact in facts_set:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_above, block_below = parts[1], parts[2]
                pos_map[block_above] = block_below
                block_above_map[block_below] = block_above
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                pos_map[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                 # Holding is a state fact, not typically a goal fact, but handle defensively
                 block = parts[1]
                 pos_map[block] = 'holding'
            # 'clear' facts implicitly define block_above_map entries as None

        # For any block not found as a value in block_above_map, it means nothing is on it.
        # This applies to blocks that should be clear in the goal, or blocks that are clear in the state.
        # Ensure all blocks from the problem are keys in the map, even if nothing is on them.
        for block in all_blocks:
             if block not in block_above_map:
                 block_above_map[block] = None # Nothing is on this block

        return pos_map, block_above_map


    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # 3. Parse current state facts to build current configuration maps
        current_pos_map, current_block_above_map = self._build_config_maps(state, self.all_blocks)

        h = 0 # Initialize heuristic value

        # 5. Iterate through each block
        for block in self.all_blocks:
            # Get goal and current positions below the block
            # Use .get() with None default for blocks not mentioned in goal_pos_map
            goal_below_block = self.goal_pos_map.get(block)
            current_below_block = current_pos_map.get(block)

            # 7. Check Condition 1: Wrong immediate position below
            # If the block has a specified position below it in the goal (is in goal_pos_map)
            # AND its current position below is different
            if goal_below_block is not None and current_below_block != goal_below_block:
                h += 1
            else:
                # 8. If Condition 1 is false, check Condition 2: Wrong block on top
                # Get goal and current blocks above the block
                # Use .get() with None default for blocks that should be/are clear
                goal_above_block = self.goal_block_above_map.get(block)
                current_above_block = current_block_above_map.get(block)

                # If the block currently on top is different from the block that should be on top
                if current_above_block != goal_above_block:
                     h += 1

        # 9. Return the total heuristic value
        return h
