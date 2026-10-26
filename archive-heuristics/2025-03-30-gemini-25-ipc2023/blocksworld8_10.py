import re
from heuristics.heuristic_base import Heuristic # Assuming this path is correct within the project structure

# Helper to parse facts like "(pred obj1 obj2)" into ["pred", "obj1", "obj2"]
def get_parts(fact):
    """Removes parentheses and splits the fact string by spaces."""
    # Handles potential extra spaces by filtering empty strings after split
    return list(filter(None, fact.strip()[1:-1].split(' ')))

class blocksworldHeuristic(Heuristic):
    """
    Heuristic for the Blocksworld domain based on misplaced blocks and clear goals.

    Summary:
    Estimates the cost to reach the goal state by counting the number of blocks
    that need to be moved. A block needs to be moved if:
    a) It is not resting on the correct block/table according to the goal.
    b) It is resting on a block that needs to be clear in the goal.
    c) It is currently stacked on top of a block identified by (a) or (b).
    Each block identified as needing to be moved contributes 2 to the heuristic
    value (1 for pick/unstack, 1 for place/stack). If the arm is holding a
    block, 1 is added initially (for the placing action).

    Assumptions:
    - The goal is specified by (on blockA blockB), (on-table blockC), and (clear blockD) predicates.
    - All blocks involved in the problem have a defined goal state (implicitly or explicitly). Blocks present in the state but not mentioned in 'on' or 'on-table' goals are assumed to need to end up on the table.
    - Each move action (pickup, putdown, stack, unstack) has a cost of 1.
    - The goal state requires the arm to be empty (usually implicitly true if all blocks are placed).

    Heuristic Initialization:
    - Parses the goal predicates from `task.goals` to build:
      - `self.goal_pos[block]`: A dictionary mapping each block to the object ('table' or another block) it should be on in the goal state.
      - `self.goal_clear`: A set containing blocks that must be clear in the goal state.
    - These structures are precomputed for efficiency during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state `node.state` satisfies all goal conditions `self.goals`. If yes, return 0.
    2. Initialize heuristic value `h = 0`.
    3. Parse the current state `frozenset` to build helper dictionaries:
       - `current_below[block]`: Maps each block to the object ('table' or another block) currently below it.
       - `current_above[object]`: Maps each object ('table' or block) to the block currently directly on top of it (if any).
       - Also identify the `held_block` (if any) and the set `all_blocks_in_state`.
    4. If a block `B` is held (`held_block` is not None), increment `h` by 1, as one action (putdown or stack) is needed to place it.
    5. Determine the set `must_move_direct` containing blocks that *directly* violate goal conditions:
       - Add any block `B` (that is not held) if its current support (`current_below.get(B)`) is different from its required goal support (`self.goal_pos.get(B, 'table')`). We default the goal position to 'table' if not explicitly specified for a block present in the state.
       - Add any block `Y` if it is currently on top of a block `X` (`current_above.get(X) == Y`) for which `(clear X)` is a goal condition (`X` is in `self.goal_clear`).
    6. Initialize the set `blocks_to_move` with all blocks from `must_move_direct`. This set will accumulate all blocks that must eventually be moved.
    7. Initialize a queue (e.g., a list used FIFO) with the blocks from `must_move_direct`. Also, keep track of blocks added to the queue (`processed_in_queue`) to avoid redundant processing.
    8. While the queue is not empty (processed using an index `idx` for efficiency):
       - Dequeue a block `B` (conceptually, by advancing `idx`).
       - Find the block `A` currently directly above `B` using `current_above`.
       - Traverse upwards from `A`: While `A` exists (i.e., we haven't reached the top of the stack above `B`):
         - If `A` has not already been identified as needing to move (`A` not in `blocks_to_move`):
           - Add `A` to `blocks_to_move`.
           - If `A` hasn't been added to the queue before (`A` not in `processed_in_queue`), enqueue `A` by appending to the list and add `A` to `processed_in_queue`. This ensures blocks above `A` are also checked later.
         - Update `A` to the block above the current `A` (`A = current_above.get(A)`).
    9. After the queue is processed, `blocks_to_move` contains all blocks that must be moved at least once. Add `2 * len(blocks_to_move)` to the heuristic value `h`.
    10. As a final check, if the calculated `h` is 0 but the state is not a goal state (this might happen in rare edge cases or if the goal involves predicates not considered, like `arm-empty`), return 1 to ensure the heuristic is non-zero for non-goal states. Otherwise, return the calculated `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.
        """
        self.goals = task.goals
        self.static = task.static # Blocksworld usually has no static facts

        # Precompute goal information from task.goals
        self.goal_pos = {} # block -> object_below / 'table'
        self.goal_clear = set() # blocks that must be clear

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            pred = parts[0]
            
            try:
                if pred == "on" and len(parts) == 3:
                    block_a, block_b = parts[1], parts[2]
                    self.goal_pos[block_a] = block_b
                elif pred == "on-table" and len(parts) == 2:
                    block_a = parts[1]
                    self.goal_pos[block_a] = 'table'
                elif pred == "clear" and len(parts) == 2:
                    block_a = parts[1]
                    self.goal_clear.add(block_a)
                # Ignore 'arm-empty' as it's usually a consequence of placing blocks
            except IndexError:
                # Handle potential malformed facts gracefully if necessary
                print(f"Warning: Skipping potentially malformed goal fact: {fact}")
                continue


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # Check for goal state first - if all goal facts are in the current state
        if self.goals <= state:
            return 0

        h = 0

        # Parse current state to understand block configuration
        current_below = {} # block -> object_below / 'table'
        current_above = {} # object_below / 'table' -> block_above
        held_block = None
        all_blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            try:
                if pred == "on" and len(parts) == 3:
                    block_a, block_b = parts[1], parts[2]
                    current_below[block_a] = block_b
                    current_above[block_b] = block_a
                    all_blocks_in_state.add(block_a)
                    all_blocks_in_state.add(block_b)
                elif pred == "on-table" and len(parts) == 2:
                    block_a = parts[1]
                    current_below[block_a] = 'table'
                    all_blocks_in_state.add(block_a)
                elif pred == "holding" and len(parts) == 2:
                    held_block = parts[1]
                    all_blocks_in_state.add(held_block)
                elif pred == "clear" and len(parts) == 2:
                     # Track blocks mentioned as clear, useful for identifying all blocks
                     all_blocks_in_state.add(parts[1])
                # We don't need to parse arm-empty for this heuristic logic
            except IndexError:
                # Handle potential malformed state facts gracefully
                print(f"Warning: Skipping potentially malformed state fact: {fact}")
                continue


        # Add cost=1 if arm is holding a block (needs one action to place it)
        if held_block is not None:
            h += 1

        # Identify blocks that must move directly due to goal violations
        must_move_direct = set()

        # Check misplaced blocks (position relative to what's below)
        for block in all_blocks_in_state:
            if block == held_block:
                continue # Skip the held block for position checks

            # Determine goal position: explicit, or default to 'table' if not specified
            # This assumes blocks existing in the state should end up on the table if not otherwise specified.
            g_pos = self.goal_pos.get(block, 'table')
            c_pos = current_below.get(block) # Block must be somewhere if not held

            if c_pos is None:
                 # This implies block exists but isn't placed (e.g. just created, or inconsistent state)
                 # Log or skip this unexpected case for robustness.
                 # print(f"Warning: Block {block} exists but has no current 'below'. State: {state}")
                 continue

            if c_pos != g_pos:
                 must_move_direct.add(block)

        # Check clear goal violations: if Y is on X, but X should be clear
        for block_to_be_clear in self.goal_clear:
            block_on_top = current_above.get(block_to_be_clear)
            if block_on_top is not None:
                # The block on top (block_on_top) must move if it's violating a clear goal for block_to_be_clear
                must_move_direct.add(block_on_top)

        # Calculate all blocks that need to move (direct violators + those stacked above them)
        blocks_to_move = set(must_move_direct)
        queue = list(must_move_direct) # Use list as a FIFO queue for blocks whose stacks need checking
        processed_in_queue = set(must_move_direct) # Track items added to queue to prevent redundant stack scans

        idx = 0
        while idx < len(queue): # Efficient list-based queue processing
            block_b = queue[idx] # Get block whose stack needs checking
            idx += 1

            block_a = current_above.get(block_b) # Get the block directly above block_b
            while block_a is not None: # Traverse up the stack starting from block_a
                if block_a not in blocks_to_move:
                    # If block_a wasn't already marked to move, mark it now
                    blocks_to_move.add(block_a)
                    # Add block_a to the queue only if it hasn't been processed yet,
                    # to check blocks above it later.
                    if block_a not in processed_in_queue:
                         queue.append(block_a)
                         processed_in_queue.add(block_a)

                # Move up the stack to check the next block
                block_a = current_above.get(block_a)


        # Add cost for all blocks identified that need moving (2 actions per block)
        h += 2 * len(blocks_to_move)

        # Final check: Ensure heuristic is non-zero for non-goal states.
        # If h calculated is 0 but it's not actually a goal state, return 1.
        if h == 0 and not (self.goals <= state):
             return 1

        return h
