import math
# Assuming the Heuristic base class is accessible via this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts like "(predicate arg1 arg2)"
def get_parts(fact):
    """Extract the components of a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(on b1 b2)" -> ['on', 'b1', 'b2']
    """
    return fact.strip()[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state. It is designed for use with Greedy Best-First Search
    and therefore does not need to be admissible. It aims to be informative and
    efficiently computable. The heuristic counts the number of blocks that are
    considered "out of place" relative to the goal configuration and estimates
    the cost to move them.

    # Assumptions
    - The goal is specified by a conjunction of `(on blockA blockB)`,
      `(on-table blockC)`, and `(clear blockD)` predicates.
    - Each block that needs to be moved from its current position to eventually
      reach the goal configuration requires approximately two actions:
      one pickup/unstack and one putdown/stack.
    - The arm must be empty in the final goal state unless the block it holds
      is the very last one to be placed. This is handled by adjusting the cost
      if a block is held.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the `task.goals` provided.
    - It builds data structures to represent the goal configuration:
        - `goal_support`: A dictionary mapping each block `b` mentioned in an
          `(on b x)` or `(on-table b)` goal to its required support (`x` or 'table').
        - `goal_clear`: A set containing all blocks `b` mentioned in a `(clear b)` goal.
    - It also stores the set `goal_blocks_support` for efficient iteration over blocks
      with defined goal positions.
    - Static facts from the task definition are not used by this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:**
        - Iterate through the facts in the current `node.state`.
        - Build `current_support`: Maps block `b` to its current support
          ('table', another block name, or 'holding' if held by the arm).
        - Build `current_on_top`: Maps block `b` to the block directly on top of it, if any.
        - Identify `held_block`: Stores the name of the block held by the arm, or `None`.
    2.  **Identify Blocks That Must Be Moved (`blocks_to_move` set):**
        - Initialize `blocks_to_move` as an empty set.
        - **Misplaced Blocks:** Add any block `b` to `blocks_to_move` if its
          `current_support` does not match its `goal_support`.
        - **Blocks Violating Goal `clear`:** For any block `g` that must be clear in the
          goal (`g` in `goal_clear`), if there is currently a block `t` on top of `g`
          (`t = current_on_top.get(g)`), then `t` must be moved. Add `t` to `blocks_to_move`.
        - **Obstructing Blocks:** Perform an upward search starting from all blocks currently
          in `blocks_to_move`. Any block found directly or indirectly above a block
          in `blocks_to_move` is also considered obstructing and must be moved. Add all
          such obstructing blocks to the `blocks_to_move` set. This is done iteratively
          (e.g., using a queue) until no more obstructing blocks are found.
    3.  **Calculate Base Cost:**
        - The base cost is estimated as `2 * len(blocks_to_move)`. Each block in the set
          is assumed to require one action to be picked up/unstacked and one action
          to be put down/stacked.
    4.  **Adjust Cost for Held Block:**
        - Check if the arm is currently holding a block (`held_block is not None`).
        - If `held_block` is in the `blocks_to_move` set: This means the block being held
          is one that we identified as needing to be moved anyway. Holding it effectively
          saves the pickup/unstack action. Therefore, subtract 1 from the total cost.
        - If `held_block` is *not* in the `blocks_to_move` set: This means the arm is
          occupied with a block that is either already correctly placed or needs to be
          moved temporarily. This held block must eventually be put down or stacked,
          costing one action. This action was not accounted for in the base cost
          (which only considered blocks *in* `blocks_to_move`). Therefore, add 1
          to the total cost.
    5.  **Return Heuristic Value:**
        - The final calculated cost is returned. It will be 0 if and only if
          `blocks_to_move` is empty and the arm is empty, signifying that the
          goal state (as interpreted by the heuristic's logic) has been reached.
          The cost is ensured to be non-negative.
    """

    def __init__(self, task):
        """Initialize the heuristic by parsing goal conditions."""
        self.goals = task.goals
        # Static facts are not needed for this heuristic.

        # Parse goal state to understand the target configuration
        self.goal_support = {} # block -> target_support ('table' or block_name)
        self.goal_clear = set() # set of blocks that must be clear
        self.goal_blocks_support = set() # blocks mentioned in on/on-table goals

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]
            try:
                if predicate == "on" and len(parts) == 3:
                    # Goal requires 'top' to be on 'bottom'
                    top, bottom = parts[1], parts[2]
                    self.goal_support[top] = bottom
                    self.goal_blocks_support.add(top)
                elif predicate == "on-table" and len(parts) == 2:
                    # Goal requires 'block' to be on the table
                    block = parts[1]
                    self.goal_support[block] = 'table'
                    self.goal_blocks_support.add(block)
                elif predicate == "clear" and len(parts) == 2:
                    # Goal requires 'block' to be clear
                    block = parts[1]
                    self.goal_clear.add(block)
            except IndexError:
                # Handle potential errors if facts are malformed
                print(f"Warning: Malformed goal fact ignored during heuristic init: {fact}")
                continue

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given node's state."""
        state = node.state

        # --- 1. Parse Current State ---
        current_support = {} # block -> current_support ('table', block_name, or 'holding')
        current_on_top = {} # block_bottom -> block_top
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            try:
                if predicate == "on" and len(parts) == 3:
                    top, bottom = parts[1], parts[2]
                    current_support[top] = bottom
                    current_on_top[bottom] = top
                elif predicate == "on-table" and len(parts) == 2:
                    block = parts[1]
                    current_support[block] = 'table'
                elif predicate == "holding" and len(parts) == 2:
                    held_block = parts[1]
                    # Mark the held block's support as 'holding' for comparison logic
                    current_support[held_block] = 'holding'
            except IndexError:
                # Silently ignore malformed facts in the state during heuristic calculation
                continue

        # --- 2. Identify Blocks That Must Be Moved ---
        blocks_to_move = set()

        # 2a. Find misplaced blocks (position relative to support is wrong)
        for block in self.goal_blocks_support:
            current_pos = current_support.get(block, None) # Use .get for safety
            goal_pos = self.goal_support.get(block) # Should exist based on init logic
            if current_pos != goal_pos:
                 blocks_to_move.add(block)

        # 2b. Find blocks violating a 'clear' goal (block on top needs moving)
        for block_to_be_clear in self.goal_clear:
            # Check if something is currently on top of this block
            block_on_top = current_on_top.get(block_to_be_clear)
            if block_on_top is not None:
                # The block on top must be moved
                blocks_to_move.add(block_on_top)

        # 2c. Find obstructing blocks (blocks above any block already in blocks_to_move)
        # Use a queue for BFS-like upward search from initially identified blocks
        queue = list(blocks_to_move)
        processed_queue_indices = 0
        while processed_queue_indices < len(queue):
            # Get the next block from the queue whose stack we need to check
            block_below = queue[processed_queue_indices]
            processed_queue_indices += 1

            # Find the block directly on top of block_below
            block_on_top = current_on_top.get(block_below)

            # If there is a block on top and it hasn't already been marked to move
            if block_on_top is not None and block_on_top not in blocks_to_move:
                # This block is obstructing, add it to the set and the queue
                blocks_to_move.add(block_on_top)
                queue.append(block_on_top) # Check above this newly added block as well

        # --- 3. Calculate Base Cost ---
        # Estimate 2 actions per block that needs to be moved
        base_cost = 2 * len(blocks_to_move)
        cost = base_cost

        # --- 4. Adjust Cost for Held Block ---
        if held_block is not None:
            if held_block in blocks_to_move:
                # Holding a block we need to move saves the pickup/unstack action cost (1)
                cost -= 1
            else:
                # Holding a block we don't need to move (or is already correct)
                # requires an extra putdown/stack action (1) not yet counted.
                cost += 1

        # --- 5. Return Heuristic Value ---
        # Ensure the cost is never negative (shouldn't happen with this logic, but safe)
        cost = max(0, cost)

        return cost
