import os
import sys
from typing import Set, Dict, Optional, FrozenSet

# Ensure the heuristics directory is in the Python path if needed
# (Adjust path based on your project structure if heuristic_base is not directly importable)
# Example:
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# Attempt to import the base class; provide a dummy if it fails
try:
    from heuristics.heuristic_base import Heuristic
    # Assuming Task and Operator classes are implicitly available or not needed by Heuristic base
    # If Node class is needed for type hint: from search.node import Node
except ImportError:
    print("Warning: Heuristic base class not found. Using dummy base class.", file=sys.stderr)
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task):
            self.task = task # Store task for access to goals etc.
        def __call__(self, node) -> int: # Use -> int type hint for clarity
            raise NotImplementedError

# Helper function to parse PDDL-style fact strings
def get_parts(fact: str) -> list[str]:
    """
    Extracts predicate and arguments from a fact string like '(on b1 b2)'.
    Removes parentheses and splits by space. Handles potential extra whitespace.
    Returns a list of strings, e.g., ["on", "b1", "b2"], or empty list for malformed input.
    """
    if not fact or not fact.startswith("(") or not fact.endswith(")"):
        return []
    # Strip whitespace, remove parentheses, split
    return fact.strip()[1:-1].split()

# Define TABLE constant for clarity when indicating a block is on the table
TABLE = "TABLE" # Using a string constant to represent the table

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain designed for Greedy Best-First Search.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It primarily counts the number of blocks that are not resting on their correct final support
    (either the correct block below them or the table, as defined in the goal). It assumes
    that each such "misplaced" block, plus any blocks currently stacked on top of it,
    needs to be moved. Each required move (pickup/unstack + putdown/stack) is estimated
    to cost 2 actions. The heuristic value is adjusted based on whether the arm is
    currently holding a block, potentially saving or costing an action. It aims for
    informativeness to guide the search effectively, not necessarily admissibility.

    # Assumptions
    - The goal state is primarily defined by `(on x y)` and `(on-table x)` predicates.
      `(clear x)` goals are considered satisfied if the block configuration matches the
      goal, meaning nothing incorrect is on top of `x`. `(arm-empty)` goals are also
      considered implicitly.
    - All block names are unique strings.
    - All actions (`pickup`, `putdown`, `stack`, `unstack`) have a uniform cost of 1.
    - The planner provides the state as a frozenset of fact strings (e.g., `'(on b1 b2)'`).
    - The `task` object provides goal facts (`task.goals`) and a method `task.goal_reached(state)`.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the `task.goals` once.
    - It builds a dictionary `goal_pos` where `goal_pos[block]` stores the name of the
      object (another block string or the constant `TABLE`) that `block` should be
      resting on in the goal state.
    - It collects the set of all unique block names (`all_blocks`) mentioned in any
      goal predicate (`on`, `on-table`, `clear`, `holding`) for later iteration.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** If `task.goal_reached(state)` is true for the current `node.state`, return 0 immediately.
    2.  **Parse Current State:** Iterate through the facts in `node.state` to determine the current configuration:
        - `current_pos[block]`: Maps each block name to what it's currently resting on (block name, `TABLE`, or `"ARM"` if held).
        - `on_top_map[block]`: Maps each block name to the block name currently directly on top of it (if any).
        - `holding`: Stores the name of the block held by the arm (or `None` if `arm-empty`).
        - `present_blocks`: Collects all block names appearing in the current state.
    3.  **Identify Blocks That Must Move (`blocks_to_move`):**
        - Initialize `blocks_to_move` as an empty set.
        - Use a helper function `get_blocks_above(block, current_on_top_map)` (with memoization) to find all blocks currently stacked above a given `block`.
        - Iterate through all relevant blocks (`blocks_to_consider` = union of goal blocks and present blocks).
        - For each `block B` not already processed:
            - Determine its current position `curr_p` and goal position `goal_p`.
            - A block `B` *needs moving* if:
                a) It is currently held by the arm (`curr_p == "ARM"`).
                b) It is currently placed on a support `curr_p` that is different from its required goal support `goal_p`. This includes cases where `goal_p` is `None` (meaning `B` shouldn't be where it is according to the goal structure).
            - If `B` needs moving:
                - Add `B` to `blocks_to_move`.
                - Add all blocks currently above `B` (using `get_blocks_above`) to `blocks_to_move` as well, because they obstruct `B`.
    4.  **Calculate Base Cost:**
        - The initial estimated cost is `cost = len(blocks_to_move) * 2`. Each block needing movement requires roughly a pickup/unstack and a putdown/stack.
    5.  **Adjust Cost for Arm State:**
        - If the arm is holding a block `H` (`holding is not None`):
            - If `H` is one of the blocks that needs to move (`H in blocks_to_move`): Decrease `cost` by 1 (saves the pickup/unstack action for `H`).
            - If `H` is *not* a block that needs to move (`H not in blocks_to_move`): Increase `cost` by 1 (the arm holds a "correct" block that must be put down first).
    6.  **Final Value:** Return the calculated `cost`, ensuring it's at least 0.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        # No static facts expected in standard blocksworld to pre-process

        self.goal_pos: Dict[str, str] = {} # block -> block/TABLE below it in goal
        self.all_blocks: Set[str] = set() # Set of all block objects mentioned in goals

        # Parse goal predicates to find final positions and all blocks
        for fact in self.task.goals: # Access goals via self.task
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            pred = parts[0]

            if pred == "on" and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                self.goal_pos[block_on_top] = block_below
                self.all_blocks.add(block_on_top)
                self.all_blocks.add(block_below)
            elif pred == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_pos[block] = TABLE
                self.all_blocks.add(block)
            elif pred == "clear" and len(parts) == 2:
                 # Ensure block is known, even if clear goal isn't directly used by goal_pos
                 self.all_blocks.add(parts[1])
            elif pred == "holding" and len(parts) == 2:
                 # A goal to be holding a block means it needs picking up if not held.
                 # This doesn't fit goal_pos, but ensures the block is known.
                 self.all_blocks.add(parts[1])
            # Ignore arm-empty goals

    def __call__(self, node) -> int: # node object expected to have a 'state' attribute
        """Calculates the heuristic value for the given state node."""
        state: FrozenSet[str] = node.state

        # 1. Goal Check
        if self.task.goal_reached(state):
            return 0

        # 2. Parse Current State
        current_pos: Dict[str, str] = {} # block -> block/TABLE/ARM below it now
        on_top_map: Dict[str, str] = {} # block -> block on top of it now
        holding: Optional[str] = None
        present_blocks: Set[str] = set() # Track all blocks existing in the current state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == "on" and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                current_pos[block_on_top] = block_below
                on_top_map[block_below] = block_on_top
                present_blocks.add(block_on_top)
                present_blocks.add(block_below)
            elif pred == "on-table" and len(parts) == 2:
                block = parts[1]
                current_pos[block] = TABLE
                present_blocks.add(block)
            elif pred == "holding" and len(parts) == 2:
                holding = parts[1]
                current_pos[holding] = "ARM" # Special marker for held block position
                present_blocks.add(holding)
            elif pred == "clear" and len(parts) == 2:
                 present_blocks.add(parts[1]) # Track block existence
            # Ignore arm-empty

        # 3. Identify Blocks That Must Move
        blocks_to_move: Set[str] = set()
        # Cache for get_blocks_above results within this single heuristic call
        memo_get_blocks_above: Dict[str, Set[str]] = {}

        def get_blocks_above(block: str, current_on_top_map: Dict[str, str]) -> Set[str]:
            """Finds all blocks currently stacked directly or indirectly above 'block'."""
            if block in memo_get_blocks_above:
                return memo_get_blocks_above[block]

            above: Set[str] = set()
            curr: str = block
            path: list[str] = [] # Simple cycle detection
            while curr in current_on_top_map:
                top_block = current_on_top_map[curr]
                # Check for immediate cycle or revisiting a block in the current upward path
                if top_block == block or top_block in path:
                    # This indicates an invalid state structure, but we should prevent infinite loops
                    print(f"Warning: Cycle detected involving block {top_block} above {block} in state.", file=sys.stderr)
                    break # Stop ascending this potentially corrupt tower
                path.append(top_block)
                above.add(top_block)
                curr = top_block

            memo_get_blocks_above[block] = above
            return above

        # Consider all blocks mentioned in goal or present in state
        blocks_to_consider: Set[str] = self.all_blocks.union(present_blocks)
        # Keep track of blocks already processed to avoid redundant checks, especially
        # for blocks added via get_blocks_above.
        processed_for_moving: Set[str] = set()

        for block in blocks_to_consider:
            # If already determined to move (e.g., was above another misplaced block) or already checked
            if block in blocks_to_move or block in processed_for_moving:
                continue
            processed_for_moving.add(block) # Mark as processed for this loop

            curr_p: Optional[str] = current_pos.get(block, None)
            # Goal position might not exist if block isn't mentioned in goal on/on-table
            goal_p: Optional[str] = self.goal_pos.get(block, None)

            needs_moving: bool = False
            if curr_p == "ARM":
                # If the block is held, it needs to be placed (moved)
                needs_moving = True
            elif curr_p is not None and curr_p != goal_p:
                # Block is placed, but its support is wrong OR it shouldn't be placed
                # according to goal_pos (goal_p is None or different).
                needs_moving = True
            # Note: If curr_p is None, the block doesn't exist in the state's placement facts,
            # which shouldn't happen if it's in blocks_to_consider unless something is wrong.

            if needs_moving:
                 blocks_to_move.add(block)
                 # Add all blocks currently above it, as they must also move
                 above_blocks = get_blocks_above(block, on_top_map)
                 blocks_to_move.update(above_blocks)
                 # Mark the blocks above as processed too, as their need-to-move status
                 # is now determined by the block below them being moved.
                 processed_for_moving.update(above_blocks)

        # 4. Calculate Base Cost
        # Each block that needs moving requires roughly 2 actions (pick/unstack + put/stack)
        cost: int = len(blocks_to_move) * 2

        # 5. Adjust Cost for Arm State
        if holding is not None:
            if holding in blocks_to_move:
                # Holding a block that needs moving saves one action (the pickup/unstack)
                # This block still needs placing, which is accounted for in the * 2 factor.
                cost -= 1
            else:
                # Holding a block that is "correct" or irrelevant according to the misplaced logic.
                # This block needs to be put down (1 action) to free the arm for necessary tasks.
                cost += 1

        # 6. Final Value
        # Ensure heuristic is non-negative. The goal check handles the 0 case.
        return max(0, cost)
