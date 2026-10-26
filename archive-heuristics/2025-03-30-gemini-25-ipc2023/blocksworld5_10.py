import sys
import os
from typing import Set, Dict, Optional, Tuple

# Assume the following classes are available from the planner's infrastructure:
# - Heuristic (base class)
# - Node (representing a search node with a state)
# - Task (representing the planning problem, including goals)
# If the structure is different, imports might need adjustment.

# Dummy classes for type hinting and basic structure if imports fail
try:
    from heuristics.heuristic_base import Heuristic
    # Assuming task representation is accessible, e.g., via node or globally
    # from planning_infra import Task, Node # Replace with actual import
except ImportError:
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node) -> int: raise NotImplementedError
    class Task:
        def __init__(self): self.goals = set()
        def goal_reached(self, state) -> bool: return self.goals <= state
    class Node:
        def __init__(self, state): self.state = state


def get_parts(fact: str) -> list[str]:
    """Extract the components of a PDDL fact string."""
    # Helper function to parse facts like '(on b1 b2)' into ['on', 'b1', 'b2']
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    configuration. It counts the number of blocks that are considered "misplaced"
    or "obstructing" and assumes each requires two actions (one pick/unstack,
    one put/stack) to correct. A block is considered misplaced if it's not in
    its final goal position relative to the block below it. A block is obstructing
    if it is currently stacked on top of a misplaced block or on top of a block
    that needs to be clear in the goal. Adjustments are made if the arm is
    currently holding a block.

    # Assumptions
    - The goal is specified by a set of `on(A, B)`, `on-table(A)`, and
      optionally `clear(A)` facts defining the desired final towers.
    - Each move action (pickup, putdown, stack, unstack) has a cost of 1.
    - The heuristic aims for informativeness for Greedy Best-First Search,
      not necessarily admissibility for A*.

    # Heuristic Initialization
    - Parses the task's goal conditions (`on`, `on-table`, `clear`) to build
      representations of the target configuration:
        - `goal_config`: Maps each block to what should be directly under it
                         (another block or 'table').
        - `goal_blocks`: Set of all blocks mentioned in goal predicates.
        - `goal_clear`: Set of blocks explicitly required to be clear.
        - `inferred_goal_clear`: Set of blocks implicitly clear (top of goal towers).
    - Stores a reference to the task object to check for goal state achievement.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** If the current state is already a goal state, return 0.
    2.  **Parse Current State:** Determine the current position and status of
        each block from the state's facts:
        - `current_below_map`: Maps block -> what's directly below it ('table', another block, or 'arm').
        - `current_above_map`: Maps block -> what's directly above it.
        - `held_block`: The block currently held by the arm, if any.
    3.  **Identify Out-of-Place Blocks:**
        - Iterate through all blocks `A` that have a defined goal position below them.
        - Check if the block/table currently below `A` (`current_below_map.get(A)`)
          matches the block/table that should be below `A` in the goal
          (`goal_config.get(A)`).
        - If they don't match, mark block `A` as "out_of_place".
    4.  **Identify Obstructing Blocks (Due to Misplacement):**
        - Start a set `must_be_moved` with all "out_of_place" blocks.
        - Iteratively find any block `C` that is currently stacked directly or
          indirectly above any block already in `must_be_moved`. Add these
          obstructing blocks `C` to the `must_be_moved` set.
    5.  **Identify Obstructing Blocks (Due to Clear Goals):**
        - Determine the set of all blocks that must be clear in the goal (explicitly
          or implicitly).
        - For each such block `A`, check if there is currently a block `C` on top of it.
        - If `C` exists and is not already in `must_be_moved`, add `C` (and all
          blocks stacked above `C`) to the `must_be_moved` set using the same
          iterative process as in step 4.
    6.  **Calculate Base Cost:**
        - The base cost is related to the total number of blocks in the final
          `must_be_moved` set (`N = len(must_be_moved)`).
    7.  **Adjust for Held Block:**
        - Check if the arm is currently holding a block (`held_block`).
        - If `held_block` exists:
            - If `held_block` is in the `must_be_moved` set: One action (pick/unstack)
              is already "paid". The cost contribution of this block is 1 (put/stack),
              and the other `N-1` blocks contribute 2 each.
              Total cost = `(N - 1) * 2 + 1`.
            - If `held_block` is NOT in `must_be_moved`: This block needs to be put
              down (cost 1) in addition to the moves required for the `N` blocks
              in `must_be_moved` (cost `N * 2`).
              Total cost = `N * 2 + 1`.
        - If the arm is empty:
            - Total cost = `N * 2`.
    8.  **Return Cost:** Return the calculated cost, ensuring it's non-negative.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing goal conditions.
        """
        self.task = task  # Store task for goal checking

        self.goal_on: Set[str] = set()
        self.goal_on_table: Set[str] = set()
        self.goal_clear: Set[str] = set() # Explicit clear goals
        self.goal_config: Dict[str, str] = {} # block -> below ('table' or block)
        self.goal_blocks: Set[str] = set() # All blocks mentioned in goals
        goal_above_map: Dict[str, str] = {} # block -> block_above in goal

        for fact in self.task.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block_a, block_b = parts[1], parts[2]
                self.goal_on.add(fact)
                self.goal_config[block_a] = block_b
                goal_above_map[block_b] = block_a
                self.goal_blocks.add(block_a)
                self.goal_blocks.add(block_b)
            elif predicate == "on-table":
                block_a = parts[1]
                self.goal_on_table.add(fact)
                self.goal_config[block_a] = 'table'
                self.goal_blocks.add(block_a)
            elif predicate == "clear":
                 block_a = parts[1]
                 self.goal_clear.add(fact)
                 # Ensure clear blocks are tracked even if not in on/on-table
                 self.goal_blocks.add(block_a)

        # Infer implicit clear goals: blocks that are top of a goal tower
        self.inferred_goal_clear: Set[str] = set()
        for block in self.goal_blocks:
            # If a block is part of the goal config but nothing is on it in the goal
            if block in self.goal_config or f'(clear {block})' in self.goal_clear:
                 if block not in goal_above_map:
                    self.inferred_goal_clear.add(block)


    def _propagate_must_move(self, blocks_to_start: Set[str],
                             current_above_map: Dict[str, str],
                             must_be_moved: Set[str]):
        """Helper to find all blocks above a given set and add them to must_be_moved."""
        queue = list(blocks_to_start)
        processed = set() # Avoid cycles and redundant checks within this propagation call

        while queue:
            block_below = queue.pop(0)
            if block_below in processed:
                continue
            processed.add(block_below)

            block_above = current_above_map.get(block_below)
            if block_above is not None:
                if block_above not in must_be_moved:
                    must_be_moved.add(block_above)
                    # Continue checking above the newly added block
                    queue.append(block_above)


    def __call__(self, node) -> int:
        """
        Compute the Blocksworld heuristic value for the given state node.
        """
        state = node.state

        # --- Step 1: Goal Check ---
        if self.task.goal_reached(state):
            return 0

        # --- Step 2: Parse Current State ---
        current_below_map: Dict[str, str] = {} # block -> below ('table'/'block'/'arm')
        current_above_map: Dict[str, str] = {} # block_below -> block_above
        held_block: Optional[str] = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block_a, block_b = parts[1], parts[2]
                current_below_map[block_a] = block_b
                current_above_map[block_b] = block_a
            elif predicate == "on-table":
                block_a = parts[1]
                current_below_map[block_a] = 'table'
            elif predicate == "holding":
                held_block = parts[1]
                current_below_map[held_block] = 'arm'

        # --- Step 3: Identify Out-of-Place Blocks ---
        out_of_place: Set[str] = set()
        for block in self.goal_blocks:
            goal_below = self.goal_config.get(block)
            # If block has no goal position defined underneath it, skip check
            if goal_below is None:
                continue

            current_below = current_below_map.get(block)
            # Treat blocks not present in current state (e.g. held) as not matching table/block goals
            if current_below != goal_below:
                out_of_place.add(block)

        # --- Step 4: Identify Obstructing Blocks (Due to Misplacement) ---
        must_be_moved: Set[str] = set(out_of_place)
        self._propagate_must_move(out_of_place, current_above_map, must_be_moved)

        # --- Step 5: Identify Obstructing Blocks (Due to Clear Goals) ---
        all_goal_clear_blocks = {get_parts(f)[1] for f in self.goal_clear} | self.inferred_goal_clear
        clear_violation_starters: Set[str] = set()

        for block_to_be_clear in all_goal_clear_blocks:
            block_above = current_above_map.get(block_to_be_clear)
            # If there's a block above it, and that block isn't already marked
            if block_above is not None and block_above not in must_be_moved:
                 clear_violation_starters.add(block_above)
                 must_be_moved.add(block_above) # Add the first offender

        # Propagate upwards for these newly identified blocks
        if clear_violation_starters:
            self._propagate_must_move(clear_violation_starters, current_above_map, must_be_moved)

        # --- Step 6 & 7: Calculate Cost ---
        cost = 0
        num_must_move = len(must_be_moved)

        if held_block is not None:
            if held_block in must_be_moved:
                # Pick/unstack is done (cost 1), others need 2 actions
                cost = (num_must_move - 1) * 2 + 1
            else:
                # must_be_moved blocks need 2 actions each + 1 for held block
                cost = num_must_move * 2 + 1
        else: # Arm is empty
            cost = num_must_move * 2

        # --- Step 8: Return Cost ---
        # Cost should be non-negative; goal state handled at start
        return max(0, cost)

