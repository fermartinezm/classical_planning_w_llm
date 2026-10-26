import os
import sys
# Assuming the heuristic is placed in a directory structure recognized by the planner
# e.g., planner_root/heuristics/blocksworld_heuristic.py
# The import path might need adjustment based on the planner's structure.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    Handles potential extra spaces and basic validation.
    """
    # Basic validation
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    # Strip whitespace around parentheses and split
    return fact.strip()[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are not "settled" in their final goal
    configuration. A block is settled if it is in its correct goal position
    (either on the table or on another block) and the block directly beneath it
    is also settled. This check proceeds recursively down to the table.
    The heuristic value is twice the number of unsettled blocks, minus one if
    the arm is currently holding a block (as the pick/unstack action is already done).
    It aims for accuracy to guide Greedy Best-First Search effectively, but is not
    guaranteed to be admissible.

    # Assumptions
    - The goal is primarily specified by `on(block, other_block)` and `on-table(block)` facts,
      defining the target towers.
    - `clear(block)` goals are ignored by this heuristic, as achieving the correct
      `on`/`on-table` structure typically satisfies the necessary `clear` conditions
      for the top blocks.
    - All blocks relevant to the final configuration are mentioned in the goal facts
      (either directly or as support blocks within the target towers).
    - Actions (`pickup`, `putdown`, `stack`, `unstack`) have a uniform cost of 1.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the goal facts (`task.goals`) provided by the planner.
    - It builds internal representations of the goal configuration:
        - `goal_on`: A dictionary mapping each block to the block it should be directly on top of in the goal state.
        - `goal_on_table`: A set containing all blocks that should be directly on the table in the goal state.
    - It identifies the set `all_blocks_in_goal`, containing all unique block names that appear in the structural goal specifications (`on` or `on-table`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** In the `__call__` method, determine the current configuration from the input `node.state`:
        - `current_on`: Create a map where keys are blocks and values are what is currently directly beneath them (either another block's name or the string 'table').
        - `held_block`: Identify which block, if any, is currently being held by the arm (value is the block name, or None if arm is empty).
        - `arm_empty`: A boolean flag indicating if the `(arm-empty)` fact is present.
    2.  **Identify Unsettled Blocks:**
        - Use a recursive helper method `_is_settled(block, current_on, held_block, settled_cache)`:
            - **Base Case:** The 'table' is considered inherently settled.
            - **Memoization:** Use `settled_cache` dictionary to store and retrieve results for previously checked blocks within the same heuristic evaluation, avoiding redundant computations.
            - **Held Block:** If the `block` being checked is the `held_block`, it cannot be settled, so return False.
            - **Goal Check:** Determine the required support (`goal_support`) for the `block` based on the `self.goal_on` and `self.goal_on_table` structures built during initialization. If the `block` does not have a defined position in the goal structure, it cannot be settled; return False.
            - **Position Check:** Get the `current_support` for the `block` from the `current_on` map derived from the state. If `current_support` does not match the `goal_support`, the block is not in its correct place relative to what's below; return False.
            - **Recursive Check:** If the block's current position matches its goal position relative to its support, the block is only truly settled if its support (`goal_support`) is also settled. Recursively call `_is_settled` on the `goal_support`.
            - **Cache & Return:** Store the result (True or False) in `settled_cache` before returning it.
        - Iterate through all blocks identified in `self.all_blocks_in_goal`.
        - For each block, call `_is_settled`. Count how many blocks return `False`. Let this count be `unsettled_count`.
    3.  **Calculate Heuristic Value:**
        - Start with a base estimate: `h = unsettled_count * 2`. This assumes each unsettled block requires at least two actions: one to pick it up (or unstack it) and one to place it down (or stack it).
        - **Arm Adjustment:** If `arm_empty` is False (meaning `held_block` is not None), subtract 1 from `h`. This adjustment reflects that the first action (pick/unstack) for the held block has effectively already been performed.
    4.  **Final Value Determination & Goal Check:**
        - Determine if the state *appears* to be a goal state based on the heuristic's criteria: `is_potentially_goal = (unsettled_count == 0 and arm_empty)`.
        - **If `is_potentially_goal` is True:**
            - Perform a stricter check: verify that all `on`, `on-table` goal facts from `self.goals` are actually present in the `state`, and that `(arm-empty)` is also present.
            - If the stricter check passes, return 0 (goal state).
            - If the stricter check fails (e.g., a `clear` goal is missing, or there's an unexpected discrepancy), return 1, as at least one action is needed.
        - **If `is_potentially_goal` is False:**
            - Handle the specific case where `unsettled_count == 0` but `arm_empty` is False. This means all goal blocks are correctly placed, but the arm is holding an extra block (not part of the goal structure). This requires one action (`putdown`). Return 1.
            - In all other non-goal cases (`unsettled_count > 0`), the calculated value `h` (which will be >= 1) represents the heuristic estimate. Return `max(1, h)` to ensure the heuristic is always at least 1 for non-goal states.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.
        """
        self.goals = task.goals
        # Blocksworld domain typically doesn't rely on static facts for this type of heuristic.

        # Parse goals to find the target configuration
        self.goal_on = {}  # Maps block -> block_below_it_in_goal
        self.goal_on_table = set() # Set of blocks on table in goal
        for fact in self.goals:
            parts = get_parts(fact)
            # Ensure parts list is not empty before accessing index 0
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # parts[1] is on top of parts[2]
                self.goal_on[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                self.goal_on_table.add(parts[1])
            # We ignore 'clear' predicates in the goal for this heuristic

        # Identify all unique blocks involved in the goal configuration's structure
        self.all_blocks_in_goal = set(self.goal_on.keys()) | set(self.goal_on.values()) | self.goal_on_table
        # Remove 'table' if it was accidentally included (e.g., as a value in goal_on)
        if 'table' in self.all_blocks_in_goal:
             self.all_blocks_in_goal.remove('table')

    def _is_settled(self, block, current_on, held_block, settled_cache):
        """
        Recursively checks if a block is 'settled': in its correct goal position
        relative to the block below it, which must also be settled.
        Uses memoization via settled_cache.
        """
        # Base case: The table is definitionally settled.
        if block == 'table':
            return True

        # Return cached result if available to avoid re-computation
        if block in settled_cache:
            return settled_cache[block]

        # A block currently held by the arm cannot be settled anywhere.
        if block == held_block:
            settled_cache[block] = False
            return False

        # Determine the block's current support (what's directly below it).
        # Returns None if the block isn't currently placed on the table or another block.
        current_support = current_on.get(block)

        # Determine the block's required goal support (what should be below it).
        goal_support = None
        if block in self.goal_on:
            goal_support = self.goal_on[block]
        elif block in self.goal_on_table:
            goal_support = 'table'
        else:
            # If a block is not mentioned in the structural goals (on/on-table),
            # it cannot be "settled" according to the goal definition.
            settled_cache[block] = False
            return False

        # Check if the block is currently on its correct goal support.
        if current_support != goal_support:
            settled_cache[block] = False
            return False

        # If the block is on the correct support, it's settled if and only if
        # the support itself is also settled. Check recursively.
        result = self._is_settled(goal_support, current_on, held_block, settled_cache)
        settled_cache[block] = result
        return result

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state
        state_facts = frozenset(state) # Use frozenset for efficient lookups

        # Parse current state to find block positions and arm status
        current_on = {}  # block -> block_below / 'table'
        held_block = None
        arm_empty = False # Assume arm not empty unless (arm-empty) fact found

        for fact in state_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                current_on[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                current_on[parts[1]] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                held_block = parts[1]
            elif predicate == 'arm-empty':
                arm_empty = True

        # If holding a block, arm cannot be empty. Sanity check.
        if held_block is not None:
            arm_empty = False

        # Count the number of blocks involved in the goal that are not yet settled.
        unsettled_count = 0
        settled_cache = {}  # Memoization cache specific to this state evaluation

        for block in self.all_blocks_in_goal:
            if not self._is_settled(block, current_on, held_block, settled_cache):
                unsettled_count += 1

        # --- Heuristic Calculation ---
        # Base cost: 2 actions per unsettled block (one to move it, one to place it)
        h = unsettled_count * 2

        # Adjustment: If arm is holding a block, the pickup/unstack action is already done.
        if not arm_empty: # i.e., held_block is not None
            h -= 1

        # --- Final Value Determination ---
        # Check if the state appears to be a goal state based on heuristic criteria
        is_potentially_goal = (unsettled_count == 0 and arm_empty)

        if is_potentially_goal:
             # Heuristic calculation suggests goal state. Verify more strictly.
             all_structural_goals_met = True
             # Check if all 'on' and 'on-table' goal facts are present in the state
             for goal_fact in self.goals:
                  parts = get_parts(goal_fact)
                  if not parts: continue
                  pred = parts[0]
                  if pred == 'on' or pred == 'on-table':
                       if goal_fact not in state_facts:
                            all_structural_goals_met = False
                            break

             # Final check: structural goals met AND arm is empty
             if all_structural_goals_met and arm_empty:
                 # Confirmed goal state
                 return 0
             else:
                 # Heuristic calculated 0, but state doesn't fully match goals
                 # (e.g., unmet 'clear' goal, or discrepancy found). Return minimum cost 1.
                 return 1
        else:
            # State is not the goal according to the heuristic calculation.
            # Handle the special case: all goal blocks settled, but arm holds an extra block.
            if unsettled_count == 0 and not arm_empty:
                 # Requires 1 action (putdown)
                 return 1
            else:
                 # unsettled_count > 0, so h >= 1.
                 # Return the calculated heuristic value, ensuring it's at least 1.
                 return max(1, h)
