import re
# Assuming the planner infrastructure provides the Heuristic base class
# If not, uncomment the following dummy class for standalone testing:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
             "(clear b1)" -> ["clear", "b1"]
             "(arm-empty)" -> ["arm-empty"]

    Args:
        fact (str): The PDDL fact string.

    Returns:
        list: A list containing the predicate name and its arguments,
              or an empty list if the fact format is invalid.
    """
    fact = fact.strip()
    if not fact.startswith("(") or not fact.endswith(")"):
        # Silently ignore invalid formats, or log a warning
        # print(f"Warning: Invalid fact format: {fact}")
        return []
    # Remove parentheses and split by whitespace
    parts = re.split(r"\s+", fact[1:-1])
    # Filter out empty strings that might result from multiple spaces
    return [part for part in parts if part]


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are not in their final goal position
    (either on the correct block or on the table) plus any blocks that are currently
    stacked above them (and thus need to be moved out of the way). Each such "misplaced"
    or "blocking" block is estimated to require two actions (one pickup/unstack,
    one putdown/stack), unless it is already held by the arm (requiring only one action).

    # Assumptions
    - The goal is specified primarily by a set of `on(block, block)` and `on-table(block)` facts.
    - `clear(block)` goals are implicitly handled by ensuring blocks are placed correctly according
      to `on`/`on-table` goals. If `clear` is explicitly in the goal state predicates,
      it's ignored by this heuristic calculation, as achieving the correct `on`/`on-table`
      structure typically satisfies necessary `clear` conditions implicitly.
    - `arm-empty` goals are also ignored, assuming the arm can be easily emptied at the end if needed.
    - Each block move operation (pickup/unstack followed by putdown/stack) costs 2 actions.

    # Heuristic Initialization
    - Parses the goal facts (`on`, `on-table`) provided in `task.goals`.
    - Builds a `goal_position` dictionary mapping each block mentioned in a goal
      to the object it should be resting on in the goal state (this can be another block
      or the special identifier 'TABLE').
    - Collects all unique block names mentioned across all goal facts into a set
      `all_blocks_in_goal` for efficient lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check for Goal State:** If the current state `node.state` already satisfies all
        goal predicates `self.goals`, the state is a goal state, and the heuristic returns 0.
    2.  **Parse Current State:** Iterate through the facts in the current state `node.state`:
        - Determine the current position of each block:
            - `on(x, y)` -> `x` is on `y`. Store this relationship.
            - `on-table(x)` -> `x` is on `TABLE`. Store this relationship.
            - `holding(x)` -> `x` is in the `ARM`. Store this and note which block is held (`held_block`).
        - Build helper data structures:
            - `current_position`: Maps each block to what is directly beneath it (another block, `TABLE`, or `ARM`).
            - `current_on_top`: Maps each object (block or `TABLE`) to the block directly on top of it, if any.
    3.  **Identify Blocks to Move:** Create an empty set `blocks_to_move`.
    4.  **Check Goal Positions:** Iterate through all blocks `b` that are part of the goal configuration (`self.all_blocks_in_goal`).
        - Compare its `current_position` (defaulting to `TABLE` if not explicitly found in the state) with its required `goal_position`.
        - If the positions don't match, add block `b` to the `blocks_to_move` set.
        - Ignore blocks that only appear as supports in the goal (i.e., `goal_position.get(b)` is None) during this check.
    5.  **Check Blocks Above Misplaced Blocks:** Use a loop that continues as long as new blocks are being added to `blocks_to_move`:
        - For each block `b` currently marked in `blocks_to_move`:
            - Use the `current_on_top` map to find if there is a block `z` directly on top of `b`.
            - If such a block `z` exists and `z` is not already in `blocks_to_move`, add `z` to `blocks_to_move` and set a flag indicating that the set was modified, ensuring the loop continues until all blocking blocks up the towers are found.
    6.  **Calculate Cost:**
        - Initialize `heuristic_value = len(blocks_to_move) * 2`. This assumes each block that needs to move requires one action to get it into the arm (pickup/unstack) and one action to place it correctly (putdown/stack).
        - **Correction for Held Block:** If a block `held_block` is currently held by the arm, and this `held_block` is also in the `blocks_to_move` set, subtract 1 from `heuristic_value`. This accounts for the fact that the pickup/unstack action for this block is already effectively completed. Ensure the heuristic value does not become negative.
    7.  **Return `heuristic_value`.**
    """

    TABLE = "TABLE" # Special identifier for the table surface
    ARM = "ARM"     # Special identifier for the robot arm location

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.

        Args:
            task: The planning task object containing goals, initial state, etc.
        """
        super().__init__(task)
        self.goals = task.goals

        self.goal_position = {} # block -> object below it in goal (block or TABLE)
        self.all_blocks_in_goal = set() # All blocks mentioned in goal facts

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip if parsing failed

            predicate = parts[0]
            # Parse 'on' goal facts: on(block, under_block)
            if predicate == "on" and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                self.goal_position[block] = under_block
                self.all_blocks_in_goal.add(block)
                self.all_blocks_in_goal.add(under_block)
            # Parse 'on-table' goal facts: on-table(block)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_position[block] = self.TABLE
                self.all_blocks_in_goal.add(block)
            # We ignore 'clear' and 'arm-empty' goals for this heuristic.

        # Ensure blocks mentioned only as supports are included in the set
        goal_supports = set(pos for pos in self.goal_position.values() if pos != self.TABLE)
        self.all_blocks_in_goal.update(goal_supports)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.

        Args:
            node: The search node containing the current state.

        Returns:
            int: The estimated cost (number of actions) to reach the goal.
        """
        state = node.state

        # --- Step 1: Check for Goal State ---
        if self.goals <= state:
            return 0

        # --- Step 2: Parse Current State ---
        current_position = {} # block -> object below it (block, TABLE, or ARM)
        current_on_top = {}   # object -> block directly on top (maps block/TABLE -> block)
        held_block = None
        # Initialize on_top map for known goal blocks and TABLE
        for b in self.all_blocks_in_goal:
            current_on_top[b] = None
        current_on_top[self.TABLE] = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            # Process 'on' facts
            if predicate == "on" and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                current_position[block] = under_block
                current_on_top[under_block] = block
                # Ensure on_top entry exists if under_block wasn't in goal
                if under_block not in current_on_top:
                    current_on_top[under_block] = block
            # Process 'on-table' facts
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_position[block] = self.TABLE
                # Ensure on_top entry exists if block wasn't in goal
                if block not in current_on_top:
                     current_on_top[block] = None
            # Process 'holding' fact
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_position[block] = self.ARM
                held_block = block
                # Ensure on_top entry exists if block wasn't in goal
                if block not in current_on_top:
                     current_on_top[block] = None
            # Process 'clear' facts (mainly to ensure block exists in on_top map)
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                if block not in current_on_top:
                     current_on_top[block] = None
            # Ignore 'arm-empty'

        # --- Step 3 & 4: Identify Blocks to Move based on Goal Mismatch ---
        blocks_to_move = set()
        for block in self.all_blocks_in_goal:
            # Default current position to TABLE if not explicitly found
            # (e.g., if state only mentions clear(b))
            current_support = current_position.get(block, self.TABLE)
            goal_support = self.goal_position.get(block)

            # Check if the block's goal position is defined and differs from current
            if goal_support is not None and current_support != goal_support:
                 blocks_to_move.add(block)

        # --- Step 5: Identify Blocks Above Misplaced Blocks ---
        newly_added = True
        while newly_added:
            newly_added = False
            blocks_to_add_this_iteration = set()
            # Iterate over a copy of the set to allow modification
            for block_to_move in list(blocks_to_move):
                # Find what's currently directly on top of this block
                block_on_top = current_on_top.get(block_to_move) # Returns None if nothing is on top

                # If there is a block on top and it's not already marked to move
                if block_on_top is not None and block_on_top not in blocks_to_move:
                    blocks_to_add_this_iteration.add(block_on_top)

            # If we found new blocks that need moving, update the set and continue looping
            if blocks_to_add_this_iteration:
                blocks_to_move.update(blocks_to_add_this_iteration)
                newly_added = True

        # --- Step 6: Calculate Cost ---
        # Each block to move costs 2 actions (pickup/unstack + putdown/stack)
        heuristic_value = len(blocks_to_move) * 2

        # --- Step 6 (Correction): Adjust for Held Block ---
        # If the block currently held is one that needs to move,
        # subtract 1 action cost because pickup/unstack is already done.
        if held_block is not None and held_block in blocks_to_move:
            heuristic_value = max(0, heuristic_value - 1) # Ensure not negative

        # --- Step 7: Return Value ---
        return heuristic_value

