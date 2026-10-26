from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal state by summing three components:
    1. The number of blocks that are not in their correct position relative to the
       block directly below them, considering the entire stack structure down
       to the table as defined by the goal state.
    2. The number of goal conditions of the form `(clear ?x)` that are not met.
    3. A penalty if the goal requires the arm to be empty but it is not.

    This heuristic is non-admissible but aims to guide the search effectively
    by prioritizing states where blocks are forming correct goal stacks,
    required blocks are clear, and the arm is free when needed.

    # Assumptions
    - The goal state defines a specific arrangement of blocks into stacks on the table.
    - Every block present in the initial state is expected to be part of the
      goal configuration, either on the table or on another block, or its
      goal is implicitly `(on-table ?x)` if not specified otherwise.
    - Blocks not explicitly mentioned in an `(on ?x ?y)` or `(on-table ?z)`
      goal fact are assumed to be `(on-table ?x)` in the goal. If such a block
      is not supposed to have anything on top of it in the goal structure,
      `(clear ?x)` is also implicitly a goal for it.

    # Heuristic Initialization
    - Parses the initial state to identify all objects (blocks).
    - Parses the goal facts to determine:
        - The desired position for each block (`self.goal_pos` mapping block to block below or 'table').
        - The set of blocks that must be clear (`self.goal_clear`).
        - Whether the arm must be empty (`self.goal_arm_empty`).
    - Ensures all objects found initially or mentioned in the goal have a
      defined goal position in `self.goal_pos` (defaulting to 'table') and
      adds implicit `(clear ?x)` goals for blocks that should be on the table
      and have nothing on top in the goal structure.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is computed as the sum of three parts:

    1. Incorrectly Stacked Blocks:
       - For each block `B`, determine if it is "correctly stacked" in the current state.
       - A block `B` is correctly stacked if:
         - It is not currently being held.
         - Its current support (the block directly below it or the table) matches its desired support from `self.goal_pos`.
         - If its desired support is another block `C`, then `C` must also be correctly stacked.
       - This check is performed recursively with memoization (`is_correctly_stacked` helper function).
       - Count the total number of blocks that are *not* correctly stacked.

    2. Unsatisfied Clear Conditions:
       - For each block `X` that must be clear according to `self.goal_clear`, check if `(clear X)` is present in the current state.
       - Count the number of such blocks that are *not* clear.

    3. Unsatisfied Arm-Empty Condition:
       - If `self.goal_arm_empty` is True, check if `(arm-empty)` is present in the current state.
       - Add 1 to the heuristic if the arm should be empty but is not.

    The total heuristic value is the sum of the counts from steps 1, 2, and 3.
    This value is 0 if and only if all goal conditions (stacking structure, clearance, arm state) are met.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information and objects.
        """
        self.goals = task.goals
        initial_state = task.initial_state

        # 1. Get all objects from the initial state
        self.all_objects = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if len(parts) > 1:
                for part in parts[1:]:
                    self.all_objects.add(part)

        # 2. Build the goal_pos mapping and identify explicit (clear ?) and (arm-empty) goals
        self.goal_pos = {}
        self.goal_clear = set()
        self.goal_arm_empty = False

        # Collect explicit goal positions and clear conditions
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                self.goal_pos[block] = under_block
            elif parts[0] == 'on-table' and len(parts) == 2:
                 block = parts[1]
                 # Only set if not already assigned by an 'on' goal
                 if block not in self.goal_pos:
                     self.goal_pos[block] = 'table'
            elif parts[0] == 'clear' and len(parts) == 2:
                 self.goal_clear.add(parts[1])
            elif parts[0] == 'arm-empty' and len(parts) == 1:
                 self.goal_arm_empty = True

        # 3. Add objects mentioned in goals but not in initial state to all_objects
        # This handles cases where the goal might involve new objects (unlikely in BW)
        # or objects only mentioned in goal facts like (clear X).
        for goal in self.goals:
             parts = get_parts(goal)
             if len(parts) > 1:
                 for part in parts[1:]:
                     self.all_objects.add(part)


        # 4. Assume blocks not explicitly in goal_pos should be on the table
        # and derive implicit (clear ?) goals.
        # Iterate over a copy because we might add to goal_pos and goal_clear
        for obj in list(self.all_objects):
            if obj not in self.goal_pos:
                 self.goal_pos[obj] = 'table'

        # Now that goal_pos is complete for all relevant objects,
        # identify which blocks should be clear in the goal (those with nothing on top).
        # This adds implicit clear goals. Explicit clear goals are already in self.goal_clear.
        blocks_that_are_supports_in_goal = set(self.goal_pos.values()) - {'table'}
        for obj in self.all_objects:
             # If a block is not a support for any other block in the goal,
             # it must be clear in the goal.
             if obj not in blocks_that_are_supports_in_goal:
                 self.goal_clear.add(obj)


        self.all_objects = list(self.all_objects) # Convert to list for consistent iteration if needed


    def is_correctly_stacked(self, block, state_facts, goal_pos, memo):
        """
        Recursively check if a block is correctly stacked according to the goal.
        Uses memoization to avoid recomputing for the same block.
        A block is correctly stacked if it's not held, its current support
        matches its goal support, and the block below it is also correctly stacked.
        """
        if block in memo:
            return memo[block]

        # Find the actual current support of the block or if it's held
        current_support = None
        is_holding = False
        for fact in state_facts:
            parts = get_parts(fact)
            if parts[0] == 'on' and len(parts) == 3 and parts[1] == block:
                current_support = parts[2]
                break
            elif parts[0] == 'on-table' and len(parts) == 2 and parts[1] == block:
                current_support = 'table'
                break
            elif parts[0] == 'holding' and len(parts) == 2 and parts[1] == block:
                is_holding = True
                break # Block is held, not on anything

        # If the block is being held, it's not correctly stacked relative to the goal structure
        if is_holding:
             memo[block] = False
             return False

        # If the block is not found in the state (shouldn't happen in valid states),
        # assume it's not correctly stacked.
        if current_support is None:
             memo[block] = False
             return False

        # Get the desired position from the goal
        # We assume block is in goal_pos because __init__ adds all objects to it
        target_pos = goal_pos[block]

        # Check if the current support matches the target support
        if current_support != target_pos:
            memo[block] = False
            return False

        # If the current support matches, check if the support itself is correctly stacked
        if target_pos == 'table':
            # Block is on the table and should be - correctly stacked
            memo[block] = True
            return True
        else:
            # Block is on another block (target_pos) and should be.
            # It's correctly stacked only if the block below it is also correctly stacked.
            # This is the recursive step.
            result = self.is_correctly_stacked(target_pos, state_facts, goal_pos, memo)
            memo[block] = result
            return result


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state # frozenset of fact strings

        memo = {} # Memoization dictionary for is_correctly_stacked

        # Part 1: Count blocks not correctly stacked
        incorrectly_stacked_count = 0
        # Iterate over all objects identified in __init__
        for obj in self.all_objects:
            # Check if the block has a defined goal position.
            # __init__ ensures all objects in all_objects are in goal_pos.
            if not self.is_correctly_stacked(obj, state, self.goal_pos, memo):
                incorrectly_stacked_count += 1

        # Part 2: Count unsatisfied (clear ?) goals
        unsatisfied_clear_goals = 0
        for block_to_be_clear in self.goal_clear:
            if f"(clear {block_to_be_clear})" not in state:
                 unsatisfied_clear_goals += 1

        # Part 3: Penalty for arm not empty if required
        arm_penalty = 0
        if self.goal_arm_empty and "(arm-empty)" not in state:
             arm_penalty = 1

        # Total heuristic is the sum of penalties
        h_value = incorrectly_stacked_count + unsatisfied_clear_goals + arm_penalty

        return h_value
