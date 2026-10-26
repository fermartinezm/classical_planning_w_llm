from typing import Dict, Set, Optional, Tuple
import sys
import os

# Ensure the directory containing 'heuristics' is in the Python path
# This might be needed if the script is run from a different directory
# Adjust the path ('..') as necessary based on your project structure
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the Heuristic base class. Define a placeholder if it fails.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder base class if the actual one is not found
    # This allows the code to be syntactically correct and testable standalone
    # Replace this with the actual import path in your environment
    class Heuristic:
        def __init__(self, task):
            """Placeholder initializer."""
            self.task = task # Store task for potential use by subclasses

        def __call__(self, node):
            """Placeholder call method."""
            # 'node' typically has attributes like 'state' and potentially 'task'
            raise NotImplementedError("Heuristic.__call__ must be implemented by subclasses")

# Helper function to parse PDDL facts
def get_parts(fact: str) -> Tuple[str, ...]:
    """
    Extracts predicate and arguments from a PDDL fact string like '(on b1 b2)'.
    Handles potential extra whitespace and returns an empty tuple for invalid formats.

    Args:
        fact: The PDDL fact string.

    Returns:
        A tuple containing the predicate name and its arguments,
        e.g., ('on', 'b1', 'b2'). Returns an empty tuple if the format is invalid.
    """
    content = fact.strip()
    # Basic validation for parentheses
    if not content.startswith("(") or not content.endswith(")"):
        return tuple()
    # Remove parentheses and strip internal whitespace
    content = content[1:-1].strip()
    # Handle empty content like "()"
    if not content:
        return tuple()
    # Split by whitespace into parts
    return tuple(content.split())


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    for the Blocksworld domain. It primarily works by counting the number of blocks
    that are not in their final correct position relative to the blocks below them
    in the target goal configuration. Each such "misplaced" block is assumed to
    require at least two actions (one pick/unstack action, one put/stack action).
    An adjustment is made if the robot arm is currently holding a block, as one
    action (the pick/unstack) has effectively already been performed for that block.

    # Assumptions
    - The goal state primarily specifies the final position (`on(A,B)` or `on-table(A)`)
      for the blocks involved in the desired towers.
    - Blocks that are present in the current state but are not mentioned in any
      `on` or `on-table` goal predicate are considered "misplaced" relative to the
      explicitly defined goal structure.
    - `clear` goals are not explicitly counted by this heuristic. Achieving the
      correct `on`/`on-table` configuration is assumed to be the main driver of
      cost, and `clear` states are often intermediate requirements or results of
      achieving the primary positional goals.
    - Each misplaced block requires a minimum of two actions. This is a simplification,
      as clearing blocks on top might require more actions in reality.
    - The heuristic is designed for use with Greedy Best-First Search and is not
      required to be admissible (i.e., it might overestimate the true cost).

    # Heuristic Initialization
    - The constructor (`__init__`) parses the goal conditions (`task.goals`) provided
      by the planner's task representation.
    - It builds internal data structures representing the target configuration:
        - `goal_on`: A dictionary mapping each block `b` to the block `c` it should
          be on (`on b c`) in the goal.
        - `goal_on_table`: A set containing all blocks that should be on the table
          (`on-table b`) in the goal.
    - It also collects the set of all unique block objects mentioned *in the goal
      predicates* (`goal_blocks`). This set helps identify which blocks have defined
      target positions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** The `__call__` method receives a search node (`node`)
        and extracts the current world state (`node.state`, a set of fact strings).
        It parses this state to determine:
        - `current_on`: A dictionary mapping blocks to the block they are currently on.
        - `current_on_table`: A set of blocks currently on the table.
        - `held_block`: The name of the block currently held by the arm, if any (`None` otherwise).
        - `state_blocks`: A set of all block objects mentioned in the current state's predicates.
    2.  **Determine All Blocks:** It computes `current_all_blocks` by taking the union
        of blocks mentioned in the goal (`goal_blocks`) and blocks present in the
        current state (`state_blocks`). This ensures all relevant blocks are considered.
    3.  **Identify Correctly Placed Blocks:** It determines the set `correct_blocks`
        containing blocks that are currently considered "correctly placed". A block `b`
        is correctly placed if:
        - Its goal is `on-table(b)` and it is currently `on-table(b)`.
        - Its goal is `on(b, c)` and it is currently `on(b, c)` AND the block `c`
          below it is also recursively determined to be correctly placed.
        - This check is performed by the recursive helper method `_is_correctly_placed`,
          which uses memoization to avoid redundant computations.
        - Importantly, if a block exists in `current_all_blocks` but does not have an
          `on` or `on-table` predicate specified in the goal, it is considered *not*
          correctly placed by the `_is_correctly_placed` method.
    4.  **Identify Misplaced Blocks:** Any block in `current_all_blocks` that is not
        found in `correct_blocks` is considered misplaced. The set `misplaced_blocks`
        is calculated as `current_all_blocks - correct_blocks`.
    5.  **Calculate Base Cost:** The initial heuristic estimate `h` is calculated as
        `len(misplaced_blocks) * 2`. This reflects the assumption that each misplaced
        block needs one action to be picked up (or unstacked) and one action to be
        placed correctly (put down or stacked).
    6.  **Adjust for Held Block:** If the arm is currently holding a block (`held_block`
        is not `None`), the heuristic value `h` is reduced by 1. This is because the
        held block is inherently misplaced from its final destination, but the action
        to pick it up or unstack it has already been completed. The remaining cost
        associated with this block is the single action needed to place it correctly.
    7.  **Goal Check:** A special check is performed if the calculation results in a
        state where there are no misplaced blocks (`misplaced_blocks` is empty) and
        the arm is empty (`held_block` is `None`). In this situation:
        - It verifies if the current state `state` actually satisfies *all* the goal
          predicates defined in `self.goals` (using `self.goals <= state`).
        - If all goals are satisfied, the heuristic returns 0, indicating a goal state.
        - If the structure matches (no misplaced blocks, arm empty) but some goal
          predicate (e.g., a `clear` goal) is not met, it returns 1. This small positive
          value ensures the search prefers this state over states with misplaced blocks
          but correctly signals that the true goal has not yet been reached.
    8.  **Final Value:** The method returns the calculated heuristic value `h`, ensuring
        it is non-negative using `max(0, h)`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.
        Stores the task goals for later checking during heuristic evaluation.

        Args:
            task: The planning task object, containing goals, initial state, etc.
                  Expected to have attributes like `task.goals`.
        """
        # Initialize the base class if it has an __init__ method
        # super().__init__(task) # Uncomment if Heuristic base class requires initialization
        self.goals = task.goals # The set of goal predicates for the task
        self.static = task.static # Static facts (usually empty in Blocksworld)

        self.goal_on: Dict[str, str] = {} # Maps block -> block it's on in goal
        self.goal_on_table: Set[str] = set() # Set of blocks on table in goal
        self.goal_blocks: Set[str] = set() # All blocks mentioned in goal predicates

        # Parse goal predicates to build the target configuration representation
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip if parsing failed

            predicate = parts[0]
            num_args = len(parts) - 1

            if predicate == "on" and num_args == 2:
                block, below = parts[1], parts[2]
                self.goal_on[block] = below
                self.goal_blocks.add(block)
                self.goal_blocks.add(below)
            elif predicate == "on-table" and num_args == 1:
                block = parts[1]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            elif predicate == "clear" and num_args == 1:
                # Record block name to ensure it's considered later, even if only in clear goal
                self.goal_blocks.add(parts[1])
            # Other predicates like 'arm-empty' are ignored for building the goal structure

    def _is_correctly_placed(self,
                             block: str,
                             current_on: Dict[str, str],
                             current_on_table: Set[str],
                             memo: Dict[str, bool]) -> bool:
        """
        Recursively checks if a block is correctly placed relative to the goal
        configuration and the blocks below it. Uses memoization for efficiency.

        A block is considered correctly placed if it meets its specific `on` or
        `on-table` goal condition, AND the block it rests on (if any) is also
        correctly placed. Blocks that exist but do not have an `on` or `on-table`
        goal defined are considered *not* correctly placed.

        Args:
            block: The name of the block to check.
            current_on: Dictionary representing the current `on` relationships.
            current_on_table: Set representing blocks currently `on-table`.
            memo: Dictionary used for memoization to store results of previous checks.

        Returns:
            True if the block is correctly placed relative to the goal structure below it,
            False otherwise.
        """
        # Return memoized result if already computed
        if block in memo:
            return memo[block]

        is_correct = False # Assume not correct by default
        if block in self.goal_on_table:
            # Case 1: The goal is for this block to be on the table
            is_correct = block in current_on_table
        elif block in self.goal_on:
            # Case 2: The goal is for this block to be on another block (`below_goal`)
            below_goal = self.goal_on[block]
            # Check if it's currently on the correct block (`below_goal`)
            if current_on.get(block) == below_goal:
                 # If yes, recursively check if the block below is also correctly placed
                 is_correct = self._is_correctly_placed(below_goal, current_on, current_on_table, memo)
            # If not on the correct block, `is_correct` remains False
        # Case 3: The block exists but has no `on` or `on-table` goal defined.
        # In this heuristic, we consider such blocks as not correctly placed relative
        # to the explicit goal structure. `is_correct` remains False.

        # Store the result in the memoization table before returning
        memo[block] = is_correct
        return is_correct

    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the given state node.

        Args:
            node: The search node, expected to have an attribute `node.state` which is
                  a set (or frozenset) of PDDL fact strings representing the current state.

        Returns:
            An integer representing the estimated number of actions to reach the goal.
            Returns 0 if the state is identified as a goal state.
            Returns a non-negative integer otherwise.
        """
        state = node.state # The current state as a set of facts

        # --- 1. Parse Current State ---
        current_on: Dict[str, str] = {} # Maps block -> block it's currently on
        current_on_table: Set[str] = set() # Set of blocks currently on the table
        held_block: Optional[str] = None # Block currently held by the arm
        state_blocks: Set[str] = set() # All blocks mentioned in the current state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            num_args = len(parts) - 1

            if predicate == "on" and num_args == 2:
                block, below = parts[1], parts[2]
                current_on[block] = below
                state_blocks.add(block)
                state_blocks.add(below)
            elif predicate == "on-table" and num_args == 1:
                block = parts[1]
                current_on_table.add(block)
                state_blocks.add(block)
            elif predicate == "holding" and num_args == 1:
                held_block = parts[1]
                state_blocks.add(held_block)
            elif predicate == "clear" and num_args == 1:
                # Record block name, ensures it's included in state_blocks
                state_blocks.add(parts[1])
            # Ignore arm-empty, etc.

        # --- 2. Determine All Blocks ---
        # Combine blocks mentioned in goals and those present in the current state
        current_all_blocks = self.goal_blocks.union(state_blocks)

        # Handle edge case: If there are no blocks at all
        if not current_all_blocks:
             # Check if the (likely empty) goal is met by the (likely empty) state
             return 0 if self.goals <= state else 1

        # --- 3. Identify Correctly Placed Blocks ---
        correct_blocks: Set[str] = set()
        memo: Dict[str, bool] = {} # Memoization cache for the recursive check

        # Check correctness for all relevant blocks
        for block in current_all_blocks:
            if self._is_correctly_placed(block, current_on, current_on_table, memo):
                correct_blocks.add(block)

        # --- 4. Identify Misplaced Blocks ---
        # Misplaced blocks are those present but not correctly placed
        misplaced_blocks = current_all_blocks - correct_blocks

        # --- 5. Calculate Base Cost ---
        # Base cost: 2 actions per misplaced block (pick/unstack + put/stack)
        h = len(misplaced_blocks) * 2

        # --- 6. Adjust for Held Block ---
        # If holding a block, one action (pick/unstack) is effectively done.
        if held_block is not None:
             # Subtract 1 from the cost, as only the placement action remains for the held block.
             h -= 1

        # --- 7. Goal Check ---
        # If the structure seems correct (no misplaced blocks, arm empty),
        # perform a final check against all goal predicates.
        if not misplaced_blocks and held_block is None:
             # Check if the current state satisfies all goal conditions
             is_goal_state = self.goals <= state
             if is_goal_state:
                 return 0 # Heuristic is 0 for a true goal state
             else:
                 # Structure matches, but some goal (e.g., 'clear') might be missing.
                 # Return 1 to indicate it's very close but not the goal yet.
                 return 1

        # --- 8. Final Value ---
        # Ensure the heuristic value is never negative
        return max(0, h)

