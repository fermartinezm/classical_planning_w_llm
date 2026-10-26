import os
import sys
from typing import Set, Dict, Optional, Tuple

# Ensure the heuristics directory is in the Python path if needed
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the Heuristic base class, define a dummy if not found
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Try to import Task class for type hinting, define dummy if not found
try:
    # Assuming Task class is defined elsewhere, e.g., in planning_task module
    # from planning_task import Task
    # Define a dummy Task for type hinting if not available
    class Task:
        goals: Set[str]
        # Add other attributes if needed by the heuristic's __init__
        def __init__(self):
            self.goals = set()

except ImportError:
     class Task:
        goals: Set[str]
        def __init__(self):
            self.goals = set()
     # Assign DummyTask to Task if real Task isn't available
     if 'Task' not in globals():
         Task = Task # Keep the dummy definition

# Define a dummy Node class for type hinting if not available
# Replace with actual Node class import if available
try:
    # from search_node import Node # Replace with actual import
    class Node:
        state: frozenset[str]
        def __init__(self, state: frozenset[str]):
            self.state = state
except ImportError:
     class Node:
        state: frozenset[str]
        def __init__(self, state: frozenset[str]):
            self.state = state
     if 'Node' not in globals():
         Node = Node # Keep the dummy definition


def parse_fact(fact: str) -> Tuple[str, ...]:
    """
    Parses a PDDL fact string like '(predicate obj1 obj2)' into a tuple.
    Removes parentheses and splits by space.

    Args:
        fact: The PDDL fact string.

    Returns:
        A tuple representing the parsed fact (predicate, arg1, arg2, ...).
        Returns an empty tuple if the input string is malformed or empty after stripping.
    """
    stripped_fact = fact.strip()
    if len(stripped_fact) < 2 or not stripped_fact.startswith('(') or not stripped_fact.endswith(')'):
        # Return empty tuple or raise error for malformed facts
        return tuple()
    # Remove parentheses and split
    return tuple(stripped_fact[1:-1].split())


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are considered "misplaced" and assigns
    a cost based on moving them. A block is considered misplaced if:
    1. It is not resting on the correct block or table as specified in the goal.
    2. It is resting on top of another block that is misplaced (and thus must also move).
    3. It is currently located where another block should be in the goal configuration (obstructing).

    The heuristic estimates 2 actions (one pickup/unstack, one putdown/stack) for
    each block that must be moved. This is adjusted if the block that needs moving
    is already held by the arm (saving the pickup/unstack action).

    # Assumptions
    - The goal is specified by a set of `(on block block)` and `(on-table block)` predicates.
    - `(clear block)` goals are implicitly handled by achieving the structural goals correctly.
    - The arm needs to be empty in the final goal state (`arm-empty` predicate).
    - All actions (pickup, putdown, stack, unstack) have a cost of 1.
    - The heuristic is designed for Greedy Best-First Search and does not need to be admissible.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's goal specification (`task.goals`).
    - It builds a `goal_on` dictionary mapping each block `b` to the block `a` or the symbol `'table'` that should be directly underneath `b` in the goal configuration.
    - It collects all unique block names mentioned in the goal predicates into `all_blocks`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** Immediately return 0 if the current state satisfies all goal conditions (`self.goals <= state`).
    2.  **Parse Current State:** Extract the current configuration from the `state` facts:
        - `current_on`: Maps block `b` to what's currently under it (`a`, `'table'`, or `'holding'`).
        - `current_what_on`: Maps block/table `a` to the block `b` currently directly on top of it.
        - `current_holding`: Stores the block held by the arm, or `None`.
        - Collect all blocks present in the current state (`blocks_in_state`).
    3.  **Identify Blocks to Move (`must_move` set):**
        a.  **Base Mismatches:** Iterate through all blocks `b` that have a defined goal position (`self.goal_on`). If the block/table currently under `b` (`current_on.get(b)`) is different from the goal base (`self.goal_on[b]`), add `b` to the `must_move` set.
        b.  **Obstructions:** Iterate through all goal relations `(on b a)` (where `a` is a block, not the table). If there is currently a block `c` on top of `a` (`current_what_on.get(a)`), and `c` is not the target block `b`, then `c` is obstructing the goal location for `b` and must be moved. Add `c` to the `must_move` set.
        c.  **Propagation:** If a block `b` is in `must_move`, any block `c` currently stacked directly on top of `b` must also eventually move. Use a queue-based approach to find all such blocks transitively and add them to the `must_move` set.
    4.  **Calculate Base Cost:** The initial estimated cost `h` is `len(must_move) * 2`, as each block in `must_move` generally requires a pickup/unstack and a putdown/stack operation.
    5.  **Adjust for Held Block:** If the arm is currently holding a block `H` (`current_holding is not None`) and this block `H` is in the `must_move` set, it means the pickup/unstack action is already effectively completed for this block in this step. Therefore, decrement the cost `h` by 1.
    6.  **Final Non-Goal Check:** If the calculated cost `h` is 0, but the state was determined not to be a goal state in Step 1, it implies a discrepancy (e.g., only arm state or clear predicates differ subtly). In this case, return 1 to ensure the search progresses past this state. Otherwise, return the calculated `h`.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by parsing the goal configuration.

        Args:
            task: The planning task object containing goals, initial state, etc.
        """
        super().__init__(task)
        self.goals: Set[str] = task.goals
        self.goal_on: Dict[str, str] = {}  # Maps block -> block | 'table' (what should be under it)
        self.all_blocks: Set[str] = set() # Keep track of all unique block names in goals

        for goal in self.goals:
            parts = parse_fact(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == 'on':
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    self.goal_on[block_on_top] = block_below
                    self.all_blocks.add(block_on_top)
                    self.all_blocks.add(block_below)
            elif predicate == 'on-table':
                if len(parts) == 2:
                    block_on_table = parts[1]
                    self.goal_on[block_on_table] = 'table'
                    self.all_blocks.add(block_on_table)
            # 'clear' and 'arm-empty' goals are handled implicitly or by final check

    def __call__(self, node: Node) -> int:
        """
        Calculates the heuristic estimate for the given state node.

        Args:
            node: The search node containing the state (frozenset of facts) to evaluate.

        Returns:
            An integer estimate of the cost (number of actions) to reach the goal.
        """
        state: frozenset[str] = node.state

        # 1. Goal Check: Check if the state is already a goal state
        is_goal_state = self.goals <= state
        if is_goal_state:
            return 0

        # 2. Parse Current State
        current_on: Dict[str, str] = {}  # Maps block -> what's under it (block | 'table' | 'holding')
        current_what_on: Dict[str, str] = {}  # Maps block/table -> what's directly on top
        current_holding: Optional[str] = None
        blocks_in_state: Set[str] = set() # Track blocks present in the current state

        for fact in state:
            parts = parse_fact(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == 'on':
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    current_on[block_on_top] = block_below
                    current_what_on[block_below] = block_on_top
                    blocks_in_state.add(block_on_top)
                    blocks_in_state.add(block_below)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block_on_table = parts[1]
                    current_on[block_on_table] = 'table'
                    blocks_in_state.add(block_on_table)
            elif predicate == 'holding':
                if len(parts) == 2:
                    current_holding = parts[1]
                    current_on[current_holding] = 'holding' # Represent being held
                    blocks_in_state.add(current_holding)

        # 3. Identify Blocks to Move (`must_move` set)
        must_move: Set[str] = set()

        # 3a. Base Mismatches: Check blocks with a defined goal position
        for block in self.goal_on: # Iterate through blocks defined in the goal structure
            current_base = current_on.get(block) # What's currently under the block
            goal_base = self.goal_on[block]      # What should be under the block

            if current_base != goal_base:
                must_move.add(block)

        # 3b. Obstructions: Check if goal locations are occupied by wrong blocks
        for block_on_top, goal_base in self.goal_on.items():
            # Only check obstructions on blocks, not the table
            if goal_base == 'table' or goal_base == 'holding':
                continue

            # Check what is currently on the goal_base location
            block_currently_on_goal_base = current_what_on.get(goal_base)

            if block_currently_on_goal_base is not None and block_currently_on_goal_base != block_on_top:
                # The goal_base location has a block on it, but it's not the correct one (block_on_top)
                must_move.add(block_currently_on_goal_base)

        # 3c. Propagation: If a block must move, anything on top of it must also move
        # Use a list as a queue for efficient processing of the propagation
        queue = list(must_move)
        # Use the must_move set itself to track processed nodes for propagation efficiently
        processed_for_propagation = must_move.copy()

        idx = 0
        while idx < len(queue):
            block_to_check = queue[idx]
            idx += 1

            # Find what block, if any, is currently on top of block_to_check
            block_on_top = current_what_on.get(block_to_check)

            if block_on_top is not None and block_on_top not in processed_for_propagation:
                must_move.add(block_on_top)
                processed_for_propagation.add(block_on_top)
                queue.append(block_on_top) # Add to the end of the queue for processing

        # 4. Calculate Base Cost
        h = len(must_move) * 2

        # 5. Adjust for Held Block
        if current_holding is not None and current_holding in must_move:
            # If the block that needs to move is already held,
            # we save one action (the pickup/unstack).
            h -= 1

        # 6. Final Non-Goal Check
        # If h is 0 but we already determined it's not a goal state, return 1.
        # This handles cases where the structure matches but e.g. arm isn't empty.
        if h == 0 and not is_goal_state:
             return 1

        # Ensure heuristic is non-negative
        return max(0, h)

