import re
from heuristics.heuristic_base import Heuristic
from typing import Dict, Set, Optional, Tuple, List

# Helper function to parse PDDL facts more robustly
def get_parts(fact_str: str) -> Tuple[str, List[str]]:
    """Extracts predicate and arguments from a PDDL fact string.

    Handles facts with zero or more arguments.
    Example: '(on b1 b2)' -> ('on', ['b1', 'b2'])
             '(clear b1)' -> ('clear', ['b1'])
             '(arm-empty)' -> ('arm-empty', [])

    Args:
        fact_str: The PDDL fact string (e.g., "(on b1 b2)").

    Returns:
        A tuple containing the predicate name (str) and a list of arguments (List[str]).

    Raises:
        ValueError: If the fact format is invalid.
    """
    fact_str = fact_str.strip()
    if not (fact_str.startswith('(') and fact_str.endswith(')')):
        raise ValueError(f"Invalid fact format: missing parentheses in '{fact_str}'")

    content = fact_str[1:-1].strip()
    if not content:
        # Handles case like "()" which might be invalid PDDL but avoids crash
        raise ValueError(f"Invalid fact format: empty content in '{fact_str}'")

    parts = content.split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by counting blocks
    that are not in their final position relative to what is underneath them,
    and blocks that are stacked on top of such misplaced blocks. It assumes
    moving each of these requires approximately two actions (one pickup/unstack,
    one putdown/stack). A minor adjustment is made if the arm is holding a block.

    # Assumptions
    - The primary cost comes from moving blocks that are not in their goal location
      relative to what's underneath them (i.e., `on(A, B)` or `on-table(A)` goals).
    - Blocks sitting on top of misplaced blocks must also be moved (unstacked and
      put down temporarily), incurring additional cost estimated as two actions.
    - Holding a block saves the initial pickup/unstack action for that block, reducing
      the estimated cost by one.
    - `clear` and `arm-empty` goal conditions are assumed to be achieved implicitly
      by satisfying the `on` and `on-table` goals.

    # Heuristic Initialization
    - Parses the task's goal conditions (`on`, `on-table`) to determine the target
      position for each block involved in these goals. This is stored in
      `goal_pos[block] = block_or_table_underneath`.
    - Identifies all unique block objects mentioned in the goal's `on` and `on-table`
      predicates, stored in `all_blocks`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Determine the current position or status of each block:
        - `current_pos[block] = 'arm'` if `holding(block)`.
        - `current_pos[block] = 'table'` if `on-table(block)`.
        - `current_pos[block] = under_block` if `on(block, under_block)`.
        Also, identify which block is currently held (`held_block`) and the current
        stacking relationships (`current_on[block_above] = block_below`).
    2.  **Identify Misplaced Blocks:** Create a set `misplaced_blocks` containing
        all blocks `b` for which `b` has a defined goal position (`b` in `goal_pos`)
        and its current position `current_pos.get(b)` is different from its
        `goal_pos[b]`. If a block with a goal position is not found in the current
        state's `on` or `holding` facts, it's assumed to be `on-table`.
    3.  **Identify Obstructing Blocks:** Create a set `blocks_on_misplaced`
        containing all blocks `c` such that `c` is currently `on` top of a block
        `b` (i.e., `current_on[c] == b`) where `b` is in `misplaced_blocks`.
        These blocks `c` obstruct the movement of `b` and likely need to be moved.
    4.  **Calculate Base Cost:** The base heuristic estimate is
        `h = 2 * len(misplaced_blocks) + 2 * len(blocks_on_misplaced)`.
        Each block in `misplaced_blocks` needs roughly 2 actions to move to its
        goal position. Each block in `blocks_on_misplaced` needs roughly 2 actions
        to be moved out of the way (unstack, putdown).
    5.  **Adjust for Held Block:** If a block `b` is currently being held
        (`held_block` is not None), it means the pickup/unstack action for `b`
        has already occurred. Since `b` is guaranteed to be in `misplaced_blocks`
        (as 'arm' is never a goal position), its cost contribution of 2 within
        `2 * len(misplaced_blocks)` is an overestimation by 1. Therefore, subtract 1
        from `h` if a block is being held.
    6.  **Return Value:** Return the final calculated heuristic value `h`, ensuring
        it is non-negative. If the state satisfies all `on` and `on-table` goals
        and the arm is empty, the heuristic should ideally return 0.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions.

        Args:
            task: The planning task object containing goals, initial state, etc.
        """
        self.goals: frozenset[str] = task.goals
        self.static: frozenset[str] = task.static # Usually empty for blocksworld

        self.goal_pos: Dict[str, str] = {} # Maps block -> block_or_table_underneath
        self.all_blocks: Set[str] = set() # Blocks involved in on/on-table goals

        # Parse goals to find target configuration and relevant blocks
        for fact in self.goals:
            try:
                predicate, args = get_parts(fact)
                if predicate == 'on':
                    if len(args) == 2:
                        block_a, block_b = args
                        self.goal_pos[block_a] = block_b
                        self.all_blocks.add(block_a)
                        self.all_blocks.add(block_b)
                    else:
                        print(f"Warning: Malformed 'on' goal fact: {fact}")
                elif predicate == 'on-table':
                    if len(args) == 1:
                        block = args[0]
                        self.goal_pos[block] = 'table'
                        self.all_blocks.add(block)
                    else:
                        print(f"Warning: Malformed 'on-table' goal fact: {fact}")
                # Ignore 'clear', 'arm-empty' goals for position mapping
            except ValueError as e:
                print(f"Warning: Skipping unparseable goal fact '{fact}': {e}")

        # Ensure all blocks mentioned as being underneath are also in all_blocks
        # This is implicitly handled as they must appear as the first arg
        # in another 'on' goal or an 'on-table' goal if they are part of the config.


    def __call__(self, node) -> int:
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: The search node containing the state.

        Returns:
            An integer estimate of the cost to reach the goal.
        """
        state: frozenset[str] = node.state
        current_pos: Dict[str, str] = {} # Maps block -> current_pos (block_below, 'table', or 'arm')
        current_on: Dict[str, str] = {} # Maps block_above -> block_below
        held_block: Optional[str] = None

        # Parse current state
        for fact in state:
            try:
                predicate, args = get_parts(fact)
                if predicate == 'on':
                    if len(args) == 2:
                        block_a, block_b = args
                        current_pos[block_a] = block_b
                        current_on[block_a] = block_b # Store direct on relationship
                    else:
                         print(f"Warning: Malformed 'on' state fact: {fact}")
                elif predicate == 'on-table':
                     if len(args) == 1:
                        block = args[0]
                        # Only set if not already set by 'on' (shouldn't happen)
                        if block not in current_pos:
                            current_pos[block] = 'table'
                     else:
                         print(f"Warning: Malformed 'on-table' state fact: {fact}")
                elif predicate == 'holding':
                    if len(args) == 1:
                        block = args[0]
                        current_pos[block] = 'arm'
                        held_block = block
                    else:
                        print(f"Warning: Malformed 'holding' state fact: {fact}")
            except ValueError as e:
                print(f"Warning: Skipping unparseable state fact '{fact}': {e}")


        misplaced_blocks: Set[str] = set()
        # Iterate over blocks that have a defined goal position
        for block in self.goal_pos:
            g_pos = self.goal_pos[block]

            # Determine current position, assuming 'table' if not explicitly mentioned
            # This handles blocks that might be clear on the table without an explicit on-table fact
            # if the state representation is minimal.
            c_pos = current_pos.get(block, 'table')

            if c_pos != g_pos:
                misplaced_blocks.add(block)

        blocks_on_misplaced: Set[str] = set()
        # current_on maps block_above -> block_below
        for block_above, block_below in current_on.items():
            if block_below in misplaced_blocks:
                blocks_on_misplaced.add(block_above)

        # Calculate heuristic value
        # Cost = 2 for each misplaced block + 2 for each block sitting on a misplaced block
        h = (2 * len(misplaced_blocks)) + (2 * len(blocks_on_misplaced))

        # Adjust if holding a block
        if held_block is not None:
            # The held block is guaranteed to be misplaced (pos='arm').
            # We counted 2 for it in the first term. Subtract 1 because pickup/unstack is done.
            # Ensure held_block was actually part of the goal config, otherwise it shouldn't affect cost.
            # The check `block in self.goal_pos` during misplaced calculation handles this.
            if held_block in misplaced_blocks:
                 h -= 1

        # Ensure heuristic is non-negative
        h = max(0, h)

        # If h is 0, does it mean goal is reached?
        # h=0 means misplaced_blocks is empty and blocks_on_misplaced is empty.
        # This implies current_pos[b] == goal_pos[b] for all b in goal_pos.
        # This satisfies all 'on' and 'on-table' goals.
        # It also implies held_block is None (or h would be -1 before max(0,h)).
        # So h=0 correctly identifies states satisfying the positional goals with an empty arm.

        return h
