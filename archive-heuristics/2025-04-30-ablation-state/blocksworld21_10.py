from heuristics.heuristic_base import Heuristic
# No need for fnmatch if we parse facts directly

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully, though valid states should conform
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the number
    of blocks that are not in their correct goal position (either on the table
    or on the correct block) plus the number of blocks that are in their correct
    goal position but have an incorrect block stacked directly on top of them.

    # Assumptions
    - The goal specifies the desired base for each block (either on the table
      or on a specific block).
    - The goal implicitly defines the required 'clear' predicates for the
      topmost blocks in the goal configuration.
    - Each block has at most one block directly on top of it at any time.
    - The heuristic assumes a well-formed state where blocks are either on the
      table or on another block, and stacking is one-to-one.

    # Heuristic Initialization
    - The heuristic pre-processes the goal facts to create two mappings:
        - `self.goal_base`: Maps each block to its required base in the goal
          ('table' or another block object).
        - `self.goal_blocks_on_top`: Maps each base (block or 'table') to the
          block that should be directly on top of it in the goal. This helps
          identify if a block has the wrong block on top or something on top
          when it should be clear.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  **Parse Current State:** Iterate through the facts in the current state
        to build two temporary mappings:
        - `current_base`: Maps each block to its current base ('table' or
          another block object).
        - `current_blocks_on_top`: Maps each base (block or 'table') to the
          block currently directly on top of it.

    2.  **Initialize Heuristic Value:** Set `h = 0`.

    3.  **Evaluate Each Block's Position:** Iterate through all blocks that
        have a specified goal position (i.e., blocks present as the first
        argument in a goal `on` or `on-table` fact).

    4.  **Check Immediate Base:** For each block `b`, compare its current base
        (`current_base.get(b)`) with its goal base (`self.goal_base[b]`).
        - If the block's current base is different from its goal base (or if
          the block's current base cannot be determined from the state facts,
          indicating a potentially malformed state like a floating block),
          increment `h` by 1. This block is misplaced relative to its base.

    5.  **Check Block on Top (if Base is Correct):** If the block `b` is
        currently on its correct goal base (`current_base.get(b) == self.goal_base[b]`),
        check the block directly on top of `b`.
        - Get the block currently on top of `b` (`current_blocks_on_top.get(b)`).
        - Get the block that should be on top of `b` in the goal
          (`self.goal_blocks_on_top.get(b)`). Note that if nothing should be
          on top (i.e., `b` should be clear), the goal map will return `None`.
        - If the block currently on top is different from the block that should
          be on top in the goal, increment `h` by 1. This covers cases where
          `b` is in the right place, but is either blocked by a wrong block
          or has something on it when it should be clear.

    6.  **Return Total Cost:** The final value of `h` is the estimated number
        of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block.
        """
        self.goals = task.goals  # Goal conditions.
        # Blocksworld domain does not typically have static facts relevant to the heuristic structure.
        # static_facts = task.static # Not used in this heuristic

        # Store goal locations for each block.
        # goal_base[block] = base_block_or_'table'
        self.goal_base = {}
        # goal_blocks_on_top[base_block_or_'table'] = block_on_top
        self.goal_blocks_on_top = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == "on":
                # Goal: (on b1 b2) -> b1 should be on b2
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_base[block] = base
                    self.goal_blocks_on_top[base] = block # Assuming one block directly on another
            elif predicate == "on-table":
                # Goal: (on-table b1) -> b1 should be on the table
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_base[block] = 'table'
                    # Note: Multiple blocks can be on the table, but goal_blocks_on_top['table']
                    # as defined here only stores the *last* block processed that should be
                    # directly on the table *if* the table were treated as a single stack point.
                    # However, the heuristic logic only uses goal_blocks_on_top to check what
                    # should be on top of a *block* base, or if a block on the table should be clear.
                    # For 'table' as a base, goal_blocks_on_top['table'] is not meaningful in this heuristic.
                    # The check `current_blocks_on_top.get(b) != self.goal_blocks_on_top.get(b)`
                    # for b='table' will compare current blocks on table (which we don't track
                    # as a single entity in current_blocks_on_top) with goal blocks on table
                    # (which isn't stored in goal_blocks_on_top this way).
                    # Let's refine: goal_blocks_on_top should only map blocks to blocks.
                    # If goal is (on-table b), goal_blocks_on_top.get(b) should be None.
                    pass # No entry in goal_blocks_on_top for blocks on table

        # Refine goal_blocks_on_top to only map block -> block
        self.goal_blocks_on_top = {}
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts:
                 continue
             if parts[0] == "on" and len(parts) == 3:
                 block, base = parts[1], parts[2]
                 self.goal_blocks_on_top[base] = block


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse current state to build current base and on-top maps
        current_base = {} # block -> base_block_or_'table'
        current_blocks_on_top = {} # base_block -> block_on_top (only for blocks, not table)
        is_clear = {} # block -> True/False

        # Initialize all blocks as not clear by default, update if clear(b) is found
        # Need to know all blocks first. Let's collect them from state and goals.
        all_blocks = set(self.goal_base.keys())

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]

            if predicate == "on":
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    current_base[block] = base
                    current_blocks_on_top[base] = block # Assuming one block directly on another
                    all_blocks.add(block)
                    all_blocks.add(base)
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_base[block] = 'table'
                    all_blocks.add(block)
            elif predicate == "clear":
                 if len(parts) == 2:
                    block = parts[1]
                    is_clear[block] = True
                    all_blocks.add(block)
            elif predicate == "holding":
                 if len(parts) == 2:
                    block = parts[1]
                    # A block being held is not on anything, nor is anything on it.
                    # Its base is effectively the arm, but for this heuristic,
                    # we can treat it as not being on a table or block.
                    # It's also not clear.
                    all_blocks.add(block)
            # Ignore arm-empty

        # Ensure all blocks seen are in is_clear map (defaulting to False)
        for block in all_blocks:
             is_clear.setdefault(block, False)


        total_cost = 0  # Initialize action cost counter.

        # Heuristic: Count blocks not on correct base + blocks on correct base but with wrong block on top
        # We only care about blocks that have a specified goal position
        for block, goal_b in self.goal_base.items():
            current_b = current_base.get(block) # Get current base, None if not found

            # Part 1: Block is not on its correct immediate base
            if current_b is None or current_b != goal_b:
                total_cost += 1
            else:
                # Part 2: Block is on correct base, but the block on top is wrong
                # This applies if the base is another block, or if the base is the table
                # and the block should be clear but isn't.
                # Check what block should be on top of 'block' in the goal
                goal_top = self.goal_blocks_on_top.get(block) # None if 'block' should be clear
                # Check what block is currently on top of 'block'
                current_top = current_blocks_on_top.get(block) # None if 'block' is clear

                if current_top != goal_top:
                     total_cost += 1


        # The heuristic should be 0 iff the goal is reached.
        # The current calculation counts mismatches. If all goal facts are true,
        # then for every block b in goal_base, current_base[b] == goal_base[b]
        # and current_blocks_on_top.get(b) == goal_blocks_on_top.get(b).
        # So total_cost will be 0.
        # If total_cost is 0, it means for every block b in goal_base,
        # current_base[b] == goal_base[b] AND current_blocks_on_top.get(b) == goal_blocks_on_top.get(b).
        # This implies all on(x,y) and on-table(x) goal facts are met.
        # It also implies that for any block b that should be clear (goal_blocks_on_top.get(b) is None),
        # current_blocks_on_top.get(b) is also None, meaning clear(b) is true.
        # So, h=0 iff all on, on-table, and clear goal facts are met.
        # What about arm-empty? The goal might require arm-empty.
        # If arm-empty is a goal, and the arm is not empty, the heuristic is 0
        # but the goal is not reached.
        # Let's add 1 if arm-empty is a goal and the arm is not empty.

        arm_empty_goal = "(arm-empty)" in self.goals
        arm_empty_state = "(arm-empty)" in state

        if arm_empty_goal and not arm_empty_state:
             total_cost += 1 # Need to put down whatever is held

        # Ensure heuristic is 0 for goal state, even if other predicates are true
        # that weren't explicitly counted (like clear(x) for non-top blocks).
        # The check h=0 iff goal state above seems mostly correct, except for arm-empty.
        # Let's add the arm-empty check.

        # Final check: if the state is the goal state, the heuristic must be 0.
        # This is guaranteed by the check h=0 iff goal state, provided the goal
        # only contains on, on-table, and clear predicates for top blocks, and arm-empty.
        # If the goal contains other predicates, this heuristic might be 0 when
        # the goal is not fully reached. However, the provided examples only have
        # on, on-table, and clear goals. Let's trust the current logic covers
        # the typical blocksworld goals.

        return total_cost

