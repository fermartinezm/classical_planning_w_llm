# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define dummy Heuristic base class if running standalone for testing
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are either in the wrong
    position relative to their support (table or another block) or have the
    wrong block on top of them (or should be clear but are not).

    # Heuristic Initialization
    - Parse goal facts to determine the desired support for each block and
      which blocks should be clear or have a specific block on top.
    - Identify all blocks involved in the problem (from initial state and goal).
    - For blocks not explicitly placed in the goal, assume they should be on
      the table and clear.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the goal configuration: For each block, determine what it should
       be directly on top of (or the table) and what should be directly on top
       of it (or if it should be clear).
    2. Identify the current configuration: For each block, determine what it is
       currently directly on top of (or the table) and what is currently
       directly on top of it (if anything).
    3. Initialize heuristic value to 0.
    4. For each block:
       a. Check if its current support (what it's on) matches its goal support.
          If not, increment the heuristic by 1.
       b. If the current support matches the goal support, check what is
          currently on top of this block.
       c. If something is currently on top:
          i. If the block should be clear in the goal, increment the heuristic by 1.
          ii. If the block should have a specific block on top in the goal,
              and the current block on top is not that specific block,
              increment the heuristic by 1.
    5. The total heuristic value is the sum of these increments.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and identifying all blocks.
        """
        super().__init__(task)

        self.goal_support = {} # block -> support_block or 'table'
        self.goal_on_top = {}  # support_block -> block_on_top (only for blocks)
        self.goal_clear_set = set() # blocks that should be clear

        all_blocks = set()

        # Parse explicit goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_on_top[support] = block # Map support block to the block on top
                all_blocks.add(block)
                all_blocks.add(support)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                all_blocks.add(block)
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                self.goal_clear_set.add(block)
                all_blocks.add(block)
            # Ignore other goal predicates if any

        # Collect blocks from initial state that might not be in goal facts
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts

             predicate = parts[0]
             if predicate in ["on", "on-table", "clear", "holding"] and len(parts) >= 2:
                 all_blocks.add(parts[1])
                 if predicate == "on" and len(parts) == 3:
                     all_blocks.add(parts[2])

        self.all_blocks = list(all_blocks) # Store as list for consistent iteration order

        # Add implicit goals for blocks not explicitly placed
        for block in self.all_blocks:
            if block not in self.goal_support:
                # If a block is not mentioned in any goal (on X Y) or (on-table X)
                # fact, assume it should be on the table and clear.
                self.goal_support[block] = 'table'
                self.goal_clear_set.add(block)
                # Ensure it's not expected to be on top of anything in the goal
                # (already handled by goal_on_top only getting 'on' facts)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        current_support = {} # block -> support_block or 'table' or 'arm'
        current_on_top = {}  # support_block -> block_on_top (only for blocks)
        current_holding = None

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
                current_on_top[support] = block # Map support block to the block on top
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
                # No entry in current_on_top for 'table' as it's not a block
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'arm'
                current_holding = block
            # Ignore clear and arm-empty for these maps

        h = 0

        # Compute heuristic based on misplacement
        for block in self.all_blocks:
            current_sup = current_support.get(block)
            goal_sup = self.goal_support.get(block) # Should always exist due to implicit goals

            # Check if the block's support is incorrect
            if current_sup != goal_sup:
                h += 1
            else: # Current support matches goal support
                # Check if something is on top that shouldn't be
                # Find block directly on top of 'block' in the current state
                block_on_top = current_on_top.get(block)

                if block_on_top is not None:
                    # Something is on top of 'block'
                    if block in self.goal_clear_set:
                        # 'block' should be clear in the goal, but has something on top
                        h += 1
                    else:
                        # 'block' should have a specific block on top in the goal
                        goal_block_on_top = self.goal_on_top.get(block)
                        # If goal_block_on_top is None, it means 'block' should have nothing
                        # on top in the goal (and is not in goal_clear_set - this implies
                        # a malformed goal where a non-clear block is not the base of a stack).
                        # Assuming well-formed goals, if block is not in goal_clear_set,
                        # goal_on_top.get(block) will return the block that should be on top.
                        if block_on_top != goal_block_on_top:
                            # 'block' has something on top, its support is correct,
                            # but the block on top is the wrong one.
                            h += 1
                # Else: block_on_top is None. If block is in goal_clear_set, this is correct.
                # If block is not in goal_clear_set, it should have something on top
                # in the goal. The block that should be on top is misplaced, and
                # will be counted when processing that block. No penalty here.


        return h
