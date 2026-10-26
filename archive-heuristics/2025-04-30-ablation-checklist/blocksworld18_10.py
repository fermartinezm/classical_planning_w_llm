from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state by counting the number of blocks that are not in their correct
    goal position, plus the number of blocks that are currently on top of
    blocks that are not in their correct goal position.

    # Heuristic Initialization
    - Extract the goal position (either on another block or on the table)
      for each block mentioned in the goal state.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1. Parse Goal State:
       - Create a mapping (dictionary) from each block to its desired base
         in the goal state. The base is either another block (if the goal is
         `(on block base)`) or the string 'table' (if the goal is
         `(on-table block)`).

    2. Parse Current State:
       - Create a mapping (dictionary) from each block to its current base
         in the state.
       - Create a mapping (dictionary) representing the current stack structure,
         e.g., `on_top[base_block] = block_on_top` if `(on block_on_top base_block)`
         is true. This helps identify blockers.

    3. Identify Misplaced Blocks:
       - Iterate through all blocks for which a goal position is defined.
       - If a block's current base is different from its goal base, mark this
         block as "misplaced".
       - Count the total number of misplaced blocks. This is the initial
         heuristic value.

    4. Identify Blocking Blocks:
       - Iterate through the current stack structure (the `on_top` mapping).
       - For each block `base_block` that has a block `top_block` on it
         (i.e., `on_top[base_block] = top_block`), check if `base_block` is
         in the set of "misplaced" blocks.
       - If `base_block` is misplaced, then `top_block` is blocking it from
         being moved to its correct position. Count `top_block` as a "blocking"
         block. A block can only block one block directly below it, but a block
         can be blocked by multiple blocks above it in a stack. We only need
         to count each *blocking* block once.

    5. Sum Costs:
       - The total heuristic value is the sum of the number of misplaced blocks
         and the number of blocking blocks. Each misplaced block needs to be
         moved. Each blocking block needs to be moved *out of the way* before
         the block below it can be moved. This heuristic is non-admissible
         but provides a reasonable estimate of the "disorder" relative to the
         goal structure.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal position for each block.
        """
        self.goals = task.goals

        # Map each block to its desired base in the goal state ('table' or another block).
        # We assume each block has exactly one goal position defined by either (on X Y) or (on-table X).
        self.goal_base = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_base[block] = base
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_base[block] = 'table'
            # We ignore clear and arm-empty goals for this heuristic

        # Get the set of all blocks mentioned in the goals
        self.goal_blocks = set(self.goal_base.keys())
        # Also include blocks that are bases in the goal, even if they aren't on top of something else
        for base in self.goal_base.values():
            if base != 'table':
                self.goal_blocks.add(base)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Parse Current State
        current_base = {} # Maps block -> its base ('table' or another block)
        current_on_top = {} # Maps base_block -> block_on_top (only the direct one)
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    current_base[block] = base
                    current_on_top[base] = block # Store who is directly on top
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_base[block] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                    held_block = parts[1]

        # If a block is held, its current base is the arm.
        if held_block:
             current_base[held_block] = 'arm' # Use 'arm' as a special base

        # Consider blocks that are in the state but not in the goal_base mapping.
        # These blocks might be on top of goal blocks and need to be moved.
        # We need to know their current base to identify misplaced blocks correctly.
        # Any block in the state that isn't in current_base must be the base of a stack on the table.
        all_blocks_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in ["on", "on-table", "holding", "clear"]:
                 if len(parts) > 1:
                     all_blocks_in_state.add(parts[1])
             # For (on ?x ?y), ?y is also a block
             if parts[0] == "on" and len(parts) == 3:
                 all_blocks_in_state.add(parts[2])

        # Ensure all blocks mentioned in the state have a current_base entry.
        # Blocks that are bases of stacks on the table won't appear as the first argument
        # of 'on' or 'on-table', but will appear as the second argument of 'on' or in 'clear'.
        # We can infer their base if they don't have one yet and aren't held.
        for block in all_blocks_in_state:
             if block not in current_base and block != held_block:
                 # If a block is not in current_base and not held, it must be on the table.
                 # This handles cases where a block is the base of a stack but not explicitly (on-table X)
                 # in the initial state if it has something on it. However, PDDL initial states
                 # usually explicitly state (on-table X) for stack bases. Let's rely on the explicit facts.
                 pass # The current_base and current_on_top maps are built from explicit facts.

        # 2. Identify Misplaced Blocks
        wrong_pos_blocks = set()
        # Only consider blocks that have a defined goal position
        for block in self.goal_blocks:
            # If a block is in the goal, it must have a goal base.
            # If it's not in the current state, it's misplaced (or held).
            # If it's in the current state, check its base.
            goal_b = self.goal_base.get(block) # Use .get() in case a block is in goal_blocks but not goal_base (e.g. a base block)
            current_b = current_base.get(block)

            # A block is misplaced if its current base is different from its goal base.
            # We only care about blocks that are in the goal structure.
            if goal_b is not None: # Block has a defined goal base
                 if current_b != goal_b:
                     wrong_pos_blocks.add(block)
            # What about blocks in the state but not in the goal? They don't have a 'goal_base'.
            # They might be blocking goal blocks. We handle this in the next step.

        # Initial heuristic: count of blocks that are in the goal structure but not in their goal position.
        h = len(wrong_pos_blocks)

        # 3. Identify Blocking Blocks
        # A block B is blocking if (on B A) is true in the state AND A is a misplaced block.
        blocking_blocks = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "on" and len(parts) == 3:
                top_block, base_block = parts[1], parts[2]
                if base_block in wrong_pos_blocks:
                    blocking_blocks.add(top_block)

        # Add the count of unique blocking blocks to the heuristic.
        h += len(blocking_blocks)

        # A block being held might also be considered 'misplaced' if it's not the block
        # that should be held in the goal (though holding goals are rare/implicit)
        # or if it's the wrong block to be holding for the next step towards the goal.
        # For simplicity in this heuristic, we just count blocks in wrong positions
        # and those blocking them. A held block will likely be put down or stacked,
        # contributing to reducing the 'misplaced' count later. We don't add a specific
        # cost for the held block itself unless it's already counted as misplaced
        # because its *eventual* goal base is wrong.

        return h

