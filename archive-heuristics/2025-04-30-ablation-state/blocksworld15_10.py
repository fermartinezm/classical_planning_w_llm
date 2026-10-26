# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential malformed facts gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting blocks that are either not in their correct goal position relative
    to their support, or are blocking other blocks that are correctly supported.
    Each such "problematic" block is estimated to require at least 2 actions
    (e.g., unstack/pickup + putdown/stack) to resolve its issue.

    # Assumptions
    - The goal state specifies a unique position (on another block or on the table)
      for every block involved in the goal.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack).
    - Assumes the cost of each action is 1.
    - Assumes block names start with 'b'.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired support for each block
      and which block should be on top of which in the goal state.
    - Collects all block names mentioned in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Parse Goal:** Iterate through the goal facts (`task.goals`).
       - Build `goal_support`: a dictionary mapping each block `X` to its desired support (`Y` if `(on X Y)` is a goal, or the string 'table' if `(on-table X)` is a goal).
       - Build `goal_block_on_top`: a dictionary mapping each block `Y` to the block `X` that should be directly on top of it in the goal (`X` if `(on X Y)` is a goal).
       - Collect all block names mentioned in the goal facts into `self.all_blocks`.

    2. **Parse Current State:** Iterate through the current state facts (`node.state`).
       - Build `current_support`: a dictionary mapping each block `X` to its current support (`Y` if `(on X Y)` is true, the string 'table' if `(on-table X)` is true, or the string 'held' if `(holding X)` is true).
       - Build `current_block_on_top`: a dictionary mapping each block `Y` to the block `X` that is currently directly on top of it (`X` if `(on X Y)` is true).
       - Identify `current_holding`: the block currently held by the arm, or `None`.
       - Collect all block names mentioned in the current state facts and add them to `self.all_blocks`.

    3. **Identify Problematic Blocks:** Initialize an empty set `problematic_blocks`.
       - If the arm is holding a block (`current_holding` is not `None`), add the held block to `problematic_blocks`. This block needs to be placed.
       - Iterate through all blocks collected in steps 1 and 2 (`self.all_blocks`). For each block `X`:
         - Get its goal support (`gs = self.goal_support.get(X)`). If a block doesn't have a goal support defined (e.g., it's not mentioned in `on` or `on-table` goals), it's likely irrelevant or implicitly can be anywhere; we skip checking its support.
         - Get its current support (`cs = current_support.get(X)`).
         - If `X` is not currently held (`cs != 'held'`) AND it has a defined goal support (`gs is not None`) AND its current support is different from its goal support (`cs != gs`), then `X` is misplaced. Add `X` to `problematic_blocks`.
         - Find the block `Z` currently on top of `X` (`blocker = current_block_on_top.get(X)`).
         - If there is a block `Z` on top of `X` (`blocker is not None`), find the block that is supposed to be on top of `X` in the goal (`goal_block_above_X = self.goal_block_on_top.get(X)`).
         - If `Z` is on top of `X` but `Z` is *not* the block that should be on top of `X` in the goal (`blocker != goal_block_above_X`), then `Z` is blocking `X`. Add `Z` to `problematic_blocks`.

    4. **Compute Heuristic Value:** The heuristic value is 2 times the number of unique blocks in the `problematic_blocks` set. This counts each problematic block once and estimates that moving it requires roughly 2 actions (e.g., unstack/pickup + putdown/stack).

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and relationships.
        """
        self.goals = task.goals

        # Data structures to store goal information
        self.goal_support = {} # block -> block_under or 'table'
        self.goal_block_on_top = {} # block_under -> block_on_top
        self.all_blocks = set() # Collect all block names mentioned in goals

        # Parse goal facts to build goal_support and goal_block_on_top
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on, block_under = parts[1], parts[2]
                    self.goal_support[block_on] = block_under
                    self.goal_block_on_top[block_under] = block_on
                    self.all_blocks.add(block_on)
                    self.all_blocks.add(block_under)
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_support[block] = 'table'
                    self.all_blocks.add(block)
            # Ignore other goal predicates like (clear ?x) or (arm-empty) for this heuristic


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Data structures to store current state information
        current_support = {} # block -> block_under or 'table' or 'held'
        current_block_on_top = {} # block_under -> block_on_top
        current_holding = None
        current_blocks = set() # Collect all block names in the current state

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on, block_under = parts[1], parts[2]
                    current_support[block_on] = block_under
                    current_block_on_top[block_under] = block_on
                    current_blocks.add(block_on)
                    current_blocks.add(block_under)
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_support[block] = 'table'
                    current_blocks.add(block)
            elif predicate == "holding":
                 if len(parts) == 2:
                    block = parts[1]
                    current_holding = block
                    current_blocks.add(block)
            elif predicate in ["clear", "arm-empty"]:
                # These predicates are not needed for the core heuristic logic
                pass
            else:
                 # Add any other objects mentioned in the state facts that look like blocks
                 # This is a heuristic way to find all blocks if not all are in goals
                 for part in parts[1:]:
                     # Simple check if it looks like an object name (starts with b)
                     # This assumes block names start with 'b'. A more robust parser
                     # would get object types from the domain/instance file.
                     # Given the examples, 'b' prefix is a reasonable heuristic assumption.
                     if isinstance(part, str) and part.startswith('b'):
                         current_blocks.add(part)


        # Combine blocks from goal (collected in __init__) and current state
        all_blocks = self.all_blocks.union(current_blocks)

        problematic_blocks = set()

        # 1. Add the held block (if any) to the set.
        if current_holding is not None:
            problematic_blocks.add(current_holding)

        # 2. Identify blocks that are misplaced or blocking.
        for block in all_blocks:
            gs = self.goal_support.get(block)
            cs = current_support.get(block)

            # If block has a goal position defined
            if gs is not None:
                # If block is not held and its current support is wrong
                if cs != 'held' and cs != gs:
                    problematic_blocks.add(block)

            # Check if there's a block on top of this block that shouldn't be there.
            blocker = current_block_on_top.get(block)
            if blocker is not None:
                goal_block_above = self.goal_block_on_top.get(block)
                # If there is a blocker, and it's not the block that should be there in the goal
                if blocker != goal_block_above:
                    problematic_blocks.add(blocker)

        # The heuristic value is 2 times the number of unique problematic blocks.
        # Each problematic block likely requires at least 2 actions (clear/pickup + place).
        h_value = 2 * len(problematic_blocks)

        return h_value
