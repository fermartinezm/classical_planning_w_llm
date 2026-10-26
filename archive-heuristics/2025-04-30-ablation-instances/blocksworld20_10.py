from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Define helper functions outside the class as they are general
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential leading/trailing whitespace and empty fact strings
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
         # Depending on expected input, could return [], raise error, or log warning
         # Assuming valid PDDL facts from the parser
         return [] # Or maybe None, but [] is safer for unpacking
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position within the goal stacks. A block is considered "in place"
    if it is on the table and should be, or if it is on the correct block
    and the block below it is also "in place". The heuristic is the total
    number of blocks that have a goal position defined for them, minus the
    number of those blocks that are currently "in place".

    # Assumptions
    - The goal state defines a set of valid stacks (or blocks on the table),
      specifying the position (on another block or on the table) for every
      block involved in the goal configuration.
    - The problem is solvable.
    - The internal state representation uses strings for facts like "(predicate arg1 arg2)".

    # Heuristic Initialization
    - Parse the goal facts (`task.goals`) to determine the desired position
      for each block (either on the table or on another specific block).
      This information is stored in `self.goal_pos`, mapping a block to the
      entity directly below it in the goal state ('table' or another block name).
    - Identify all unique blocks mentioned as the *subject* of an 'on' or
      'on-table' goal fact. These are the blocks whose final positions are
      explicitly specified in the goal and contribute to the heuristic count.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (`node.state`):

    1. Parse the current state facts to determine the current position of each
       block (on another block, on the table, or held by the arm). Store this
       in a dictionary `current_pos`.
    2. Initialize an empty set `in_place_blocks`. This set will store blocks
       that are determined to be in their correct goal position relative to
       what's below them, recursively defined.
    3. **Base Case for "In Place":** Iterate through all blocks that have a
       goal position defined (`self.blocks_with_goal_pos`). If a block `b`'s
       goal position is the 'table' (`self.goal_pos[b] == 'table'`), check if
       `(on-table b)` is true in the current state (i.e., `current_pos.get(b) == 'table'`).
       If it is, add `b` to `in_place_blocks`.
    4. **Recursive Step for "In Place":** Repeatedly iterate until no new blocks
       are added to `in_place_blocks` in a full pass:
       Initialize a temporary set `added_this_pass`.
       For each block `b` that has a goal position defined and is not yet in
       `in_place_blocks`:
         Let `under_block_goal` be the goal position for `b` (`self.goal_pos[b]`).
         If `under_block_goal` is not 'table':
           Check if `b` is currently on `under_block_goal` (`current_pos.get(b) == under_block_goal`)
           AND if `under_block_goal` is already in `in_place_blocks`.
           If both conditions are met, add `b` to `added_this_pass`.
       Add all blocks from `added_this_pass` to `in_place_blocks`. If `added_this_pass`
       was empty, the iteration stops.
    5. The heuristic value is the count of blocks that have a goal position
       defined for them (`self.blocks_with_goal_pos`) but are *not* present in the
       final `in_place_blocks` set.
    6. In the goal state, all blocks with goal positions will satisfy the
       "in place" definition, resulting in a heuristic value of 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and identifying
        the set of blocks whose goal positions are specified.
        """
        self.goals = task.goals
        self.static = task.static # Static facts are empty in blocksworld

        # Determine the goal position for each block.
        # goal_pos[block] = block_below_it (or 'table')
        self.goal_pos = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                self.goal_pos[block] = under_block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_pos[block] = 'table'
            # Ignore (clear ...) and (arm-empty) goals for goal_pos mapping

        # The set of blocks whose final position is explicitly defined in the goal.
        # These are the blocks we count towards the heuristic if they are not "in place".
        self.blocks_with_goal_pos = set(self.goal_pos.keys())


    def __call__(self, node):
        """Compute the number of blocks not in their goal-relative position."""
        state = node.state

        # Map current position for each block
        current_pos = {}
        # We don't need to track clear, holding, arm-empty for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                current_pos[block] = under_block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'table'
            # Facts like (holding b1) mean b1 is not on anything or table,
            # so it won't be in current_pos unless we explicitly add it,
            # but the .get(block) == target check handles this (returns None).


        # --- Compute "in place" blocks ---
        in_place_blocks = set()

        # Step 3: Base Case for "In Place" (Blocks on table in goal)
        # Iterate only over blocks that have a goal position defined
        for block in self.blocks_with_goal_pos:
             if self.goal_pos[block] == 'table':
                 # Check if it's currently on the table
                 if current_pos.get(block) == 'table':
                     in_place_blocks.add(block)

        # Step 4: Recursive Step for "In Place" (Blocks on other blocks in goal)
        while True:
            added_this_pass = set()
            # Iterate only over blocks that have a goal position defined
            for block in self.blocks_with_goal_pos:
                if block not in in_place_blocks:
                    under_block_goal = self.goal_pos[block]
                    # Check if the goal is to be on another block
                    if under_block_goal != 'table':
                        # Check if it's currently on the correct block
                        if current_pos.get(block) == under_block_goal:
                            # Check if the block below it is in place
                            if under_block_goal in in_place_blocks:
                                added_this_pass.add(block)

            if not added_this_pass:
                break # No new blocks added, convergence
            in_place_blocks.update(added_this_pass)

        # Step 5 & 6: Heuristic is the count of blocks with goal positions that are NOT in place.
        not_in_place_count = len(self.blocks_with_goal_pos) - len(in_place_blocks)

        # The heuristic value should be 0 if and only if the state is a goal state.
        # The "in place" definition ensures that if all goal facts (on/on-table) are true,
        # all blocks with goal positions will be marked as in place, resulting in 0.
        # If the state is not a goal state, at least one goal fact is false.
        # If an (on/on-table) goal fact is false for a block with a goal position,
        # that block cannot be marked "in place" unless it's the base of a correct stack
        # built on a block that *is* in place, which contradicts the goal being false.
        # Thus, at least one block with a goal position will not be in place, and the
        # heuristic will be > 0.

        return not_in_place_count
