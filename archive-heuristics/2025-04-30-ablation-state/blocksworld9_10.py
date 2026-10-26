from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential leading/trailing whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Handle unexpected fact format, maybe raise error or log warning
         # Assuming valid PDDL facts for now
         return [] # Or raise ValueError(f"Invalid fact format: {fact)")
    return fact_str[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position relative to the block directly below them (or the table)
    OR do not have the correct block directly on top of them (or should be clear),
    as defined by the goal state. It counts how many blocks relevant to the goal
    are "misplaced" in terms of their immediate parent or child in the stack structure.

    # Assumptions:
    - The goal state defines the desired stack structure using `(on ?x ?y)`,
      `(on-table ?x)`, and `(clear ?x)` predicates.
    - Blocks mentioned in the goal predicates are the relevant blocks for the heuristic.
    - The heuristic counts a block as contributing to the cost if its current
      parent (what it's on) is different from its goal parent, OR if its current
      child (what's on it) is different from its goal child (or if it should be clear but isn't).

    # Heuristic Initialization
    - Extracts the desired parent (`goal_parent`) and child (`goal_child`) for
      each block mentioned in the goal predicates.
      - `goal_parent[B]` is the block `Y` if `(on B Y)` is a goal, or 'table'
        if `(on-table B)` is a goal. It's undefined otherwise.
      - `goal_child[B]` is the block `X` if `(on X B)` is a goal. If no block
        `X` has `(on X B)` as a goal, then `goal_child[B]` is 'clear'.
    - Identifies the set of all blocks (`self.goal_blocks`) mentioned in any
      goal predicate.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost `total_cost` to 0.
    2. Create dictionaries `current_pos` and `current_child` to represent the
       current stack structure. Also track all blocks encountered in the state.
       Iterate through the facts in the current state:
       - If a fact is `(on ?x ?y)`, record `current_pos[?x] = ?y` and
         `current_child[?y] = ?x`. Add `?x` and `?y` to the set of blocks in state.
       - If a fact is `(on-table ?x)`, record `current_pos[?x] = 'table'`. Add `?x` to the set of blocks in state.
       - If a fact is `(holding ?x)`, record `current_pos[?x] = 'holding'`. Add `?x` to the set of blocks in state.
       - Ignore `(clear ?x)` and `(arm-empty)` facts for building pos/child maps directly.
    3. Infer `current_child` for blocks that are clear in the current state.
       A block `B` is clear if it is present in the state but is not a key
       in the `current_child` map (meaning nothing is currently on it).
       For such blocks, record `current_child[B] = 'clear'`.
    4. Iterate through each block `B` in the set `self.goal_blocks` (all blocks
       relevant to the goal).
    5. For each block `B`, retrieve its goal parent `gp` from `self.goal_parent`
       (defaults to `None` if not specified by `on`/`on-table` goals).
    6. For each block `B`, retrieve its goal child `gc` from `self.goal_child`.
    7. Retrieve the block's current parent `cp` from `current_pos` (defaults
       to `None` if not found, e.g., not on anything, table, or holding).
    8. Retrieve the block's current child `cc` from `current_child` (defaults
       to `None` if not found, e.g., nothing is on it, or it's holding).
    9. Check if the block is "misplaced" relative to its parent OR child:
       - If `gp` is defined and `cp != gp`, increment `total_cost`.
       - If `gc` is defined and `cc != gc`, increment `total_cost`.
       Note: A block contributes at most 2 to the cost (1 for parent mismatch, 1 for child mismatch).
    10. Return the final `total_cost` as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal parent and child relationships.
        """
        self.goals = task.goals

        self.goal_parent = {} # {block: parent_block or 'table'}
        self.goal_child = {} # {block: child_block or 'clear'}
        self.goal_blocks = set() # All blocks mentioned in goal predicates

        # Collect all blocks mentioned in goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            # Add all arguments as potential goal blocks
            for part in parts[1:]:
                 # Basic check if it looks like an object name (starts with letter)
                 if part and part[0].isalpha():
                    self.goal_blocks.add(part)

        # Determine goal_parent from 'on' and 'on-table' goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_parent[block] = 'table'

        # Determine goal_child for all goal_blocks
        # A block B's goal_child is X if (on X B) is a goal. Otherwise, it's 'clear'.
        for block in list(self.goal_blocks): # Iterate over a copy as we might add blocks
             # Find if any block X has (on X block) as a goal
             child = None
             for goal in self.goals:
                 parts = get_parts(goal)
                 if parts and parts[0] == "on" and len(parts) == 3 and parts[2] == block:
                     child = parts[1]
                     break # Found the block that should be on top

             if child is not None: # Some block X should be on B
                 self.goal_child[block] = child
                 # Ensure the child block is also considered if it wasn't already
                 if child and child[0].isalpha():
                     self.goal_blocks.add(child)
             else: # No block should be on B according to 'on' goals
                 self.goal_child[block] = 'clear'


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Track current position and child for all blocks in the state
        current_pos = {} # {block: parent_block or 'table' or 'holding'}
        current_child = {} # {block: child_block or 'clear'}
        all_blocks_in_state = set() # All blocks mentioned as subject or object in relevant facts

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, parent = parts[1], parts[2]
                current_pos[block] = parent
                current_child[parent] = block
                all_blocks_in_state.add(block)
                all_blocks_in_state.add(parent)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'table'
                all_blocks_in_state.add(block)
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'holding'
                all_blocks_in_state.add(block)
            # Ignore (clear ?x) and (arm-empty) facts for building pos/child maps directly.

        # Infer current_child for blocks that are clear
        # A block B is clear if it is present in the state but is not a key
        # in the current_child map (meaning nothing is currently on it).
        blocks_that_are_parents_in_state_on = set(current_child.keys())
        blocks_that_are_clear_in_state = all_blocks_in_state - blocks_that_are_parents_in_state_on

        for block in blocks_that_are_clear_in_state:
             current_child[block] = 'clear'

        total_cost = 0
        # Iterate over all blocks that are relevant in the goal
        for block in self.goal_blocks:
            gp = self.goal_parent.get(block) # Can be None if block only appears in (clear B) goal or as parent in (on X B)
            gc = self.goal_child.get(block) # Can be 'clear' or a block name

            # Get current position and child, default to None if block is not in state
            cp = current_pos.get(block)
            cc = current_child.get(block) # Can be 'clear' or a block name or None

            # Check if parent is wrong OR child is wrong
            parent_wrong = (gp is not None and cp != gp)
            child_wrong = (gc is not None and cc != gc) # gc is always defined for goal_blocks

            if parent_wrong or child_wrong:
                total_cost += 1

        return total_cost
