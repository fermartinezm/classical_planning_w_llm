# from fnmatch import fnmatch # Not used in the final version
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their correct goal position, blocks that
    are currently on top of a block that is not in its correct goal position,
    and blocks that are required to be clear in the goal state but are not.
    It assigns costs based on the estimated effort to correct these issues.

    # Assumptions:
    - The goal specifies the desired position (on another block or on the table)
      for a subset of blocks using `(on X Y)` or `(on-table Z)` predicates.
    - The goal may specify that certain blocks must be clear using `(clear C)` predicates.
    - Blocks not mentioned in goal 'on' or 'on-table' predicates do not have a
      specific required final position, but might still need to be moved if they
      are blocking a block that *does* have a required position.
    - The cost estimates (2 for moving a block to its goal position, 1 for unstacking
      a block from a misplaced block, 1 for clearing a block that needs to be clear
      in the goal) are rough approximations designed to guide search effectively,
      not to be admissible.

    # Heuristic Initialization
    - Parses the goal facts (`task.goals`) to identify the target support for each block
      (`goal_below`, `goal_on_table`) and blocks that must be clear (`goal_clear`).
    - Parses the initial state facts (`task.init`) to collect all unique objects
      present in the problem instance. This ensures all blocks are considered
      during heuristic computation, even if they are not explicitly mentioned
      in the goal facts (though they must be in the initial state facts).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize heuristic value `h` to 0.
    2. Parse the current state (`node.state`) to determine the current support
       for each block (`current_below`, `current_on_table`, `current_holding`)
       and which blocks are clear (`current_clear`). Also build `current_above`
       (the inverse mapping of `current_below`) to quickly find blocks on top.
    3. Identify "misplaced" blocks and add cost: Iterate through all objects
       identified during initialization. For each block, determine its goal support
       (if specified in `goal_below` or `goal_on_table`) and its current support
       (from `current_below`, `current_on_table`, or `current_holding`).
       If the block has a goal support specified and its current support is different,
       add the block to a set of `misplaced_blocks` and add 2 to `h`. This cost
       estimates the actions needed to move the block (e.g., unstack/pickup + stack/putdown).
    4. Add penalty for blocks blocking misplaced blocks: Iterate through the set
       of `misplaced_blocks`. For each misplaced block, check if there is a block
       currently on top of it using the `current_above` mapping. If a block is
       found on top, add 1 to `h`. This cost estimates the action needed to unstack
       the blocking block.
    5. Add penalty for unclear goal blocks: Iterate through the set of blocks
       that must be clear in the goal (`goal_clear`). If a block is not clear
       in the current state (`current_clear`), add 1 to `h`. This cost estimates
       the action needed to unstack something from the block to make it clear.
    6. Return the total `h` value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and all objects.
        """
        self.goals = task.goals
        self.initial_state = task.init # Get initial state to find all objects

        self.goal_below = {}
        self.goal_on_table = set()
        self.goal_clear = set()
        self.all_objects = set() # Collect all unique objects

        # Collect objects from initial state facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 # Assume any part after the predicate is an object
                 for obj in parts[1:]:
                     self.all_objects.add(obj)

        # Collect objects and goal conditions from goal state facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                obj1, obj2 = parts[1], parts[2]
                self.goal_below[obj1] = obj2
                self.all_objects.add(obj1)
                self.all_objects.add(obj2)
            elif predicate == "on-table" and len(parts) == 2:
                obj = parts[1]
                self.goal_on_table.add(obj)
                self.all_objects.add(obj)
            elif predicate == "clear" and len(parts) == 2:
                obj = parts[1]
                self.goal_clear.add(obj)
                self.all_objects.add(obj)
            # Ignore other goal predicates if any (like arm-empty)

        # Remove potential type indicators like '- object'
        self.all_objects.discard('-')
        self.all_objects.discard('object')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Parse current state into dictionaries/sets for quick lookup
        current_below = {}
        current_on_table = set()
        current_holding = set()
        current_clear = set()
        current_above = {} # Inverse mapping of current_below

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                obj1, obj2 = parts[1], parts[2]
                current_below[obj1] = obj2
                current_above[obj2] = obj1 # Build inverse mapping
            elif predicate == "on-table" and len(parts) == 2:
                obj = parts[1]
                current_on_table.add(obj)
            elif predicate == "holding" and len(parts) == 2:
                obj = parts[1]
                current_holding.add(obj)
            elif predicate == "clear" and len(parts) == 2:
                obj = parts[1]
                current_clear.add(obj)
            # Ignore other state predicates (like arm-empty)

        misplaced_blocks = set()

        # 1. Check position for blocks that have a goal position
        # Iterate over all objects found in init/goal
        for block in self.all_objects:
             # Determine goal support
            goal_sup = None
            if block in self.goal_below:
                goal_sup = self.goal_below[block]
            elif block in self.goal_on_table:
                goal_sup = 'table'

            # Determine current support
            current_sup = None
            if block in current_below:
                current_sup = current_below[block]
            elif block in current_on_table:
                current_sup = 'table'
            elif block in current_holding:
                current_sup = 'held'
            # Note: If a block exists but is not mentioned in any of the above
            # state predicates, current_sup remains None. This shouldn't happen
            # in valid Blocksworld states, but the code handles it by not
            # considering its position if goal_sup is also None.

            # If block has a goal position and is not in that position
            if goal_sup is not None and current_sup != goal_sup:
                 misplaced_blocks.add(block)
                 h += 2 # Cost to move the block (pickup/unstack + putdown/stack)

        # 2. Add penalty for blocks on top of misplaced blocks
        for block in misplaced_blocks:
            # Find the block directly on top of this misplaced block
            block_above = current_above.get(block)
            if block_above is not None:
                h += 1 # Cost to unstack the block above

        # 3. Add penalty for blocks that need to be clear but aren't
        for block in self.goal_clear:
            if block not in current_clear:
                h += 1 # Cost to unstack something from it

        return h
