from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

class blocksworldHeuristic: # Inherit from Heuristic in the actual environment
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are either not on their
    correct base according to the goal state, or have an incorrect block (or
    any block when they should be clear) on top of them. It counts the number
    of "misplaced" or "incorrectly blocked" blocks that are part of the goal
    configuration.

    # Assumptions
    - The heuristic assumes standard Blocksworld actions (pick-up, put-down,
      stack, unstack) with unit cost.
    - The goal specifies the desired configuration of blocks using `on` and
      `on-table` predicates, and `clear` for the top blocks of goal stacks.
    - Blocks not mentioned in the goal are considered "junk" and their final
      position is not explicitly optimized, but their presence can increase
      the heuristic if they block goal blocks.

    # Heuristic Initialization
    The constructor parses the goal conditions (`task.goals`) to build the
    target configuration:
    - `self.goal_below`: A dictionary mapping each block in the goal to the
      block it should be directly on top of, or 'table'.
    - `self.goal_above`: A dictionary mapping each block in the goal to the
      block that should be directly on top of it, or 'nothing' if it's a
      top block in a goal stack.
    - `self.goal_blocks`: A set of all blocks explicitly mentioned in the goal
      `on` or `on-table` predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state (`node.state`) to determine the current position
       of every block. This involves building:
       - `current_below`: A dictionary mapping each block in the state to the
         block it is currently on, or 'table', or 'arm' if held.
       - `current_above`: A dictionary mapping each block in the state to the
         block currently on top of it, or 'nothing' if it is clear. Also tracks
         what the arm is holding.
       - Identify all blocks present in the current state.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block `block` that is part of the goal configuration
       (`self.goal_blocks`).
    4. For each `block`, determine its current support (`current_below[block]`)
       and its goal support (`self.goal_below[block]`).
    5. For each `block`, determine what is currently on top of it
       (`current_above[block]`) and what should be on top of it according to
       the goal (`self.goal_above[block]`). Default to 'nothing' if a block
       is not specified as being under something in the goal.
    6. Check if the block contributes to the heuristic value:
       - If the block's current support is different from its goal support
         (`current_below.get(block) != self.goal_below.get(block)`), increment `h`.
         (This covers blocks on the wrong base or held by the arm).
       - Otherwise (the block is on its correct base or table), check if it is
         incorrectly blocked: If there is a block currently on top of it
         (`current_above.get(block, 'nothing') != 'nothing'`) AND the block
         currently on top is NOT the block that should be on it according to
         the goal (`current_above.get(block, 'nothing') != self.goal_above.get(block, 'nothing')`), increment `h`.
         (This covers blocks in the right place but blocked by the wrong block,
         or blocked when they should be clear).
    7. The total value of `h` after checking all goal blocks is the heuristic
       estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration.
        """
        # In the actual environment, you would call super().__init__(task)
        # super().__init__(task)
        self.task = task # Keep task reference if needed later, though goal is extracted
        self.goals = task.goals # Keep goals reference if needed later

        self.goal_below = {}
        self.goal_above = {}
        self.goal_blocks = set()

        # Parse goal facts to build the desired stack structure
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block_on = parts[1]
                block_under = parts[2]
                self.goal_below[block_on] = block_under
                self.goal_above[block_under] = block_on
                self.goal_blocks.add(block_on)
                self.goal_blocks.add(block_under)
            elif predicate == 'on-table':
                block_on_table = parts[1]
                self.goal_below[block_on_table] = 'table'
                self.goal_blocks.add(block_on_table)
            # 'clear' and 'arm-empty' in goal are implicitly handled by
            # deriving goal_above = 'nothing' for top blocks.

        # Identify blocks that should be clear in the goal (top blocks)
        # Any block in goal_blocks that is not a value in goal_above should be clear
        for block in list(self.goal_blocks): # Iterate over a copy if modifying the set
             if block not in self.goal_above:
                 self.goal_above[block] = 'nothing'

        # Ensure all goal blocks have an entry in goal_below, defaulting to None
        # This handles cases where a block might be mentioned only as being *under* something,
        # which is unusual but handled by .get(block) in __call__.
        # Standard blocksworld goals explicitly state on-table for stack bases or on X for others.
        # So goal_below should be populated for all goal_blocks.
        # No explicit fill needed here, .get() handles missing keys safely.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        h = 0

        # 1. Parse the current state to determine block positions
        current_below = {}
        current_above = {}
        all_objects_in_state = set()

        # First pass: Identify all objects
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            # Collect all arguments that look like objects (not keywords)
            for arg in parts[1:]:
                if arg not in ['table', 'arm', 'arm-empty']:
                    all_objects_in_state.add(arg)

        # Initialize current_below and current_above for all objects found
        for obj in all_objects_in_state:
            current_below[obj] = None # Use None initially, will be set to 'table', block, or 'arm'
            current_above[obj] = 'nothing' # Assume clear until something is found on top

        current_above['arm'] = 'nothing' # Initialize arm state

        # Second pass: Populate current_below and current_above based on facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block_on = parts[1]
                block_under = parts[2]
                current_below[block_on] = block_under
                current_above[block_under] = block_on
            elif predicate == 'on-table':
                block_on_table = parts[1]
                current_below[block_on_table] = 'table'
            elif predicate == 'holding':
                held_block = parts[1]
                current_below[held_block] = 'arm'
                current_above['arm'] = held_block
            # 'clear' and 'arm-empty' facts are redundant if on/on-table/holding are parsed.

        # 3. Iterate through each block that is part of the goal configuration
        for block in self.goal_blocks:
            goal_b = self.goal_below.get(block)
            goal_a = self.goal_above.get(block, 'nothing') # Default to 'nothing' if not specified as being under something

            # Get current position, default to None if block isn't in state structure
            # (e.g., if it's a goal block but not present in the state facts parsed)
            current_b = current_below.get(block)
            current_a = current_above.get(block, 'nothing') # Default to 'nothing' if block isn't in state or is clear

            # 6. Check if the block contributes to the heuristic value (Idea 13 Logic)

            # Condition 1: Misplaced base
            misplaced_base = (current_b != goal_b)

            # Condition 2: Incorrectly blocked
            # A block is incorrectly blocked if something is on it (current_a is a block, not 'nothing')
            # AND what's on it is not what should be on it (current_a != goal_a)
            incorrectly_blocked = (current_a != 'nothing' and current_a != goal_a)

            if misplaced_base or incorrectly_blocked:
                h += 1

        return h
