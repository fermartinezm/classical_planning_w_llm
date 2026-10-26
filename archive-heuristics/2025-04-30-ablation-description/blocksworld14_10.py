# from heuristics.heuristic_base import Heuristic # Assumed provided

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    # Remove outer parentheses
    if fact_str.startswith('(') and fact_str.endswith(')'):
        fact_str = fact_str[1:-1]
    # Split by whitespace
    return fact_str.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing three components:
    1. The number of blocks whose current support (the block directly below them,
       the table, or the arm) is different from their desired support in the goal state.
    2. The number of blocks that are required to be clear in the goal state but
       are not clear in the current state.
    3. A penalty if the arm is required to be empty in the goal state but is
       currently holding a block.

    # Assumptions
    - The goal state specifies the desired configuration of some blocks using
      `(on X Y)` and `(on-table Z)` predicates.
    - The goal state may also specify that certain blocks must be clear using
      `(clear X)` predicates.
    - The goal state may require the arm to be empty using `(arm-empty)`.
    - The heuristic only considers blocks explicitly mentioned in the goal's
      `(on X Y)`, `(on-table Z)`, or `(clear X)` predicates.

    # Heuristic Initialization
    - Parse the goal conditions (`task.goals`) to identify:
        - The desired support for each block (another block or the table) based on `(on X Y)` and `(on-table X)` predicates. Store this in `self.goal_support`.
        - The set of blocks that must be clear based on `(clear X)` predicates. Store this in `self.goal_clear`.
        - Whether the arm must be empty based on the `(arm-empty)` predicate. Store this in `self.goal_arm_empty`.
        - The set of all blocks involved in any of the above goal predicates. Store this in `self.goal_blocks`.
    - Static facts (`task.static`) are not used in this heuristic as Blocksworld has none relevant to state configuration.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the heuristic value `h_value` to 0.
    2. Parse the current state (`node.state`) to build temporary data structures:
       - `current_support`: A dictionary mapping each block to its current support
         (the block it's `on`, or the string 'table').
       - `current_clear`: A set containing the names of all blocks that are currently clear.
       - `current_holding`: The name of the block currently held by the arm, or `None` if the arm is empty.
       Iterate through all facts in the state to populate these structures.
    3. Calculate the penalty for misplaced blocks:
       - Iterate through each `block` in the set `self.goal_blocks`.
       - Look up the `desired_support` for this block from `self.goal_support`. If the block is not in `self.goal_support` (meaning its position is not specified in the goal, only its clarity might be), skip the support check for this block.
       - Determine the `current_sup` for the block:
         - If `current_holding` is equal to the `block`, the `current_sup` is the string 'arm'.
         - Otherwise, look up the block in the `current_support` dictionary. The value found is the `current_sup` (either a block name or 'table'). If the block is not found in `current_support` (which implies it's not on anything or the table, potentially an invalid state or it's the block being held, which is handled), the `current_sup` remains `None`.
       - If the `desired_support` was found (i.e., the block's position is specified in the goal) and the `current_sup` is different from the `desired_support`, increment `h_value` by 1.
    4. Calculate the penalty for blocks that should be clear but aren't:
       - Iterate through each `block` in the set `self.goal_clear`.
       - If the `block` is not present in the `current_clear` set, increment `h_value` by 1.
    5. Calculate the penalty for the arm not being empty when required:
       - If `self.goal_arm_empty` is `True` and `current_holding is not None`:
            h_value += 1
    6. Return the final `h_value`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        super().__init__(task)

        # Store goal support for each block: block -> support_block or 'table'
        self.goal_support = {}
        # Store blocks that must be clear in the goal
        self.goal_clear = set()
        # Store whether the arm must be empty in the goal
        self.goal_arm_empty = False
        # Store all blocks involved in goal on/on-table/clear predicates
        self.goal_blocks = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: # Skip empty parts from malformed facts if any
                continue
            predicate = parts[0]

            if predicate == "on":
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_support[block] = support
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(support) # Add the support block too
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_support[block] = 'table'
                    self.goal_blocks.add(block)
            elif predicate == "clear":
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_clear.add(block)
                    self.goal_blocks.add(block) # A block that needs to be clear is also a goal block
            elif predicate == "arm-empty":
                 if len(parts) == 1:
                    self.goal_arm_empty = True
            # Ignore other potential goal predicates if any (like object types)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        h_value = 0

        # Build current state info
        current_support = {}
        current_clear = set()
        current_holding = None # Track if arm is holding something

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty parts
                continue
            predicate = parts[0]

            if predicate == "on":
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    current_support[block] = support
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_support[block] = 'table'
            elif predicate == "clear":
                if len(parts) == 2:
                    block = parts[1]
                    current_clear.add(block)
            elif predicate == "holding":
                if len(parts) == 2:
                    current_holding = parts[1]
            # We don't need arm-empty from state explicitly, just check current_holding

        # Calculate penalty for misplaced blocks
        # Iterate only through blocks that are part of the goal configuration or need clearing
        for block in self.goal_blocks:
            # Find the block's desired support in the goal
            desired_support = self.goal_support.get(block)

            # If the block has a desired support (i.e., it's part of an on/on-table goal)
            if desired_support is not None:
                 # Find the block's current support
                 current_sup = None
                 if current_holding == block:
                     current_sup = 'arm' # Use a special value for arm
                 else:
                     current_sup = current_support.get(block) # Gets 'table' or block name, or None if not found

                 # If current support doesn't match desired support, add penalty
                 if current_sup != desired_support:
                     h_value += 1

        # Calculate penalty for blocks that should be clear but aren't
        for block in self.goal_clear:
            if block not in current_clear:
                h_value += 1

        # Calculate penalty for arm not being empty when required
        if self.goal_arm_empty and current_holding is not None:
            h_value += 1

        return h_value
