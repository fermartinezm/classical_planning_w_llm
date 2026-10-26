from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing two components:
    1. A cost for blocks that are not in their correct goal position relative to their
       goal parent block or the table, or are currently held.
    2. A penalty for blocks that are currently on top of other blocks in a configuration
       that is not part of the goal state.

    # Assumptions
    - The goal specifies the desired position for some blocks, either on another
      block (`on`) or on the table (`on-table`).
    - The heuristic primarily focuses on achieving the correct stack configurations
      specified by the `on` and `on-table` goal predicates.
    - 'clear' and 'arm-empty' goals are treated as conditions that are typically
      met as a consequence of achieving the main stacking goals, and are not
      directly counted in the heuristic value.

    # Heuristic Initialization
    - Parses the goal state (`task.goals`) to determine the desired parent (another
      block or 'table') for each block that is part of an `on` or `on-table` goal
      predicate. This mapping is stored in `self.goal_parent`.
    - Stores the set of all `(on X Y)` goal facts in `self.goal_on_facts` for efficient lookup.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is computed as follows:

    1. **Parse Current State:**
       - Iterate through the facts in the current state (`node.state`).
       - Build a map `current_parent` where `current_parent[block]` is the block
         it is currently on, or 'table' if it's on the table.
       - Build a set `current_on_facts` containing all `(on X Y)` facts true in the state.
       - Identify the block currently held by the arm, if any (`held_block`).

    2. **Initialize Cost:** Set `total_cost = 0`.

    3. **Cost for Misplaced or Held Blocks:**
       - Iterate through each block `B` that is present as the first argument
         in an `on` or `on-table` goal predicate (i.e., each block in `self.goal_parent.keys()`).
       - Get the goal parent/table for `B` (`goal_p = self.goal_parent[B]`).
       - Get the current parent/table for `B` (`current_p = current_parent.get(B)`).
       - If `B` is the block currently held (`held_block == B`):
         - Add 1 to `total_cost`. This represents the action needed to place the block
           (either `stack` or `putdown`).
       - If `B` is not held and its current position (`current_p`) is different
         from its goal position (`goal_p`):
         - Add 2 to `total_cost`. This represents the actions needed to pick up/unstack
           the block (1 action) and then place it correctly (1 action).

    4. **Cost for Blocking Blocks:**
       - Iterate through each `(on X Y)` fact present in the current state (`current_on_facts`).
       - For each such fact, check if the exact fact `(on X Y)` is present in the
         set of goal `on` facts (`self.goal_on_facts`).
       - If `(on X Y)` is true in the state but is *not* a goal fact:
         - Add 2 to `total_cost`. This represents the estimated cost to move block `X`
           out of the way (unstack `X` from `Y` and place `X` somewhere else).

    5. **Return Total Cost:** The final `total_cost` is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information.

        Args:
            task: The planning task object containing goals and static facts.
        """
        self.goals = task.goals

        # Build goal_parent map: block -> parent_block or 'table'
        # This maps each block that needs to be on another block or the table
        # to its desired parent or 'table'.
        self.goal_parent = {}
        # Store goal on facts for quick lookup when checking for blocking blocks.
        self.goal_on_facts = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                # Goal is (on block parent)
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
                self.goal_on_facts.add(goal)
            elif predicate == "on-table":
                # Goal is (on-table block)
                block = parts[1]
                self.goal_parent[block] = 'table'
            # 'clear' and 'arm-empty' goals are not explicitly handled by this heuristic's counts.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the cost to reach the goal.
        """
        state = node.state

        # If the state is a goal state, the heuristic is 0.
        # This check is important for correctness and efficiency.
        if self.goals <= state:
             return 0

        # Build current_parent map: block -> parent_block or 'table'
        # This maps each block that is currently on another block or the table
        # to its current parent or 'table'.
        current_parent = {}
        # Store current on facts for quick lookup when checking for blocking blocks.
        current_on_facts = set()
        # Identify the block currently held by the arm.
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, parent = parts[1], parts[2]
                current_parent[block] = parent
                current_on_facts.add(fact)
            elif predicate == "on-table":
                block = parts[1]
                current_parent[block] = 'table'
            elif predicate == "holding":
                held_block = parts[1]
            # Ignore 'clear' and 'arm-empty' state facts for the heuristic calculation.

        total_cost = 0

        # Component 1: Cost for blocks not in their goal position relative to parent/table, or held.
        # Iterate through blocks whose final position is specified in the goal.
        for block in self.goal_parent:
            goal_p = self.goal_parent[block]
            # Get current parent, defaults to None if the block is not on anything (e.g., held or not in state - though not in state shouldn't happen).
            current_p = current_parent.get(block)

            if held_block == block:
                 # The block is currently held. It needs to be placed correctly.
                 # This typically costs 1 action (stack or putdown).
                 total_cost += 1
            elif current_p != goal_p:
                 # The block is not held and is in the wrong place relative to its goal parent/table.
                 # It needs to be picked up/unstacked (1 action) and then placed correctly (1 action).
                 total_cost += 2

        # Component 2: Cost for blocks that are currently on top of others incorrectly.
        # Iterate through all 'on' facts currently true in the state.
        for fact in current_on_facts:
            # Check if this exact 'on' fact is NOT a goal 'on' fact.
            if fact not in self.goal_on_facts:
                # This fact (on X Y) means block X is on block Y, but this configuration
                # is not desired in the goal. Block X is blocking Y or is simply misplaced
                # on top of Y. X needs to be moved out of the way.
                # Moving X requires unstacking X (1 action) and placing it somewhere else (1 action).
                # Add 2 actions as a penalty for this blocking/incorrectly placed block X.
                total_cost += 2

        return total_cost
