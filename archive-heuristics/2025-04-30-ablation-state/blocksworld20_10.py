from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    if fact.startswith("("):
        return fact[1:-1].split()
    return [fact] # Should not happen for facts, but safe

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the number of blocks that are not in their correct position relative to the block/table below them in the goal state, or that have an incorrect block stacked directly on top of them. It also adds a cost if the arm is holding a block and should be empty in the goal.

    # Assumptions
    - All blocks mentioned in the initial state and goal are the objects of the problem.
    - The goal specifies the final position (on another block or on the table) for every block.
    - The goal implicitly requires the arm to be empty unless a block is explicitly specified as held (which is not standard in Blocksworld goals). We assume the goal is always arm-empty.

    # Heuristic Initialization
    - Parses the goal state to determine the desired block configuration:
        - `goal_below`: Mapping from each block to the block it should be directly on, or 'table'.
        - `goal_above`: Mapping from each block to the block that should be directly on it, or None if it should be clear.
    - Collects all objects (blocks) involved in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. For a given state, determine the current configuration of blocks:
       - `current_below`: Mapping from each block to the block it is directly on, or 'table', or 'arm' if held.
       - `current_above`: Mapping from each block to the block that is directly on it, or None if clear.
       - Identify the block currently held by the arm, if any.
    2. Initialize a counter `unhappy_count` to 0.
    3. For each block in the problem:
       - Determine the block's current base (what it's on or if it's held) and its goal base.
       - If the current base is different from the goal base, increment `unhappy_count`.
       - If the current base is the same as the goal base, check the block directly on top. Determine the block currently on top and the block that should be on top in the goal. If they differ, increment `unhappy_count`.
    4. Check the state of the arm:
       - If the arm is holding a block, and the goal requires the arm to be empty (which is assumed), increment `unhappy_count`.
    5. The heuristic value is `unhappy_count`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        self.goal_below = {}
        self.goal_above = {}
        self.objects = set()

        # Collect objects from initial state and goals
        # Collect all unique arguments from all facts
        all_facts = set(task.initial_state) | set(task.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            # Add all parts that are not common predicates/keywords as objects
            self.objects.update(p for p in parts if p not in {'on', 'on-table', 'clear', 'arm-empty', 'holding', 'table'})

        # Parse goal structure based on identified objects
        for goal in task.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                # Ensure block is a recognized object and base is a recognized object or 'table'
                if block in self.objects and (base in self.objects or base == 'table'):
                     self.goal_below[block] = base
                     if base in self.objects: # Only blocks can have something stacked on them
                         self.goal_above[base] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                if block in self.objects:
                    self.goal_below[block] = 'table'
            # 'clear' goals are handled by initializing goal_above later

        # Initialize goal_above for blocks that should be clear (not mentioned as base in any 'on' goal)
        for obj in self.objects:
             if obj not in self.goal_above:
                 self.goal_above[obj] = None # Should be clear in the goal

        # Note: This heuristic assumes every object in self.objects has a goal location
        # specified by an 'on' or 'on-table' predicate in the goals.
        # If an object exists but is not in self.goal_below, its goal location is unknown
        # and this heuristic might behave unexpectedly. Valid BW problems should define
        # the goal location for all blocks.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        current_below = {}
        current_above = {}
        held_block = None
        # arm_is_empty = False # Not strictly needed if we track held_block

        # Build current state structure
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_below[block] = base
                current_above[base] = block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                held_block = parts[1]
                # current_below[held_block] = 'arm' # Not strictly needed for this logic
            # elif predicate == 'arm-empty':
            #      arm_is_empty = True # Not strictly needed

        unhappy_count = 0

        # Check blocks
        for block in self.objects:
            # Get goal state for this block
            # Use .get() with a default like 'unknown' to handle blocks not in goal,
            # although the assumption is all objects are in goal.
            goal_base = self.goal_below.get(block)
            goal_block_above = self.goal_above.get(block)

            # Determine current state of the block
            if held_block == block:
                current_base = 'arm'
                current_block_above = None # Nothing is on a held block
            else:
                # If block is not held, it must be on something or on the table
                current_base = current_below.get(block)
                # If current_base is None here, it means the block is not held,
                # not on anything, and not on the table, which is an invalid state.
                # Assuming valid states, current_base will be a block or 'table'.

                current_block_above = current_above.get(block, None) # None if clear

            # Check unhappiness conditions
            # Condition 1: Incorrect base
            if current_base != goal_base:
                unhappy_count += 1
            # Condition 2: Correct base, but incorrect block above
            # This check applies if the block is currently on its correct base.
            # The base could be 'table' or another block.
            # If the base is 'table', current_above.get(block, None) will be None (correctly).
            # goal_above.get(block) will be None if the block should be clear on the table,
            # or a block name if something should be on it.
            # The comparison `current_block_above != goal_block_above` works correctly.
            elif current_base == goal_base and current_block_above != goal_block_above:
                 unhappy_count += 1

        # Check arm state (implicit goal: arm-empty)
        # If the arm is holding something (held_block is not None), and the goal is arm-empty (assumed), add cost.
        if held_block is not None:
             unhappy_count += 1 # Cost to put down or stack the held block

        return unhappy_count
