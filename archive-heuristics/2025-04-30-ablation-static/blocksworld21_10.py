from heuristics.heuristic_base import Heuristic
# fnmatch is not used in this heuristic logic
# from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not yet in their
    correct final position as part of a correctly built stack segment.
    It counts the number of blocks that are *not* part of a stack segment
    that matches the goal configuration and is rooted on a block that is
    correctly placed on the table according to the goal.

    # Assumptions
    - The goal state specifies a desired configuration of blocks stacked
      on each other or placed on the table.
    - All blocks mentioned in the initial state or goal state are relevant
      and have a defined goal position (either on the table or on another block).
    - The heuristic does not explicitly count 'clear' or 'arm-empty' goals,
      assuming they are satisfied when the block configuration is correct.

    # Heuristic Initialization
    - Parses the goal facts to identify the desired 'on' relationships
      (`goal_on` map) and blocks that should be on the table (`goal_on_table` set).
    - Collects all unique objects (blocks) mentioned in the initial state
      and goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If yes, the heuristic is 0.
    2. Parse the current state to determine the current 'on' relationships
       (`current_on` map), blocks currently on the table (`current_on_table` set),
       and the block currently being held (`holding`). Note that 'holding' is
       not directly used in the final heuristic calculation but is part of
       parsing the state's block configuration.
    3. Initialize a set `correctly_stacked_blocks`. This set will store blocks
       that are in their correct position relative to the block below them,
       AND the block below them is also considered correctly stacked (recursively).
    4. Identify the base cases for `correctly_stacked_blocks`: Add any block `B`
       to this set if `(on-table B)` is a goal fact AND `(on-table B)` is true
       in the current state. These are the correctly placed foundations.
    5. Iteratively expand the set `correctly_stacked_blocks`: Repeat the following
       until no new blocks are added to the set in a full pass:
       For every goal fact `(on A B)`:
       - Check if block `B` (the base) is already in the `correctly_stacked_blocks` set.
       - Check if block `A` (the block on top) is currently directly on block `B`
         in the current state (i.e., `current_on.get(A) == B`).
       - If both conditions are met, it means the stack segment `A` on `B` is
         correct according to the goal and is built upon a correctly stacked base.
         Add block `A` to the `correctly_stacked_blocks` set if it's not already there.
         Set a flag to indicate that a change occurred.
    6. The heuristic value is the total number of blocks minus the number of blocks
       in the `correctly_stacked_blocks` set. This counts blocks that are either
       misplaced or are part of a stack built on a wrong foundation.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        self.goals = task.goals

        self.goal_on = {}
        self.goal_on_table = set()
        all_objects = set()

        # Parse goal facts to build goal configuration maps/sets
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_on[block] = base
                all_objects.add(block)
                all_objects.add(base)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_on_table.add(block)
                all_objects.add(block)
            # Ignore 'clear' and 'arm-empty' goals for this heuristic's core logic

        # Collect all objects from the initial state as well
        # This assumes all relevant objects appear in initial or goal state facts.
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Add all arguments as objects, ignoring the predicate name
             for part in parts[1:]:
                 all_objects.add(part)

        self.all_objects = frozenset(all_objects)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Step 1: Check if the current state is the goal state
        if self.goals <= state:
             return 0

        # Step 2: Parse current state facts
        current_on = {}
        current_on_table = set()
        # holding = None # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_on[block] = base
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_on_table.add(block)
            # 'holding' fact is not needed for this specific heuristic calculation
            # elif predicate == "holding" and len(parts) == 2:
            #     holding = parts[1]
            # Ignore 'clear' and 'arm-empty' in current state parsing for this heuristic

        # Step 3: Initialize correctly_stacked_blocks set
        correctly_stacked_blocks = set()

        # Step 4: Add blocks correctly placed on the table according to the goal
        for block in self.goal_on_table:
            if block in current_on_table:
                correctly_stacked_blocks.add(block)

        # Step 5: Iteratively add blocks correctly stacked on correctly stacked blocks
        changed = True
        while changed:
            changed = False
            # Iterate through goal 'on' facts to find correctly stacked blocks
            # Use list(self.goal_on.items()) to iterate over a copy if needed,
            # but we are only adding to correctly_stacked_blocks, not modifying goal_on.
            for block_on_top, block_below in self.goal_on.items():
                # Check if the block below is already correctly stacked
                if block_below in correctly_stacked_blocks:
                    # Check if the block on top is currently directly on the block below
                    # Use .get() for safety in case block_on_top is not in current_on
                    if current_on.get(block_on_top) == block_below:
                         # block_on_top is correctly stacked on block_below
                         if block_on_top not in correctly_stacked_blocks:
                             correctly_stacked_blocks.add(block_on_top)
                             changed = True

        # Step 6: Compute the heuristic value
        # The heuristic is the number of blocks that are *not* correctly stacked.
        h = len(self.all_objects) - len(correctly_stacked_blocks)

        # The explicit goal check at the beginning ensures h=0 iff goal.
        # If the state is not a goal state, h must be > 0 based on the logic,
        # unless there are no objects, in which case h=0, which is correct
        # for an empty problem.

        return h
