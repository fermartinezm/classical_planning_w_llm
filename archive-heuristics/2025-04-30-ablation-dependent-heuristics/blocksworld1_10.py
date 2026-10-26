import re

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal by counting the
        number of blocks that are either in the wrong goal position or are
        currently on top of a block that itself needs to be moved (either
        because it's in the wrong position or because something on top of it
        needs to be moved). This count represents the minimum number of blocks
        that must be moved out of the way or into place. Each such block
        is estimated to require 2 actions (e.g., unstack/pickup + stack/putdown).
        The heuristic value is 2 times this count.

    Assumptions:
        - The input state and task are valid representations for the Blocksworld domain.
        - Goal facts primarily define the desired 'on' and 'on-table' relationships.
        - All blocks present in the initial state are also relevant to the goal
          (i.e., they appear in goal facts or initial state facts).
        - The domain follows standard STRIPS semantics for Blocksworld actions.
        - Blocks are identified by strings (e.g., 'b1', 'b2').

    Heuristic Initialization:
        The constructor pre-processes the goal facts to determine the desired
        final position for each block ('on' another block or 'on-table').
        It also identifies the set of all blocks involved in the problem
        by collecting unique object names from the initial state and goal facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Parse the state facts to determine the current position of each block
           (which block it is 'on', if it is 'on-table', or if it is 'holding').
           Also, build a map indicating which block is currently directly on top
           of another block.
        2. Identify the set of blocks that are currently in a different position
           than their specified goal position. Let's call this the 'out_of_place_set'.
           A block B is considered 'out of place' if its goal position is defined,
           and its current position (on another block, on the table, or held)
           is different from its goal position.
        3. Initialize a set called 'must_be_moved_set' with all blocks from the
           'out_of_place_set'. These blocks definitely need to be moved.
        4. Propagate the 'must_be_moved' status upwards through the stacks:
           Repeatedly iterate through all blocks. If a block X is currently
           on top of a block B (i.e., state contains '(on X B)') and block B
           is in the 'must_be_moved_set', then block X also needs to be moved
           (to get it out of the way). Add X to the 'must_be_moved_set'.
           Continue this propagation until no new blocks are added to the set
           in a full iteration.
        5. The heuristic value is the total number of blocks in the final
           'must_be_moved_set' multiplied by 2. This estimates that each block
           that is misplaced or blocking a misplaced block requires approximately
           two actions (one to pick it up/unstack, one to put it down/stack it
           elsewhere) to resolve its situation relative to the goal structure.
    """
    def __init__(self, task):
        self.goal_pos = {}
        self.all_blocks = set()

        # Parse goal facts to build goal_pos and collect all blocks
        for fact_string in task.goals:
            predicate, args = self._parse_fact_parts(fact_string)
            if predicate == 'on':
                self.goal_pos[args[0]] = args[1]
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif predicate == 'on-table':
                self.goal_pos[args[0]] = 'table'
                self.all_blocks.add(args[0])
            # Ignore 'clear' and 'arm-empty' in goals for structure heuristic

        # Collect all blocks from initial state as well
        for fact_string in task.initial_state:
             predicate, args = self._parse_fact_parts(fact_string)
             # Arguments can be blocks or 'table' or 'arm-empty'. Filter for blocks.
             # Assuming block names don't contain '-' or start with 'arm'.
             for arg in args:
                 if arg != 'arm-empty' and arg != 'table': # Exclude non-block terms
                    self.all_blocks.add(arg)

    def __call__(self, state, task):
        # If it's a goal state, heuristic is 0
        if task.goal_reached(state):
            return 0

        current_pos = {} # Maps block -> block_below or 'table' or 'holding'
        current_on_map = {} # Maps block_above -> block_below
        # currently_held is implicitly captured in current_pos

        # Parse state facts to build current_pos and current_on_map
        for fact_string in state:
            predicate, args = self._parse_fact_parts(fact_string)
            if predicate == 'on':
                block_above, block_below = args
                current_pos[block_above] = block_below
                current_on_map[block_above] = block_below
            elif predicate == 'on-table':
                block = args[0]
                current_pos[block] = 'table'
            elif predicate == 'holding':
                 block = args[0]
                 current_pos[block] = 'holding' # Represent held state

        # 1. Identify blocks in the wrong goal position
        out_of_place_set = set()
        for block in self.all_blocks:
            goal_target = self.goal_pos.get(block)

            # If the block has a defined goal position
            if goal_target is not None:
                current_target = current_pos.get(block)

                # If the block's current position is different from its goal position
                # This covers cases where current_target is None (e.g., block is not in state facts, shouldn't happen for blocks in all_blocks)
                # or where current_target is defined but doesn't match goal_target.
                if current_target != goal_target:
                     out_of_place_set.add(block)

        # 2. Initialize must_be_moved_set
        must_be_moved_set = set(out_of_place_set)

        # 3. Propagate must_be_moved status upwards
        changed = True
        while changed:
            changed = False
            # Iterate through all blocks to find ones on top of blocks that must be moved
            for block_above in self.all_blocks:
                 # If block_above is not already marked as must_be_moved
                 if block_above not in must_be_moved_set:
                     # Find the block below it in the current state using the on_map
                     block_below = current_on_map.get(block_above)
                     # If block_above is on some block_below, and block_below must be moved
                     if block_below is not None and block_below in must_be_moved_set:
                         # Then block_above must also be moved
                         must_be_moved_set.add(block_above)
                         changed = True # Set changed, need another iteration

        # 4. Calculate heuristic value
        # Each block that must be moved requires at least 2 actions (pickup/unstack + putdown/stack)
        heuristic_value = len(must_be_moved_set) * 2

        return heuristic_value

    def _parse_fact_parts(self, fact_string):
        """Helper to parse a PDDL fact string into predicate and arguments."""
        # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
        # Example: '(arm-empty)' -> ['arm-empty']
        return fact_string.strip('()').split()

