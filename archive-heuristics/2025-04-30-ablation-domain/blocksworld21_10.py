from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(arm-empty)" or "(clear b1)"
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Fallback, though PDDL facts should be parenthesized
    return fact.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting
    blocks that are not in their correct final position within the goal stacks
    and adding a penalty for blocks that are obstructing correctly placed blocks below them.
    The heuristic value is the sum of two components:
    1. The number of blocks that are not part of a "correct goal stack prefix".
    2. The number of blocks that are currently on top of a block that *is*
       part of a correct goal stack prefix, but the top block itself is *not*.

    # Assumptions
    - The goal state is defined by a set of (on ?x ?y) and (on-table ?x) predicates,
      forming one or more stacks and blocks on the table.
    - Each block has a unique target position in the goal (either on another block or on the table).
    - Blocks mentioned in the initial state but not in goal (on) or (on-table) predicates
      are assumed to have a goal of being on the table.

    # Heuristic Initialization
    - Parses the goal facts to determine the target position for each block, storing
      this in `self.goal_pos`.
    - Identifies all blocks involved in the problem by examining initial and goal states,
      storing them in `self.all_blocks`.
    - For any block found that is not assigned a goal position by an (on) or (on-table)
      goal fact, its goal position is implicitly set to 'table'.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Parse Current State:** Iterate through the facts in the current state to
        determine the current position of each block (`current_pos`: mapping block
        to the block below it or 'table' or 'arm' if held). Also, create an inverse
        mapping (`current_blocks_on_top`: mapping block to the block directly on top of it).
        Identify the block being held, if any.

    2.  **Compute `is_in_goal_stack`:** Determine for each block whether it is part
        of a "correct goal stack prefix". A block `B` is in a correct goal stack
        prefix if:
        -   Its goal position is 'table' (`self.goal_pos[B] == 'table'`) AND its
            current position is 'table' (`current_pos[B] == 'table'`).
        -   OR its goal position is on block `A` (`self.goal_pos[B] == A`) AND its
            current position is on block `A` (`current_pos[B] == A`) AND block `A`
            is already determined to be in a correct goal stack prefix (`is_in_goal_stack[A]` is True).
        This is computed using a bottom-up approach, starting with blocks correctly
        placed on the table according to the goal, and propagating this property
        upwards through the current stacks that match the goal stack structure.

    3.  **Count Misplaced Blocks Relative to Goal Stacks:** Calculate the first component
        of the heuristic: the number of blocks `B` for which `is_in_goal_stack[B]` is False.
        These are blocks that are either in the wrong position or are on top of
        blocks that are not correctly placed relative to the goal structure.

    4.  **Calculate Blocking Penalty:** Calculate the second component of the heuristic:
        Iterate through all blocks `A`. If block `A` is in a correct goal stack prefix
        (`is_in_goal_stack[A]` is True), check if there is a block `B` currently
        stacked directly on top of `A` (`current_blocks_on_top[A] == B`). If such a
        block `B` exists AND `B` is NOT in a correct goal stack prefix
        (`is_in_goal_stack[B]` is False), then `B` is blocking a correctly placed
        stack. Increment the blocking penalty count.

    5.  **Sum Components:** The total heuristic value is the sum of the count from
        step 3 and the penalty from step 4.

    6.  **Goal State Check:** If the current state is the goal state (checked using
        `node.task.goal_reached(state)`), the heuristic value is 0. This overrides
        the calculated value to ensure the heuristic is 0 only at the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and identifying all blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        # Static facts are not needed for this heuristic in blocksworld.
        # static_facts = task.static

        self.goal_pos = {}
        self.all_blocks = set()

        # Collect all blocks mentioned in initial state and goal state
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 for obj in parts[1:]:
                     if obj != 'table':
                          self.all_blocks.add(obj)

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                block, target = parts[1], parts[2]
                self.goal_pos[block] = target
                self.all_blocks.add(block)
                self.all_blocks.add(target)
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                self.goal_pos[block] = 'table'
                self.all_blocks.add(block)
            # (clear ?) goals are ignored for goal_pos mapping

        # Assume blocks not mentioned in goal (on) or (on-table) should be on the table
        # This is a common convention in blocksworld benchmarks.
        for block in list(self.all_blocks): # Iterate over a copy
             if block not in self.goal_pos:
                 self.goal_pos[block] = 'table'


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 6. Goal State Check: Return 0 if the goal is reached.
        if node.task.goal_reached(state):
             return 0

        # 1. Parse current state
        current_pos = {}
        # Map block -> block_on_top (inverse of current_pos)
        current_blocks_on_top = {block: None for block in self.all_blocks}
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            if parts[0] == 'on':
                block, target = parts[1], parts[2]
                current_pos[block] = target
                current_blocks_on_top[target] = block
            elif parts[0] == 'on-table':
                block = parts[1]
                current_pos[block] = 'table'
            elif parts[0] == 'holding':
                current_holding = parts[1]
                current_pos[current_holding] = 'arm' # Represent holding state

        # 2. Determine is_in_goal_stack
        is_in_goal_stack = {block: False for block in self.all_blocks}
        queue = []

        # Add blocks whose goal is on-table and are currently on-table
        for block in self.all_blocks:
            if self.goal_pos.get(block) == 'table' and current_pos.get(block) == 'table':
                is_in_goal_stack[block] = True
                queue.append(block)

        # Propagate up the goal stacks
        # Use a queue for breadth-first propagation up the stacks
        q_index = 0
        while q_index < len(queue):
            base_block = queue[q_index]
            q_index += 1

            # Find blocks that should be on top of base_block in the goal
            for block in self.all_blocks:
                if self.goal_pos.get(block) == base_block:
                    # Check if block is currently on base_block and base_block is in goal stack
                    if current_pos.get(block) == base_block and is_in_goal_stack[base_block]:
                         if not is_in_goal_stack[block]: # Avoid adding duplicates to queue
                            is_in_goal_stack[block] = True
                            queue.append(block)

        # 3. Count blocks not in goal stack prefix
        misplaced_in_stack_count = sum(1 for block in self.all_blocks if not is_in_goal_stack[block])

        # 4. Calculate blocking penalty
        blocking_penalty = 0
        for block in self.all_blocks:
            # Find the block directly on top of 'block' in the current state
            block_on_top = current_blocks_on_top.get(block)
            if block_on_top is not None: # If there is a block on top
                # Check if 'block' is in a correct goal stack prefix
                if is_in_goal_stack.get(block, False):
                    # Check if the block on top is NOT in a correct goal stack prefix
                    if not is_in_goal_stack.get(block_on_top, False):
                        blocking_penalty += 1

        # 5. Total heuristic value
        heuristic_value = misplaced_in_stack_count + blocking_penalty

        return heuristic_value
