from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into predicate and arguments."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions required to reach the goal state.
    This heuristic is non-admissible and designed to guide a greedy best-first search.
    It counts the number of blocks that are not in their correct goal position
    relative to their base, or have the wrong block on top. It also penalizes
    having a block in the arm if the goal is not reached.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal state.

        Heuristic Initialization:
        Parses the goal facts to build data structures representing the desired
        final configuration of blocks:
        - goal_base_map: Maps each block to the block it should be directly on top of
                         in the goal state, or 'table' if it should be on the table.
                         {block_on_top: block_below or 'table'}
        - goal_stack_map: Maps each block to the block that should be directly on top
                          of it in the goal state. {block_below: block_on_top}
        - goal_clear_set: Set of blocks that should be clear (have nothing on top)
                          in the goal state.
        Static facts are ignored as the Blocksworld domain provided has no static facts.
        """
        self.task = task # Store task to check goal_reached
        self.goals = task.goals

        self.goal_base_map = {}
        self.goal_stack_map = {}
        self.goal_clear_set = set()

        # Parse goal facts to build goal configuration maps
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                self.goal_base_map[block_on_top] = block_below
                self.goal_stack_map[block_below] = block_on_top
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_base_map[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear_set.add(block)

        # Identify all blocks that are part of the goal configuration (either on something or on table)
        self.goal_blocks = set(self.goal_base_map.keys())

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state using `self.task.goal_reached(state)`. If yes, the heuristic is 0.
        2. Build data structures representing the current state by iterating through its facts:
           - current_base_map: Maps each block to its current support (block below,
                               'table', or 'arm'). {block: support}
           - current_stack_map: Maps each block to the block currently on top of it.
                                {block_below: block_on_top}
           - held_block: The block currently held by the arm, or None.
        3. Initialize the heuristic value `h` to 0.
        4. Iterate through each block that is part of the goal configuration (`self.goal_blocks`).
        5. For each block `b` in `self.goal_blocks`:
           a. Find its desired goal base (`goal_base = self.goal_base_map[b]`).
           b. Find its current base (`current_base = current_base_map.get(b)`).
           c. If `current_base` is different from `goal_base`, increment `h`. This block is on the wrong base (or held when it shouldn't be).
           d. If `current_base` is the same as `goal_base`:
              i. Find the block that should be on top of `b` in the goal (`goal_block_on_top = self.goal_stack_map.get(b)`).
              ii. Find the block currently on top of `b` (`current_block_on_top = current_stack_map.get(b)`).
              iii. If `current_block_on_top` is different from `goal_block_on_top`, increment `h`. This block has the wrong block on top (or should be clear but isn't, or shouldn't be clear but is).
        6. If the arm is currently holding a block (`held_block is not None`), increment `h` by 1. This penalizes having the arm occupied, as it often needs to be empty to make progress towards the goal. This penalty is applied only if the state is not the goal state (handled by step 1).
        7. Return the final heuristic value `h`.

        Assumptions:
        - The input state is a frozenset of PDDL fact strings.
        - The goal is a frozenset of PDDL fact strings defining the desired block configuration.
        - Blocks not explicitly mentioned in goal (on/on-table) facts are not strictly part of the required goal structure, and the heuristic focuses on achieving the specified goal structure.
        - (clear X) goals typically apply to blocks X that are the intended top of a goal stack or alone on the table. The heuristic handles the case where a block on its correct base has the wrong block on top (which includes the case where it should be clear but isn't).
        - The heuristic is non-admissible and aims to reduce expanded nodes in greedy best-first search.

        @param node: The search node containing the state.
        @return: The estimated cost (heuristic value) to reach the goal.
        """
        state = node.state

        # Heuristic is 0 only for goal states
        if self.task.goal_reached(state):
            return 0

        # Build current state maps
        current_base_map = {}
        current_stack_map = {}
        held_block = None
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                current_base_map[block_on_top] = block_below
                current_stack_map[block_below] = block_on_top
            elif predicate == 'on-table':
                current_base_map[parts[1]] = 'table'
            elif predicate == 'holding':
                held_block = parts[1]
                # Represent held block's status; it doesn't have a 'base' in the usual sense.
                # Storing 'arm' allows comparison with goal_base.
                current_base_map[parts[1]] = 'arm'
            # 'clear' and 'arm-empty' facts are not needed for base/stack maps

        h = 0

        # Count blocks in goal configuration that are misplaced
        for block in self.goal_blocks:
            goal_base = self.goal_base_map[block]
            current_base = current_base_map.get(block)

            if current_base != goal_base:
                # Block is on the wrong base (or held when it shouldn't be)
                h += 1
            else: # current_base == goal_base (Block is on the correct base)
                goal_block_on_top = self.goal_stack_map.get(block)
                current_block_on_top = current_stack_map.get(block)

                if current_block_on_top != goal_block_on_top:
                    # Block has the wrong block on top (or should be clear but isn't, etc.)
                    h += 1

        # Add penalty if arm is not empty (and we are not in the goal state)
        # The goal_reached check at the start ensures this is only applied in non-goal states.
        if held_block is not None:
             h += 1

        return h
