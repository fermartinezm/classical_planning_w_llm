from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by counting
        the number of blocks that are not in their correct goal position
        relative to their support (table or another block), plus the number
        of blocks that have an incorrect block stacked directly on top of them
        relative to the goal state, plus one if the arm state does not match
        the goal arm state.

    Assumptions:
        - The PDDL domain is Blocksworld as provided.
        - Goal states specify the desired positions of blocks using
          '(on X Y)' or '(on-table Z)' predicates, and optionally the arm state
          using '(arm-empty)' or '(holding X)'.
        - The set of objects (blocks) is static and can be determined from
          the initial state and goal state facts.
        - The heuristic is used for greedy best-first search and does not
          need to be admissible, but should be efficiently computable and
          provide a good estimate to minimize expanded nodes.

    Heuristic Initialization:
        In the __init__ method, the heuristic pre-processes the goal state
        and initial state to build data structures representing the desired
        configuration and the set of all blocks:
        - `all_blocks`: A set of all block objects in the problem.
        - `goal_pos`: A dictionary mapping each block to its desired position.
                      The value is a tuple: ('on', support_block) or
                      ('on-table', None).
        - `goal_above`: A dictionary mapping each block to the block that
                        should be directly on top of it in the goal state,
                        or None if nothing should be on it.
        - `goal_arm_state`: Represents the desired arm state: ('empty', None)
                            or ('holding', block).

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Parse the current state facts to determine the current position
           and the block directly above for each block, and the current arm state.
           - Create `current_pos`: block -> ('on', support) or ('on-table', None) or ('holding', None).
           - Create `current_above`: support -> block_above.
           - Determine `current_arm_state`: ('empty', None) or ('holding', block).
        3. Iterate through all blocks identified during initialization (`all_blocks`).
        4. For each block `b`:
           a. Determine its current position (`current_pos.get(b, ('unknown', None))`)
              and its goal position (`goal_pos.get(b, ('unknown', None))`).
           b. If the current position is different from the goal position,
              increment `h` by 1. This counts blocks that are on the wrong
              support (or held when they shouldn't be, or on table when they
              should be on a block, etc.).
           c. Determine the block currently above `b` (`current_above.get(b, None)`)
              and the block that should be above `b` in the goal state
              (`goal_above.get(b, None))`.
           d. If the block currently above `b` is different from the block
              that should be above `b` in the goal state, increment `h` by 1.
              This counts blocks that have the wrong block on top, or have
              something on top when they should be clear, or are clear when
              something should be on top.
        5. Compare the current arm state (`current_arm_state`) with the goal
           arm state (`goal_arm_state`). If they are different, increment `h` by 1.
        6. Return the final value of `h`.

    The heuristic is 0 if and only if for every block, its position is correct
    AND the block above it is correct AND the arm state matches the goal arm state.
    This corresponds exactly to all relevant goal state facts being satisfied
    ('(on X Y)', '(on-table Z)', '(clear W)', and arm state).
    """
    def __init__(self, task):
        self.goals = task.goals

        self.all_blocks = set()
        self.goal_pos = {} # block -> ('on', support_block) or ('on-table', None)
        self.goal_above = {} # support_block -> block_above
        self.goal_arm_state = ('empty', None) # Default goal arm state

        # Collect all blocks from initial state and goal state
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] in ['on', 'on-table', 'holding']:
                self.all_blocks.add(parts[1])
                if parts[0] == 'on':
                    self.all_blocks.add(parts[2])

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                self.goal_pos[block] = ('on', support)
                self.goal_above[support] = block
                self.all_blocks.add(block)
                self.all_blocks.add(support)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_pos[block] = ('on-table', None)
                self.all_blocks.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_state = ('empty', None)
            elif predicate == 'holding':
                block = parts[1]
                self.goal_arm_state = ('holding', block)
                self.all_blocks.add(block)

        # Ensure all blocks from initial state that weren't in goal_pos get a default
        # goal position. In standard BW, all blocks have a goal location, but this
        # makes the heuristic more robust. Assume if not in goal_pos, its goal is on_table.
        # This might not be strictly correct for all BW variants, but is a reasonable default.
        # However, the problem description implies standard BW where all blocks have a goal location.
        # Let's stick to the assumption that all blocks appear in goal on/on-table facts.
        # If a block appears in initial state but not goal facts, its goal_pos will be missing.
        # The .get() with default ('unknown', None) handles this, but comparing ('unknown', None)
        # might not be ideal. Let's rely on standard BW problem structure.

    def __call__(self, node):
        state = node.state

        current_pos = {} # block -> ('on', support) or ('on-table', None) or ('holding', None)
        current_above = {} # support_block -> block_above
        current_arm_state = ('empty', None)

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_pos[block] = ('on', support)
                current_above[support] = block
            elif predicate == 'on-table':
                block = parts[1]
                current_pos[block] = ('on-table', None)
            elif predicate == 'holding':
                block = parts[1]
                current_pos[block] = ('holding', None)
                current_arm_state = ('holding', block)
            # clear and arm-empty are used, but don't define block positions directly

        h = 0

        # Compare current state to goal state for each block
        for block in self.all_blocks:
            # Get goal position. If block not in goal_pos (shouldn't happen in standard BW),
            # treat its goal position as unknown, which will likely cause a mismatch.
            goal_p = self.goal_pos.get(block, ('unknown', None))
            # Get current position. If block not in current_pos (e.g., held),
            # its position is defined by holding. If not on/on-table/holding, this is an invalid state.
            # Use .get() with a default that ensures mismatch if block is missing.
            current_p = current_pos.get(block, ('missing', None))


            # Check if block's own position is incorrect
            if current_p != goal_p:
                h += 1

            # Get goal block above, default to None (meaning should be clear)
            goal_a = self.goal_above.get(block, None)
            # Get current block above, default to None (meaning is clear)
            current_a = current_above.get(block, None)

            # Check if the block directly above is incorrect
            if current_a != goal_a:
                 h += 1

        # Check arm state mismatch
        if current_arm_state != self.goal_arm_state:
             h += 1

        # The heuristic is 0 iff all goal conditions related to block positions,
        # blocks above, and arm state are met. This corresponds to the goal state.
        # If the state is not the goal state, at least one goal fact is false,
        # which should result in h > 0.

        return h
