# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
# If running standalone, define a dummy Heuristic class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch

# Helper function to parse fact strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
        This heuristic estimates the cost to reach the goal by counting
        the number of blocks that are not in their correct goal position
        (either on another block or on the table) and the number of blocks
        that are goal tops but are not clear. It aims to penalize states
        where blocks are misplaced relative to their goal base or where
        goal tops are blocked.

    Assumptions:
        - The input state and goal are valid Blocksworld states represented
          as frozensets of strings.
        - Goal facts include (on ?x ?y), (on-table ?x), and potentially (clear ?x).
        - (arm-empty) goal fact is ignored by the heuristic.
        - The goal configuration forms valid stacks (no cycles, unique base for stacks).

    Heuristic Initialization:
        The constructor processes the goal facts to precompute:
        - goal_pos: A dictionary mapping each block that appears as the first
                    argument in an (on ?x ?y) or (on-table ?x) goal fact to
                    the block it should be on (?y) or 'table' respectively.
        - all_goal_objects: A set of all block names that appear in any goal fact
                          (on, on-table, or clear).
        - goal_tops: A set of block names that appear in any goal fact but do not
                     appear as the block being stacked upon (second argument)
                     in any (on ?x ?y) goal fact. These are the blocks that should
                     be at the top of their respective goal stacks or on the table
                     with nothing on them, according to the goal.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to determine the current position of each block
           (on which block, on the table, or held) and which block is currently
           on top of each block.
           - Create current_pos: map block -> block_below or 'table' or 'holding'.
           - Create current_top: map block -> block_on_top or None.
        2. Initialize the heuristic value `h` to 0.
        3. Iterate through each block that is involved in any goal fact (`all_goal_objects`).
        4. For each block `B`:
           a. Check if `B` has a defined goal position (i.e., it's a key in `goal_pos`).
              If yes, find its expected position (`goal_pos[B]`). Find its current
              position (`current_pos.get(B)`). If the block is not found in the
              current state's position map (meaning it's not on anything, on table,
              or held - potentially an invalid state or block not relevant to state parsing)
              or if its current position does not match the expected position,
              increment `h` by 1. This penalizes blocks that are not on the correct
              block below them or not on the table when they should be.
           b. Check if `B` is a goal top (i.e., it's in the `goal_tops` set).
              If yes, find what is currently on top of `B` (`current_top.get(B)`).
              If there is any block on top (`current_top.get(B)` is not None),
              increment `h` by 1. This penalizes goal tops that are not clear.
        5. Return the final value of `h`.

        This heuristic is non-admissible. It is 0 if and only if all blocks
        with a goal position are in that position, and all goal tops are clear,
        which corresponds to the goal state (assuming arm-empty is ignored).
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals

        self.goal_pos = {} # Map block -> block_below or 'table'
        blocks_underneath_in_goal = set()
        self.all_goal_objects = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, under_block = parts[1], parts[2]
                self.goal_pos[block] = under_block
                blocks_underneath_in_goal.add(under_block)
                self.all_goal_objects.add(block)
                self.all_goal_objects.add(under_block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_pos[block] = 'table'
                self.all_goal_objects.add(block)
            elif parts[0] == 'clear':
                 block = parts[1]
                 self.all_goal_objects.add(block)
            # Ignore 'arm-empty' goal

        # Identify goal tops: blocks in goals that are not underneath any other block in goal 'on' facts
        self.goal_tops = self.all_goal_objects - blocks_underneath_in_goal


    def __call__(self, node):
        state = node.state

        current_pos = {} # Map block -> block_below or 'table' or 'holding'
        current_top = {} # Map block -> block_on_top or None

        # Build current_pos and current_top maps from state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                top, bottom = parts[1], parts[2]
                current_pos[top] = bottom
                current_top[bottom] = top
            elif parts[0] == 'on-table':
                block = parts[1]
                current_pos[block] = 'table'
            elif parts[0] == 'holding':
                block = parts[1]
                current_pos[block] = 'holding'
            # 'clear' and 'arm-empty' don't define position or top/bottom relationship directly

        h = 0

        # Consider blocks mentioned in goals
        for block in self.all_goal_objects:
            # Penalty 1: Block is not in its correct goal position (on/on-table)
            if block in self.goal_pos:
                expected_pos = self.goal_pos[block]
                # Check if the block's current position matches the expected position
                # Use .get() with a default to handle blocks not found in current_pos (e.g., held)
                if current_pos.get(block) != expected_pos:
                     h += 1

            # Penalty 2: Block is a goal top but is not clear
            if block in self.goal_tops:
                 # Check if anything is on top of this block in the current state
                 # Use .get() with a default (None) for blocks not in current_top keys
                 if current_top.get(block) is not None:
                      h += 1

        return h
