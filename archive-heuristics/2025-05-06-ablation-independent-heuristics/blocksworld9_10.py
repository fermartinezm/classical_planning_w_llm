import re
from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available via import or in the same scope
# from task import Operator, Task

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding parentheses and split by space
    # Handle cases like '(on-table b1)' or '(arm-empty)'
    parts = fact_string.strip("()").split()
    return tuple(parts)

def get_objects_from_facts(facts):
    """Collects all unique object names from a set of facts."""
    objects = set()
    for fact_string in facts:
        parts = parse_fact(fact_string)
        # Arguments are parts[1:]
        for arg in parts[1:]:
            # Objects are typically not predicates or type symbols
            if arg not in ['on', 'on-table', 'clear', 'arm-empty', 'holding', '-', 'object']: # Added '-' and 'object'
                 objects.add(arg)
    return objects

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the number of actions required to reach the goal
        by counting local mismatches between the current state and the goal state
        regarding block positions and stack structure. It counts:
        1. Blocks that are currently on the wrong base (another block or the table)
           compared to their desired base in the goal.
        2. Blocks that currently have the wrong block directly on top of them
           compared to the block that should be directly on top in the goal
           (or if something is on top when nothing should be).
        3. A penalty if the arm is not empty in the current state but should be
           empty in the goal state.

    Assumptions:
        - The heuristic is designed for the standard Blocksworld domain with
          actions pickup, putdown, stack, and unstack.
        - It assumes a STRIPS planning task representation.
        - It assumes all objects relevant to the problem are mentioned in the
          initial state or the goal state.
        - It assumes the goal state defines a complete or partial configuration
          of blocks. Blocks not explicitly given a position in the goal are
          assumed to have no specific required base or block on top in the goal.
        - It assumes the goal typically includes `(arm-empty)`.

    Heuristic Initialization:
        The constructor pre-processes the goal state to build two mappings:
        - `goal_on_map`: Maps each block (that has a specific base in the goal)
          to its desired base (either another block name or the string 'table').
        - `goal_above_map`: Maps each block (that should have another block
          directly on top of it in the goal) to the name of the block that
          should be directly on top. This assumes at most one block is directly
          on top in the goal configuration.
        It also collects a set of all unique block names present in the initial
        state or the goal state.
        It checks if `(arm-empty)` is a goal fact.
        Static facts are not used by this specific heuristic.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Parse the state facts to build similar mappings for the current state:
           - `state_on_map`: Maps each block (that is on another block or the table)
             to its current base.
           - `state_above_map`: Maps each block (that has another block directly
             on top) to the name of the block currently directly on top.
           - Identify the block currently held by the arm, if any.
        2. Initialize the heuristic value `h` to 0.
        3. Iterate through the set of all relevant blocks identified during initialization.
        4. For each block `b`:
           a. Determine its goal base (`goal_base`) from `goal_on_map`. If the block
              is not in `goal_on_map`, its goal base is considered None.
           b. Check if the block is currently on its goal base. A block is on its
              goal base if it's not held AND its base in `state_on_map` matches
              `goal_base`.
           c. If the block has a defined goal base (`goal_base is not None`) AND it
              is NOT currently on its goal base, increment `h` by 1.
              Note: This correctly handles blocks that are held, as they are not
              on any base, and thus not on their goal base (unless goal_base could be 'arm', which it cannot).
           d. Determine the block that should be directly on top of `b` in the goal
              (`goal_block_on_top`) from `goal_above_map`. If nothing should be on top,
              this is None.
           e. Determine the block that is currently directly on top of `b`
              (`current_block_on_top`) from `state_above_map`. If nothing is on top,
              this is None.
           f. If there is currently a block on top of `b` (`current_block_on_top is not None`)
              AND this block is different from the block that should be on top
              (`current_block_on_top != goal_block_on_top`), increment `h` by 1.
        5. Check if `(arm-empty)` is a goal fact (`self.goal_arm_empty` is True). If it is,
           and a block is currently held (`holding_block is not None`), increment `h` by 1.
        6. Return the final value of `h`.
        This heuristic is 0 if and only if the state matches the goal configuration
        in terms of block positions, blocks on top, and arm state.
    """
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_on_map = {} # block -> base (block or 'table')
        self.goal_above_map = {} # base -> block_on_top (block or None)
        self.all_blocks = set()
        self.goal_arm_empty = False

        # Collect all blocks from initial state and goals
        self.all_blocks.update(get_objects_from_facts(task.initial_state))
        self.all_blocks.update(get_objects_from_facts(task.goals))

        # Parse goal facts to build goal_on_map and goal_above_map
        for fact_string in task.goals:
            parts = parse_fact(fact_string)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                self.goal_on_map[block] = base
                # Check if base is a block (not 'table') before adding to goal_above_map
                # Ensure base is a block object that exists
                if base in self.all_blocks:
                     self.goal_above_map[base] = block # Assumes unique block on top in goal
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_on_map[block] = 'table'
            elif parts[0] == 'arm-empty':
                 self.goal_arm_empty = True
            # Ignore clear goals for these maps - they are implicitly handled by above/on-table facts

    def __call__(self, node):
        state = node.state
        h = 0

        # Parse state facts
        state_on_map = {} # block -> base (block or 'table')
        state_above_map = {} # base -> block_on_top
        holding_block = None

        for fact_string in state:
            parts = parse_fact(fact_string)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                state_on_map[block] = base
                state_above_map[base] = block # Assumes unique block on top in state
            elif parts[0] == 'on-table':
                block = parts[1]
                state_on_map[block] = 'table'
            elif parts[0] == 'holding':
                holding_block = parts[1]

        # Heuristic calculation based on Idea 33 refined
        for block in self.all_blocks:
            goal_base = self.goal_on_map.get(block)

            # Check condition 3a: Block is on the wrong base (if it has a goal base)
            if goal_base is not None:
                is_on_goal_base = False
                if holding_block != block: # If block is held, it's not on any base
                    current_base = state_on_map.get(block)
                    if current_base == goal_base:
                        is_on_goal_base = True

                if not is_on_goal_base:
                    h += 1

            # Check condition 3b: Block has the wrong block on top (if something is on it)
            current_block_on_top = state_above_map.get(block)
            goal_block_on_top = self.goal_above_map.get(block)

            if current_block_on_top is not None and current_block_on_top != goal_block_on_top:
                 h += 1
            # Also add cost if nothing is on top currently, but something should be in goal?
            # No, the block that should be on top will be counted when checking its own position.

        # Check condition 5: Arm state mismatch
        if self.goal_arm_empty and holding_block is not None:
             h += 1

        # If the state is the goal state, h must be 0.
        # If h=0, then:
        # - For all blocks with goal_base, current_base must match (implies correct on/on-table facts).
        # - For all blocks, if something is on top, it must be the correct block (implies correct on facts, and correct clear facts for blocks that should be clear).
        # - If goal_arm_empty, holding_block must be None.
        # These conditions together imply the state is the goal state for standard blocksworld.

        return h
