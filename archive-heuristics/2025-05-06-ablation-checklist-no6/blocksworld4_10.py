from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of structural mismatches between the current state
    and the goal state. It counts two types of mismatches for each block:
    1. The block is not on the correct block or table below it, if its goal
       position is specified.
    2. The block does not have the correct block or clear condition immediately
       above it, if its goal top condition is specified.

    This heuristic is non-admissible and designed for greedy best-first search.
    It returns 0 if the goal state is reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block
        and identifying all blocks in the problem instance.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Build goal_below and goal_above maps
        self.goal_below = {} # block -> block_below_or_table
        self.goal_above = {} # block -> block_above_or_clear

        # Identify all blocks from initial state and goal
        all_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, below = parts[1], parts[2]
                self.goal_below[block] = below
                self.goal_above[below] = block # Assuming only one block can be on another in goal
                all_blocks.add(block)
                all_blocks.add(below)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_below[block] = 'table'
                all_blocks.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_above[block] = 'clear'
                all_blocks.add(block)
            # Other predicates like 'holding' or 'arm-empty' are state predicates,
            # not typically used to define the desired structure in Blocksworld goals.

        for fact in self.initial_state:
             parts = get_parts(fact)
             for part in parts[1:]:
                 if part != 'arm-empty': # arm-empty is not a block
                     all_blocks.add(part)

        # Remove 'table' from the set of blocks as it's a location, not a block object
        if 'table' in all_blocks:
             all_blocks.remove('table')

        self.all_blocks = list(all_blocks) # Store as list for consistent iteration order

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        task = node.task # Access task object to check goal_reached

        # Return 0 if the goal state is reached
        if task.goal_reached(state):
             return 0

        # Build current_below and current_above maps based on the current state
        current_below = {}
        current_above = {}

        # Find the block being held, if any
        held_block = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'holding':
                held_block = parts[1]
                break

        # Populate current_below and current_above for all blocks in the problem instance
        for block in self.all_blocks:
            if block == held_block:
                current_below[block] = 'holding'
                current_above[block] = 'holding'
            else:
                # Find what's below the block
                found_below = False
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'on' and parts[1] == block:
                        current_below[block] = parts[2]
                        found_below = True
                        break # Found its base
                    elif parts[0] == 'on-table' and parts[1] == block:
                        current_below[block] = 'table'
                        found_below = True
                        break # Found its base
                # If not found_below, the block must be the held_block, which is handled above.
                # If a block is in all_blocks but not in state facts as on/on-table/holding,
                # this indicates an inconsistent state representation. We assume valid states.

                # Find what's above the block
                found_above = False
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'on' and parts[2] == block:
                        current_above[block] = parts[1]
                        found_above = True
                        break # Found block on top
                # If nothing is on top, check if it's clear
                if not found_above:
                    if f'(clear {block})' in state:
                        current_above[block] = 'clear'
                        found_above = True
                    # If not found_above and not clear, state is inconsistent (a block
                    # is neither clear nor has something on it). We assume valid states.


        h = 0
        # Iterate through all blocks in the problem instance to calculate mismatches
        for block in self.all_blocks:
            # Cost for incorrect base:
            # If the goal specifies what should be below this block, and the current state differs.
            goal_base = self.goal_below.get(block, None)
            current_base = current_below.get(block, None)

            if goal_base is not None and current_base != goal_base:
                h += 1

            # Cost for incorrect top:
            # If the goal specifies what should be on top of this block (or that it should be clear),
            # and the current state differs.
            goal_top = self.goal_above.get(block, None)
            current_top = current_above.get(block, None)

            if goal_top is not None and current_top != goal_top:
                 h += 1

        return h
