# Assuming heuristic_base.py exists and defines a Heuristic base class
# Example:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic # Uncomment if running in a specific framework

# Define a dummy Heuristic base class if not provided
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting blocks that are in the wrong immediate position, blocks that
    are wrongly stacked on other blocks, and blocks that are part of the goal
    configuration but are not clear.

    # Assumptions
    - The goal specifies a configuration of blocks stacked on each other or the table.
    - The heuristic counts conditions that need to be fixed, assuming each fix
      requires approximately one action or a small constant number of actions.
      It is non-admissible.

    # Heuristic Initialization
    - Extracts the goal configuration: which block should be on which other block
      or on the table (`goal_on` mapping).
    - Identifies all blocks involved in the goal configuration (`goal_blocks` set).
    - Stores goal facts related to `on` and `on-table` for quick lookup.
    - Collects all possible block objects from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is the sum of three components:

    1.  **Misplaced Immediate Count:** Count the number of blocks `B` that are part
        of the goal configuration (`B` is in `goal_on`) but are not currently
        in their correct immediate position relative to the block or table below them
        as specified by the goal (`current_on[B]` is not `goal_on[B]`).

    2.  **Not Clear Goal Block Count:** Count the number of blocks `B` that are part
        of the goal configuration (`B` is in `goal_blocks`) but are currently
        not clear (`(clear B)` is false in the state). These blocks need to be cleared
        to allow other blocks to be stacked on them or to be picked up themselves.

    3.  **Wrongly Stacked Count:** Count the number of blocks `Y` that are currently
        stacked on top of another block `B` (`(on Y B)` is true in the state), but
        this specific `on` relationship `(on Y B)` is *not* part of the goal
        configuration. These blocks `Y` are obstructing the goal structure and
        need to be moved out of the way.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration details.
        """
        self.goals = task.goals
        self.goal_on = {} # Map block -> block_below or 'table'
        self.goal_blocks = set() # Blocks mentioned in goal on/on-table/clear
        self.goal_on_facts = set() # Set of (on Y B) goal facts
        self.goal_on_table_facts = set() # Set of (on-table B) goal facts

        # Extract goal structure and identify goal blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, underob = parts[1], parts[2]
                self.goal_on[block] = underob
                self.goal_blocks.add(block)
                self.goal_blocks.add(underob)
                self.goal_on_facts.add(goal)
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_on[block] = 'table'
                self.goal_blocks.add(block)
                self.goal_on_table_facts.add(goal)
            elif parts[0] == "clear":
                 block = parts[1]
                 self.goal_blocks.add(block) # Include blocks from clear goals

        # Collect all possible blocks from initial state and goals
        self.all_blocks = set(self.goal_blocks)
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Add all arguments that look like objects (start with a letter)
             for part in parts[1:]:
                 if part and part[0].isalpha():
                     self.all_blocks.add(part)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Parse current state
        current_on = {} # Map block -> block_below or 'table' or 'arm'
        current_clear = set() # Set of clear blocks
        current_on_facts = set() # Set of (on Y B) state facts
        # current_holding = None # The block being held, or None (not needed for this heuristic)

        # Initialize all blocks as not clear by default
        for block in self.all_blocks:
            current_on[block] = None # Default location unknown
            # current_clear is a set, default is not in set (not clear)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, underob = parts[1], parts[2]
                current_on[block] = underob
                current_on_facts.add(fact)
            elif parts[0] == "on-table":
                block = parts[1]
                current_on[block] = 'table'
            elif parts[0] == "clear":
                block = parts[1]
                current_clear.add(block)
            elif parts[0] == "holding":
                block = parts[1]
                current_on[block] = 'arm'
                # current_holding = block # Not needed

        heuristic_value = 0

        # Component 1: Misplaced Immediate Count
        # Count blocks that are part of the goal configuration but not in their
        # correct immediate position (on the right block or table).
        for block in self.goal_on: # Only consider blocks that have a goal_on specified
            goal_loc = self.goal_on[block]
            current_loc = current_on.get(block) # Use .get to handle blocks not in state (e.g., in arm)

            # If block is supposed to be on table but isn't
            if goal_loc == 'table' and current_loc != 'table':
                 heuristic_value += 1
            # If block is supposed to be on another block but isn't
            elif goal_loc != 'table' and current_loc != goal_loc:
                 heuristic_value += 1

        # Component 2: Not Clear Goal Block Count
        # Count blocks that are part of the goal configuration but are not clear.
        # These need to be cleared to allow operations.
        for block in self.goal_blocks: # Only consider blocks relevant to the goal
             if block not in current_clear:
                 heuristic_value += 1

        # Component 3: Wrongly Stacked Count
        # Count blocks that are stacked on others in the state, but this stacking
        # is not part of the goal configuration. These need to be moved.
        for fact in current_on_facts:
             if fact not in self.goal_on_facts:
                 heuristic_value += 1

        # Ensure heuristic is 0 only for goal states
        if task.goal_reached(state):
             return 0

        return heuristic_value
