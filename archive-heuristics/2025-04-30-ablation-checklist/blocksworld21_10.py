from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not currently
    on their correct support (another block or the table) as specified
    in the goal state. It counts each block that is misplaced relative
    to the object or table directly below it in the goal configuration.

    # Assumptions:
    - The goal specifies the desired support (block or table) for some blocks.
    - Blocks that are not specified in 'on' or 'on-table' goals are ignored
      by this heuristic regarding their position, but their presence might
      obstruct goal achievement (this heuristic doesn't explicitly model
      clearing costs, but misplaced blocks needing clearing contribute
      implicitly if the blocks on top are also misplaced).

    # Heuristic Initialization
    - Extracts the desired support (block or table) for each block
      mentioned in the goal state's `on` or `on-table` predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the goal configuration: For each block `B` that is the
        first argument of an `(on B C)` or `(on-table B)` goal predicate,
        record its desired support (`C` or 'table').
    2.  Identify the current configuration: For each block `B`, find what
        it is currently on (`(on B C)` in state) or if it's on the table
        (`(on-table B)` in state) or held (`(holding B)` in state).
    3.  Initialize the heuristic value `h = 0`.
    4.  Iterate through each block `B` for which a goal support was identified
        in step 1.
    5.  For each such block `B`, compare its current support (from step 2)
        with its desired goal support (from step 1).
    6.  If the current support does *not* match the goal support, increment `h`.
        A block being held (`holding`) means it's not on a block or table,
        so it won't match any standard `on` or `on-table` goal support.
    7.  Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support configurations.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Build a dictionary mapping each block to its desired support (block or 'table')
        # based on the goal state.
        self.goal_config = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                # Goal is (on block1 block2)
                block1, block2 = parts[1], parts[2]
                self.goal_config[block1] = block2
            elif parts[0] == "on-table":
                # Goal is (on-table block)
                block = parts[1]
                self.goal_config[block] = 'table'
        
        # Note: Blocks only appearing in (clear block) goals are not included
        # in goal_config as their support isn't explicitly specified, only that
        # nothing should be on them. This heuristic focuses on the 'on' structure.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # Build a dictionary mapping each block to its current support (block, 'table', or 'arm')
        # based on the current state.
        current_support_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                # Fact is (on block1 block2)
                block1, block2 = parts[1], parts[2]
                current_support_map[block1] = block2
            elif parts[0] == "on-table":
                # Fact is (on-table block)
                block = parts[1]
                current_support_map[block] = 'table'
            elif parts[0] == "holding":
                # Fact is (holding block)
                block = parts[1]
                current_support_map[block] = 'arm' # Special marker for being held

        heuristic_value = 0

        # Iterate through the blocks whose goal position is specified
        for block, goal_support in self.goal_config.items():
            # Find the current support for this block.
            # Use .get() because the block might not be on anything or the table
            # if it's currently held, or if it's not in the state facts at all
            # (though in blocksworld, blocks are usually always on something or held).
            current_support = current_support_map.get(block)

            # If the current support does not match the goal support, increment heuristic.
            # This counts blocks that are not on the correct block or table.
            if current_support != goal_support:
                heuristic_value += 1

        return heuristic_value

