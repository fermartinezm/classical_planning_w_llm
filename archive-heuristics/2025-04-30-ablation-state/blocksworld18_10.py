from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# The match function is not strictly used in the heuristic logic but was in examples.
# Keeping it for completeness or if required by the environment.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of unsatisfied goal positioning predicates ('on'
    and 'on-table') and the number of blocks that are incorrectly positioned
    on top of blocks that are part of the goal configuration.

    # Assumptions
    - Standard Blocksworld rules apply.
    - The goal specifies a configuration of blocks stacked on each other or the table.
    - The arm is typically empty in the goal state (implicitly handled as 'holding'
      facts are not goal predicates).

    # Heuristic Initialization
    - Parse the goal state to identify the desired 'on' relationships and
      blocks that should be on the table. This defines the goal stack
      configuration.
    - Create mappings to quickly look up the desired block below and block above
      for any block involved in the goal configuration.
    - Identify the set of all blocks involved in the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state: Determine the current position of every block
       (on another block, on the table, or held by the arm).
    2. Identify the goal configuration: From the parsed goal, create mappings
       indicating which block should be on which other block (`goal_on_map`)
       and which blocks should be on the table (`goal_on_table_set`). Also,
       create a reverse mapping (`goal_above_map`) to quickly find which block
       (if any) should be on a given block in the goal. Identify the set of
       all blocks involved in these goal predicates (`goal_blocks`).
    3. Calculate the first component of the heuristic: Iterate through the
       desired 'on' relationships in `goal_on_map`. For each `(block_above, block_below)`
       pair, if `block_above` is not currently on `block_below`, increment the
       heuristic count. Then, iterate through the blocks in `goal_on_table_set`.
       For each such block, if it is not currently on the table, increment the
       heuristic count.
    4. Calculate the second component of the heuristic (blocking blocks):
       Iterate through the current 'on' relationships in the state (block X is
       currently on block B). If block B is part of the goal configuration
       (`block_B` is in `goal_blocks`), check if block X is the block that is
       supposed to be on B according to the goal (`goal_above_map.get(block_B)`).
       If B is in the goal configuration and X is NOT the correct block to be on B
       in the goal, then X is considered a blocking block that needs to be moved.
       Increment the heuristic count for each such blocking block X.
    5. The total heuristic value is the sum of the counts from steps 3 and 4.
       A heuristic value of 0 indicates the goal state has been reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Build goal configuration maps
        self.goal_on_map = {} # block -> block_it_should_be_on
        self.goal_on_table_set = set() # {block}
        self.goal_above_map = {} # block_below -> block_above (for goal)

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "on":
                block_above, block_below = parts[1], parts[2]
                self.goal_on_map[block_above] = block_below
                # Assuming unique blocks above in goal for simplicity of this heuristic
                self.goal_above_map[block_below] = block_above
            elif parts and parts[0] == "on-table":
                block = parts[1]
                self.goal_on_table_set.add(block)
            # Ignore 'clear' and 'arm-empty' goals for building stack structure

        # Identify all blocks involved in the goal configuration (those mentioned in on or on-table goals)
        self.goal_blocks = set(self.goal_on_map.keys()) | self.goal_on_table_set
        self.goal_blocks.update(v for v in self.goal_on_map.values() if v != 'table')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_on_map = {} # block -> block_it_is_on
        current_on_table_set = set() # {block}
        # 'holding' and 'clear' facts are not directly used in this heuristic's counts

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "on":
                current_on_map[parts[1]] = parts[2]
            elif parts and parts[0] == "on-table":
                current_on_table_set.add(parts[1])

        h = 0

        # Part 1: Count unsatisfied goal 'on' and 'on-table' predicates.
        for block_above, block_below in self.goal_on_map.items():
            # Check if block_above is currently on block_below
            if current_on_map.get(block_above) != block_below:
                h += 1

        for block_on_table in self.goal_on_table_set:
            # Check if block_on_table is currently on the table
            if block_on_table not in current_on_table_set:
                 # If it's not on the table, it must be on another block or held.
                 # In either case, the 'on-table' goal is not met.
                h += 1

        # Part 2: Count blocks blocking a goal stack block
        # A block X is blocking if it is currently on top of a block B,
        # and B is part of the goal configuration, AND X is NOT the block
        # that should be on B in the goal.
        for block_X, block_B in current_on_map.items(): # block_X is currently on block_B
            if block_B in self.goal_blocks: # block_B is part of a goal stack
                goal_above = self.goal_above_map.get(block_B) # What block should be on block_B in the goal? (or None)
                if block_X != goal_above:
                    # block_X is on block_B, block_B is in goal_blocks, and block_X is not the correct block to be on block_B.
                    # block_X is blocking.
                    h += 1

        return h
