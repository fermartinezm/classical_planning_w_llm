from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real environment, this would be imported.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

# Helper functions (assuming they are available or included)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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

    Estimates the number of actions needed based on misplaced blocks and obstructions.
    Counts:
    1. Goal blocks that are not on their correct base, plus blocks currently on top of them.
    2. Blocks on top of correctly based goal blocks that are not supposed to be there
       according to the goal stack structure.
    3. Penalty if the arm is holding a block.

    This heuristic is non-admissible and aims to guide a greedy best-first search
    by prioritizing states where more blocks are in their correct positions or
    fewer blocks are obstructing goal positions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        super().__init__(task)

        # Build goal_below map and identify goal_blocks
        self.goal_below = {}
        self.goal_blocks = set()
        # Find all blocks mentioned in 'on' goals
        blocks_in_on_goals = set()
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                x, y = get_parts(goal)[1:]
                self.goal_below[x] = y
                blocks_in_on_goals.add(x)
                blocks_in_on_goals.add(y)

        # Find blocks that are bases of goal stacks (on-table and not supporting anything in goal)
        goal_bases = set()
        # Collect all blocks that are *above* something in the goal (i.e., appear as the first argument in an 'on' goal)
        blocks_above_in_goal = set(self.goal_below.keys())
        for goal in self.goals:
            if match(goal, "on-table", "*"):
                z = get_parts(goal)[1]
                # Check if z is NOT a block that something else is ON in the goal
                if z not in blocks_above_in_goal:
                     self.goal_below[z] = 'table'
                     goal_bases.add(z)

        self.goal_blocks = blocks_in_on_goals | goal_bases

        # Build goal_blocks_above map (blocks that should be directly or indirectly above a block in the goal stack)
        self.goal_blocks_above = {}
        # Start tracing up from bases
        for base in self.goal_blocks:
            if self.goal_below.get(base) == 'table': # Found a base
                current = base
                # Trace up the stack from the base
                block_above = None
                for goal in self.goals:
                    if match(goal, "on", "*", current):
                        block_above = get_parts(goal)[1]
                        break

                while block_above is not None:
                    if current not in self.goal_blocks_above:
                        self.goal_blocks_above[current] = set()
                    self.goal_blocks_above[current].add(block_above)
                    # Add everything that should be above block_above
                    if block_above in self.goal_blocks_above:
                         self.goal_blocks_above[current].update(self.goal_blocks_above[block_above])

                    # Move up one level
                    next_block_above = None
                    for goal in self.goals:
                         if match(goal, "on", "*", block_above):
                             next_block_above = get_parts(goal)[1]
                             break
                    current = block_above
                    block_above = next_block_above


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current_below and current_above maps
        current_below = {}
        current_above = {}
        all_blocks_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                x, y = parts[1:]
                current_below[x] = y
                current_above[y] = x
                all_blocks_in_state.add(x)
                all_blocks_in_state.add(y)
            elif parts[0] == "on-table":
                x = parts[1]
                current_below[x] = 'table'
                all_blocks_in_state.add(x)
            elif parts[0] == "holding":
                x = parts[1]
                current_below[x] = 'holding'
                all_blocks_in_state.add(x)

        h = 0

        # Part 1: Penalize blocks in goal stacks that are on the wrong base, plus everything on top of them.
        misplaced_goal_blocks = set()
        for block in self.goal_blocks:
            # A goal block is misplaced if it's not in the state, or its current base is not its goal base.
            if block not in all_blocks_in_state or current_below.get(block) != self.goal_below.get(block):
                misplaced_goal_blocks.add(block)

        for block in misplaced_goal_blocks:
            h += 1 # Cost to move the block itself
            # Add cost for clearing blocks on top of this block in the current stack.
            Y = current_above.get(block)
            while Y is not None:
                h += 1
                Y = current_above.get(Y)

        # Part 2: Penalize blocks on top of correctly based goal blocks if the block on top is not supposed to be there.
        correctly_based_goal_blocks = self.goal_blocks - misplaced_goal_blocks

        for block in correctly_based_goal_blocks:
            # Check if the block is actually in the state and correctly based
            if block in all_blocks_in_state and current_below.get(block) == self.goal_below.get(block):
                Y = current_above.get(block)
                while Y is not None:
                    # Y is on top of block. Is Y supposed to be above block in the goal?
                    # Y is supposed to be above block if block is in goal_blocks_above and Y is in goal_blocks_above[block].
                    # If Y is not in goal_blocks_above[block], then Y is an obstruction.
                    if block not in self.goal_blocks_above or Y not in self.goal_blocks_above[block]:
                         # Y is an obstruction on a correctly placed block.
                         h += 1 # Cost to move Y out of the way.
                    # Move up the current stack
                    Y = current_above.get(Y)

        # Part 3: Add penalty if arm is not empty.
        # Holding a block implies it's not in its final place and requires an action.
        if "(arm-empty)" not in state:
            h += 1

        return h
