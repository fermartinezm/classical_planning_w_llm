from fnmatch import fnmatch
# Assuming Heuristic base class is available in 'heuristics.heuristic_base'
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.strip():
        return []
    # Remove surrounding parentheses and split by whitespace
    # Ensure fact starts with '(' and ends with ')'
    fact_stripped = fact.strip()
    if fact_stripped.startswith('(') and fact_stripped.endswith(')'):
        return fact_stripped[1:-1].split()
    else:
        # Handle facts without parentheses if necessary, though PDDL facts usually have them
        # print(f"Warning: Fact does not have expected parentheses format: {fact}")
        return fact_stripped.split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed by counting:
    1. Blocks that are not on their correct goal base (block or table).
    2. Blocks that are on top of another block in the state, but are not
       the block that should be on top according to the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal structure and all blocks."""
        super().__init__(task) # Call base class constructor

        # Build the goal stack structure: block -> block_below_it_in_goal or 'table'
        self.goal_below = {}
        # Build the goal stack structure: block -> block_above_it_in_goal or None
        self.goal_above = {}
        self.all_blocks = set()

        # Extract blocks and goal structure from goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                self.goal_below[block] = below_block
                self.goal_above[below_block] = block # Assuming unique block above
                self.all_blocks.add(block)
                self.all_blocks.add(below_block)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_below[block] = 'table'
                self.all_blocks.add(block)
            elif len(parts) > 1: # Add blocks from other goal predicates like 'clear'
                 self.all_blocks.add(parts[1])


        # Extract blocks from initial state facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or malformed facts
             predicate = parts[0]
             # Consider predicates that involve objects
             if predicate in ['on', 'holding', 'clear', 'on-table'] and len(parts) > 1:
                 self.all_blocks.add(parts[1])
             if predicate == 'on' and len(parts) > 2:
                 self.all_blocks.add(parts[2])

        # For blocks not explicitly in goal 'on' or 'on-table' facts, assume goal is on table
        # Ensure all blocks identified are in goal_below and goal_above maps
        for block in list(self.all_blocks): # Iterate over a copy as we might add to goal_below
             if block not in self.goal_below:
                  self.goal_below[block] = 'table' # Default goal is on table
             if block not in self.goal_above:
                  self.goal_above[block] = None # Default goal is nothing on top


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state # frozenset of facts

        # Build the current stack structure: block -> block_below_it_in_state or 'table' or 'holding'
        current_below = {}
        # Build current_above: block -> block_above_it_in_state or None
        current_above = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                current_below[block] = below_block
                current_above[below_block] = block # Assuming unique block on top
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                 block = parts[1]
                 current_below[block] = 'holding' # Special location for held block
                 # A held block cannot have anything below it in the stack structure
                 # and nothing can be on top of it.

        h = 0
        for block in self.all_blocks:
            current_base = current_below.get(block) # Get current base, None if block is not on/on-table/holding

            # Rule 1: Block is in the wrong place relative to its base
            # If block is holding, its base is 'holding'. Its goal base is never 'holding'.
            # So if current_base is 'holding', it's always misplaced according to this rule.
            # If current_base is None, the block is not in a standard location (on/on-table/holding).
            # This implies an invalid state representation or the block doesn't exist.
            # Assuming valid states, current_base will be 'table', a block, or 'holding' for existing blocks.
            # If a block exists but is not in current_below, it's misplaced.
            # We can treat None as a base different from any goal base.
            goal_base = self.goal_below.get(block) # Should always exist due to __init__ logic

            if current_base != goal_base:
                 h += 1

            # Rule 2: A block on top is wrong/blocking
            # This check only applies if the block is *not* holding something,
            # as nothing can be on top of a held block.
            if current_base != 'holding':
                block_on_top_in_state = current_above.get(block)
                block_on_top_in_goal = self.goal_above.get(block) # Should always exist due to __init__ logic

                if block_on_top_in_state is not None: # There is a block on top in the state
                     if block_on_top_in_goal is None or block_on_top_in_state != block_on_top_in_goal:
                          # The block on top is wrong or shouldn't be there
                          h += 1 # Add 1 for the blocking block

        return h
