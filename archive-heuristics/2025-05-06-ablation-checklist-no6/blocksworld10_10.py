from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and empty facts
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return [] # Should not happen with valid PDDL state facts
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to achieve the goal configuration
    of blocks. It counts unsatisfied goal position predicates (on, on-table)
    and blocks that are obstructing goal-relevant positions.

    Heuristic value is 0 only for goal states.
    Heuristic value is finite for solvable states.
    Designed for greedy best-first search (not admissible).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the goal stack structure.
        """
        self.goals = task.goals

        # Build goal structure mappings
        self.goal_below = {} # block -> block_below or 'table'
        self.goal_on_top = {} # block_below -> block_on_top (reverse mapping for 'on' goals)
        self.goal_clear = set() # blocks that should be clear in the goal

        # Collect goal position facts and build mappings
        self.goal_position_facts = set() # Store (on A B) and (on-table X) goal facts

        # Collect all blocks that are at the bottom of an 'on' goal fact
        blocks_in_goal_on_bottom = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                block_on_top, block_below = parts[1], parts[2]
                self.goal_below[block_on_top] = block_below
                self.goal_on_top[block_below] = block_on_top
                self.goal_position_facts.add(goal)
                blocks_in_goal_on_bottom.add(block_below)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_below[block] = 'table'
                self.goal_position_facts.add(goal)
            elif predicate == 'clear':
                 self.goal_clear.add(parts[1])

        # Identify blocks that are relevant as potential bases for obstructions.
        # These are blocks that are below something in a goal stack
        # OR are explicitly supposed to be clear in the goal.
        self.relevant_base_blocks = blocks_in_goal_on_bottom | self.goal_clear
        self.relevant_base_blocks.discard('table') # 'table' cannot be a base for 'on' facts

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state # state is a frozenset of fact strings

        # Parse current state to build 'on_top' mapping and find held block
        current_on_top = {} # block_below -> block_on_top (only for 'on' facts)
        held_block = None

        # Convert state frozenset to a set for faster lookups
        state_set = set(state)

        for fact in state_set:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                block_on_top, block_below = parts[1], parts[2]
                current_on_top[block_below] = block_on_top
            elif predicate == 'holding':
                held_block = parts[1]
            # Ignore 'on-table', 'clear', 'arm-empty' for these mappings

        h = 0

        # 1. Count unsatisfied goal position predicates (on, on-table) (cost 2 each)
        # This counts blocks that are not on their correct immediate support (or table)
        for goal_fact in self.goal_position_facts:
            if goal_fact not in state_set:
                h += 2

        # 2. Count blocks that are obstructing goal-relevant positions (cost 2 each)
        # Iterate through blocks that are currently on top of other blocks.
        for block_below, block_on_top in current_on_top.items():
            B = block_below
            C = block_on_top

            # Check if B is a relevant base block (below something in goal or supposed to be clear)
            if B in self.relevant_base_blocks:
                # Find the block A that is supposed to be on B in the goal, if any.
                goal_block_on_top = self.goal_on_top.get(B) # None if B should be clear or is top of goal stack

                # C is an obstruction if C is not the block that is supposed to be on B in the goal.
                if C != goal_block_on_top:
                    h += 2

        # 3. Add cost if the arm is holding a block (cost 1)
        # This penalizes having the arm busy, as it might need to be free
        # to move other blocks counted in steps 1 or 2.
        if held_block is not None:
            h += 1

        # The heuristic is 0 if and only if:
        # 1. All goal position facts (on, on-table) are satisfied.
        # 2. There are no obstructions on relevant base blocks. This implies that for any block B that is below something in the goal or supposed to be clear, either nothing is on B, or the correct block A is on B (if B is below A in the goal). Combined with step 1, this means all goal 'on' and 'on-table' facts are correctly formed, and any block supposed to be clear is indeed clear of incorrect blocks.
        # 3. The arm is empty.
        # These conditions together define a goal state in Blocksworld.

        return h
