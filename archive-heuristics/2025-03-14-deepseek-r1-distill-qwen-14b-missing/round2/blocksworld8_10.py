from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of blocks that are not in their correct position in the goal stack and the number of blocks above them in the current stack that are not part of the goal stack above them.

    # Assumptions:
    - The goal is to build a specific stack of blocks as defined in the goal facts.
    - Each block can only be moved one at a time.
    - The heuristic counts the number of blocks that need to be moved to achieve the goal stack.

    # Heuristic Initialization
    - Extract the goal stack from the goal facts.
    - Build the current stack from the current state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal stack from the goal facts.
    2. Build the current stack from the current state.
    3. For each block in the goal stack, count the number of blocks above it in the current stack that are not part of the goal stack above it.
    4. Sum these counts to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting the goal stack."""
        self.task = task
        # Extract the goal stack
        goal_facts = [fact for fact in task.goals if fact.startswith('(on ')]
        self.goal_on = {}
        for fact in goal_facts:
            x, y = self.parse_on_fact(fact)
            self.goal_on[x] = y
        # Build the goal stack
        current = 'table'
        self.goal_stack = []
        while current in self.goal_on:
            next_block = self.goal_on[current]
            self.goal_stack.append(next_block)
            current = next_block

    def parse_on_fact(self, fact):
        """Parse an 'on' fact into (x, y)."""
        parts = fact[1:-1].split()
        return parts[1], parts[2]

    def __call__(self, node):
        """Compute the heuristic value."""
        state = node.state
        # Build the current stack
        current_on = {}
        for fact in state:
            if fact.startswith('(on '):
                x, y = self.parse_on_fact(fact)
                current_on[x] = y
        # Build the current stack
        current_stack = []
        current = 'table'
        while current in current_on:
            next_block = current_on[current]
            current_stack.append(next_block)
            current = next_block
        # Compute the heuristic
        heuristic = 0
        for block in self.goal_stack:
            # Find the index of the block in the current stack
            index = None
            for i, b in enumerate(current_stack):
                if b == block:
                    index = i
                    break
            if index is None:
                # The block is not in the current stack
                heuristic += 1
                continue
            # Count the number of blocks above it that are not in the goal stack above it
            for b in current_stack[index + 1:]:
                if b not in self.goal_stack[self.goal_stack.index(block) + 1:]:
                    heuristic += 1
        return heuristic
