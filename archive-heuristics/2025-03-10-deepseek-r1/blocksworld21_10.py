from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class blocksworld21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - The number of blocks that need to be moved from their current positions.
    - The number of blocks above each misplaced block that must be moved first.
    - The number of blocks in the goal's support chain that are not yet correctly placed.

    # Assumptions
    - The goal specifies a set of 'on', 'on-table', and 'clear' predicates.
    - Blocks not mentioned in the goal can be in any state and are ignored.
    - Each block mentioned in the goal must be moved to its specified position and clear if required.

    # Heuristic Initialization
    - Extract the goal's 'on', 'on-table', and 'clear' predicates.
    - For each block in the goal, determine its goal support and whether it needs to be clear.
    - Precompute the support chain for each block in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block in the goal:
        a. Check if it's on the correct support and clear (if required).
        b. If not, calculate the number of blocks above it (current_above) that must be moved.
        c. Determine the number of blocks in its goal support chain (required_below) that are misplaced.
        d. Add 2 * (current_above + required_below + 1) to the heuristic value.
    2. Sum the costs for all misplaced blocks to get the total heuristic estimate.
    """

    def __init__(self, task):
        self.goal_supports = {}
        self.goal_clear = set()
        self.support_chains = {}

        # Extract goal predicates
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_supports[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_supports[block] = 'table'
            elif parts[0] == 'clear':
                self.goal_clear.add(parts[1])

        # Build support chains for each block in goal_supports
        for block in self.goal_supports:
            chain = []
            current = block
            while True:
                current_support = self.goal_supports.get(current, None)
                if current_support is None or current_support == 'table':
                    break
                if current_support not in self.goal_supports:
                    break
                chain.append(current_support)
                current = current_support
            self.support_chains[block] = chain

    def __call__(self, node):
        state = node.state

        # Extract current on, on-table, and clear information
        current_on = {}
        current_on_table = set()
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_on_table.add(parts[1])
            elif parts[0] == 'clear':
                current_clear.add(parts[1])

        # Build tower_above to compute current_above for each block
        tower_above = {}
        for C, B in current_on.items():
            if B not in tower_above:
                tower_above[B] = []
            tower_above[B].append(C)

        current_above = {}
        def compute_above(B):
            if B in current_above:
                return current_above[B]
            total = 0
            for C in tower_above.get(B, []):
                total += 1 + compute_above(C)
            current_above[B] = total
            return total

        # Compute current_above for all blocks in current_on or current_on_table
        processed = set()
        for block in current_on_table:
            if block not in processed:
                compute_above(block)
                processed.add(block)
        for block in current_on:
            if block not in processed:
                compute_above(block)
                processed.add(block)

        # Calculate heuristic value
        heuristic_value = 0

        for block in self.goal_supports:
            # Check if block is correctly placed
            correctly_placed = True

            # Check on/on-table
            goal_support = self.goal_supports[block]
            if goal_support == 'table':
                if block not in current_on_table:
                    correctly_placed = False
            else:
                if current_on.get(block, None) != goal_support:
                    correctly_placed = False

            # Check clear if required
            if block in self.goal_clear:
                if block not in current_clear:
                    correctly_placed = False

            if not correctly_placed:
                # Compute current_above for this block
                ca = current_above.get(block, 0)

                # Compute required_below
                required_below = 0
                for s in self.support_chains.get(block, []):
                    # Check if s is correctly placed
                    s_goal_support = self.goal_supports.get(s, None)
                    s_correct = True
                    if s_goal_support == 'table':
                        if s not in current_on_table:
                            s_correct = False
                    else:
                        if current_on.get(s, None) != s_goal_support:
                            s_correct = False
                    if s in self.goal_clear and s not in current_clear:
                        s_correct = False
                    if not s_correct:
                        required_below += 1

                # Add to heuristic
                heuristic_value += 2 * (ca + required_below + 1)

        return heuristic_value
