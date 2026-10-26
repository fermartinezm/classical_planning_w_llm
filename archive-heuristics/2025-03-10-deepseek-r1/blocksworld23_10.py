from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class blocksworld23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state by considering:
    - Blocks that are not in their required 'on' or 'on-table' positions.
    - Blocks that are required to be 'clear' but have other blocks stacked on them.

    # Assumptions
    - The goal is a conjunction of 'on', 'on-table', and 'clear' predicates.
    - Moving a block requires two actions (pickup and putdown/stack) if it's misplaced.
    - Clearing a block requires two actions per block stacked on top of it (unstack and putdown).

    # Heuristic Initialization
    - Extracts 'on', 'on-table', and 'clear' goals from the task.
    - Identifies all blocks involved in the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Current Support Analysis**: Determine where each block is currently placed.
    2. **Clear Blocks Check**: Identify which blocks are currently clear.
    3. **Support Chain Counting**: For each block, count how many blocks are stacked above it.
    4. **On/On-Table Cost Calculation**: For each block in 'on' or 'on-table' goals, compute the cost to move it to its goal position, considering dependencies.
    5. **Clear Cost Calculation**: For each block in 'clear' goals, add the cost to clear it by moving all blocks above it.
    """

    def __init__(self, task):
        self.on_goals = {}  # {block: target_support}
        self.on_table_goals = set()
        self.clear_goals = set()
        self.goal_blocks = set()

        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.on_goals[block] = support
                self.goal_blocks.update([block, support])
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.on_table_goals.add(block)
                self.goal_blocks.add(block)
            elif parts[0] == 'clear' and len(parts) == 2:
                block = parts[1]
                self.clear_goals.add(block)
                self.goal_blocks.add(block)

    def __call__(self, node):
        state = node.state
        current_support = {}
        clear_blocks = set()

        # Extract current supports and clear blocks from the state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = None
            elif parts[0] == 'clear' and len(parts) == 2:
                clear_blocks.add(parts[1])

        # Build support_to_blocks to find blocks stacked above
        support_to_blocks = defaultdict(list)
        for block, support in current_support.items():
            if support is not None:
                support_to_blocks[support].append(block)

        # Calculate number of blocks above each block using memoization
        memo_count = {}
        def count_above(block):
            if block in memo_count:
                return memo_count[block]
            total = 0
            for b in support_to_blocks.get(block, []):
                total += 1 + count_above(b)
            memo_count[block] = total
            return total

        for block in set(current_support.keys()).union(support_to_blocks.keys()):
            if block not in memo_count:
                count_above(block)

        # Calculate cost for 'on' and 'on-table' goals
        memo = {}
        def compute_cost(block):
            if block in memo:
                return memo[block]
            
            cost = 0
            if block in self.on_goals:
                goal_support = self.on_goals[block]
                curr_support = current_support.get(block, None)
                if curr_support != goal_support:
                    cost += 2 + compute_cost(goal_support)
                else:
                    # Check if the support is correctly placed
                    if goal_support is not None:
                        cost += compute_cost(goal_support)
            elif block in self.on_table_goals:
                if current_support.get(block, None) is not None:
                    cost += 2
            memo[block] = cost
            return cost

        on_on_table_cost = 0
        processed = set()
        for block in self.on_goals.keys() | self.on_table_goals:
            if block not in processed:
                on_on_table_cost += compute_cost(block)
                processed.add(block)

        # Calculate cost for 'clear' goals
        clear_cost = 0
        for block in self.clear_goals:
            if block not in clear_blocks:
                clear_cost += 2 * memo_count.get(block, 0)

        return on_on_table_cost + clear_cost
