from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworld3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - Blocks that are not in their goal position (on or on-table) require 2 actions (pickup and putdown/stack) plus the number of blocks above them that need to be unstacked.
    - Blocks that are required to be clear but have blocks on top require moving each of those blocks, adding 2 actions plus the number of blocks above them for each such block.

    # Assumptions
    - Each block can be part of only one stack.
    - Moving a block requires unstacking all blocks above it first.
    - The heuristic does not account for the order of moving blocks but estimates the total actions needed.

    # Heuristic Initialization
    - Extract the goal conditions for on, on-table, and clear predicates.
    - Store these in dictionaries and sets for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block required to be on another block (goal_on):
        a. Check if it's currently on the correct block.
        b. If not, add 2 actions (pickup and stack) plus the number of blocks above it.
    2. For each block required to be on the table (goal_on_table):
        a. Check if it's currently on the table.
        b. If not, add 2 actions (pickup and putdown) plus the number of blocks above it.
    3. For each block required to be clear (goal_clear):
        a. Check if it's currently clear.
        b. If not, add for each block on top: 2 actions plus the number of blocks above that block.
    """

    def __init__(self, task):
        self.goal_on = {}  # Maps block to its goal support
        self.goal_on_table = set()  # Blocks that must be on the table
        self.goal_clear = set()  # Blocks that must be clear

        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on' and len(parts) == 3:
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == 'on-table' and len(parts) == 2:
                self.goal_on_table.add(parts[1])
            elif parts[0] == 'clear' and len(parts) == 2:
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        state = node.state
        current_on = {}
        current_on_table = set()

        for fact in state:
            if fact.startswith('(on ') and fact.endswith(')'):
                parts = fact[1:-1].split()
                current_on[parts[1]] = parts[2]
            elif fact.startswith('(on-table ') and fact.endswith(')'):
                parts = fact[1:-1].split()
                current_on_table.add(parts[1])

        # Compute number of blocks above each block
        num_blocks_above = {}
        blocks = set(current_on.keys()).union(current_on_table)
        for block in blocks:
            count = 0
            current = block
            while True:
                found = False
                for fact in state:
                    if fact.startswith('(on ') and fact.endswith(')'):
                        parts = fact[1:-1].split()
                        if parts[2] == current:
                            current = parts[1]
                            count += 1
                            found = True
                            break
                if not found:
                    break
            num_blocks_above[block] = count

        heuristic_value = 0

        # Check on and on-table goals
        for block, target in self.goal_on.items():
            if current_on.get(block, None) != target:
                heuristic_value += 2 + num_blocks_above.get(block, 0)

        for block in self.goal_on_table:
            if block not in current_on_table:
                heuristic_value += 2 + num_blocks_above.get(block, 0)

        # Check clear goals
        for block in self.goal_clear:
            is_clear = True
            for fact in state:
                if fact.startswith('(on ') and fact.endswith(')'):
                    parts = fact[1:-1].split()
                    if parts[2] == block:
                        is_clear = False
                        break
            if not is_clear:
                # Find all blocks on top of 'block'
                current = block
                blocks_on_top = []
                while True:
                    found = False
                    for fact in state:
                        if fact.startswith('(on ') and fact.endswith(')'):
                            parts = fact[1:-1].split()
                            if parts[2] == current:
                                blocks_on_top.append(parts[1])
                                current = parts[1]
                                found = True
                                break
                    if not found:
                        break
                for blk in blocks_on_top:
                    heuristic_value += 2 + num_blocks_above.get(blk, 0)

        return heuristic_value
