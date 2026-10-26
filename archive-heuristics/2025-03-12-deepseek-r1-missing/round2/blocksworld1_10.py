from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    1. Blocks not in their correct position (2 actions per block).
    2. Extra blocks on top of correctly placed blocks (1 action per block).
    3. Blocks on top of blocks that need to be clear (1 action per block).

    # Assumptions
    - The goal specifies the desired 'on' and 'on-table' relationships for relevant blocks.
    - 'clear' predicates in the goal are explicitly checked, requiring no blocks on top of specified blocks.
    - The heuristic assumes that moving a block takes 2 actions (pickup and stack/putdown), and removing a block from on top takes 1 action (unstack).

    # Heuristic Initialization
    - Extract 'on', 'on-table', and 'clear' predicates from the goal.
    - Build mappings for each block's goal parent (block it should be on) and goal children (blocks that should be on it).
    - Track which blocks need to be clear according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block in the goal's 'on' or 'on-table' predicates:
        a. If the block is not on its goal parent, add 2 to the heuristic.
        b. If the block is correctly placed but has extra blocks on top, add 1 per extra block.
    2. For each block in the goal's 'clear' predicates:
        a. Add 1 to the heuristic for each block currently on top of it.
    """
    def __init__(self, task):
        self.goal_parent = {}  # Maps each block to its goal parent (None for on-table)
        self.goal_children = defaultdict(set)  # Maps each block to its goal children
        self.goal_clear = set()  # Blocks that must be clear

        # Parse goal predicates
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                self.goal_parent[child] = parent
                self.goal_children[parent].add(child)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parent[block] = None
            elif parts[0] == 'clear':
                block = parts[1]
                self.goal_clear.add(block)

    def __call__(self, node):
        state = node.state
        current_parent = {}
        current_children = defaultdict(set)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                current_parent[child] = parent
                current_children[parent].add(child)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_parent[block] = None

        h = 0

        # Check misplaced blocks
        for block, goal_p in self.goal_parent.items():
            current_p = current_parent.get(block)
            if current_p != goal_p:
                h += 2

        # Check extra blocks on top of correctly placed blocks
        for block in self.goal_parent:
            if current_parent.get(block) == self.goal_parent.get(block):
                current_kids = current_children.get(block, set())
                goal_kids = self.goal_children.get(block, set())
                for kid in current_kids:
                    if kid not in goal_kids:
                        h += 1

        # Check clear conditions
        for block in self.goal_clear:
            h += len(current_children.get(block, set()))

        return h
