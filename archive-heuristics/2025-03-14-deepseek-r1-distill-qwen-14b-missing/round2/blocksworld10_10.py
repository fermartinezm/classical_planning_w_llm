from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state by counting the number of blocks that are out of place and need to be moved. Each block that needs to be moved contributes at least two actions (pickup and putdown).

    # Assumptions:
    - The goal state specifies a specific stack configuration on the table.
    - Blocks not part of the target stack on the table need to be moved.
    - Blocks in the target stack but not correctly ordered also need adjustment.

    # Heuristic Initialization
    - Extract the target stack configuration from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the target stack structure from the goal conditions.
    2. For each block, determine if it is in the correct position:
       - Blocks on the table not part of the target stack need to be moved.
       - Blocks in the target stack but not correctly ordered need adjustment.
    3. Count the number of blocks that need to be moved. Each contributes at least two actions.
    4. Sum the actions required to rearrange all blocks into the target configuration.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """
            Check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(on b1 b2)").
            - `args`: The pattern to match (e.g., "on", "*", "*").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        goal_stack = []
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                obj1, obj2 = goal[3:-3].split()
                goal_stack.append((obj1, obj2))

        current_blocks = {}
        for fact in state:
            if match(fact, "on", "*", "*"):
                obj1, obj2 = fact[3:-3].split()
                current_blocks[obj1] = obj2
            elif match(fact, "on-table", "*"):
                obj = fact[7:-7]
                current_blocks[obj] = "table"

        arm_holding = any(match(fact, "holding", "*") for fact in state)
        clear_table = any(match(fact, "clear", "table") for fact in state)

        heuristic = 0

        if not self.goals.issubset(state):
            target_blocks = [goal[3:-3].split()[0] for goal in self.goals if match(goal, "on", "*", "*")]
            target_blocks_set = set(target_blocks)
            current_on_table = [obj for obj, loc in current_blocks.items() if loc == "table"]

            blocks_to_move = []
            for obj in current_blocks:
                if obj not in target_blocks_set:
                    blocks_to_move.append(obj)
                else:
                    if obj in current_on_table:
                        blocks_to_move.append(obj)

            for obj in blocks_to_move:
                if arm_holding:
                    heuristic += 2  # Put down and pick up
                else:
                    heuristic += 2  # Pick up and put down

            if clear_table and len(blocks_to_move) > 0:
                heuristic += 1  # Move to the correct position

        return heuristic
