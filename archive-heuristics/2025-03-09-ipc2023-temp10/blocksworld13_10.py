from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class blocksworld13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state in the Blocksworld domain.
    It considers the number of blocks that are not in their goal positions, the number of blocks that have
    blocks on top of them that should be clear, and the number of blocks that are clear but should not be.

    # Assumptions
    - Each block needs to be moved at most once to its final position.
    - Moving a block involves picking it up, potentially putting down an intermediate block, and stacking it.
    - Clearing a block involves unstacking blocks from it.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal state from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal state and current state information.
    2. Identify blocks that are not in their goal positions (either on the table or on another block).
    3. Identify blocks that are in the correct position but have blocks on top of them that should be clear in the goal.
    4. Identify blocks that are clear in the current state but should have a block on top of them in the goal.
    5. Sum the number of misplaced blocks, blocks needing to be cleared, and blocks needing to be stacked upon.
    6. Return the total count as the heuristic estimate.  Each misplaced block, uncleared block, and unstacked block
       is assumed to require at least one action (pickup/putdown/stack/unstack).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
            return all(fnmatch(part, arg, )) for part, arg in zip(parts, args)

        misplaced_blocks = 0
        uncleared_blocks = 0
        unstacked_blocks = 0

        # Goal state information
        goal_ons = set()
        goal_on_table = set()
        goal_clear = set()

        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                goal_ons.add(goal)
            elif match(goal, "on-table", "*"):
                goal_on_table.add(goal)
            elif match(goal, "clear", "*"):
                goal_clear.add(goal)

        # Current state information
        current_ons = set()
        current_on_table = set()
        current_clear = set()

        for fact in state:
            if match(fact, "on", "*", "*"):
                current_ons.add(fact)
            elif match(fact, "on-table", "*"):
                current_on_table.add(fact)
            elif match(fact, "clear", "*"):
                current_clear.add(fact)

        # Misplaced blocks
        for on in current_ons:
            if on not in goal_ons:
                misplaced_blocks += 1
        for on_table in current_on_table:
            if on_table not in goal_on_table:
                misplaced_blocks += 1

        # Uncleared blocks
        for on in goal_ons:
            parts = on[1:-1].split()
            block_to_clear = parts[1]
            if "(clear " + block_to_clear + ")" not in current_clear:
                uncleared_blocks += 1

        # Unstacked blocks
        for clear in goal_clear:
            parts = clear[1:-1].split()
            block_to_stack = parts[1]
            
            found = False
            for on in goal_ons:
                parts_on = on[1:-1].split()
                if parts_on[2] == block_to_stack:
                    found = True
                    break
            
            if not found and "(on-table " + block_to_stack + ")" not in goal_on_table:
                continue

            if "(clear " + block_to_stack + ")" in current_clear:
                unstacked_blocks += 1

        heuristic_value = misplaced_blocks + uncleared_blocks + unstacked_blocks

        return heuristic_value
