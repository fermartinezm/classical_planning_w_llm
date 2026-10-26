from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(on b1 b2)' it returns ['b1', 'b2'].
    """
    fact_content = fact[1:-1].split()
    return fact_content[1:]  # Return objects, skip predicate name

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    For each block that is not in its goal position, it adds 1 to the heuristic value.
    Additionally, for each block that is on top of another block in the current state, but should not be in the goal state, it adds another 1 to the heuristic value.
    This heuristic is admissible if we consider each misplaced block requires at least one action to move to its correct position. However, for greedy best-first search, admissibility is not required, and we aim for efficiency and good guidance.

    # Assumptions:
    - The goal state is defined by a set of `on` and `on-table` predicates.
    - We are minimizing the number of actions.

    # Heuristic Initialization
    - Extract the goal `on` and `on-table` predicates from the task.
    - Store the goal positions for each block in a dictionary for efficient lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Parse the goal state to identify the desired position of each block (either `on-table` or `on` another block). Store this in `goal_block_positions`.
    3. For each block in the problem:
        a. Determine the current position of the block from the current state.
        b. Determine the goal position of the block from `goal_block_positions`.
        c. If the current position is not the goal position, increment the heuristic value by 1.
    4. For each `on(block1, block2)` predicate in the current state:
        d. Check if `on(block1, block2)` is part of the goal `on` predicates.
        e. If `on(block1, block2)` is not in the goal predicates, increment the heuristic value by 1.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_on_table = set()
        self.goal_on = set()
        self.goal_clear = set()

        for goal in self.goals:
            if goal.startswith('(on-table'):
                self.goal_on_table.add(get_objects_from_fact(goal)[0])
            elif goal.startswith('(on'):
                block1, block2 = get_objects_from_fact(goal)
                self.goal_on.add((block1, block2))
            elif goal.startswith('(clear'):
                self.goal_clear.add(get_objects_from_fact(goal)[0])


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0
        current_on_table = set()
        current_on = set()
        current_clear = set()

        for fact in state:
            if fact.startswith('(on-table'):
                current_on_table.add(get_objects_from_fact(fact)[0])
            elif fact.startswith('(on'):
                block1, block2 = get_objects_from_fact(fact)
                current_on.add((block1, block2))
            elif fact.startswith('(clear'):
                current_clear.add(get_objects_from_fact(fact)[0])

        blocks_in_goal = set()
        for goal in self.goals:
            if goal.startswith('(on-table'):
                blocks_in_goal.add(get_objects_from_fact(goal)[0])
            elif goal.startswith('(on'):
                blocks_in_goal.add(get_objects_from_fact(goal)[0])
                blocks_in_goal.add(get_objects_from_fact(goal)[1])
            elif goal.startswith('(clear'):
                blocks_in_goal.add(get_objects_from_fact(goal)[0])

        all_blocks = set()
        for fact in state:
            if fact.startswith('(on-table'):
                all_blocks.add(get_objects_from_fact(fact)[0])
            elif fact.startswith('(on'):
                all_blocks.add(get_objects_from_fact(fact)[0])
                all_blocks.add(get_objects_from_fact(fact)[1])
            elif fact.startswith('(clear'):
                all_blocks.add(get_objects_from_fact(fact)[0])
            elif fact.startswith('(holding'):
                all_blocks.add(get_objects_from_fact(fact)[0])

        for block in all_blocks:
            goal_on_table_block = block in self.goal_on_table
            current_on_table_block = block in current_on_table

            if goal_on_table_block != current_on_table_block:
                goal_on_relation = None
                for b1, b2 in self.goal_on:
                    if b1 == block:
                        goal_on_relation = (b1, b2)
                        break
                current_on_relation = None
                for b1, b2 in current_on:
                    if b1 == block:
                        current_on_relation = (b1, b2)
                        break

                if goal_on_relation is None and not goal_on_table_block:
                     heuristic_value += 1
                elif goal_on_relation is not None:
                    if current_on_relation != goal_on_relation:
                        heuristic_value += 1
                elif goal_on_table_block:
                    if not current_on_table_block:
                        heuristic_value += 1

        for on_relation in current_on:
            if on_relation not in self.goal_on:
                heuristic_value += 1

        return heuristic_value
