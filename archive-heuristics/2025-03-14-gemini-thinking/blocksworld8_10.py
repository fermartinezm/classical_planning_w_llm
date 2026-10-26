from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(on b1 b2)' it returns ['b1', 'b2'].
    Ignores the predicate name.
    """
    parts = fact_str[1:-1].split()
    return parts[1:]

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of blocks that are not in their goal positions.
    A block is considered misplaced if it is not on the block it is supposed to be on
    in the goal state, or if it is supposed to be on the table but is not, or vice versa.
    It also considers blocks that are on top of a block that should be clear in the goal.

    # Assumptions:
    - The goal state is defined by a set of `on` and `on-table` predicates.
    - `clear` predicates in the goal are also considered for misplaced blocks.

    # Heuristic Initialization
    - Extracts the goal `on` and `on-table` relationships to determine the desired
      stack configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Parse the goal state to determine the desired `on` and `on-table` relationships.
       Store these goal relationships in a dictionary for easy lookup, e.g., goal_positions = {'b1': 'b2', 'b2': 'table', ...}.
       Blocks that should be on the table will have 'table' as their goal position.
    3. For each block present in the goal configuration:
       a. Determine its current position from the current state. Check if it is on another block or on the table.
       b. Determine its goal position from the parsed goal conditions.
       c. If the current position is different from the goal position, increment the heuristic value.
    4. Additionally, for each goal `clear(b)`:
       a. Check if `clear(b)` is true in the current state.
       b. If `clear(b)` is not true in the current state, it means there are blocks on top of `b` that need to be moved. Increment the heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                parts = get_objects_from_fact(goal)
                block1, block2 = parts[0], parts[1]
                self.goal_on[block1] = block2
            elif goal.startswith('(on-table '):
                block = get_objects_from_fact(goal)[0]
                self.goal_on_table.add(block)
            elif goal.startswith('(clear '):
                block = get_objects_from_fact(goal)[0]
                self.goal_clear.add(block)

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state."""
        state = node.state
        heuristic_value = 0
        current_on = {}
        current_on_table = set()
        current_clear = set()
        all_blocks_in_state = set()

        for fact in state:
            if fact.startswith('(on '):
                parts = get_objects_from_fact(fact)
                block1, block2 = parts[0], parts[1]
                current_on[block1] = block2
                all_blocks_in_state.add(block1)
                all_blocks_in_state.add(block2)
            elif fact.startswith('(on-table '):
                block = get_objects_from_fact(fact)[0]
                current_on_table.add(block)
                all_blocks_in_state.add(block)
            elif fact.startswith('(clear '):
                block = get_objects_from_fact(fact)[0]
                current_clear.add(block)
                all_blocks_in_state.add(block)
            elif fact.startswith('(holding '):
                block = get_objects_from_fact(fact)[0]
                all_blocks_in_state.add(block)


        blocks_in_goal = set(self.goal_on.keys()) | set(self.goal_on.values()) | self.goal_on_table | self.goal_clear
        all_blocks = all_blocks_in_state | blocks_in_goal

        for block in all_blocks:
            goal_parent = self.goal_on.get(block, 'table') if block in self.goal_on or block in self.goal_on_table else None
            current_parent = current_on.get(block, 'table') if block in current_on or block in current_on_table else None

            if goal_parent is not None: # Block is in goal configuration
                if goal_parent == 'table':
                    if block not in current_on_table:
                        heuristic_value += 1
                else: # goal_parent is a block
                    if current_on.get(block) != goal_parent:
                        heuristic_value += 1

        for block in self.goal_clear:
            if block not in current_clear:
                heuristic_value += 1

        return heuristic_value
