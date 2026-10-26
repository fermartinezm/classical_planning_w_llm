from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(on b1 b2)' it returns ['b1', 'b2'].
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[1:]  # Return objects, skip predicate name

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    A block is considered in its goal position if it satisfies the 'on' or 'on-table' predicates
    as specified in the goal. For each block that is not in its goal position, the heuristic
    increments the estimated cost by one. This heuristic is admissible in relaxed planning,
    but here we use it as a domain-dependent heuristic for greedy best-first search, so admissibility is not required.

    # Assumptions:
    - The goal is defined by a set of 'on' and 'on-table' predicates.
    - We only consider the 'on' and 'on-table' goal predicates to determine the goal positions of blocks.
    - 'clear' and 'arm-empty' goal predicates are not directly used in this heuristic calculation,
      but achieving the 'on' and 'on-table' goals implicitly addresses the 'clear' conditions.

    # Heuristic Initialization
    - The heuristic initializes by parsing the goal predicates to determine the desired 'on' and 'on-table'
      relationships between blocks in the goal state.
    - It creates a mapping of each block to its intended 'under' block in the goal configuration,
      and a set of blocks that should be on the table in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to zero.
    2. Parse the goal predicates to identify the desired 'on' and 'on-table' relationships.
       Store these goal relationships in `goal_on` (block -> block_under) and `goal_on_table` (set of blocks).
    3. For each block mentioned in the goal predicates:
       a. Determine the goal position of the block:
          - If the block is in `goal_on`, its goal position is on the block specified in `goal_on`.
          - If the block is in `goal_on_table` and not in `goal_on`, its goal position is on the table.
       b. Determine the current position of the block in the given state:
          - Check for `(on block block_under)` predicate in the current state.
          - If not found, check for `(on-table block)` predicate in the current state.
       c. Compare the goal position with the current position.
          - If the current position does not match the goal position, increment the heuristic value by one.
    4. Return the total heuristic value, which represents the estimated number of misplaced blocks.
    """

    def __init__(self, task):
        """Initialize the blocksworld heuristic by parsing goal predicates."""
        self.goals = task.goals
        self.goal_on = {}
        self.goal_on_table = set()
        blocks_in_goal = set()

        for goal in self.goals:
            if goal.startswith('(on '):
                parts = get_objects_from_fact(goal)
                block_over, block_under = parts[0], parts[1]
                self.goal_on[block_over] = block_under
                blocks_in_goal.add(block_over)
                blocks_in_goal.add(block_under)
            elif goal.startswith('(on-table '):
                block = get_objects_from_fact(goal)[0]
                self.goal_on_table.add(block)
                blocks_in_goal.add(block)
        self.blocks_in_goal = blocks_in_goal


    def __call__(self, node):
        """Estimate the number of misplaced blocks in the current state."""
        state = node.state
        misplaced_blocks = 0

        for block in self.blocks_in_goal:
            goal_pos = None
            current_pos = None

            if block in self.goal_on:
                goal_pos = self.goal_on[block]
            elif block in self.goal_on_table:
                goal_pos = 'table'

            for fact in state:
                if fact.startswith('(on '):
                    parts = get_objects_from_fact(fact)
                    if parts[0] == block:
                        current_pos = parts[1]
                        break
                elif fact.startswith('(on-table '):
                    parts = get_objects_from_fact(fact)
                    if parts[0] == block:
                        current_pos = 'table'
                        break

            if goal_pos == 'table':
                if current_pos != 'table':
                    misplaced_blocks += 1
            elif goal_pos is not None: # goal_pos is a block
                if current_pos != goal_pos:
                    misplaced_blocks += 1
            elif goal_pos is None and current_pos is not None: # Block should not be on anything in goal, but is in current state
                misplaced_blocks += 1
            elif goal_pos is None and current_pos is None and 'on' in str(self.goals) or 'on-table' in str(self.goals):
                # Case where block is in goal but not specified as on-table or on-another-block.
                # This case might not be directly covered by the simple misplaced block count, but for simplicity, we can ignore it or consider it correctly placed if not explicitly misplaced.
                pass


        return misplaced_blocks
