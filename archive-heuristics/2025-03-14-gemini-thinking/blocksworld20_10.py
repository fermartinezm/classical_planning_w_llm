from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Blocksworld domain.
    It counts the number of blocks that are not in their goal positions and the number of blocks that are not clear when they should be in the goal.

    # Assumptions:
    - The problem is solvable in the Blocksworld domain using the given actions.
    - The goal state is well-defined in terms of 'on', 'on-table', and 'clear' predicates.

    # Heuristic Initialization
    - Extracts the goal predicates from the task definition.
    - Preprocesses the goal predicates to easily access the desired 'on' and 'on-table' relationships for each block and the 'clear' goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract goal state information:
       - For each block, determine its goal position (on another block or on the table).
       - Identify blocks that are required to be clear in the goal state.
    3. For each block in the domain:
       - Determine its current position (on another block or on the table) from the current state.
       - Compare the current position with the goal position. If they differ, increment the heuristic value.
    4. For each block that is required to be clear in the goal state:
       - Check if it is clear in the current state. If not, increment the heuristic value.
    5. Return the accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_on = {}
        self.goal_on_table = {}
        self.goal_clear = set()
        self.blocks = set()

        for goal in self.goals:
            predicate_name, *objects = self._extract_objects_from_fact(goal)
            if predicate_name == 'on':
                block, underblock = objects
                self.goal_on[block] = underblock
                self.blocks.add(block)
                self.blocks.add(underblock)
            elif predicate_name == 'on-table':
                block = objects[0]
                self.goal_on_table[block] = True
                self.blocks.add(block)
            elif predicate_name == 'clear':
                block = objects[0]
                self.goal_clear.add(block)
                self.blocks.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0
        current_on = {}
        current_on_table = {}
        current_clear = set()
        block_set = set()

        for fact in state:
            predicate_name, *objects = self._extract_objects_from_fact(fact)
            if predicate_name == 'on':
                block, underblock = objects
                current_on[block] = underblock
                block_set.add(block)
                block_set.add(underblock)
            elif predicate_name == 'on-table':
                block = objects[0]
                current_on_table[block] = True
                block_set.add(block)
            elif predicate_name == 'clear':
                block = objects[0]
                current_clear.add(block)
                block_set.add(block)
            elif predicate_name == 'holding':
                block = objects[0]
                block_set.add(block)

        for block in self.blocks.union(block_set): # Iterate over all blocks in goal and current state
            goal_below = self.goal_on.get(block, 'table') if block in self.goal_on or block in self.goal_on_table else None
            current_below = current_on.get(block, 'table') if block in current_on or block in current_on_table else None

            goal_on_table_block = block in self.goal_on_table and self.goal_on_table[block]
            current_on_table_block = block in current_on_table and current_on_table[block]

            if goal_on_table_block:
                goal_below = 'table'
            elif block in self.goal_on:
                goal_below = self.goal_on[block]
            else:
                goal_below = None


            if current_on_table_block:
                current_below = 'table'
            elif block in current_on:
                current_below = current_on[block]
            else:
                current_below = None


            if goal_below == 'table':
                if current_below != 'table':
                    heuristic_value += 1
            elif goal_below is not None and goal_below != 'table':
                if current_below != goal_below:
                    heuristic_value += 1


        for block in self.goal_clear:
            if block not in current_clear:
                heuristic_value += 1

        return heuristic_value

    def _extract_objects_from_fact(self, fact):
        """Extract predicate name and objects from a PDDL fact string."""
        parts = fact[1:-1].split() # Remove parentheses and split
        predicate_name = parts[0]
        objects = parts[1:]
        return predicate_name, *objects
