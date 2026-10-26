import sys
import os
from collections import defaultdict

# Assume the Task class is available from the provided code-file-task
# from task import Task, Operator # Not needed for the heuristic itself

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
        This heuristic estimates the cost to reach the goal by counting the number
        of blocks that are not "correctly placed" within the goal configuration.
        A block is considered "correctly placed" if it is on the table and
        supposed to be on the table according to the goal, OR if it is on
        another block Y and supposed to be on Y according to the goal, AND Y
        is itself correctly placed. The heuristic value is the total number
        of blocks minus the number of correctly placed blocks. This captures
        the idea that blocks higher up in a goal stack cannot be placed correctly
        until the blocks below them are correctly placed.

    Assumptions:
        - The input state and goal are valid for the Blocksworld domain as defined
          in the provided PDDL.
        - The goal specifies the desired position (on the table or on another block)
          for all blocks involved in the problem.
        - The state representation uses strings for facts, e.g., '(on b1 b2)'.
        - The Task object provides initial_state, goals, and operators. Static facts
          are provided but are typically empty for Blocksworld and not used by this heuristic.

    Heuristic Initialization:
        The heuristic constructor precomputes the set of all blocks involved in the
        task by inspecting the initial state and goal facts. It also builds a map
        representing the desired support for each block based on the goal facts
        (e.g., block X should be on block Y, or block Z should be on the table).
        A map from a block Y to the list of blocks X that should be on top of Y
        in the goal is also precomputed for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state using task.goal_reached(state).
           If it is, the heuristic value is 0.
        2. If not a goal state, compute the set of "correctly placed" blocks.
           A block X is correctly placed if:
           a) (on-table X) is a goal fact AND (on-table X) is true in the current state.
           b) (on X Y) is a goal fact AND (on X Y) is true in the current state AND Y is
              already in the set of correctly placed blocks.
        3. This set is computed iteratively:
           - Start with an empty set `goal_placed`.
           - Initialize a set `newly_placed` with all blocks X satisfying condition (a).
           - While `newly_placed` is not empty:
             - Add all blocks from `newly_placed` to `goal_placed`.
             - Create an empty set `candidate_set`.
             - For each block Y in `newly_placed`:
               - Find all blocks X such that (on X Y) is a goal fact (using the precomputed map).
               - For each such X:
                 - If (on X Y) is true in the current state AND X is not already in `goal_placed`:
                   - Add X to `candidate_set`.
             - Set `newly_placed` to `candidate_set`.
        4. The heuristic value is the total number of blocks minus the size of the `goal_placed` set.
           This counts the number of blocks that are not part of a correctly built
           stack segment starting from the table according to the goal.
    """

    def __init__(self, task, static_facts):
        """
        Initializes the heuristic, precomputing goal structure and blocks.

        Args:
            task: The planning task object.
            static_facts: A frozenset of static facts (ignored in this heuristic).
        """
        self.task = task
        self.all_blocks = self._get_all_blocks(task.initial_state, task.goals)
        self.goal_support_map = self._get_goal_support_map(task.goals)
        self.goal_on_map = self._get_goal_on_map(task.goals)


    def _get_all_blocks(self, initial_state, goals):
        """Extracts all unique block names from initial state and goal facts."""
        blocks = set()
        # Collect blocks from initial state facts
        for fact in initial_state:
            parts = fact.strip('()').split()
            # Facts like (predicate arg1 arg2 ...)
            # Objects are the arguments, excluding the predicate name
            if len(parts) > 1:
                # Ignore 'arm-empty' and 'holding' predicates as they don't involve blocks as arguments in the same way
                if parts[0] not in ['arm-empty', 'holding']:
                     for obj in parts[1:]:
                         blocks.add(obj)
                elif parts[0] == 'holding' and len(parts) == 2:
                     blocks.add(parts[1]) # Block being held is an object
        # Collect blocks from goal facts
        for fact in goals:
             parts = fact.strip('()').split()
             if len(parts) > 1:
                 # Ignore 'clear' and 'arm-empty' in goals for block identification
                 if parts[0] not in ['clear', 'arm-empty']:
                     for obj in parts[1:]:
                         blocks.add(obj)
                 elif parts[0] == 'clear' and len(parts) == 2:
                      blocks.add(parts[1]) # Block that should be clear is an object

        return frozenset(blocks)

    def _get_goal_support_map(self, goals):
        """Builds a map from block to its required support in the goal."""
        goal_support = {}
        for fact in goals:
            parts = fact.strip('()').split()
            if len(parts) == 2 and parts[0] == 'on-table':
                block = parts[1]
                goal_support[block] = 'table'
            elif len(parts) == 3 and parts[0] == 'on':
                block_on_top = parts[1]
                block_below = parts[2]
                goal_support[block_on_top] = block_below
        return goal_support

    def _get_goal_on_map(self, goals):
        """Builds a map from a block to a list of blocks that should be directly on top of it in the goal."""
        goal_on_map = defaultdict(list)
        for fact in goals:
             parts = fact.strip('()').split()
             if len(parts) == 3 and parts[0] == 'on':
                 block_on_top = parts[1]
                 block_below = parts[2]
                 goal_on_map[block_below].append(block_on_top)
        return goal_on_map


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The heuristic value (integer).
        """
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 2. & 3. Compute the set of correctly placed blocks (GoalPlaced)
        goal_placed = set()
        newly_placed = set()

        # Add blocks that are correctly on the table according to the goal
        for block in self.all_blocks:
            goal_support = self.goal_support_map.get(block)
            if goal_support == 'table':
                on_table_fact = f'(on-table {block})'
                if on_table_fact in state:
                    newly_placed.add(block)

        # Iteratively add blocks that are correctly on top of already goal_placed blocks
        while newly_placed:
            goal_placed.update(newly_placed)
            candidate_set = set()
            for block_below in newly_placed:
                # Find blocks X that should be on block_below according to the goal
                blocks_to_be_on_below = self.goal_on_map.get(block_below, [])
                for block_on_top in blocks_to_be_on_below:
                    # Check if block_on_top is currently on block_below
                    on_fact = f'(on {block_on_top} {block_below})'
                    if on_fact in state and block_on_top not in goal_placed:
                        candidate_set.add(block_on_top)
            newly_placed = candidate_set

        # 4. Heuristic value
        h_value = len(self.all_blocks) - len(goal_placed)

        # Ensure heuristic is non-negative (should be guaranteed by logic, but defensive)
        return max(0, h_value)
