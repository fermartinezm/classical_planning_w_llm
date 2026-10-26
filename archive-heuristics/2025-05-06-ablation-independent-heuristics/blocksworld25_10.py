# Need to import Heuristic and Task from wherever they are defined in the planner structure.
# Assuming they are in a 'heuristics' and 'task' module respectively.
# The provided code snippets suggest this structure.
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is needed for structure, though not directly used in __call__ after init


class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the Blocksworld domain.
    This heuristic counts the number of blocks that are not in their correct
    final position as defined by the goal state. A block's position is
    considered correct if it is not being held and is placed on the correct
    block or the table, and recursively, the block below it is also in its
    correct position. For blocks not explicitly placed in goal stacks or on
    the table by goal facts, the goal is assumed to be on the table and clear.
    The heuristic value is the total count of blocks that fail this recursive
    correctness check.

    Assumptions:
    - This heuristic is non-admissible and designed for greedy best-first search.
    - Blocks not explicitly mentioned as being 'on' another block or 'on-table'
      in the goal facts are assumed to have a goal position of being on the
      table and clear.
    - The goal state consists of a set of stacks and blocks on the table, and
      possibly specific blocks needing to be clear or the arm needing to be empty.
    - The heuristic value is 0 if and only if the state is a goal state.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic pre-processes the goal facts
    to build a map (`goal_parent_map`) indicating the desired parent (another
    block or 'table') for each block that is part of a goal 'on' or 'on-table'
    fact. It also stores the sets of blocks explicitly required to be 'on-table'
    (`goal_on_table_set`) and 'clear' (`goal_clear_set`). All objects present
    in the task definition are identified by examining all possible ground facts
    (`task.facts`).

    Step-By-Step Thinking for Computing Heuristic:
    1.  Extract the current state facts from the provided node (`node.state`).
    2.  For each block in the problem, determine if it is in its "correct final
        position" using a recursive helper function (`_is_correct`).
        - A block is correct if it's not being held AND:
            - If the goal explicitly puts it 'on-table', it is currently 'on-table'.
            - If the goal explicitly puts it 'on' another block (parent), it is
              currently 'on' that parent AND the parent block is also correct.
            - If the block is not explicitly placed by any goal 'on' or 'on-table'
              fact, it is currently 'on-table' AND 'clear'.
        - Memoization is used in the recursive helper to avoid redundant calculations.
    3.  The raw heuristic value is the count of blocks for which the `_is_correct`
        check returns False.
    4.  As a final check to strictly adhere to the requirement that the heuristic
        is 0 only for goal states:
        - If the state is a goal state (`node.task.goal_reached(state)` is True),
          return 0.
        - Otherwise (state is not a goal), return the raw calculated heuristic
          value. If the raw value is 0, return 1 instead to ensure non-goal
          states have a heuristic value of at least 1.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        # Extract all objects from the task facts. task.facts should contain all possible ground facts.
        self.all_objects = self._extract_objects_from_facts(task.facts)

        # Pre-process goal facts
        self.goal_parent_map = {} # child -> parent (block or 'table')
        self.goal_on_table_set = set() # blocks explicitly goal to be on table
        self.goal_clear_set = set() # blocks explicitly goal to be clear

        for goal_fact in self.goals:
            if goal_fact.startswith('(on '):
                parts = goal_fact[len('(on '):-1].split()
                if len(parts) == 2:
                    child, parent = parts[0], parts[1]
                    self.goal_parent_map[child] = parent
            elif goal_fact.startswith('(on-table '):
                parts = goal_fact[len('(on-table '):-1].split()
                if len(parts) == 1:
                    block = parts[0]
                    self.goal_on_table_set.add(block)
                    self.goal_parent_map[block] = 'table' # Also add to parent map
            elif goal_fact.startswith('(clear '):
                 parts = goal_fact[len('(clear '):-1].split()
                 if len(parts) == 1:
                     block = parts[0]
                     self.goal_clear_set.add(block)
            # Ignore (arm-empty) goal for position correctness check

    def _extract_objects_from_facts(self, facts):
        """Helper to extract all unique object names from a set of facts."""
        objects = set()
        for fact in facts:
            # Fact is like '(predicate obj1 obj2)'
            # Remove surrounding brackets and split by space
            parts = fact[1:-1].split()
            # Ignore predicate name (parts[0])
            for obj in parts[1:]:
                objects.add(obj)
        return objects

    def _is_correct(self, block, state, goal_parent_map, goal_on_table_set, goal_clear_set, memo):
        """Helper to recursively check if a block is in its correct final position."""
        if block in memo:
            return memo[block]

        # A block is correct if:
        # 1. It's not holding.
        # 2. If goal is (on-table block), it is on the table.
        # 3. If goal is (on block parent), it is on parent AND parent is correct.
        # 4. If block is not in goal on/on-table, it is on table AND clear.

        is_holding = any(f == f'(holding {block})' for f in state)
        if is_holding:
            memo[block] = False
            return False

        current_p = None
        is_on_table = False
        is_clear = any(f == f'(clear {block})' for f in state)

        for fact in state:
            if fact == f'(on-table {block})':
                current_p = 'table'
                is_on_table = True
            elif fact.startswith('(on '):
                 parts = fact[len('(on '):-1].split()
                 if len(parts) == 2 and parts[0] == block:
                     current_p = parts[1] # Block is on another block

        is_in_goal_on = block in goal_parent_map and goal_parent_map[block] != 'table'
        is_in_goal_on_table = block in goal_on_table_set

        result = False
        if is_in_goal_on: # Goal is (on block goal_parent_map[block])
            goal_p = goal_parent_map[block]
            if current_p == goal_p:
                # Check if the parent is also correct
                result = self._is_correct(goal_p, state, goal_parent_map, goal_on_table_set, goal_clear_set, memo)
            else:
                result = False
        elif is_in_goal_on_table: # Goal is (on-table block)
            result = is_on_table
        else: # Block is not explicitly in a goal 'on' or 'on-table' fact
             # Assume goal is (on-table block) and (clear block)
             result = is_on_table and is_clear

        memo[block] = result
        return result


    def __call__(self, node):
        state = node.state

        # Calculate heuristic: Count blocks that are not in their correct final configuration
        h = 0
        is_correct_map = {}
        memo = {} # Memoization for recursive _is_correct
        for block in self.all_objects:
            is_correct_map[block] = self._is_correct(block, state, self.goal_parent_map, self.goal_on_table_set, self.goal_clear_set, memo)

        h = sum(1 for block in self.all_objects if not is_correct_map[block])

        # Safeguard: Ensure heuristic is 0 only for goal states
        # This check is crucial because the recursive 'is_correct' might not
        # capture all goal conditions like (arm-empty).
        if node.task.goal_reached(state):
             return 0 # Must be 0 if it's the goal state
        elif h == 0:
             # If not the goal state, but heuristic is 0, return 1
             return 1
        else:
             # Heuristic > 0 and not the goal state, return heuristic
             return h
