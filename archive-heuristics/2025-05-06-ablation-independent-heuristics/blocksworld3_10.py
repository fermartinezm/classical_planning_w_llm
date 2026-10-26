# Assuming Heuristic base class and Task class are available from the planner environment
# from heuristics.heuristic_base import Heuristic
# from task import Task

class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the Blocksworld domain. It estimates the
    cost to reach the goal by counting the number of blocks that are in the
    wrong location or have the wrong block stacked on top, plus a penalty
    if the arm is not empty when it should be.

    Assumptions:
    - The domain is Blocksworld with standard predicates (on, on-table, clear,
      holding, arm-empty) and actions (pickup, putdown, stack, unstack).
    - The goal state typically specifies the desired configuration of blocks
      using (on A B) and (on-table C) predicates, and often includes (arm-empty)
      or implies it via (clear X) for the top block. This heuristic assumes
      (arm-empty) is a goal if it appears in the task's goal set.
    - Each block is either on another block, on the table, or being held.
    - At most one block can be on top of another block.
    - At most one block can be held at a time.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes the goal state provided
    in the Task object. It extracts:
    - `desired_on`: A dictionary mapping a block to the block it should be on
      according to the goal (e.g., {'b1': 'b2'} if (on b1 b2) is a goal).
    - `desired_on_table`: A set of blocks that should be on the table
      according to the goal.
    - `desired_on_top`: A dictionary mapping a block to the block that should
      be directly on top of it according to the goal (e.g., {'b2': 'b1'} if
      (on b1 b2) is a goal).
    - `goal_arm_empty`: A boolean indicating if (arm-empty) is a goal fact.
    - `all_objects`: A set of all block objects present in the domain,
      extracted by parsing all possible ground facts from the Task object.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic value is computed as follows:
    1. Parse the current state to determine:
       - `current_support`: A dictionary mapping each block to its current
         support ('table' or another block). Blocks being held are not
         assigned a support in this mapping initially.
       - `current_on_top`: A dictionary mapping each block to the block
         currently on top of it (None if clear).
       - `is_holding`: The block currently being held, or None if arm is empty.
       - `is_arm_empty`: Boolean indicating if arm is empty.
    2. Initialize the heuristic value `h` to 0.
    3. Count "misplaced" blocks: Iterate through all blocks. If a block has
       a desired location (on another block or on the table) specified in the
       goal, and its current location is different, increment `h`. Blocks being
       held are considered not to be in their desired location if they have one.
    4. Count "obstructing" blocks: Iterate through all blocks X. If there is
       a block Y currently on top of X:
       - If X is supposed to have a specific block Z on top (`X in desired_on_top`),
         and Y is not Z, increment `h`.
       - If X is supposed to be on the table (`X in desired_on_table`), meaning
         nothing should be on X, increment `h`.
       - If X is not involved as a support in the goal structure (`X not in
         desired_on_top` and `X not in desired_on_table`), and something is
         on it, increment `h` as Y is obstructing X.
    5. Add arm penalty: If (arm-empty) is a goal and the arm is currently
       holding a block, increment `h`.
    6. The final value of `h` is the heuristic estimate.

    The heuristic is 0 if and only if the state is the goal state (assuming
    (arm-empty) is a goal or implied). It is finite for all solvable states.
    It is not admissible as it counts multiple types of "wrongness" which may
    each require multiple actions to fix, and these counts are simply summed.
    """

    def __init__(self, task):
        super().__init__(task)
        self.desired_on = {}
        self.desired_on_table = set()
        self.desired_on_top = {}  # support -> block_on_top
        self.goal_arm_empty = False
        self.all_objects = set()

        # Parse goal facts and extract objects
        for fact_str in task.goals:
            if fact_str == '(arm-empty)':
                self.goal_arm_empty = True
            elif fact_str.startswith('(on-table '):
                block = fact_str[len('(on-table '):-1]
                self.desired_on_table.add(block)
            elif fact_str.startswith('(on '):
                parts = fact_str[len('(on '):-1].split()
                if len(parts) == 2: # Ensure correct format
                    block = parts[0]
                    support = parts[1]
                    self.desired_on[block] = support
                    self.desired_on_top[support] = block  # Assumes unique block on top in goal

        # Extract all objects from all possible facts in the domain
        for fact_str in task.facts:
            # Remove parentheses and split
            parts = fact_str[1:-1].split()
            # Skip predicate, remaining parts are objects
            for obj in parts[1:]:
                self.all_objects.add(obj)

    def __call__(self, node):
        state = node.state

        current_support = {}  # block -> support_block or 'table'
        current_on_top = {}  # support_block -> block_on_top
        is_holding = None
        is_arm_empty = False

        # Parse current state facts
        for fact in state:
            if fact == '(arm-empty)':
                is_arm_empty = True
            elif fact.startswith('(holding '):
                is_holding = fact[len('(holding '):-1]
            elif fact.startswith('(on-table '):
                block = fact[len('(on-table '):-1]
                current_support[block] = 'table'
            elif fact.startswith('(on '):
                parts = fact[len('(on '):-1].split()
                if len(parts) == 2: # Ensure correct format
                    block = parts[0]
                    support = parts[1]
                    current_support[block] = support
                    current_on_top[support] = block  # Assumes unique block on top in state

        h = 0

        # Count misplaced blocks
        for block in self.all_objects:
            current_loc = current_support.get(block) # None if not on table/block (must be holding)

            if block in self.desired_on:
                if current_loc != self.desired_on[block]:
                    h += 1
            elif block in self.desired_on_table:
                if current_loc != 'table':
                    h += 1
            # Blocks not in desired_on or desired_on_table are not explicitly positioned in the goal structure.
            # We don't penalize their location directly, only if they obstruct or are obstructed.

        # Count obstructing blocks
        for block in self.all_objects:
            block_on_top = current_on_top.get(block) # None if nothing on top

            if block_on_top is not None: # Something is on 'block'
                Y = block_on_top
                if block in self.desired_on_top: # 'block' should have desired_on_top[block] on it
                    if Y != self.desired_on_top[block]:
                        h += 1 # Wrong block Y is on X
                elif block in self.desired_on_table: # Nothing should be on 'block' (as it's a base)
                    h += 1 # Block Y is on 'block', but 'block' should be clear
                else: # 'block' is not a goal support (not in desired_on_top, not in desired_on_table)
                    # If something is on 'block', it's obstructing.
                    h += 1 # Block Y is on 'block', obstructing 'block'

        # Add cost if arm is not empty but should be
        if self.goal_arm_empty and is_holding is not None:
            h += 1

        return h
