# Need to import Heuristic and Task.
# Based on the provided code structure, they are likely in these paths.
from heuristics.heuristic_base import Heuristic
# Task is used in __init__, but not directly in __call__.
# However, the base class might expect it or it's good practice to import.
# from task import Task # Let's assume Heuristic base class handles task initialization

# Helper function to parse fact strings
def parse_fact(fact_str):
    """Removes surrounding brackets and splits by space."""
    # Handle potential empty fact string or malformed string defensively
    if not fact_str or not fact_str.startswith('(') or not fact_str.endswith(')'):
        # In a real planner, logging or error handling would be better
        # print(f"Warning: Malformed fact string: {fact_str}") # For debugging
        return None, []
    parts = fact_str[1:-1].split()
    if not parts:
        # print(f"Warning: Empty fact string after removing brackets: {fact_str}") # For debugging
        return None, []
    return parts[0], parts[1:] # predicate, arguments


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    The heuristic estimates the required number of actions to reach the goal
    state by combining the number of unsatisfied goal predicates with a penalty
    for blocks that are in the wrong position according to the goal and are
    buried under other blocks. It aims to guide a greedy best-first search
    efficiently.

    Assumptions:
    - The state representation uses strings like '(predicate arg1 arg2)'.
    - PDDL predicates used in the state and goal are 'on', 'on-table', 'clear',
      'holding', and 'arm-empty'.
    - Blocksworld objects are represented by strings (e.g., 'b1', 'b2').
    - The goal state is fully specified by the set of goal predicates provided
      in the task.
    - Every block in the problem instance is either on another block, on the
      table, or held by the arm. The state representation is assumed to be
      complete regarding the position of blocks (via 'on' or 'on-table' facts,
      or implicitly by 'holding').
    - The goal configuration defined by 'on' and 'on-table' predicates forms
      acyclic stacks.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic processes the task information:
    - Stores the set of all goal predicates (`self.goal_facts`).
    - Parses goal predicates to separate 'on' and 'on-table' goals for easier access.
    - Identifies the set of all blocks that are explicitly mentioned as arguments
      in goal 'on' or 'on-table' predicates (`self.goal_blocks`).
    - Extracts all object names from the task's facts (`self.all_objects`). This
      is done by looking for arguments in any fact, assuming they are objects
      if they are lowercase strings not matching known predicates or 'table'.
      It also explicitly adds objects found in 'object' type facts if they exist.
      Finally, it ensures all blocks from `self.goal_blocks` are included in
      `self.all_objects`.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (`node.state`):
    1. Initialize the heuristic value `h` to 0.
    2. Add 1 to `h` for every goal predicate (of any type: 'on', 'on-table',
       'clear', 'arm-empty') that is not present in the current state. This is
       the standard goalcount component.
    3. If `h` is 0 after step 2, the state is the goal state, so return 0 immediately.
    4. Parse the current state predicates to build data structures representing
       the current configuration of blocks:
       - `current_on_top`: a dictionary mapping a block to the block immediately
         on top of it.
       - `current_below`: a dictionary mapping a block to the block or 'table'
         immediately below it.
       - `current_holding`: the block currently held by the arm, or None.
       - `current_clear`: a set of blocks that are clear.
       - `current_arm_empty`: a boolean indicating if the arm is empty.
    5. Identify blocks that are "not in place" according to the goal 'on' and
       'on-table' relations. A block `B` is considered "in place" if:
       - `(on-table B)` is a goal predicate AND `(on-table B)` is true in the state.
       - OR `(on A B)` is a goal predicate (for some block A) AND `(on A B)` is true in the state.
       Iterate through all blocks `B` that are involved in goal 'on' or 'on-table'
       predicates (`self.goal_blocks`).
    6. For each block `B` identified in step 5 as "not in place":
       Calculate the number of blocks currently stacked directly on top of `B`
       in the current state using the `current_on_top` structure. This count
       represents the number of blocks that need to be moved off `B` before `B`
       can potentially be moved to its correct position.
       Add this count to `h`. This adds a penalty for misplaced blocks that are
       buried under others.
    7. Return the final calculated value of `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goal_facts = set(task.goals)

        self.goal_on_preds = set()
        self.goal_on_table_preds = set()
        self.goal_blocks = set()

        # Parse goal predicates to identify goal structure and involved blocks
        for goal_fact_str in self.goal_facts:
            pred, args = parse_fact(goal_fact_str)
            if pred == 'on':
                if len(args) == 2:
                    block, below = args
                    self.goal_on_preds.add(goal_fact_str)
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(below)
            elif pred == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    self.goal_on_table_preds.add(goal_fact_str)
                    self.goal_blocks.add(block)
            # Note: clear and arm-empty goals don't involve blocks in a structural way
            # for the penalty part of this heuristic, but are included in goal_facts.

        # Extract all object names from task facts
        self.all_objects = set()
        # Assuming task.facts contains all ground facts, including potential type facts like '(object b1)'
        # and also facts from the initial state.
        for fact_str in task.facts:
             pred, args = parse_fact(fact_str)
             if pred == 'object':
                 # If 'object' facts exist, args should be a list of object names
                 self.all_objects.update(args)
             else:
                 # Collect objects from predicate arguments as a fallback/completeness check
                 # Assume objects are lowercase strings not matching known predicates or 'table'
                 # This is a heuristic and might need adjustment based on actual PDDL object naming
                 known_keywords = {'table', 'arm-empty', 'clear', 'holding', 'on', 'on-table'}
                 for arg in args:
                     if isinstance(arg, str) and arg.islower() and arg not in known_keywords:
                          self.all_objects.add(arg)

        # Ensure all goal blocks are in the set of all objects found
        self.all_objects.update(self.goal_blocks)


    def __call__(self, node):
        state = node.state
        h = 0

        # --- Step 2: Goalcount for all goal predicates ---
        # Add 1 for each goal predicate not in state
        for goal_fact in self.goal_facts:
            if goal_fact not in state:
                h += 1

        # --- Step 3: Check if goal state ---
        # If h is already 0, it's the goal state, no need for further calculation
        if h == 0:
            return 0

        # --- Step 4: Parse current state structure ---
        current_on_top = {}
        current_below = {}
        current_holding = None
        current_clear = set()
        current_arm_empty = False

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'on':
                if len(args) == 2:
                    block, below = args
                    current_below[block] = below
                    current_on_top[below] = block # Assuming only one block on top
            elif pred == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    current_below[block] = 'table'
            elif pred == 'holding':
                if len(args) == 1:
                    current_holding = args[0]
            elif pred == 'clear':
                 if len(args) == 1:
                     current_clear.add(args[0])
            elif pred == 'arm-empty':
                 current_arm_empty = True

        # --- Step 5 & 6: Penalty for misplaced and buried blocks ---
        # Identify blocks that are "not in place" and calculate penalty
        for block in self.goal_blocks:
            is_b_in_place = False

            # Check if goal is (on-table B) and it's true in the state
            goal_on_table_b = f'(on-table {block})'
            if goal_on_table_b in self.goal_on_table_preds and goal_on_table_b in state:
                 is_b_in_place = True

            # Check if goal is (on A B) for some A and it's true in the state
            # Find the specific goal (on A B) where block is the upper block
            goal_on_b_pred = None
            for pred_str in self.goal_on_preds:
                 _, args = parse_fact(pred_str)
                 if len(args) == 2 and args[0] == block: # If block is the upper block in this goal 'on' predicate
                     goal_on_b_pred = pred_str
                     break # Found the specific goal (on block X)

            if goal_on_b_pred is not None and goal_on_b_pred in state:
                 is_b_in_place = True

            # If the block is not in its goal-specified immediate position
            if not is_b_in_place:
                # Calculate penalty: number of blocks on top of 'block'
                current = current_on_top.get(block)
                penalty = 0
                while current is not None:
                    penalty += 1
                    current = current_on_top.get(current)
                h += penalty

        # --- Step 7: Return final heuristic value ---
        return h
