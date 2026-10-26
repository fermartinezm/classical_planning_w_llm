# from heuristics.heuristic_base import Heuristic # Assuming this exists

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static
        self.init = task.init # Need initial state facts to get all objects

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# Need fnmatch for consistency with examples, although not strictly needed for simple facts
from fnmatch import fnmatch

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are part of a "misplaced" stack structure and estimating
    2 actions (pickup/unstack + putdown/stack) per such block, adjusting if the
    arm is already holding a block that needs moving.

    # Assumptions:
    - Standard Blocksworld actions and predicates.
    - Goal states specify desired `on` and `on-table` relationships for relevant blocks.
    - Blocks present in the initial state but not explicitly given an `on` or `on-table`
      goal position are assumed to belong on the table in the goal state.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Identify all objects present in the initial state by parsing initial facts.
    - Parse goal facts to determine the desired position (`on` or `on-table`) for
      each object that appears in the goal.
    - For objects from the initial state not explicitly given a goal position,
      assume their goal position is `on-table`.
    - Store these goal positions in a map (`goal_pos`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current position (`on`, `on-table`, or `holding`) for every
       object present in the initial state based on the current state facts.
    2. Identify the block currently held by the arm, if any.
    3. Compute the set of blocks that "need moving". A block B needs moving if:
       a. Its current position relative to the block below it (`current_pos[B]`)
          is different from its goal position relative to the block below it (`goal_pos[B]`).
       b. OR, it is currently in the correct position relative to the block below it
          (`current_pos[B] == goal_pos[B]`), but the block below it (which is
          `goal_pos[B]`, and must be another block, not 'table') is itself a block
          that needs moving.
       This is computed iteratively: Start with blocks satisfying condition (a),
       then repeatedly add blocks satisfying condition (b) until no new blocks are added.
       All objects from the initial state are considered in this calculation.
    4. Count the total number of blocks identified as needing moving.
    5. The base heuristic value is 2 times this count. This estimates 2 actions
       (pickup/unstack and putdown/stack) for each block that is part of a
       misplaced stack structure.
    6. If the arm is currently holding a block, and that block is in the set of
       blocks needing moving, subtract 1 from the base heuristic value. This accounts
       for the fact that the pickup/unstack action for this block is already completed.
    7. Return the final calculated value.
    """

    def __init__(self, task):
        # Assuming Heuristic base class exists and is initialized correctly
        # super().__init__(task) # Use this if Heuristic is a proper base class
        self.task = task # Store task to access initial state later
        self.goals = task.goals
        # task.static is frozenset() for blocksworld, no static facts needed for this heuristic

        # 1. Get all objects from task.init by parsing all terms that are not predicates or keywords
        self.all_objects_in_init = set()
        # List all predicate names and keywords used in Blocksworld PDDL
        keywords = {'on', 'on-table', 'clear', 'holding', 'arm-empty', 'table', 'object'}
        for fact in task.init:
            parts = get_parts(fact)
            for part in parts:
                # If a part is not a known keyword, assume it's an object name
                if part not in keywords:
                    self.all_objects_in_init.add(part)

        # 2. Build goal_pos map from on and on-table goal facts.
        self.goal_pos = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                obj, underob = parts[1], parts[2]
                self.goal_pos[obj] = underob
            elif predicate == 'on-table':
                obj = parts[1]
                self.goal_pos[obj] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for goal_pos mapping

        # 3. For any object O from task.init not in goal_pos keys, set goal_pos[O] = 'table'.
        # This handles blocks present initially but not explicitly placed in the goal.
        for obj in self.all_objects_in_init:
            if obj not in self.goal_pos:
                self.goal_pos[obj] = 'table' # Default goal is on the table

    def __call__(self, node):
        state = node.state

        # 1. Determine current position for all objects in init
        current_pos = {}
        holding_obj = None

        # Initialize current_pos for all objects in init
        for obj in self.all_objects_in_init:
             current_pos[obj] = 'unknown' # Default until found in state

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                obj, underob = parts[1], parts[2]
                if obj in self.all_objects_in_init:
                    current_pos[obj] = underob
            elif predicate == 'on-table':
                obj = parts[1]
                if obj in self.all_objects_in_init:
                    current_pos[obj] = 'table'
            elif predicate == 'holding':
                obj = parts[1]
                if obj in self.all_objects_in_init:
                    current_pos[obj] = 'holding'
                    holding_obj = obj
            # Ignore clear and arm-empty in state for position tracking

        # 3. Compute needs_moving iteratively
        needs_moving = {}
        for obj in self.all_objects_in_init:
             needs_moving[obj] = False

        # Initial set: blocks whose current base is wrong compared to goal base
        initial_needs_moving_set = set()
        for obj in self.all_objects_in_init:
            current_b = current_pos.get(obj, 'unknown') # Should be in current_pos if in init
            goal_b = self.goal_pos.get(obj, 'table') # Should be in goal_pos by init logic

            # A block needs moving if its current base is not its goal base
            if current_b != goal_b:
                 initial_needs_moving_set.add(obj)

        for obj in initial_needs_moving_set:
            needs_moving[obj] = True

        # Propagate needs_moving upwards: if a block is on the correct base (according to goal)
        # but that base needs moving, then this block also needs moving.
        changed = True
        while changed:
            changed = False
            # Iterate over all objects in init
            for obj in self.all_objects_in_init:
                if not needs_moving[obj]:
                    current_b = current_pos.get(obj, 'unknown')
                    goal_b = self.goal_pos.get(obj, 'table')

                    # Check if it's in the correct position relative to below AND
                    # the block below it (in the goal stack) needs moving.
                    # This applies only if the goal position is on another block.
                    if current_b == goal_b and goal_b != 'table' and goal_b != 'holding': # goal_b can't be holding
                         block_below_in_goal = goal_b
                         # Check if the block below is in the needs_moving set
                         # We need to check if block_below_in_goal is an object that could need moving
                         # (i.e., it's in the initial state objects).
                         if block_below_in_goal in self.all_objects_in_init and needs_moving.get(block_below_in_goal, False):
                             needs_moving[obj] = True
                             changed = True

        # 4. Count blocks needing moving
        num_blocks_needing_moving = sum(1 for obj in self.all_objects_in_init if needs_moving[obj])

        # 5. Base heuristic cost
        total_cost = 2 * num_blocks_needing_moving

        # 6. Adjust for held block
        # If the arm is holding a block that needs moving, one action (pickup/unstack) is already done.
        if holding_obj is not None and needs_moving.get(holding_obj, False):
             total_cost -= 1

        # Ensure heuristic is non-negative
        return max(0, total_cost)
