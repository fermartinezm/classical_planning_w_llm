from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Estimates the cost to reach the goal by counting discrepancies in block
    positions and the blocks on top of them, relative to the goal state.

    Docstring Sections:
    Summary:
        This heuristic counts the number of "incorrect" relationships in the
        current state compared to the goal state. An incorrect relationship
        is defined in two ways for blocks that are part of the goal structure:
        1. A block is not on the correct support (another block or the table)
           as specified in the goal.
        2. A block *is* on the correct support, but the block directly on top
           of it is not the one specified in the goal (or the block should be
           clear in the goal but is not clear in the state).
        Additionally, a penalty is added if the arm is holding a block when
        the goal requires the arm to be empty. Each discrepancy adds 1 to the
        heuristic value.

    Assumptions:
        - The input state and goal are valid Blocksworld states/goals.
        - Every block in a valid state is either on another block, on the table,
          or held by the arm.
        - The goal does not contain contradictory predicates (e.g., (on A B) and
          (clear B) for the same B).
        - The Heuristic base class is available and provides the expected structure
          with __init__(self, task) and __call__(self, node).
        - The task object provides task.goals (frozenset of goal fact strings)
          and node.state (frozenset of state fact strings).

    Heuristic Initialization:
        In the __init__ method, the heuristic pre-processes the goal facts to
        build data structures that allow for quick lookup of the desired state
        for each block involved in the goal.
        - `goal_support`: A dictionary mapping a block to its desired support
          (either another block's name or the string 'table') based on (on ?x ?y)
          and (on-table ?x) goal predicates.
        - `goal_block_on`: A dictionary mapping a block to the block that should
          be directly on top of it based on (on ?x ?y) goal predicates. If a block
          should be clear according to a (clear ?x) goal predicate, its entry is
          set to None.
        - `goal_blocks`: A set containing the names of all blocks that appear in
          any (on ?x ?y), (on-table ?x), or (clear ?x) goal predicate. This set
          defines which blocks are relevant to the goal structure.
        - It also checks if (arm-empty) is a goal.

    Step-By-Step Thinking for Computing Heuristic:
        In the __call__ method, for a given state:
        1. Parse the current state facts to build temporary data structures:
           - `current_pos`: A dictionary mapping a block to its current support
             (another block's name, 'table', or 'arm' if held) based on (on ?x ?y),
             (on-table ?x), and (holding ?x) state facts.
           - `current_block_on`: A dictionary mapping a block to the block
             currently directly on top of it based on (on ?x ?y) state facts.
             If a block is clear, it will not be a key in this dictionary, or its
             value would effectively be None (handled by .get()).
           - `holding_obj`: Stores the name of the block being held, or None if arm is empty.
           - `is_arm_empty`: Boolean indicating if (arm-empty) is in the state.
        2. Initialize the heuristic value `h` to 0.
        3. Iterate through each block in the pre-calculated `goal_blocks` set.
        4. For each block:
           a. If the block is the special 'arm' key (used when (arm-empty) is a goal):
              Check if (arm-empty) is a goal and if the arm is not empty in the current state.
              If both are true, increment `h`. Continue to the next block.
           b. For regular blocks:
              Retrieve the desired support (`goal_support.get(block)`) and the current
              support (`current_pos.get(block)`).
              If the current support does not match the desired support, increment `h`.
              This counts blocks that are not on the correct base.
           c. If the current support *does* match the desired support:
              Retrieve the desired block on top (`goal_block_on.get(block)`) and the
              current block on top (`current_block_on.get(block)`).
              If the current block on top does not match the desired block on top
              (this includes cases where something is on a block that should be clear,
              or the wrong block is on top, or the correct block is missing from the top),
              increment `h`. This counts blocks that are on the correct base but have
              an incorrect configuration directly above them.
        5. Return the final value of `h`.

    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating goal information.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals

        # Pre-calculate goal positions and goal blocks on top
        self.goal_support = {}
        self.goal_block_on = {}
        # Blocks explicitly mentioned in on/on-table/clear goals, plus 'arm' if arm-empty is a goal
        self.goal_blocks = set()

        # First pass: Process 'on' and 'on-table' goals to determine desired support
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                self.goal_support[obj] = underob
                self.goal_block_on[underob] = obj # obj should be on underob
                self.goal_blocks.add(obj)
                self.goal_blocks.add(underob)
            elif predicate == "on-table":
                obj = parts[1]
                self.goal_support[obj] = 'table'
                self.goal_blocks.add(obj)

        # Second pass: Process 'clear' goals. These indicate nothing should be on the block.
        # This pass ensures goal_block_on is None if (clear B) is a goal.
        # In valid PDDL, if (clear B) is a goal, no (on A B) should be a goal.
        for goal in self.goals:
             parts = get_parts(goal)
             predicate = parts[0]
             if predicate == "clear":
                 obj = parts[1]
                 self.goal_block_on[obj] = None # Nothing should be on obj
                 self.goal_blocks.add(obj) # Add block to goal_blocks even if only clear goal exists

        # Add (arm-empty) to goal_blocks if it's a goal, using a special key
        if "(arm-empty)" in self.goals:
             self.goal_blocks.add('arm')


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The heuristic value (estimated cost to goal).
        """
        state = node.state

        # Build current position and current block on top maps
        current_pos = {} # Maps block -> support (block or 'table' or 'arm')
        current_block_on = {} # Maps block -> block_on_top (None if clear)
        holding_obj = None
        is_arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                current_pos[obj] = underob
                current_block_on[underob] = obj # obj is on underob
            elif predicate == "on-table":
                obj = parts[1]
                current_pos[obj] = 'table'
            elif predicate == "holding":
                holding_obj = parts[1]
                current_pos[holding_obj] = 'arm'
            elif predicate == "clear":
                # If clear(obj) is true, nothing is on obj.
                # The absence of obj as a key in current_block_on means nothing is on it.
                pass # current_block_on.get(obj) will return None if obj is not a key
            elif predicate == "arm-empty":
                 is_arm_empty = True


        h = 0

        # Iterate through blocks and the arm state that are part of the goal structure
        for item in self.goal_blocks:
            if item == 'arm': # Handle the arm-empty goal separately
                 # Check if (arm-empty) is a goal (already ensured by adding 'arm' to goal_blocks)
                 if not is_arm_empty: # Arm should be empty but isn't
                      h += 1
                 continue # Move to the next item in goal_blocks

            # Handle blocks
            block = item
            goal_support = self.goal_support.get(block)
            current_support = current_pos.get(block) # None if block is not in state (shouldn't happen in valid states)

            # Check if the block is on the correct support
            if current_support != goal_support:
                 h += 1
            else: # current_support == goal_support
                 # The block is on the correct support. Now check the block on top.
                 goal_top_block = self.goal_block_on.get(block) # None if clear goal or no block on top in goal
                 current_top_block = current_block_on.get(block) # None if no block is on it in state

                 if current_top_block != goal_top_block:
                     h += 1

        return h

