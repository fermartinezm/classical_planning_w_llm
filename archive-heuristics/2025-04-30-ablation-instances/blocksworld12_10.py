# from fnmatch import fnmatch # Not used
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct goal position within the goal stack structure, plus 1 if the
    arm needs to be empty in the goal but isn't. A block is considered
    correctly placed only if it is on its correct goal parent (another
    block or the table) AND its goal parent is also correctly placed
    (recursively), or if it has no specified goal position and is on the table.

    # Assumptions
    - The goal state defines a set of desired stack configurations and potentially
      requires the arm to be empty.
    - Blocks must be stacked from the bottom up.
    - The goal state is reachable.
    - Standard Blocksworld goals are used (on, on-table, arm-empty, clear).
    - Blocks not explicitly mentioned in 'on' or 'on-table' goals are implicitly
      meant to be on the table in the goal state.

    # Heuristic Initialization
    - Extract the goal configuration for each block explicitly mentioned in
      '(on ...)' or '(on-table ...)' goal facts. This creates a mapping
      `goal_parent_map` from block to its desired parent (another block or 'table').
    - Identify all unique blocks present in the initial state and goal facts.
      This set represents all blocks in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to create a mapping `goal_parent_map` where
       `goal_parent_map[block]` is the block or 'table' that `block` should
       be directly on top of in the goal state, for blocks explicitly mentioned
       in 'on' or 'on-table' goals.
    2. Identify all blocks in the problem instance by collecting unique objects
       from the initial state and goal facts.
    3. For a given state, parse the current facts to create a mapping
       `current_parent_map` where `current_parent_map[block]` is the block
       or 'table' that `block` is currently directly on top of, or 'holding'
       if the arm is holding it.
    4. Initialize the heuristic count `total_misplaced_count` to 0.
    5. For each block `B` identified in step 2:
       - Determine its goal parent: If `B` is in `goal_parent_map`, its goal parent
         is `goal_parent_map[B]`. Otherwise, its implicit goal parent is 'table'.
       - Check if `B` is "correctly stacked" relative to this goal parent. A block `B`
         is correctly stacked if:
         - Its current parent matches its goal parent AND
         - If the goal parent is another block `P`, then `P` must also be correctly stacked.
         - If the block is currently `holding`, it is not correctly stacked.
       - Use memoization to efficiently compute the `is_correctly_stacked` status
         for each block.
       - If `B` is not correctly stacked, increment `total_misplaced_count`.
    6. Check if `(arm-empty)` is a goal fact and if the arm is not empty in the current state.
       If both are true, increment `total_misplaced_count` by 1 (representing the `putdown` action needed).
    7. The heuristic value is `total_misplaced_count`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal parent mapping and
        identifying all blocks.
        """
        self.task = task # Store task to check goal_reached
        self.goals = task.goals

        # Build the goal_parent_map: block -> desired_parent (block or 'table')
        # Only includes blocks explicitly mentioned in 'on' or 'on-table' goals.
        self.goal_parent_map = {}
        # Collect all unique objects from initial state and goal facts
        self.all_problem_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent_map[block] = parent
                self.all_problem_blocks.add(block)
                self.all_problem_blocks.add(parent) # Parent is also a block
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parent_map[block] = 'table'
                self.all_problem_blocks.add(block)
            elif len(parts) > 1: # Add objects from other goal types like (clear B)
                 self.all_problem_blocks.add(parts[1])


        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts[1:]: # Add all objects mentioned in initial state facts
                 if not part.startswith('?'): # Avoid parameters
                     self.all_problem_blocks.add(part)

        # Remove 'table' if it was added as an object name
        self.all_problem_blocks.discard('table')


    def is_correctly_stacked(self, block, current_parent_map, memo):
         """
         Recursive helper with memoization to check if a block is correctly stacked
         relative to its goal position.
         """
         if block in memo:
              return memo[block]

         # Determine the goal parent for this block
         # Default to 'table' if the block is not explicitly in the goal_parent_map
         goal_parent = self.goal_parent_map.get(block, 'table')

         # Find the current parent of the block
         current_parent = current_parent_map.get(block)

         # If the block is not in the state at all, it's not correctly placed.
         # (This shouldn't happen if all_problem_blocks is correctly populated
         # and state is valid, but defensive check).
         if current_parent is None:
              memo[block] = False
              return False

         # If the block is being held, it's not correctly stacked in a final position
         if current_parent == 'holding':
             memo[block] = False
             return False

         # Check if the current parent matches the goal parent
         if current_parent != goal_parent:
             memo[block] = False
             return False

         # If the current parent is the table and it matches the goal parent,
         # the block is correctly placed.
         if goal_parent == 'table':
             memo[block] = True
             return True

         # If the current parent is another block and it matches the goal parent,
         # the block is correctly placed only if the parent block is also correctly stacked.
         # This is the recursive step.
         # Ensure the parent block exists in the problem (it should if it's a goal parent)
         # This check is mostly defensive; goal_parent should be in all_problem_blocks
         # if it came from a goal fact.
         if goal_parent in self.all_problem_blocks:
             memo[block] = self.is_correctly_stacked(goal_parent, current_parent_map, memo)
         else:
             # This case indicates an issue with problem definition or parsing.
             # A goal parent block should always be a known block.
             # Treat as not correctly stacked.
             memo[block] = False # Should not happen in valid problems

         return memo[block]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached first for efficiency and correctness (h=0 iff goal)
        if self.task.goal_reached(state):
             return 0

        # Build the current_parent_map: block -> current_parent (block, 'table', or 'holding')
        current_parent_map = {}
        # Initialize all blocks to have no parent recorded yet
        for block in self.all_problem_blocks:
             current_parent_map[block] = None # Use None to indicate not found yet

        # Populate map from state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                if block in self.all_problem_blocks: # Only track known blocks
                    current_parent_map[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                if block in self.all_problem_blocks: # Only track known blocks
                    current_parent_map[block] = 'table'
            elif parts[0] == 'holding':
                 block = parts[1]
                 if block in self.all_problem_blocks: # Only track known blocks
                    current_parent_map[block] = 'holding'
            # Ignore other predicates like (clear) and (arm-empty) for this map


        total_misplaced_count = 0
        memo = {} # Memoization for recursive calls

        # Check correctness for all blocks identified in the problem
        for block in self.all_problem_blocks:
             if not self.is_correctly_stacked(block, current_parent_map, memo):
                 total_misplaced_count += 1

        # Add cost for arm-empty goal if needed and not satisfied
        arm_empty_goal = "(arm-empty)" in self.goals
        arm_empty_state = "(arm-empty)" in state
        if arm_empty_goal and not arm_empty_state:
             # If the arm is not empty, it's holding something.
             # To achieve arm-empty, a putdown action is needed.
             # This adds 1 to the heuristic.
             # Check if currently holding something to be sure (should be true if not arm-empty)
             # This check is redundant in Blocksworld as not arm-empty implies holding,
             # but harmless.
             holding_something = any(get_parts(fact)[0] == 'holding' for fact in state)
             if holding_something: # Only add cost if actually holding something
                  total_misplaced_count += 1

        # The heuristic should be 0 only at the goal state.
        # We already handled the goal state returning 0 at the beginning.
        # For any non-goal state, the heuristic must be > 0.
        # The logic ensures this for standard Blocksworld goals.

        return total_misplaced_count
