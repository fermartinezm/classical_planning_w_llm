from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the heights
        of all stacks that are rooted at a block that is currently in the wrong
        position relative to its goal. A block is considered a "root" of an
        incorrect stack if it is misplaced (its current support is not its goal
        support) AND it is currently on a stable base (either the table or a
        block that is correctly positioned relative to its goal). The heuristic
        counts the number of blocks in each such stack segment, summing these
        counts. This captures the idea that an entire incorrect stack segment
        needs to be moved to fix the position of the base block.

    Assumptions:
        - The input state and goal are represented as frozensets of PDDL fact strings.
        - The goal consists primarily of (on ?x ?y) and (on-table ?x) predicates,
          defining the desired stack structure. (clear ?x) goals are implicitly handled
          if the stack structure is correct.
        - All objects mentioned in the initial state are blocks.
        - The heuristic base class provides the task object with goals and static facts.
        - All blocks present in the initial state are expected to be part of the
          goal stack configuration defined by (on) and (on-table) goal facts.
        - The robot can only hold one block at a time.

    Heuristic Initialization:
        The constructor extracts all blocks mentioned in the initial state facts
        ((on), (on-table), (clear), (holding)).
        It builds a mapping `goal_under` from the goal facts, where `goal_under[block]`
        is the block that `block` should be directly on top of in the goal, or the string 'table'.
        Blocks present in the initial state but not explicitly positioned in the goal
        (via on/on-table) are not considered in the misplaced check.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Parse the current state facts to determine the current position of each block
           and the block currently on top of each block. Create mappings `current_under`
           and `current_top`. Handle blocks that are held by representing their
           `current_under` as 'arm'.
        2. Identify which blocks are "incorrectly placed": A block B is incorrectly placed
           if it is present in the `goal_under` mapping AND its current position
           (`current_under.get(B)`) is not equal to its desired goal position (`goal_under[B]`).
        3. Identify the "roots" of incorrect stacks: A block R is a root if it is
           incorrectly placed AND it is currently on a stable base. A stable base is
           either the table ('table'), the arm ('arm' for a held block), or a block A
           that is currently positioned correctly relative to its goal (`current_under.get(A) == goal_under.get(A)`).
        4. Initialize the heuristic value `h` to 0.
        5. Maintain a set `counted_blocks` to avoid double-counting blocks in overlapping stacks.
        6. Iterate through all blocks identified in the initial state.
        7. For each block `B`:
           a. Check if `B` is incorrectly placed using the `is_incorrectly_placed` map.
           b. Check if `B` is a root based on the definition in step 3.
           c. If `B` is a root and has not been counted yet (i.e., not in `counted_blocks`):
              i. Calculate the height of the stack segment starting from `B` upwards
                 in the current state using the `current_top` mapping. This height
                 includes `B` itself.
              ii. Add this height to `h`.
              iii. Add all blocks in this stack segment to `counted_blocks`.
        8. The final value of `h` is the heuristic estimate.

        This heuristic is efficiently computable (linear in the number of blocks and facts)
        and provides a non-admissible estimate that correlates with the number of blocks
        that need to be moved because they are part of a misplaced stack segment starting
        from a stable base.
    """
    def __init__(self, task):
        self.goals = task.goals

        # Extract all objects (blocks) from the initial state facts
        self.blocks = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 # Add all arguments as blocks
                 self.blocks.update(parts[1:])

        # Build goal_under mapping from goal facts
        self.goal_under = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "on":
                block, under_block = parts[1], parts[2]
                self.goal_under[block] = under_block
            elif predicate == "on-table":
                block = parts[1]
                self.goal_under[block] = 'table'
            # Ignore (clear ?x) goals for the stack structure definition

        # Ensure all blocks found are in the goal_under map, even if their goal is table
        # and not explicitly listed for all blocks. This handles cases where only
        # specific stacks are mentioned in the goal. Blocks not in goal_under
        # will not be considered "misplaced" by this heuristic's definition.
        # For standard Blocksworld, all blocks in init are usually in goal_under.
        for block in list(self.blocks): # Iterate over a copy as blocks might be added
             if block not in self.goal_under:
                 # If a block is in init but not in goal_under, its goal position is unknown.
                 # We cannot determine if it's misplaced relative to the goal structure.
                 # For standard Blocksworld, this case is rare/invalid.
                 # If it happens, this block won't contribute to the 'misplaced' count.
                 pass # Or assign a default goal, e.g., 'table'? Let's stick to goal facts.


    def get_stack_height(self, block, current_top_map):
        """Calculates the height of the stack segment starting from block upwards."""
        height = 0
        current = block
        # Traverse upwards using the current_top map
        while current is not None:
            height += 1
            current = current_top_map.get(current)
        return height

    def __call__(self, node):
        state = node.state

        # Build current_under and current_top mappings from state facts
        current_under = {}
        current_top = {} # Maps block -> block on top of it
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "on":
                block, under_block = parts[1], parts[2]
                current_under[block] = under_block
                current_top[under_block] = block # Map the block below to the block on top
            elif predicate == "on-table":
                block = parts[1]
                current_under[block] = 'table'
            elif predicate == "holding":
                 block = parts[1]
                 held_block = block
                 current_under[block] = 'arm' # Represent held block as being 'on arm'

        # Infer current_top for blocks that are clear (not mentioned as being under another block)
        # and are not the held block
        for block in self.blocks:
             if block not in current_top and block != held_block:
                 current_top[block] = None # Explicitly mark as top of stack

        # Identify incorrectly placed blocks (only consider blocks with a goal position)
        is_incorrectly_placed = {}
        for block in self.blocks:
            if block in self.goal_under:
                 is_incorrectly_placed[block] = (current_under.get(block) != self.goal_under[block])
            # Blocks not in goal_under are not marked as incorrectly_placed by this heuristic

        # Find roots of incorrect stacks
        # A root is a misplaced block that is on a stable base (table, arm, or correctly placed block)
        roots = set()
        for block in self.blocks:
            # Check if block is incorrectly placed according to the goal structure
            if is_incorrectly_placed.get(block, False):
                under_block = current_under.get(block)

                if under_block == 'table':
                    # Misplaced block on the table is a root
                    roots.add(block)
                elif under_block == 'arm':
                    # Misplaced held block is a root (stack height 1)
                    roots.add(block)
                elif under_block is not None and under_block in self.blocks: # Check if under_block is a block we track
                    # Misplaced block on another block
                    # Check if the block it's on is correctly placed
                    if not is_incorrectly_placed.get(under_block, False):
                         roots.add(block)
                # If under_block is None, it's an inconsistency or block not in state? Assume valid states.


        h = 0
        counted_blocks = set() # To avoid double counting blocks in overlapping stacks

        # Iterate through all blocks that could potentially be the bottom of a stack
        # (i.e., they are on the table or held)
        potential_bottoms = [b for b in self.blocks if current_under.get(b) in ['table', 'arm']]

        for block in potential_bottoms:
             # Check if this block is a root of an incorrect stack
             # and if its stack hasn't been counted yet (via a lower root)
             if block in roots and block not in counted_blocks:
                  # Calculate stack height above this root
                  height = self.get_stack_height(block, current_top)
                  h += height
                  # Mark all blocks in this stack as counted (part of this root's stack)
                  current = block
                  while current is not None:
                       counted_blocks.add(current)
                       current = current_top.get(current)

        # If the state is the goal state, is_incorrectly_placed will be empty for all blocks
        # in goal_under. Roots will be empty. h will be 0. Correct.
        # If the goal is empty, goal_under is empty, is_incorrectly_placed is empty, roots are empty, h is 0. Correct.

        return h
