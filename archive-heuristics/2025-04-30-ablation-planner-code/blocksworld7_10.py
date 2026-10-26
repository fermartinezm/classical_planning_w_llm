from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts blocks that are part of a goal stack but are not in their correct
    position relative to the correctly built part of the stack below them, and
    blocks that are currently on top of a block that should be the top of a goal stack.
    Each such block is estimated to require at least 2 actions (pickup/unstack + stack/putdown).
    An adjustment is made if the arm is currently holding a block, as the first action
    (pickup/unstack) for that block is already completed.

    # Assumptions
    - The goal predicates define a set of disjoint stacks of blocks, potentially
      including single blocks on the table.
    - Goal stacks are rooted at blocks that are specified as being `on-table` in the goal.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) are used.

    # Heuristic Initialization
    The heuristic parses the goal predicates to build the structure of the goal stacks
    and identify the top block of each goal stack. It also stores goal `on` and
    `on-table` predicates in sets for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the position of each block (what it is `on` or if it's `on-table`)
       and whether the arm is `holding` a block. Store this in a `current_on` map and `current_held` variable.
    2. Initialize the heuristic value `h` to 0.
    3. Identify the set of blocks `S` that are "out of place". This set includes:
       a. Blocks that are part of a goal stack but are not in the correctly built segment
          starting from the bottom of that goal stack in the current state.
          For each goal stack `[B_1, B_2, ..., B_k]` (where `B_1` is the bottom block):
          Find the largest index `j` (0 <= j <= k) such that the segment `[B_1, ..., B_j]`
          is present in the current state (i.e., `(on-table B_1)` is a goal and true in state
          if j>=1, and `(on B_i B_{i-1})` is a goal and true in state for i=2..j).
          The blocks `B_{j+1}, ..., B_k` are added to the set `S`.
       b. Blocks that are currently on top of a block that is the designated top of a goal stack.
          For each block `T` and `W` such that `(on T W)` is true in the current state,
          if `W` is the top block of any goal stack, add `T` to the set `S`.
    4. The base heuristic estimate is `2 * |S|`. This is because each block in `S` typically
       requires at least one pickup/unstack action and one stack/putdown action (2 actions).
    5. Adjust the heuristic based on the block currently held by the arm (`current_held`):
       - If the arm is holding a block `B`:
         - If `B` is in the set `S`: The pickup/unstack action for `B` is already done.
           The cost for `B` is 1 (stack/putdown). The total cost is `2 * (|S| - 1) + 1 = 2*|S| - 1`.
         - If `B` is not in the set `S`: `B` is not a block that needs to be moved for
           goal stack construction or clearing goal tops according to our main criteria.
           However, it must be put down (1 action) to free the arm for other necessary moves.
           The total cost is `2 * |S| + 1`.
       - If the arm is empty, the base heuristic `2 * |S|` is used.
    6. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack structures and goal tops.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are empty for Blocksworld and not needed for this heuristic.
        # static_facts = task.static

        self.goal_stacks = []
        self.goal_tops = set()
        goal_on_map = {} # maps support -> block_on_top for goal
        goal_bottoms = set()
        self.goal_on_facts = set() # Store goal (on X Y) facts for quick lookup
        self.goal_on_table_facts = set() # Store goal (on-table X) facts for quick lookup

        # Pass 1: Identify goal 'on' relationships and potential bottoms/tops
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on_top, support = parts[1], parts[2]
                goal_on_map[support] = block_on_top
                self.goal_on_facts.add(tuple(parts)) # Store as tuple for set hashability
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                goal_bottoms.add(block)
                self.goal_on_table_facts.add(tuple(parts)) # Store as tuple
            # Ignore 'clear' goals for stack structure building

        # Build stacks starting from identified bottoms
        used_blocks_in_stack = set() # Blocks already added to a goal stack
        for bottom in list(goal_bottoms): # Iterate over a copy
            if bottom in used_blocks_in_stack:
                 continue

            stack = [bottom]
            used_blocks_in_stack.add(bottom)
            current = bottom
            while current in goal_on_map:
                next_block = goal_on_map[current]
                # Check for cycles or blocks already used in another stack (indicates non-standard goal)
                if next_block in used_blocks_in_stack:
                    # Assuming standard BW goals are disjoint stacks rooted at on-table blocks
                    # If we encounter a block already used, something is non-standard.
                    # Stop building this stack to avoid infinite loops or incorrect structures.
                    stack = [] # Discard this invalid stack segment
                    # print(f"Warning: Block {next_block} appears in multiple goal stacks or forms complex structure.")
                    break

                stack.append(next_block)
                used_blocks_in_stack.add(next_block)
                current = next_block

            if stack: # Only add valid, non-empty stacks
                self.goal_stacks.append(stack)
                self.goal_tops.add(stack[-1])

        # Add any goal_bottoms that weren't used as bases for larger stacks as single-block stacks
        # This handles goals like just (on-table A)
        for bottom in goal_bottoms:
             if bottom not in used_blocks_in_stack:
                 self.goal_stacks.append([bottom])
                 self.goal_tops.add(bottom)
                 used_blocks_in_stack.add(bottom)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        current_on = {} # maps block -> support (block or 'table')
        current_held = None
        # current_clear = set() # Not needed for this heuristic calculation

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on_top, support = parts[1], parts[2]
                current_on[block_on_top] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                current_held = parts[1]
            # Ignore 'clear' and 'arm-empty' for this calculation

        out_of_place_blocks = set()

        # 3a. Blocks in goal stacks not in the correctly built suffix
        for goal_stack in self.goal_stacks:
            k = len(goal_stack)
            j = 0 # Length of the correctly built suffix from the bottom

            if k > 0:
                bottom_block = goal_stack[0]
                # Check if the goal is (on-table bottom_block) and it's true in state
                is_goal_on_table = ('on-table', bottom_block) in self.goal_on_table_facts

                # A stack segment starts correctly only if the bottom is correctly placed on the table
                if is_goal_on_table and current_on.get(bottom_block) == 'table':
                     j = 1 # Bottom block is correctly placed on the table

                     # Check blocks above the bottom
                     for i in range(1, k): # i is the index in the stack list (0-based)
                         # goal_stack[i] should be on goal_stack[i-1]
                         block_on_top = goal_stack[i]
                         support = goal_stack[i-1]
                         # Check if (on block_on_top support) is a goal and is true in state
                         is_goal_on = ('on', block_on_top, support) in self.goal_on_facts

                         if is_goal_on and current_on.get(block_on_top) == support:
                             j = i + 1 # This block is also correctly placed relative to the one below
                         else:
                             break # The correct segment ends here

            # Blocks from index j to k-1 are misplaced relative to the correct stack below them
            for i in range(j, k):
                 out_of_place_blocks.add(goal_stack[i])

        # 3b. Blocks currently on top of goal stack top blocks
        # Iterate through all blocks that are currently on top of something
        for block_on_top, support in current_on.items():
            if support in self.goal_tops:
                 out_of_place_blocks.add(block_on_top)

        # 4. Base heuristic cost
        h = 2 * len(out_of_place_blocks)

        # 5. Adjust based on the held block
        if current_held is not None:
             if current_held in out_of_place_blocks:
                 h -= 1 # Pickup/unstack for this block is already done
             else:
                 h += 1 # This block needs to be put down to free the arm

        return h
