from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though planner states should be clean.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting blocks that are not in their correct goal position relative
    to the block below them, plus any blocks currently stacked on top of
s    those misplaced blocks. Each such block is estimated to require two actions
    (one to pick/unstack, one to putdown/stack).

    # Assumptions
    - All actions have a unit cost.
    - Blocks not explicitly mentioned in goal 'on' or 'on-table' facts are
      assumed to have a goal of being on the table.
    - The heuristic is non-admissible and designed to guide a greedy best-first
      search by prioritizing states that appear structurally closer to the goal stacks.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired block-below relationships
      (`goal_below` map) and identifies all relevant objects in the problem
      (from initial and goal states).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current block-below relationships
       (`current_below` map), the block currently held (`current_held`), and
       the inverse mapping (`current_above`).
    2. Identify the set of blocks (`wrong_base`) whose immediate base is incorrect
       according to the goal. This includes:
       - Blocks explicitly in the goal's `on` or `on-table` facts but currently
         on a different block or the table.
       - Blocks not explicitly in the goal's `on` or `on-table` facts but
         currently not on the table (assuming their implicit goal is on-table).
       - Any block currently held by the arm.
    3. Initialize a set `blocks_to_move_or_clear` with the blocks from `wrong_base`.
    4. Perform a breadth-first traversal starting from blocks in `wrong_base`
       using the `current_above` mapping. Any block found to be currently
       stacked on top of a block already in `blocks_to_move_or_clear` is added
       to the set `blocks_to_move_or_clear`. These blocks must be moved out
       of the way before the blocks below them can be fixed.
    5. The heuristic value is calculated as `2 * len(blocks_to_move_or_clear)`.
       This estimates that each block needing to be moved or cleared requires
       approximately two actions (e.g., unstack/pickup + putdown/stack).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure and objects.

        @param task: The planning task object.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state # Need initial state to find all objects

        # Map block -> block_below or 'table' in the goal state
        self.goal_below = {}
        # Set of all objects involved in the problem
        self.all_objects = set()

        # Parse initial state to find all objects
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] != 'arm-empty': # Ignore arm-empty predicate
                 # Add all arguments as objects
                self.all_objects.update(parts[1:])

        # Parse goal state to find all objects and goal structure
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty or malformed goals

            predicate = parts[0]
            args = parts[1:]
            # Add all arguments as objects
            self.all_objects.update(args)

            if predicate == 'on' and len(args) == 2:
                block, block_below = args
                self.goal_below[block] = block_below
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_below[block] = 'table'
            # Ignore 'clear' goals for the goal_below mapping, they are implicitly handled
            # by the stack structure definition.

        # Ensure all objects mentioned in goal_below are in all_objects
        self.all_objects.update(self.goal_below.keys())
        self.all_objects.update(self.goal_below.values())
        self.all_objects.discard('table') # 'table' is a location, not an object


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated number of actions to reach the goal.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Parse current state to build relationships
        current_below = {} # block -> block_below or 'table'
        current_above = {} # block_below or 'table' -> set of blocks on top
        current_held = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, block_below = args
                current_below[block] = block_below
                current_above.setdefault(block_below, set()).add(block)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_below[block] = 'table'
                current_above.setdefault('table', set()).add(block)
            elif predicate == 'holding' and len(args) == 1:
                current_held = args[0]
            # Ignore 'clear' and 'arm-empty' for this heuristic calculation

        # Identify blocks that are in the wrong position relative to their goal base
        wrong_base = set()

        for obj in self.all_objects:
            # Determine the desired base for this block
            # If obj is not in goal_below, assume its goal is on-table
            desired_base = self.goal_below.get(obj, 'table')

            # Determine the current base for this block
            current_base = current_below.get(obj)
            if current_held == obj:
                current_base = 'arm' # Use a special value for held blocks

            # A block needs fixing if its current base is different from its desired base
            # Note: A block cannot be on 'arm' in the goal, so held blocks always need fixing.
            if current_base != desired_base:
                 wrong_base.add(obj)


        # Identify all blocks that need to be moved or cleared from on top of
        # blocks that need fixing. This propagates upwards from wrong_base.
        blocks_to_move_or_clear = set(wrong_base)

        # Use a queue for breadth-first propagation
        q = deque(list(wrong_base)) # Use deque for efficient popping
        visited = set(q) # Keep track of blocks added to the set/queue

        while q:
            b_below = q.popleft() # Get a block that needs fixing or clearing

            # Find blocks currently on top of b_below
            if b_below in current_above:
                for b_on in current_above[b_below]:
                    # Any block on top of a block that needs fixing/clearing
                    # must also be moved out of the way.
                    if b_on not in visited:
                        blocks_to_move_or_clear.add(b_on)
                        visited.add(b_on)
                        q.append(b_on)

        # The heuristic value is the number of blocks identified, multiplied by 2
        # to estimate the pickup/unstack + putdown/stack actions needed per block.
        h = 2 * len(blocks_to_move_or_clear)

        return h
