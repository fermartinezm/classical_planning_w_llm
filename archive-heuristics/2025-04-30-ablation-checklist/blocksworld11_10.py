from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    # A simpler check is just zipping and checking all elements match.
    # This assumes args length is <= parts length, which is true for typical predicates.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the "distance" to the goal by counting blocks
    that are not in their correct position within the goal stacks, and
    blocks that are currently blocking access to such misplaced blocks.
    It is designed to be informative for greedy best-first search and is not admissible.

    # Heuristic Initialization
    - Parses the goal state to determine the target support (the block or table
      a block should be directly on) for each block that is part of a goal stack.
    - Collects all block names involved in the problem from goals and initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the target support for each block based on the goal state.
       Build a mapping from block to its target support (another block or 'table').
       Blocks not mentioned in goal 'on' or 'on-table' predicates are ignored for target support.
    2. Identify all blocks present in the problem instance by collecting all objects
       mentioned in relevant predicates ('on', 'on-table', 'clear', 'holding')
       in the initial state and goals.
    3. In the current state, determine the current support for each block
       (what it is directly on, or 'table'). Also, identify which blocks are
       currently on top of which others.
    4. Compute the set of blocks that are "in goal stack position". A block B is
       in goal stack position if:
       - (on-table B) is a goal AND (on-table B) is true in the state.
       - OR (on B UnderB) is a goal AND (on B UnderB) is true in the state AND
         UnderB is also in goal stack position.
       This is computed iteratively starting from blocks correctly on the table
       according to the goal.
    5. Identify blocks that are "not in place" (`NotInPlace`) - these are all blocks
       identified in step 2 minus those in `InGoalStackPosition`. These blocks
       are not part of the correctly built goal structure and likely need to be moved.
    6. Identify blocks that are "blocking misplaced" (`BlockingMisplaced`) - these
       are blocks C such that (on C B) is true in the current state and B is in
       `NotInPlace`. These blocks must be moved *first* to access B.
    7. The heuristic value is the sum of the number of blocks in `NotInPlace` and
       the number of blocks in `BlockingMisplaced`.
    8. Add a penalty of 1 if the arm is holding a block and the goal requires
       the arm to be empty. This accounts for the necessary 'putdown' action.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure and block names.
        """
        super().__init__(task)

        self.target_support = {}
        self.all_blocks = set()

        # Extract target support and collect block names from goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, support = parts[1], parts[2]
                self.target_support[block] = support
                self.all_blocks.add(block)
                self.all_blocks.add(support)
            elif parts[0] == "on-table":
                block = parts[1]
                self.target_support[block] = 'table'
                self.all_blocks.add(block)
            # (clear ?) goals don't define stack structure, ignore for target_support

        # Add any blocks from initial state not mentioned in goals
        # This ensures we consider all blocks in the problem instance
        for fact in self.initial_state:
             parts = get_parts(fact)
             # Consider objects in any relevant predicate
             if parts[0] in ["on", "on-table", "clear", "holding"]:
                 for obj in parts[1:]:
                     self.all_blocks.add(obj)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Build current support and on_relations from the current state
        current_support = {}
        on_relations = set() # Store (C, B) if C is on B
        is_holding = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, support = parts[1], parts[2]
                current_support[block] = support
                on_relations.add((block, support))
            elif parts[0] == "on-table":
                block = parts[1]
                current_support[block] = 'table'
            elif parts[0] == "holding":
                is_holding = parts[1]
            # (clear ?) facts are not needed for this heuristic calculation

        # Compute InGoalStackPosition set
        # A block B is in_goal_stack_position if it's on its target support
        # AND its target support is also in_goal_stack_position (recursively),
        # or if it's targeted for the table and is on the table.
        in_goal_stack_position = set()
        queue = [] # Use list as a simple queue for breadth-first propagation

        # Start with blocks that should be on the table according to the goal
        # AND are currently on the table in the state.
        for block in self.all_blocks:
            if self.target_support.get(block) == 'table':
                if current_support.get(block) == 'table':
                    in_goal_stack_position.add(block)
                    queue.append(block)

        # Propagate up the goal stacks: if B should be on UnderB (goal), and B is currently on UnderB (state),
        # and UnderB is already identified as in_goal_stack_position, then B is also in_goal_stack_position.
        while queue:
            under_b = queue.pop(0) # Dequeue

            # Find blocks that *should* be on under_b according to the goal
            blocks_to_check = [b for b, support in self.target_support.items() if support == under_b]

            for block in blocks_to_check:
                # Check if block is *currently* on under_b
                if current_support.get(block) == under_b:
                    if block not in in_goal_stack_position:
                        in_goal_stack_position.add(block)
                        queue.append(block) # Enqueue the newly identified block

        # Compute NotInPlace: Blocks that are not part of the correctly built goal stacks
        not_in_place = self.all_blocks - in_goal_stack_position

        # Compute BlockingMisplaced: Blocks that are currently on top of a block in NotInPlace
        blocking_misplaced = set()
        for c, b in on_relations: # For every (on C B) fact in the state
            if b in not_in_place:
                blocking_misplaced.add(c)

        # Heuristic value: Sum of blocks not in place and blocks blocking misplaced blocks
        # Each block not in place needs to be moved. Each block blocking a misplaced
        # block also needs to be moved first.
        h = len(not_in_place) + len(blocking_misplaced)

        # Add a penalty if the arm is holding a block and the goal requires arm-empty
        # This represents the cost of putting the block down, which is often necessary
        # before other actions can be taken.
        if is_holding is not None and "(arm-empty)" in self.goals:
             h += 1

        return h
