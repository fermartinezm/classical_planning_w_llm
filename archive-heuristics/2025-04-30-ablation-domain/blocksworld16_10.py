from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Depending on expected input robustness, could raise error or return empty
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
    # Ensure the number of parts matches the number of arguments for a valid match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position relative to the block below them in the goal stack,
    considering only blocks that are part of a correctly built stack segment
    starting from the table. The heuristic value is the count of blocks
    specified in the goal configuration that are not "correctly placed"
    in this structural sense, plus a penalty if the arm is not empty.

    # Assumptions
    - The goal state defines a desired configuration of blocks in stacks on the table.
    - The heuristic focuses on achieving the correct relative positions within goal stacks.
    - The cost of actions is uniform (implicitly 1).
    - The arm should be empty in the goal state (standard for blocksworld stack goals).

    # Heuristic Initialization
    - Extracts the desired support for each block from the goal conditions, creating
      a `goal_on` mapping (block -> support_block or 'table').
    - Identifies which blocks are intended to be at the top of goal stacks (`goal_tops`).
    - Collects all objects (blocks) present in the problem instance from initial and goal states.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the goal facts to create a mapping `goal_on` where `goal_on[B]` is the block
       or 'table' that block B should be directly on top of in the goal state.
       Also, identify the set of blocks that are at the top of goal stacks (`goal_tops`).
    2. Parse the current state facts to create a mapping `current_on` where
       `current_on[B]` is the block or 'table' that block B is currently directly on top of.
       Identify the set of blocks that are currently clear (`current_clear`).
       Track if the arm is currently holding a block (`is_holding`). If a block B is held,
       `current_on[B]` is set to a special value like 'holding'.
    3. Determine the set of blocks that are "correctly placed" according to the goal structure.
       A block B is correctly placed if:
       - B is specified in the `goal_on` mapping (i.e., it's part of the goal configuration).
       - Its current support (`current_on[B]`) matches its goal support (`goal_on[B]`).
       - If its goal support is a block Y, then Y must also be in the set of correctly placed blocks.
       - If B is a goal top block (`B` in `goal_tops`), it must be currently clear (`B` in `current_clear`).
       This set is computed iteratively:
       - Start with blocks in `goal_on` that should be on the table and currently are, and are clear if they are goal tops.
       - In each subsequent step, add blocks in `goal_on` that are currently on their goal support block, where that support block is already in the correctly placed set, and the block itself is clear if it is a goal top.
       - Repeat until no new blocks can be added to the correctly placed set.
    4. The heuristic value is the count of blocks that are present in the `goal_on` mapping
       but are NOT in the set of correctly placed blocks.
    5. Add a penalty of 1 to the heuristic if the arm is currently holding a block (`is_holding` is not None),
       as an empty arm is typically required to complete the goal configuration.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships,
        identifying goal top blocks, and collecting all objects.
        """
        self.goals = task.goals

        # 1. Parse goal facts to create goal_on mapping and identify goal_tops
        self.goal_on = {}
        is_goal_support = set() # Blocks that are supports for other blocks in the goal
        all_objects = set() # Collect all objects mentioned in goal and initial state

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "on":
                # (on b1 b2) -> b1 should be on b2
                block, support = parts[1], parts[2]
                self.goal_on[block] = support
                is_goal_support.add(support)
                all_objects.add(block)
                all_objects.add(support)
            elif predicate == "on-table":
                # (on-table b3) -> b3 should be on 'table'
                block = parts[1]
                self.goal_on[block] = 'table'
                all_objects.add(block)
            # Ignore (clear X) and (arm-empty) goals for the goal_on mapping

        # Identify blocks that are goal tops (are in goal_on but not a support for any other block in goal_on)
        self.goal_tops = {block for block in self.goal_on if block not in is_goal_support}

        # Collect objects from initial state facts as well
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             # Assuming arguments are objects
             all_objects.update(parts[1:])

        self.objects = frozenset(all_objects)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the number of blocks not correctly placed in goal stacks
        and the state of the arm.
        """
        state = node.state

        # 2. Parse current state facts to get current_on, current_clear, is_holding
        current_on = {}
        current_clear = set()
        is_holding = None # Track the block being held, if any

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "on":
                # (on b1 b2) -> b1 is on b2
                block, support = parts[1], parts[2]
                current_on[block] = support
            elif predicate == "on-table":
                # (on-table b3) -> b3 is on 'table'
                block = parts[1]
                current_on[block] = 'table'
            elif predicate == "clear":
                block = parts[1]
                current_clear.add(block)
            elif predicate == "holding":
                 # (holding b15) -> b15 is held
                 held_block = parts[1]
                 is_holding = held_block
                 current_on[held_block] = 'holding' # Use a special marker

        # 3. Determine the set of blocks that are "correctly placed"
        correctly_placed = set()
        placed_this_iter = set()

        # Initialize with blocks correctly placed on the table
        for block in self.goal_on: # Only consider blocks that have a goal position defined
            if self.goal_on[block] == 'table':
                 # Check if block has a current position defined and it's on the table
                 if block in current_on and current_on[block] == 'table':
                    # Check clearance if it's a goal top
                    if block in self.goal_tops:
                        if block in current_clear:
                            placed_this_iter.add(block)
                    else: # Not a goal top, clearance doesn't matter for its own placement status relative to support
                         placed_this_iter.add(block)


        correctly_placed.update(placed_this_iter)

        # Iteratively add blocks correctly placed on already correctly placed blocks
        while placed_this_iter:
            next_iter_placed = set()
            # Check all blocks in goal_on that are not yet correctly placed
            for block in self.goal_on:
                if block not in correctly_placed:
                    if self.goal_on[block] != 'table': # Must be on a block
                        goal_support = self.goal_on[block]
                        # Check if block has a current position defined and it matches the goal support
                        if block in current_on and current_on[block] == goal_support:
                            # Check if the goal support block is already correctly placed
                            if goal_support in correctly_placed:
                                # Check clearance if it's a goal top
                                if block in self.goal_tops:
                                    if block in current_clear:
                                        next_iter_placed.add(block)
                                else: # Not a goal top
                                     next_iter_placed.add(block)

            # If no new blocks were placed in this iteration, stop
            if not next_iter_placed:
                break

            correctly_placed.update(next_iter_placed)
            placed_this_iter = next_iter_placed


        # 4. Heuristic value is the count of blocks in goal_on that are NOT correctly placed.
        h = sum(1 for block in self.goal_on if block not in correctly_placed)

        # 5. Add penalty of 1 if the arm is not empty (assuming arm-empty is implicitly desired)
        if is_holding is not None:
             h += 1

        return h
