from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    by counting the number of blocks that are in a 'wrong' state. A block is
    considered 'wrong' if it is being held, is on the incorrect support
    (table or another block) according to the goal, or is on the correct
    support but has the wrong block stacked directly on top of it. This
    heuristic is non-admissible and designed to guide a greedy best-first search.

    # Assumptions
    - The heuristic is designed for standard Blocksworld problems where the goal
      specifies the final position (on-table or on another block) for relevant
      blocks and implicitly or explicitly specifies which blocks should be clear.
    - All blocks mentioned in the goal are assumed to be 'objects'.
    - The heuristic value is 0 if and only if the state is a goal state.
    - The heuristic value is finite for all reachable states.

    # Heuristic Initialization
    - Parses the goal facts to build two maps representing the desired stack structure:
        - `goal_below_map`: Maps each block to the object (or 'table') it should
          be directly on top of in the goal state.
        - `goal_above_map`: Maps each block to the object that should be directly
          on top of it in the goal state, or 'clear' if nothing should be on it.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state facts to determine the current configuration of blocks:
       - Build `current_below_map`: Maps each block to the object (or 'table' or 'arm')
         it is currently directly on top of.
       - Build `current_above_map`: Maps each block to the object currently directly
         on top of it, or 'clear' if nothing is on it.
       - Identify `current_holding`: The block currently held by the arm, or None.
       - Identify all blocks present in the current state.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block that is part of the goal configuration (i.e., appears
       in the `goal_below_map`).
    4. For each block `B`:
       - Get its goal support (`goal_below_map[B]`) and goal block on top (`goal_above_map[B]`).
       - Get its current support (`current_below_map.get(B)`) and current block on top (`current_above_map.get(B, 'clear')`).
       - Check if `B` is currently being held (`current_below_map.get(B) == 'arm'`). If yes,
         increment `h` by 1 (it needs to be placed).
       - If `B` is not being held:
         - Check if its current support matches its goal support. If they do not match,
           increment `h` by 1 (the block is on the wrong support).
         - If its current support *does* match its goal support:
           - Check if the block currently on top of `B` matches the block that should be
             on top of `B` in the goal. If they do not match (including cases where one
             is 'clear' and the other isn't), increment `h` by 1 (the block is on the
             correct support but is obstructed or has the wrong block on top).
    5. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        goal configuration maps.
        """
        super().__init__(task) # Call base class constructor

        self.goal_below_map = {}
        self.goal_above_map = {}
        all_goal_blocks = set()

        # Parse goal facts to build goal configuration maps
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                x, y = parts[1], parts[2]
                self.goal_below_map[x] = y
                self.goal_above_map[y] = x
                all_goal_blocks.add(x)
                all_goal_blocks.add(y)
            elif predicate == 'on-table' and len(parts) == 2:
                z = parts[1]
                self.goal_below_map[z] = 'table'
                all_goal_blocks.add(z)
            # (clear) goals are implicitly handled by goal_above_map

        # Determine which blocks should be clear in the goal
        for block in all_goal_blocks:
            if block not in self.goal_above_map:
                self.goal_above_map[block] = 'clear' # Block should be clear in goal

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_below_map = {}
        current_above_map = {}
        current_holding = None
        all_current_blocks = set()

        # Parse current state facts to build current configuration maps
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                x, y = parts[1], parts[2]
                current_below_map[x] = y
                current_above_map[y] = x
                all_current_blocks.add(x)
                all_current_blocks.add(y)
            elif predicate == 'on-table' and len(parts) == 2:
                z = parts[1]
                current_below_map[z] = 'table'
                all_current_blocks.add(z)
            elif predicate == 'holding' and len(parts) == 2:
                b = parts[1]
                current_holding = b
                current_below_map[b] = 'arm'
                all_current_blocks.add(b)
            # (clear) and (arm-empty) facts are not needed for building below/above maps directly

        # Determine which blocks are clear in the current state
        for block in all_current_blocks:
             if block not in current_above_map:
                 current_above_map[block] = 'clear' # Block is clear in current state

        h = 0 # Initialize heuristic value

        # Iterate through blocks that are part of the goal configuration
        for block in self.goal_below_map.keys():
            goal_support = self.goal_below_map[block]
            goal_on_top = self.goal_above_map.get(block, 'clear') # What should be on top in goal

            # Get current support, default to None if block is not in current state maps
            # (This shouldn't happen for blocks in goal_below_map in valid states,
            # but .get is safer)
            current_support = current_below_map.get(block)
            current_on_top = current_above_map.get(block, 'clear') # What is on top now

            if current_support == 'arm':
                # Block is being held, needs to be placed
                h += 1
            elif current_support is not None: # Block is on table or another block
                if current_support != goal_support:
                    # Block is on the wrong support
                    h += 1
                else: # current_support == goal_support
                    # Block is on the correct support, check what's on top
                    if current_on_top != goal_on_top:
                        # Block is on correct support, but has wrong block on top
                        h += 1
            # If current_support is None, the block isn't in the state facts as on/on-table/holding.
            # This might indicate an issue with state representation or an unreachable state.
            # For robustness, we could potentially add a large penalty, but assuming valid
            # states for blocks in goal_below_map, this case shouldn't occur.

        return h
