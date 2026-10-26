# Assume heuristic_base.py contains:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and has a Heuristic class
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual file is not available
# This is just for making the code runnable standalone if needed for testing
class Heuristic:
    def __init__(self, task):
        self.task = task
    def __call__(self, node):
        raise NotImplementedError("Heuristic must implement __call__")

# Utility functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to reach the goal state.
    Counts blocks that are in the wrong position relative to their base
    OR have the wrong block on top of them (or shouldn't have anything on them).

    # Summary
    This heuristic counts two types of discrepancies between the current state
    and the goal state for each block:
    1. The block's immediate base (the block it's on, or the table) is not
       the desired base according to the goal.
    2. The block immediately on top of it is not the desired block according
       to the goal (or something is on it when it should be clear).
    The total heuristic value is the sum of these discrepancies across all blocks.

    # Heuristic Initialization
    - Parses the goal state to build maps representing the desired base for
      each block (`goal_config`) and the desired block immediately above
      each block/table (`goal_above_map`).
    - Extracts all objects from the initial state and goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the immediate base for each block
       (`current_config`) and the block immediately above each block/table
       (`current_above_map`). Also identify if any block is being held.
    2. Initialize a counter `heuristic_cost` to 0.
    3. Iterate through all objects in the problem.
    4. For each object `obj`:
       a. Check its base position:
          - Get the desired base (`desired_base`) from `goal_config`.
          - Get the current base (`current_base`) from `current_config`
            (handling the case where the block is being held).
          - If `desired_base` is specified in the goal and `current_base`
            does not match `desired_base`, increment `heuristic_cost`.
       b. Check the block immediately above it:
          - Get the desired block above (`desired_above`) from `goal_above_map`
            ('nothing' if the block should be clear).
          - Get the current block above (`current_above`) from `current_above_map`
            ('nothing' if the block is clear).
          - If `desired_above` is specified (i.e., the block is a base in the
            goal config or should be clear) and `current_above` does not match
            `desired_above`, increment `heuristic_cost`.
    5. Return the total `heuristic_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing the goal state to determine
        the desired configuration.
        """
        super().__init__(task) # Call base class constructor

        self.goals = task.goals

        # Parse goal predicates to build the desired configuration
        self.goal_config = {} # maps block -> block_below or 'table'
        self.goal_above_map = {} # maps block_below -> block_above or 'nothing'

        # Collect all objects mentioned in the goal and initial state
        self.all_objects = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                obj, base = parts[1], parts[2]
                self.goal_config[obj] = base
                self.goal_above_map[base] = obj
                self.all_objects.add(obj)
                self.all_objects.add(base)
            elif predicate == "on-table":
                obj = parts[1]
                self.goal_config[obj] = 'table'
                self.all_objects.add(obj)
            elif predicate == "clear":
                obj = parts[1]
                # If a block should be clear, nothing should be above it
                self.goal_above_map[obj] = 'nothing'
                self.all_objects.add(obj)
            # Ignore arm-empty goal for this heuristic's structure

        # Add objects from initial state that might not be in the goal predicates
        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts[1:]:
                 # Simple check to identify potential objects
                 if part not in ['table', 'arm-empty'] and not part.startswith('('):
                      self.all_objects.add(part)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Parse current state
        current_config = {} # maps block -> block_below or 'table' or 'arm'
        current_above_map = {} # maps block_below -> block_above
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                obj, base = parts[1], parts[2]
                current_config[obj] = base
                current_above_map[base] = obj
            elif predicate == "on-table":
                obj = parts[1]
                current_config[obj] = 'table'
            elif predicate == "holding":
                current_holding = parts[1]
                current_config[current_holding] = 'arm' # Represent holding as being on 'arm'
            # 'clear' and 'arm-empty' predicates are implicitly handled by
            # checking if a block is in current_above_map or if current_holding is None.

        heuristic_cost = 0

        # Iterate through all objects to check their placement relative to the goal
        for obj in self.all_objects:
            # Check base position
            desired_base = self.goal_config.get(obj)
            current_base = current_config.get(obj) # Can be block, 'table', or 'arm'

            # If the block's base is specified in the goal
            if desired_base is not None:
                 # If the block is being held, it's not in its goal base position
                 if current_holding == obj:
                     heuristic_cost += 1
                 # If the block is not being held, check if its base matches the goal base
                 elif current_base != desired_base:
                     heuristic_cost += 1

            # Check block above position
            # We only care about the block above if the current block is a base in the goal config
            # or if the current block should be clear in the goal.
            desired_above = self.goal_above_map.get(obj) # 'nothing' if should be clear, block if something should be on it, None if not a base in goal_above_map keys

            if desired_above is not None: # This block is a base in the goal config or should be clear
                 current_above = current_above_map.get(obj, 'nothing') # 'nothing' if nothing is on it

                 if current_above != desired_above:
                     heuristic_cost += 1

        return heuristic_cost
