from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and provides a base class named Heuristic
# from heuristics.heuristic_base import Heuristic

# If running standalone or base class is not available, define a dummy:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(arm-empty)'
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Fallback, though valid PDDL facts should have parentheses
    return fact.split()


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

    Estimates the cost based on the number of blocks that are not part
    of a correctly built stack segment from the bottom up, relative to
    the goal configuration.

    Heuristic value = 2 * (Total number of blocks - Number of "happy" blocks)

    A block X is "happy" if:
    1. Its current base (the block directly below it or the table) is the same
       as its goal base.
    2. If its goal base is another block Y, then Y must also be "happy".
       If its goal base is the table, this condition is met.

    This heuristic is not admissible but aims to guide greedy best-first search
    efficiently by prioritizing states where more blocks are in their correct
    relative positions within the goal stacks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        # Call the base class constructor if necessary (depends on Heuristic definition)
        # super().__init__(task)

        self.goals = task.goals
        self.initial_state = task.initial_state

        # 1. Extract all objects (blocks) present in the initial state.
        self.objects = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            # Consider predicates that involve objects
            if parts and parts[0] in ['on', 'on-table', 'holding', 'clear']:
                 # Objects are arguments to these predicates (skip predicate name)
                 for obj in parts[1:]:
                     self.objects.add(obj)

        # 2. Build the goal configuration: map each block to its desired base (block or 'table').
        self.goal_base = {}
        # First, handle explicit 'on' and 'on-table' goals
        explicitly_placed_in_goal = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                block, base = parts[1], parts[2]
                if block in self.objects: # Only care about objects from initial state
                    self.goal_base[block] = base
                    explicitly_placed_in_goal.add(block)
                    # The base block might also be an object we need to track
                    if base in self.objects:
                         explicitly_placed_in_goal.add(base)
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                if block in self.objects: # Only care about objects from initial state
                    self.goal_base[block] = 'table'
                    explicitly_placed_in_goal.add(block)
            # Ignore 'clear' and 'arm-empty' goals for base configuration

        # 3. For any object from the initial state not explicitly placed in goal 'on' or 'on-table',
        #    assume its goal base is 'table'.
        for obj in self.objects:
             if obj not in explicitly_placed_in_goal:
                 self.goal_base[obj] = 'table'

        # Update self.objects to only include those that have a defined goal_base.
        # This handles cases where the initial state might contain objects not relevant to the goal structure.
        # (Although in standard Blocksworld, all objects usually participate in the goal structure).
        # This also ensures we don't try to find a goal_base for something like 'table' or 'arm'.
        self.objects = {obj for obj in self.objects if self.goal_base.get(obj) is not None}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build the current configuration: map each block to its current base ('table', block, or 'arm').
        current_base = {}
        # Initialize all objects we care about to have no known base yet
        for obj in self.objects:
            current_base[obj] = None

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'on':
                block, base = parts[1], parts[2]
                if block in self.objects: # Only track objects we care about
                    current_base[block] = base
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                if block in self.objects: # Only track objects we care about
                    current_base[block] = 'table'
            elif parts and parts[0] == 'holding':
                block = parts[1]
                if block in self.objects: # Only track objects we care about
                    current_base[block] = 'arm'

        # Identify "happy" blocks iteratively.
        happy_blocks = set()
        # Iterate up to the number of objects. In each iteration, if a block becomes happy,
        # it's because its base is happy. This propagates happiness up the stacks.
        # Max depth of any stack is number of objects.
        # Using a fixed number of iterations equal to the number of objects guarantees convergence
        # because in each successful iteration, at least one new block is added to happy_blocks,
        # and a block, once happy, remains happy.
        for _ in range(len(self.objects) + 1): # Add 1 for safety margin
            newly_happy = set()
            for obj in self.objects:
                if obj not in happy_blocks:
                    goal_b = self.goal_base.get(obj)
                    current_b = current_base.get(obj)

                    # A block is happy if its current base matches its goal base...
                    # Ensure both goal_b and current_b are known (obj is in self.objects and has a base in state)
                    if current_b is not None and goal_b is not None and current_b == goal_b:
                        # ...AND if the goal base is a block, that block must be happy.
                        if goal_b == 'table':
                            newly_happy.add(obj)
                        elif goal_b in happy_blocks:
                            newly_happy.add(obj)

            if not newly_happy:
                break # No new happy blocks found, convergence
            happy_blocks.update(newly_happy)

        # The heuristic is 2 * (number of unhappy blocks).
        num_unhappy_blocks = len(self.objects) - len(happy_blocks)

        # The heuristic is 0 iff the goal is reached.
        # If the goal is reached, all goal facts are true.
        # This implies current_base matches goal_base for all blocks involved in 'on'/'on-table' goals.
        # Blocks not in 'on'/'on-table' goals are assumed to be on the table in the goal.
        # If they are on the table in the state, current_base matches goal_base ('table').
        # By induction from the table up, all blocks will be happy. So num_unhappy_blocks = 0.
        # If the state is NOT the goal, at least one goal fact is false.
        # If an (on X Y) goal is false, either X is not on Y, or Y is not in its goal position.
        # If X is not on Y, current_base[X] != goal_base[X], so X is unhappy.
        # If X is on Y, but Y is not in its goal position, Y is unhappy, making X unhappy.
        # So, if the state is not the goal, at least one block must be unhappy.
        # Thus, num_unhappy_blocks > 0 for non-goal states.
        # The heuristic is 0 iff the goal is reached.

        heuristic_value = 2 * num_unhappy_blocks

        return heuristic_value
