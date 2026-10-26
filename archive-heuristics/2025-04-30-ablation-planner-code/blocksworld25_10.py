from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def get_blocks_on_top_recursive(block, on_map):
    """
    Helper function to find all blocks stacked directly or indirectly on top of a given block.
    Args:
        block: The block to start checking from (the base).
        on_map: A dictionary mapping block_below -> block_on_top in the current state.
    Returns:
        A set of blocks that are on top of the given block.
    """
    blocks = set()
    current = block
    while current in on_map:
        block_above = on_map[current]
        blocks.add(block_above)
        current = block_above # Move up the stack
    return blocks


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by identifying blocks
    that are either in the wrong final position or need to be clear but are not,
    and summing the number of these "problem base" blocks plus the number of
    blocks stacked on top of them.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table)
      for some blocks using 'on' or 'on-table' facts.
    - The goal may specify that certain blocks must be clear using 'clear' facts.
    - Blocks not explicitly mentioned in goal 'on', 'on-table', or 'clear' facts
      do not directly contribute to the heuristic value, but can be "in the way".
    - The cost of fixing a problem involves moving the problem base block and
      moving any blocks currently stacked on top of it. Each such block (the base
      itself and each block on top) is estimated to require at least one action.

    # Heuristic Initialization
    - Extracts the desired final position (on another block or on the table) for
      each block explicitly mentioned in the goal's 'on' or 'on-table' conditions.
      This information is stored in `self.goal_positions`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Build a map representing the current stack structure from the state facts.
       This map, `current_on_relations_forward`, stores which block is directly
       on top of which block (mapping block_below -> block_on_top).
    3. Identify the set of "problem base" blocks. A block `B` is added to this set if:
       a. `B` is mentioned in a goal 'on' or 'on-table' fact, AND its current
          position (on another block or on the table) does not match its goal position.
       OR
       b. `(clear B)` is a goal fact, AND `(clear B)` is not true in the current state.
    4. Add the number of blocks in the "problem base" set to the total cost.
       (Each problem base block needs to be moved or cleared).
    5. Identify the set of all blocks that are currently stacked directly or
       indirectly on top of *any* block in the "problem base" set. Use the
       `current_on_relations_forward` map to traverse stacks upwards from each
       problem base block.
    6. Add the number of blocks in this set (blocks on top of problem bases)
       to the total cost. (Each block on top needs to be moved out of the way).
    7. The total accumulated cost is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for blocks.
        """
        super().__init__(task)
        self.goal_positions = {}
        # Extract goal positions from goal facts
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                block, under_block = parts[1], parts[2]
                self.goal_positions[block] = under_block
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_positions[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for position tracking in self.goal_positions

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Build current 'on' relationships (block_below -> block_on_top)
        current_on_relations_forward = {}
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_on_top, block_below = parts[1], parts[2]
                current_on_relations_forward[block_below] = block_on_top
            elif parts[0] == 'clear':
                current_clear.add(parts[1])

        problem_base_blocks = set() # Blocks that are the root of a problem (wrong position or not clear)

        # Identify blocks whose goal position is not met
        for block, goal_pos in self.goal_positions.items():
            goal_fact_met = False
            if goal_pos == 'table':
                if f'(on-table {block})' in state:
                    goal_fact_met = True
            else: # goal_pos is another block
                if f'(on {block} {goal_pos})' in state:
                    goal_fact_met = True

            if not goal_fact_met:
                problem_base_blocks.add(block)

        # Identify blocks that need to be clear but aren't
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts[0] == 'clear':
                 block_needs_clear = parts[1]
                 if block_needs_clear not in current_clear:
                     problem_base_blocks.add(block_needs_clear) # Add block itself if not clear and needs to be

        # Cost 1: Number of blocks that are the base of a problem
        total_cost = len(problem_base_blocks)

        # Cost 2: Number of blocks on top of problem bases
        blocks_on_top_of_problem_bases = set()

        # Collect all blocks on top of each problem base block
        for block in problem_base_blocks:
             blocks_on_top_of_problem_bases.update(get_blocks_on_top_recursive(block, current_on_relations_forward))

        total_cost += len(blocks_on_top_of_problem_bases)

        return total_cost
