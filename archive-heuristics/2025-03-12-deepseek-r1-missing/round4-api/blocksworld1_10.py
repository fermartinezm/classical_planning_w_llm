from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    Estimates the number of actions required to achieve the goal by checking 'on', 'on-table', and 'clear' predicates.
    - For each unsatisfied 'on' or 'on-table' predicate, add 2 actions.
    - For each 'clear' predicate, add 2 actions per block on the target block.

    # Assumptions
    - The goal specifies required 'on', 'on-table', and 'clear' predicates.
    - Moving a block requires two actions (pickup/stack or unstack/putdown).
    - Blocks not mentioned in the goal can be in any state.

    # Heuristic Initialization
    - Extract 'on', 'on-table', and 'clear' predicates from the goal.

    # Step-By-Step Thinking
    1. Check each 'on' goal: add 2 if not satisfied.
    2. Check each 'on-table' goal: add 2 if not satisfied.
    3. For each 'clear' goal, count blocks on it and add 2 per block.
    """

    def __init__(self, task):
        self.goal_ons = {}  # {block: target_block}
        self.goal_ontables = set()  # Blocks requiring on-table
        self.goal_clears = set()  # Blocks requiring clear

        for goal in task.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, under = parts[1], parts[2]
                self.goal_ons[block] = under
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_ontables.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clears.add(block)

    def __call__(self, node):
        state = node.state
        heuristic_value = 0

        # Check unsatisfied 'on' predicates
        for block, target in self.goal_ons.items():
            required_fact = f'(on {block} {target})'
            if required_fact not in state:
                heuristic_value += 2

        # Check unsatisfied 'on-table' predicates
        for block in self.goal_ontables:
            required_fact = f'(on-table {block})'
            if required_fact not in state:
                heuristic_value += 2

        # Check unsatisfied 'clear' predicates
        for block in self.goal_clears:
            blocks_on = sum(1 for fact in state if match(fact, 'on', '*', block))
            heuristic_value += blocks_on * 2

        return heuristic_value
