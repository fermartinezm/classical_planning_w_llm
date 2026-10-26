# The base class Heuristic is assumed to be available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Assuming the Heuristic base class is available via import.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example fact: '(on b1 b2)' -> parts = ['on', 'b1', 'b2']
    # Example fact: '(arm-empty)' -> parts = ['arm-empty']
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by summing two components:
    1. The number of goal predicates that are not satisfied in the current state.
    2. The number of 'on' or 'on-table' facts in the current state that are not part of the goal state.

    This heuristic is non-admissible but aims to guide a greedy best-first search
    by penalizing states that are far from the goal in terms of unsatisfied conditions
    and incorrect block placements.

    # Heuristic Initialization
    - The constructor receives the planning task object.
    - It extracts and stores the set of goal facts for efficient lookup during heuristic computation.
    - Static facts are not needed for this heuristic in the Blocksworld domain.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Iterate through each goal fact defined in the task (`self.goal_facts`).
       - For each `goal_fact`, check if it is present in the current state (`node.state`).
       - If `goal_fact` is NOT in the current state, it means this goal condition is not yet met. Increment `h` by 1. This component counts how many desired conditions (like a block being on another, on the table, or clear, or the arm being empty) are yet to be achieved.
    3. Iterate through each fact in the current state (`node.state`).
       - For each `fact`, parse its components to identify the predicate and arguments.
       - If the predicate is 'on' and it has two arguments (e.g., `(on block1 block2)`), check if this specific `on` fact is present in the set of goal facts (`self.goal_facts`). If it is NOT a goal fact, it means this block (`block1`) is currently placed on a block (`block2`) incorrectly according to the goal. Increment `h` by 1.
       - If the predicate is 'on-table' and it has one argument (e.g., `(on-table block1)`), check if this specific `on-table` fact is present in the set of goal facts (`self.goal_facts`). If it is NOT a goal fact, it means this block (`block1`) is currently on the table incorrectly according to the goal. Increment `h` by 1.
       - Facts with other predicates (like 'clear', 'arm-empty', 'holding') do not contribute to this second component.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal facts."""
        # Store goal facts as a set for efficient lookup.
        self.goal_facts = set(task.goals)
        # Blocksworld domain does not have static facts relevant to this heuristic.
        # task.static is available but not used here.

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state # Current world state is a frozenset of facts (strings).

        h = 0

        # 1. Count unsatisfied goal predicates.
        # Iterate through the stored goal facts.
        for goal_fact in self.goal_facts:
            if goal_fact not in state:
                h += 1

        # 2. Count incorrectly placed blocks (wrong 'on' or 'on-table' relations).
        # Iterate through all facts in the current state.
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            # Check if the fact is an 'on' predicate with two arguments.
            if predicate == 'on' and len(parts) == 3:
                # If this 'on' relationship is not a goal fact, it's incorrect.
                if fact not in self.goal_facts:
                    h += 1
            # Check if the fact is an 'on-table' predicate with one argument.
            elif predicate == 'on-table' and len(parts) == 2:
                 # If this 'on-table' relationship is not a goal fact, it's incorrect.
                 if fact not in self.goal_facts:
                     h += 1

        return h
