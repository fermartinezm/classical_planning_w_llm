from fnmatch import fnmatch
# Assuming Heuristic base class is available in a 'heuristics' directory
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure fact is a string
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for invalid format
        return []
    return fact[1:-1].split()


# The match function from examples is not strictly needed for this heuristic
# but can be kept as a utility if needed later or for consistency.
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     """
#     parts = get_parts(fact)
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by summing two main components:
    1. The number of goal facts that are not currently true in the state.
    2. The number of 'on' relationships in the current state that are not part of the goal configuration.

    This heuristic is non-admissible but aims to guide the search towards states that satisfy more goal conditions and have fewer incorrect block placements.

    # Assumptions
    - The goal is a conjunction of facts, typically including 'on', 'on-table', 'clear', and possibly 'arm-empty'.
    - The standard Blocksworld actions (pickup, putdown, stack, unstack) are used.

    # Heuristic Initialization
    - Stores the set of goal facts for efficient lookup during heuristic computation.
    - Static facts are not relevant for this heuristic in the Blocksworld domain.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the heuristic value `h` to 0.
    2. Component 1 (Unsatisfied Goals): Iterate through each fact in the task's goal list. If a goal fact is not present in the current state, increment `h`. This counts how many goal conditions still need to be achieved.
    3. Component 2 (Wrong Stacks): Iterate through each fact in the current state. If a state fact is an '(on ?x ?y)' predicate and this exact fact is NOT present in the task's goal list, it means block ?x is incorrectly placed on block ?y according to the goal. Increment `h` for each such 'on' fact. This penalizes incorrect stacking structures that must be dismantled.
    4. Return the total heuristic value `h`.

    This heuristic is 0 if and only if the state is the goal state:
    - If the state is the goal state, all goal facts are in the state (Component 1 is 0). Also, any '(on ?x ?y)' fact in the goal state must be a goal fact (as the goal defines the final stack configuration), so no state '(on ?x ?y)' fact will be absent from the goals (Component 2 is 0). Thus, h=0.
    - If h=0, then all goal facts are in the state (Component 1 is 0). Also, all state '(on ?x ?y)' facts are present in the goals (Component 2 is 0). This combination implies the state contains all goal facts and no 'extra' or 'wrong' on-relations. In Blocksworld, this is sufficient to be the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal facts set.
        """
        self.goals = task.goals
        # Store goal facts as a set for quick lookup in __call__
        self._goal_facts_set = set(self.goals)
        # Static facts are not used in this heuristic for Blocksworld.
        # self.static = task.static # Not needed

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings)

        total_cost = 0  # Initialize heuristic cost

        # Component 1: Unsatisfied Goals
        # Count how many goal facts are not present in the current state.
        for goal_fact in self.goals:
            if goal_fact not in state:
                total_cost += 1

        # Component 2: Wrong Stacks
        # Count 'on' facts in the state that are not part of the goal configuration.
        for state_fact in state:
            parts = get_parts(state_fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            # Check if the fact is an 'on' predicate and if it's not a goal fact
            if predicate == "on" and state_fact not in self._goal_facts_set:
                 total_cost += 1

        return total_cost
