# Assuming Heuristic base class is available and imported as:
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Define the blocksworldHeuristic class inheriting from the assumed Heuristic base class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct
    goal position, specifically counting the number of desired 'on' and 'on-table'
    predicates from the goal that are not currently true in the state. It focuses
    on the structural goal configuration rather than transient conditions like 'clear'
    or 'arm-empty'.

    # Assumptions
    - The goal specifies the desired final arrangement of blocks using 'on' and 'on-table' predicates.
    - Achieving a correct 'on' or 'on-table' relationship is a fundamental step towards the goal.
    - Each unmet 'on' or 'on-table' goal predicate represents at least one required action (or sequence of actions)
      to correct the block's position.

    # Heuristic Initialization
    - Extracts all 'on' and 'on-table' predicates from the goal state definition provided in the task.
    - Stores these desired predicates in sets (`self.goal_on_facts`, `self.goal_ontable_facts`)
      for efficient lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state from the input `node`. The state is represented as a frozenset of PDDL fact strings.
    2. Initialize a counter `unmet_conditions` to 0. This will store the heuristic value.
    3. Iterate through each desired 'on' predicate stored during initialization (`self.goal_on_facts`).
    4. For the current goal 'on' predicate (e.g., `'(on b1 b2)'`), check if this exact string is present in the current state `state`.
    5. If the goal 'on' predicate string is NOT found in the current state, increment `unmet_conditions` by 1.
    6. Iterate through each desired 'on-table' predicate stored during initialization (`self.goal_ontable_facts`).
    7. For the current goal 'on-table' predicate (e.g., `'(on-table b5)'`), check if this exact string is present in the current state `state`.
    8. If the goal 'on-table' predicate string is NOT found in the current state, increment `unmet_conditions` by 1.
    9. The final value of `unmet_conditions` is the heuristic estimate for the current state. Return this value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal 'on' and 'on-table' facts.
        """
        # Call the base class constructor to store task goals and static facts
        super().__init__(task)

        # Sets to store the specific 'on' and 'on-table' facts required by the goal
        self.goal_on_facts = set()
        self.goal_ontable_facts = set()

        # Iterate through all goal predicates defined in the task
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Ensure the fact is valid and has enough parts
            if not parts:
                continue

            predicate = parts[0]

            # Check if it's an 'on' predicate with two arguments
            if predicate == 'on' and len(parts) == 3:
                self.goal_on_facts.add(goal_fact)
            # Check if it's an 'on-table' predicate with one argument
            elif predicate == 'on-table' and len(parts) == 2:
                self.goal_ontable_facts.add(goal_fact)
            # Other goal predicates like 'clear' or 'arm-empty' are ignored by this heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        The heuristic is the count of goal 'on' and 'on-table' conditions
        that are not met in the current state.
        """
        # The current state is accessed via the node object
        state = node.state # state is a frozenset of fact strings

        # Initialize the count of unmet goal conditions
        unmet_conditions = 0

        # Count how many required 'on' facts are missing from the current state
        for goal_on_fact in self.goal_on_facts:
            if goal_on_fact not in state:
                unmet_conditions += 1

        # Count how many required 'on-table' facts are missing from the current state
        for goal_ontable_fact in self.goal_ontable_facts:
            if goal_ontable_fact not in state:
                unmet_conditions += 1

        # The total count of missing 'on' and 'on-table' facts is the heuristic value
        return unmet_conditions
