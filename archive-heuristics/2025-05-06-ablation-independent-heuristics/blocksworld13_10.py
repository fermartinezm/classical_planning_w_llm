from heuristics.heuristic_base import Heuristic
from task import Task


class blocksworldHeuristic(Heuristic):
    """
    Summary:
        A domain-dependent heuristic for the Blocksworld domain.
        It estimates the number of actions required to reach the goal state
        by summing three components:
        1. The number of goal facts specifying block positions ((on ?x ?y) or (on-table ?x))
           that are not satisfied in the current state.
        2. The number of 'on' facts in the current state that are not part of
           the desired goal configuration ((on ?x ?y) where (on ?x ?y) is not a goal fact).
        3. A penalty of 1 if the arm is holding any block, as the arm must be
           empty in the goal state.

    Assumptions:
        - Assumes the standard Blocksworld domain predicates: clear, on-table,
          arm-empty, holding, on.
        - Assumes goal states are defined by a set of facts, typically including
          (on ?x ?y), (on-table ?x), and (clear ?x) for the top blocks.
        - Assumes goal states implicitly require the arm to be empty.
        - Assumes goal facts are well-formed strings following the PDDL format
          like '(predicate arg1 arg2)'.

    Heuristic Initialization:
        The constructor processes the goal facts provided in the Task object.
        It identifies and stores:
        - `self.goal_position_facts`: A set containing all goal facts that are
          either '(on ?x ?y)' or '(on-table ?x)'. These represent the desired
          final positions of blocks relative to what's below them.
        - `self.goal_on_facts`: A set containing only the goal facts that are
          '(on ?x ?y)'. This is used to quickly check if a state '(on ?x ?y)'
          fact is a desired goal fact.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (represented as a frozenset of fact strings):
        1. Initialize the heuristic value components: `h1 = 0`, `h2 = 0`, `h_arm = 0`.
        2. Calculate `h1`: Iterate through each fact string in `self.goal_position_facts`.
           If a goal position fact string is not found in the current state's
           frozenset, increment `h1`. This counts how many required position
           relationships are currently unsatisfied.
        3. Calculate `h2`: Iterate through each fact string in the current state's
           frozenset. If a state fact string starts with `'(on '` (indicating it's
           an '(on ?x ?y)' fact), check if this state fact string is present in
           `self.goal_on_facts`. If the state '(on ?x ?y)' fact is *not* found
           in `self.goal_on_facts`, increment `h2`. This counts how many blocks
           are stacked on other blocks in a way that is not part of the goal
           stack configuration.
        4. Calculate `h_arm`: Iterate through each fact string in the current state's
           frozenset. If a state fact string starts with `'(holding '`, set `h_arm`
           to 1 and break the loop (assuming only one block can be held at a time).
           This adds a penalty if the arm is not empty.
        5. The total heuristic value for the state is `h1 + h2 + h_arm`.
        6. The heuristic returns 0 if and only if the state is a goal state:
           - If the state is a goal state, all goal facts (including position facts)
             are true, so `h1` is 0.
           - If the state is a goal state, any `(on ?x ?y)` fact present must be
             a goal `(on ?x ?y)` fact (otherwise a `(clear ?y)` goal would be violated
             or the state would contradict the goal structure), so `h2` is 0.
           - If the state is a goal state, the arm must be empty, so `h_arm` is 0.
           - Thus, for a goal state, `h1 + h2 + h_arm = 0`.
           - If the state is not a goal state, at least one goal fact is missing,
             or a fact exists that prevents a goal fact from being true (like
             `(on X B)` when `(clear B)` is a goal), or the arm is not empty.
             In these cases, at least one of `h1`, `h2`, or `h_arm` will be greater
             than 0, resulting in a non-zero heuristic value.
        7. The heuristic value is finite for any solvable state as the number of
           facts is finite.
    """

    def __init__(self, task: Task):
        """
        Initializes the Blocksworld heuristic by parsing goal facts.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        super().__init__()
        # Heuristic Initialization
        self.goal_position_facts = set()
        self.goal_on_facts = set()

        # Parse goal facts to identify position goals and on goals
        for goal_fact_str in task.goals:
            # Extract predicate and arguments, ignoring surrounding brackets
            parts = goal_fact_str.strip('()').split()
            if not parts: # Handle empty fact string if any
                continue
            predicate = parts[0]

            if predicate == 'on' or predicate == 'on-table':
                self.goal_position_facts.add(goal_fact_str)
                if predicate == 'on':
                    self.goal_on_facts.add(goal_fact_str)

    def __call__(self, node):
        """
        Computes the heuristic value for a given state node.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated heuristic value (integer).
        """
        state = node.state
        # Step-By-Step Thinking for Computing Heuristic

        # 1. Count unsatisfied goal position facts
        h1 = 0
        for goal_fact in self.goal_position_facts:
            if goal_fact not in state:
                h1 += 1

        # 2. Count state 'on' facts that are not goal 'on' facts
        h2 = 0
        for state_fact in state:
            # Check if the fact is an 'on' predicate efficiently
            if state_fact.startswith('(on '):
                if state_fact not in self.goal_on_facts:
                    h2 += 1

        # 3. Add penalty if arm is not empty
        h_arm = 0
        # Check for any fact like '(holding ?x)'
        for state_fact in state:
            if state_fact.startswith('(holding '):
                 h_arm = 1
                 break # Found a holding fact, arm is not empty

        # Total heuristic value
        h_value = h1 + h2 + h_arm

        return h_value

