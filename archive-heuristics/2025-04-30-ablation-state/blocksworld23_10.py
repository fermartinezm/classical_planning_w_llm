# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential empty facts or malformed input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts, heuristic will ignore them
        return []
    return fact[1:-1].split()

# Assuming Heuristic base class is available as specified in the problem description.
# If running this code standalone, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# Assuming the Heuristic base class is imported from a framework like this:
# from heuristics.heuristic_base import Heuristic


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting unsatisfied goal conditions (specifically 'on' and 'on-table')
    and adding a cost for blocks that are currently obstructing the desired
    stack configuration. It is designed for greedy best-first search and is
    not admissible.

    # Assumptions
    - The goal state is defined primarily by `(on ?x ?y)` and `(on-table ?x)`
      predicates. `(clear ?x)` and `(arm-empty)` goals are implicitly handled
      or ignored as they are often consequences of achieving the main stacking goals.
    - Each action (pickup, putdown, stack, unstack) has a cost of 1 in the
      underlying planning problem. The heuristic estimates sequences of these
      actions.

    # Heuristic Initialization
    - The heuristic extracts the desired 'on' relationships from the goal state
      to build a mapping `goal_above`. This mapping indicates, for each block
      that acts as a base in the goal, which block should be directly on top of it.
      This helps identify blocks that are blocking goal positions in the current state.
    - The set of all goal facts is stored for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the heuristic value `h` to 0.
    2. Create a set of the facts true in the current state for efficient lookup.
    3. Parse the current state to identify the current 'on' relationships. Build a
       mapping `current_above` where `current_above[base] = top` if `(on top base)`
       is true in the state. This identifies which blocks are currently supporting others.
    4. Iterate through each goal fact provided in the task (`self.goals`):
       - If the goal fact is `(on ?b ?u)` and it is *not* present in the current state,
         add 2 to `h`. This estimates the cost of achieving this specific stack,
         representing actions like getting block `?b` into the arm and then stacking
         it onto `?u`.
       - If the goal fact is `(on-table ?b)` and it is *not* present in the current state,
         add 1 to `h`. This estimates the cost of achieving this specific table position,
         representing an action like putting block `?b` down onto the table. A cost of 1
         is used here, assuming the cost of getting the block into the arm is potentially
         accounted for by other parts of the heuristic (like clearing costs or 'on' goals).
       - Ignore unsatisfied `(clear ?x)` or `(arm-empty)` goals.
    5. Iterate through the blocks that are currently acting as bases for other blocks
       in the current state (i.e., the keys in the `current_above` mapping):
       - For each base block `?u`, identify the block currently on top of it, `?b`
         (`?b = current_above[?u]`).
       - Check if block `?b` is the block that is supposed to be on top of `?u`
         in the goal state (using the `self.goal_above` mapping).
       - If `?b` is *not* the block that should be on top of `?u` in the goal state
         (either nothing should be on `?u`, or a different block should be),
         add 1 to `h`. This represents the estimated cost of moving block `?b`
         off of `?u` because it is obstructing the desired goal configuration.
    6. Return the total calculated value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Build a mapping from base block to the block that should be on top in the goal.
        # This helps identify blocks that are blocking goal positions.
        self.goal_above = {}
        # Store the goal facts as a set for quick lookup during __call__
        self.goal_facts_set = set()

        for goal in self.goals:
            self.goal_facts_set.add(goal)
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_above[base] = block
            # Ignore clear and arm-empty goals for the structure mapping

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        # Build a mapping from base block to the block currently on top.
        current_above = {}
        # Create a set of current facts for quick lookup
        current_facts_set = set(state)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_above[base] = block
            # on-table and holding facts are not needed for current_above,
            # but are part of current_facts_set for goal checking.

        h = 0

        # 1. Count unsatisfied goal facts (on and on-table)
        # We give different weights based on the type of goal predicate.
        for goal in self.goals:
            if goal not in current_facts_set:
                parts = get_parts(goal)
                if not parts: continue # Skip malformed goals

                predicate = parts[0]
                if predicate == "on" and len(parts) == 3:
                    # Unsatisfied (on B U) goal. Estimated cost: 2 actions (e.g., pickup/unstack B, stack B U).
                    h += 2
                elif predicate == "on-table" and len(parts) == 2:
                    # Unsatisfied (on-table B) goal. Estimated cost: 1 action (e.g., putdown B).
                    h += 1
                # Ignore unsatisfied clear or arm-empty goals for heuristic calculation

        # 2. Count blocks that are currently blocking other blocks or are on the wrong base
        # Iterate through blocks that are currently bases for something.
        # These are the keys in the current_above mapping.
        for base, current_top in current_above.items():
             # Find what should be on top of this base in the goal
             goal_top = self.goal_above.get(base)

             # If there is something on top of 'base' currently (`current_top` is not None),
             # and it's NOT the block that is supposed to be there in the goal (or nothing is supposed to be there)
             # Note: current_above only contains entries where current_top is not None.
             if goal_top is None or goal_top != current_top:
                 # This 'current_top' block needs to be moved off 'base'.
                 # Add 1 for the task of unstacking/moving 'current_top'.
                 h += 1

        return h
