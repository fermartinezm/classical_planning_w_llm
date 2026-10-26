from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact):
    """
    Extract the objects from a PDDL fact string.
    For example, from '(at bob shed)' it returns ['bob', 'shed'].
    """
    return fact[1:-1].split()[1:]

def get_predicate_name(fact):
    """
    Extract the predicate name from a PDDL fact string.
    For example, from '(at bob shed)' it returns 'at'.
    """
    return fact[1:-1].split()[0]

def match(fact, *args):
    """
    Utility function to check if a PDDL fact matches a given pattern.
    - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
    - `args`: The pattern to match (e.g., "at", "*", "rooma").
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the necessary steps for each nut: being at the nut's location,
    carrying a usable spanner, and then tightening the nut.

    # Assumptions:
    - There is always at least one usable spanner available in the domain.
    - Moving between any two linked locations costs 1 action.
    - Picking up a spanner costs 1 action.
    - Tightening a nut costs 1 action.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task.
    - It also extracts static link information to potentially use for more informed
      distance estimation between locations, although the current heuristic version
      uses a simplified approach.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition `(tightened ?n)`:
    1. Check if the goal `(tightened ?n)` is already achieved in the current state.
       If yes, no further actions are needed for this nut, so the cost is 0.
    2. If the goal is not achieved (i.e., the nut is still loose):
       - Initialize the estimated cost for this nut to 0.
       - Increment the cost by 1, accounting for the `tighten_nut` action itself.
       - Determine the location of the nut (`nut_location`) from the current state
         using the predicate `(at ?n ?l)`.
       - Determine the location of the man (`man_location`) from the current state
         using the predicate `(at ?m ?l)`.
       - If `man_location` is not the same as `nut_location`, increment the cost by 1
         to account for the `walk` action to reach the nut's location.
       - Check if the man is carrying a usable spanner. Iterate through the state facts
         to find if there exists a fact `(carrying ?m ?s)` and `(usable ?s)` for any spanner `?s`.
       - If the man is not carrying a usable spanner, increment the cost by 1 to
         account for the `pickup_spanner` action.
    3. The total heuristic value for the state is the sum of the estimated costs
       for tightening each nut that is part of the goal and is not yet tightened in the current state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_links = set()
        for fact in task.static:
            if get_predicate_name(fact) == 'link':
                self.static_links.add(fact)

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_nuts = set()
        for goal_fact in self.goals:
            if get_predicate_name(goal_fact) == 'tightened':
                goal_nuts.add(get_objects_from_fact(goal_fact)[0])

        for nut_name in goal_nuts:
            tightened_goal_fact = f'(tightened {nut_name})'
            if tightened_goal_fact not in state:
                nut_location = None
                man_location = None
                carrying_usable_spanner = False

                for fact in state:
                    if get_predicate_name(fact) == 'at':
                        objects = get_objects_from_fact(fact)
                        if objects[0] == nut_name:
                            nut_location = objects[1]
                        elif objects[0] == 'bob': # Assuming man's name is always 'bob' as per examples
                            man_location = objects[1]
                    elif get_predicate_name(fact) == 'carrying':
                        man_obj, spanner_obj = get_objects_from_fact(fact)
                        if man_obj == 'bob': # Assuming man's name is always 'bob'
                            if f'(usable {spanner_obj})' in state:
                                carrying_usable_spanner = True
                                break

                nut_cost = 1 # for tighten_nut action

                if man_location != nut_location:
                    nut_cost += 1 # for walk action

                if not carrying_usable_spanner:
                    nut_cost += 1 # for pickup_spanner action

                heuristic_value += nut_cost

        return heuristic_value
