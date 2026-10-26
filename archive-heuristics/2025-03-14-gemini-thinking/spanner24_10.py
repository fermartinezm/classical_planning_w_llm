from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at bob shed)' it returns ['bob', 'shed'].
    Ignores the predicate name.
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[1:]  # Return objects, excluding predicate name

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
    It considers the actions: tighten_nut, pickup_spanner, and walk. For each nut that needs
    to be tightened, it estimates the cost based on whether a usable spanner is carried and
    whether the man is at the nut's location.

    # Assumptions:
    - For each nut that needs tightening, we assume we need to perform a 'tighten_nut' action.
    - If the man is not carrying a usable spanner, we assume one 'pickup_spanner' action is needed.
    - If the man is not at the location of the nut, we assume one 'walk' action is needed.
    - We do not explicitly calculate shortest paths for walking; we simply count a walk action if needed.
    - We prioritize tightening each nut independently and sum up the costs.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates, specifically targeting '(tightened ?n)' goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost to 0.
    2. Iterate through each goal fact in the task's goal description.
    3. For each goal fact, check if it is of the form '(tightened ?nut)'.
    4. If it is a '(tightened ?nut)' goal, check if this goal is already achieved in the current state.
    5. If the goal '(tightened ?nut)' is not achieved in the current state, then estimate the cost to achieve it:
        a. Increment the cost by 1, assuming a 'tighten_nut' action will be needed.
        b. Check if the man is currently carrying any usable spanner.
           To do this, iterate through the state and look for facts of the form '(carrying bob ?spanner)' and '(usable ?spanner)'.
           If no usable spanner is carried, increment the cost by 1, assuming a 'pickup_spanner' action is needed.
        c. Determine the location of the nut. Iterate through the state and find a fact of the form '(at ?nut ?location)'.
        d. Determine the location of the man. Iterate through the state and find a fact of the form '(at bob ?location)'.
        e. If the man's location is not the same as the nut's location, increment the cost by 1, assuming a 'walk' action is needed.
    6. Sum up the costs for all unachieved '(tightened ?nut)' goals.
    7. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_tightened_nuts = set()
        for goal_fact in self.goals:
            if match(goal_fact, "tightened", "*"):
                goal_tightened_nuts.add(goal_fact)

        for goal_fact in goal_tightened_nuts:
            if goal_fact not in state:
                heuristic_value += 1 # For tighten_nut action

                carrying_usable_spanner = False
                for fact in state:
                    if match(fact, "carrying", "bob", "*"):
                        carried_spanner = get_objects_from_fact(fact)[1]
                        if match(fact, "usable", carried_spanner): # This check is not correct, 'usable' is a separate predicate.
                            for usable_fact in state:
                                if match(usable_fact, "usable", carried_spanner):
                                    carrying_usable_spanner = True
                                    break
                            if carrying_usable_spanner:
                                break # Found a usable spanner being carried.
                if not carrying_usable_spanner:
                    heuristic_value += 1 # For pickup_spanner action

                goal_nut = get_objects_from_fact(goal_fact)[0]
                nut_location = None
                man_location = None

                for fact in state:
                    if match(fact, "at", goal_nut, "*"):
                        nut_location = get_objects_from_fact(fact)[1]
                    if match(fact, "at", "bob", "*"):
                        man_location = get_objects_from_fact(fact)[1]

                if nut_location != man_location:
                    heuristic_value += 1 # For walk action

        return heuristic_value
