from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at bob shed)' it returns ['bob', 'shed'].
    """
    fact_content = fact_str[1:-1] # Remove parentheses
    return fact_content.split()[1:] # Split by space and ignore predicate name

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the actions needed to get a usable spanner, move to the nut's location,
    and then tighten the nut for each nut that is currently loose.

    # Assumptions:
    - For each loose nut, we need to perform at least one 'tighten_nut' action.
    - If the man is not carrying a usable spanner, we assume we need one 'pickup_spanner' action.
    - If the man is not at the nut's location, we assume we need one 'walk' action.
    - We do not explicitly calculate shortest paths, but rather assume a single 'walk' action is sufficient if locations differ.
    - We prioritize tightening each loose nut independently and sum up the estimated costs.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base Heuristic class.
    - Static facts are not explicitly used in this simplified heuristic calculation,
      although a more sophisticated version could use 'link' facts to estimate walk costs.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Identify all nuts that are currently 'loose' in the state.
    3. For each loose nut:
       a. Check if the nut is already 'tightened' in the goal state. If yes, no cost for this nut.
       b. If the nut is still 'loose' in the current state (and needs to be tightened):
          i.  Increment the heuristic value by 1 (for the 'tighten_nut' action itself).
          ii. Check if the man is currently carrying a usable spanner.
              - If not, increment the heuristic value by 1 (for a 'pickup_spanner' action).
          iii.Determine the location of the nut and the man.
              - If the man's location is not the same as the nut's location, increment the heuristic value by 1 (for a 'walk' action).
    4. Return the total accumulated heuristic value.

    This heuristic is a simplification and does not guarantee admissibility, but it aims
    to provide a reasonable estimate of the remaining actions in the spanner domain
    for guiding a greedy best-first search.
    """

    def __init__(self, task):
        """Initialize the spanner heuristic."""
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0
        loose_nuts = set()
        tightened_nuts_goal = set()

        # Identify loose nuts in the current state
        for fact in state:
            if fact.startswith('(loose '):
                nut_name = get_objects_from_fact(fact)[0]
                loose_nuts.add(nut_name)

        # Identify tightened nuts in the goal state
        for goal_fact in self.goals:
            if goal_fact.startswith('(tightened '):
                nut_name = get_objects_from_fact(goal_fact)[0]
                tightened_nuts_goal.add(nut_name)

        for nut_name in loose_nuts:
            if f'(tightened {nut_name})' in state: # Check if already tightened in current state (should not happen if state is consistent with actions)
                continue # Nut already tightened, no cost

            nut_location = None
            man_location = None
            carrying_usable_spanner = False
            man_name = None
            usable_spanner_name = None

            # Get nut location
            for fact in state:
                if fact.startswith(f'(at {nut_name} '):
                    nut_location = get_objects_from_fact(fact)[1]
                    break

            # Get man location and check carrying usable spanner
            for fact in state:
                if fact.startswith('(at '):
                    objects = get_objects_from_fact(fact)
                    if len(objects) == 2 and 'man' in self.task.type_dict.get(objects[0],[]): # Identify man
                        man_name = objects[0]
                        man_location = objects[1]
                        break

            for fact in state:
                if fact.startswith('(carrying '):
                    objects = get_objects_from_fact(fact)
                    if objects[0] == man_name:
                        spanner_name = objects[1]
                        if f'(usable {spanner_name})' in state:
                            carrying_usable_spanner = True
                            usable_spanner_name = spanner_name
                            break

            nut_cost = 0
            nut_cost += 1 # For tighten_nut action

            if not carrying_usable_spanner:
                nut_cost += 1 # For pickup_spanner action

            if man_location != nut_location:
                nut_cost += 1 # For walk action

            heuristic_value += nut_cost

        return heuristic_value
