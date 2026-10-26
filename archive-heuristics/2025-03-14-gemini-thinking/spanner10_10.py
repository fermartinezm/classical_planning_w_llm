from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at bob shed)' it returns ['bob', 'shed'].
    """
    fact_content = fact_str[1:-1]  # Remove parentheses
    return fact_content.split()[1:] # Split by space and ignore predicate name

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal.
    It considers the necessary steps for each nut: having a usable spanner, being at the nut's location, and performing the tighten action.

    # Assumptions:
    - The goal is always to tighten a set of nuts.
    - There is only one man (bob).
    - We only consider the cost of tightening each nut independently, without considering interactions between tightening different nuts.

    # Heuristic Initialization
    - Extracts the goal nuts from the task definition.
    - Extracts static link information to potentially use for more advanced heuristics (currently not used in this simple version).

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact that specifies a nut should be tightened:
    1. Check if the nut is already tightened in the current state. If yes, no cost is added for this nut.
    2. If the nut is not tightened, estimate the cost to tighten it:
        a. Check if the man 'bob' is carrying a usable spanner. If not, assume 1 action is needed to acquire a usable spanner (pickup).
        b. Find the location of the nut and the man. If they are not at the same location, assume 1 action is needed to move to the nut's location (walk).
        c. Add 1 action for the 'tighten_nut' action itself.
    3. The total heuristic value is the sum of the costs estimated for each nut that is not yet tightened in the current state but should be tightened in the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_nuts = set()
        for goal_fact in self.goals:
            if goal_fact.startswith('(tightened'):
                nut_name = get_objects_from_fact(goal_fact)[0]
                self.goal_nuts.add(nut_name)
        self.links = set()
        for static_fact in self.static_facts:
            if static_fact.startswith('(link'):
                self.links.add(static_fact)


    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        for nut_name in self.goal_nuts:
            goal_tightened_fact = f'(tightened {nut_name})'
            if goal_tightened_fact not in state:
                # Nut is not tightened in the current state, estimate cost to tighten it
                cost_for_nut = 0

                # Check if carrying a usable spanner
                carrying_usable_spanner = False
                for fact in state:
                    if fact.startswith('(carrying bob'):
                        carried_spanner = get_objects_from_fact(fact)[1]
                        if f'(usable {carried_spanner})' in state:
                            carrying_usable_spanner = True
                            break
                if not carrying_usable_spanner:
                    cost_for_nut += 1 # Assume 1 action to get a usable spanner

                # Check if man and nut are at the same location
                man_location = None
                nut_location = None
                for fact in state:
                    if fact.startswith('(at bob'):
                        man_location = get_objects_from_fact(fact)[1]
                    elif fact.startswith(f'(at {nut_name}'):
                        nut_location = get_objects_from_fact(fact)[1]

                if man_location != nut_location:
                    cost_for_nut += 1 # Assume 1 action to walk to the nut location

                cost_for_nut += 1 # Action to tighten the nut
                heuristic_value += cost_for_nut

        return heuristic_value
