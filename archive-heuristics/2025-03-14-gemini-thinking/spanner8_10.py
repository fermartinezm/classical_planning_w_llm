from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at bob shed)' it returns ['bob', 'shed'].
    Ignores the predicate name.
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[1:]

def get_predicate_name(fact_str):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, from '(at bob shed)' it returns 'at'.
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[0]

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the actions needed for each nut individually and sums them up.
    The heuristic accounts for the need to walk to the nut's location and to carry a usable spanner.

    # Assumptions:
    - For each loose nut, we need to perform a 'tighten_nut' action.
    - To perform 'tighten_nut', the man must be at the nut's location and carrying a usable spanner.
    - We assume that for each loose nut, we minimally need to walk to the nut's location (if not already there) and pickup a spanner (if not already carrying one).

    # Heuristic Initialization
    - The heuristic initializes by identifying the goal nuts (nuts that need to be tightened).
    - It also extracts static link information to potentially use for pathfinding (although not used in this simplified heuristic).

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is required to be tightened in the goal and is currently loose:
    1. Initialize the estimated cost for this nut to 0.
    2. Check if the man is at the location of the nut. If not, increment the cost by 1 (estimate for 'walk' action).
    3. Check if the man is carrying a usable spanner. If not, increment the cost by 1 (estimate for 'pickup_spanner' action).
    4. Finally, increment the cost by 1 for the 'tighten_nut' action itself.
    5. Sum up the costs for all nuts that need to be tightened and are currently loose.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_nuts = set()
        for goal in self.goals:
            if get_predicate_name(goal) == 'tightened':
                self.goal_nuts.add(get_objects_from_fact(goal)[0])
        self.links = set()
        for fact in self.static_facts:
            if get_predicate_name(fact) == 'link':
                self.links.add(tuple(get_objects_from_fact(fact)))
                self.links.add(tuple(reversed(get_objects_from_fact(fact)))) # links are bidirectional in this simple heuristic assumption, although not necessarily in general

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        current_loose_nuts = set()
        nut_locations = {}
        man_location = None
        carried_spanner = None
        usable_spanners = set()

        for fact in state:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'loose':
                current_loose_nuts.add(objects[0])
                nut_locations[objects[0]] = None # location will be determined later
            elif predicate == 'at':
                if len(objects) == 2: # at(locatable, location)
                    locatable_obj = objects[0]
                    location_obj = objects[1]
                    if locatable_obj == 'bob': # assuming 'bob' is the man
                        man_location = location_obj
                    elif locatable_obj in self.goal_nuts: # if it's a nut we are interested in
                        nut_locations[locatable_obj] = location_obj
                    # spanner locations are not directly used in this heuristic but could be in a more sophisticated one
            elif predicate == 'carrying':
                carried_spanner = objects[1] # assuming objects[0] is 'bob'
            elif predicate == 'usable':
                usable_spanners.add(objects[0])


        for nut in self.goal_nuts:
            if nut in current_loose_nuts:
                nut_cost = 1 # for tighten_nut action
                nut_location = nut_locations.get(nut)

                if man_location != nut_location:
                    nut_cost += 1 # for walk action

                is_carrying_usable_spanner = False
                if carried_spanner is not None and carried_spanner in usable_spanners:
                    is_carrying_usable_spanner = True

                if not is_carrying_usable_spanner:
                    usable_spanner_available = False
                    for fact in state:
                        if get_predicate_name(fact) == 'at' and len(get_objects_from_fact(fact)) == 2:
                            locatable_obj = get_objects_from_fact(fact)[0]
                            location_obj = get_objects_from_fact(fact)[1]
                            if locatable_obj in usable_spanners and location_obj == nut_location:
                                usable_spanner_available = True
                                break # assuming at least one usable spanner is enough, could be refined to check if man is at the spanner location
                    if not is_carrying_usable_spanner: # double check to avoid double counting if already carrying
                        nut_cost += 1 # for pickup_spanner action

                heuristic_value += nut_cost

        return heuristic_value
