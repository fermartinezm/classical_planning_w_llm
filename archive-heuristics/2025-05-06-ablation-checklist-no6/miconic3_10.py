from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have enough parts to match the args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost as the sum of:
    1. Twice the number of passengers not yet served (estimating 1 board + 1 depart per passenger).
    2. The minimum vertical travel distance required to span all floors where pickups or dropoffs are needed, starting from the current lift floor.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals # Goal conditions (served passengers)
        static_facts = task.static # Facts that are not affected by actions.

        # 1. Parse floor levels from 'above' facts
        # The predicate (above f_lower f_higher) means f_higher is immediately above f_lower.
        self.floor_to_level = {}
        above_facts = [fact for fact in static_facts if match(fact, "above", "*", "*")]

        all_floors = set()
        floors_that_are_above = set() # Floors that appear as the second argument in (above f_lower f_higher)
        above_relations = [] # Store (f_lower, f_higher) pairs

        for fact in above_facts:
            parts = get_parts(fact)
            f_lower, f_higher = parts[1], parts[2]
            all_floors.add(f_lower)
            all_floors.add(f_higher)
            floors_that_are_above.add(f_higher)
            above_relations.append((f_lower, f_higher))

        # Find the lowest floor (a floor that is not above any other floor)
        # Assumes there is at least one floor and exactly one lowest floor
        lowest_floor = (all_floors - floors_that_are_above).pop()

        # Build the level mapping using BFS
        level = 1
        current_level_floors = {lowest_floor}
        self.floor_to_level[lowest_floor] = level

        while len(self.floor_to_level) < len(all_floors):
            level += 1
            next_level_floors = set()
            for f_lower in current_level_floors:
                for rel_lower, rel_higher in above_relations:
                    if rel_lower == f_lower and rel_higher not in self.floor_to_level:
                        self.floor_to_level[rel_higher] = level
                        next_level_floors.add(rel_higher)
            current_level_floors = next_level_floors

        # 2. Store passenger destinations
        self.passenger_destin = {}
        # Get all passengers mentioned in the goal (served)
        self.goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Find destin facts for these passengers in static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                 p, f_destin = parts[1], parts[2]
                 if p in self.goal_passengers:
                    self.passenger_destin[p] = f_destin


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Find current lift location
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break
        # If lift location is not found, something is wrong with the state, return infinity or a large value
        if current_floor is None:
             return float('inf') # Should not happen in valid states

        current_level = self.floor_to_level[current_floor]

        # Identify passengers not yet served
        passengers_not_served = {
            p for p in self.goal_passengers if '(served ' + p + ')' not in state
        }

        # Identify floors where actions are needed (pickups or dropoffs)
        required_floors = set()

        # Floors for pickups (passengers waiting at origin)
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f_origin = get_parts(fact)[1], get_parts(fact)[2]
                if p in passengers_not_served:
                    required_floors.add(f_origin)

        # Floors for dropoffs (passengers boarded)
        for fact in state:
            if match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                if p in passengers_not_served:
                    # Get destination floor from pre-calculated destinations
                    f_destin = self.passenger_destin.get(p)
                    if f_destin: # Should always exist for goal passengers
                         required_floors.add(f_destin)

        # If there are no passengers not served, required_floors should be empty,
        # and the heuristic would have returned 0 already.
        # If passengers_not_served is not empty, required_floors must also be non-empty
        # (a not-served passenger is either waiting at an origin or boarded going to a destin).

        # Calculate estimated vertical travel
        required_levels = {self.floor_to_level[f] for f in required_floors}
        all_relevant_levels = required_levels | {current_level}

        min_level = min(all_relevant_levels)
        max_level = max(all_relevant_levels)

        # Estimated travel is the distance needed to span all relevant floors
        estimated_vertical_travel = max_level - min_level

        # Calculate heuristic value
        # Estimate: 2 actions per not-served passenger (board + depart) + estimated vertical travel
        # This is a non-admissible estimate designed to prioritize states where
        # more passengers are closer to being served and the lift is well-positioned.
        h = len(passengers_not_served) * 2 + estimated_vertical_travel

        return h
