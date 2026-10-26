from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departing actions for each passenger.

    # Assumptions:
    - Each passenger must be boarded before being served.
    - The lift can move one floor at a time, with each move counting as an action.
    - Boarding and departing each passenger each count as one action.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger, check if they are already served. If so, no actions are needed.
    2. For passengers not served:
       a. If boarded, calculate the distance from the lift's current floor to their destination and add the necessary actions to move and serve them.
       b. If not boarded, calculate the distance from the lift's current floor to their origin, add actions to board them, then calculate the distance from the origin to their destination and add actions to serve them.
    3. Sum the actions required for all passengers to get the total estimated actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting passenger origins and destinations from static facts."""
        self.passenger_info = {}
        static_facts = task.static

        # Extract origin and destination for each passenger
        for fact in static_facts:
            if fact.startswith('(origin '):
                parts = fact[1:-1].split()
                p = parts[1]
                origin = int(parts[2][1:])  # Remove 'f' prefix
                if p not in self.passenger_info:
                    self.passenger_info[p] = {'origin': origin}
                else:
                    self.passenger_info[p]['origin'] = origin
            elif fact.startswith('(destin '):
                parts = fact[1:-1].split()
                p = parts[1]
                dest = int(parts[2][1:])
                if p not in self.passenger_info:
                    self.passenger_info[p] = {'dest': dest}
                else:
                    self.passenger_info[p]['dest'] = dest

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_lift_floor = None

        # Find the current lift floor
        for fact in state:
            if fact.startswith('(lift-at '):
                current_lift_floor = int(fact.split()[2][1:])  # Remove 'f' prefix
                break

        if current_lift_floor is None:
            return 0  # No lift position found, assume it's already at the goal

        total_actions = 0

        for p in self.passenger_info:
            served = f'(served {p})' in state
            boarded = f'(boarded {p})' in state

            if not served:
                if boarded:
                    dest = self.passenger_info[p]['dest']
                    distance = abs(current_lift_floor - dest)
                    total_actions += distance + 1  # Move to destination and depart
                else:
                    origin = self.passenger_info[p]['origin']
                    dest = self.passenger_info[p]['dest']
                    distance_to_origin = abs(current_lift_floor - origin)
                    distance_to_dest = abs(origin - dest)
                    total_actions += distance_to_origin + 1 + distance_to_dest + 1  # Move, board, move, depart

        return total_actions
