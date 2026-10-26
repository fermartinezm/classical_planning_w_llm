from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departure actions.

    # Assumptions:
    - The lift can move one floor per action (up or down).
    - Each boarding and departure action counts as one step.
    - Passengers not yet served are either waiting to be boarded or need to be transported to their destination.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current position of the lift.
    2. For each passenger not served:
       a. If the passenger is boarded, calculate the distance from the lift's current position to their destination and add the departure action.
       b. If the passenger is not boarded, calculate the distance from the lift's current position to their origin, add the boarding action, then calculate the distance from the origin to their destination, and add the departure action.
    3. Sum all these distances and actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract destination for each passenger
        self.passenger_dest = {}
        for fact in self.static:
            if fact.startswith('(destin '):
                parts = fact[1:-1].split()
                p, dest = parts[1], parts[2]
                self.passenger_dest[p] = dest

    def __call__(self, node):
        """Estimate the minimum cost to serve all passengers."""
        state = node.state

        # Extract current lift position
        current_lift = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_lift = int(fact[1:-1].split()[1])
                break

        if current_lift is None:
            return float('inf')

        cost = 0

        # Collect all passengers
        passengers = set()
        for fact in state:
            if fact.startswith('(boarded '):
                p = fact[1:-1].split()[1]
                passengers.add(p)
            elif fact.startswith('(served '):
                p = fact[1:-1].split()[1]
                passengers.add(p)
            elif fact.startswith('(origin '):
                p = fact[1:-1].split()[1]
                passengers.add(p)

        for p in passengers:
            # Check if served
            served = any(fact.startswith('(served {})'.format(p)) for fact in state)
            if served:
                continue

            # Check if boarded
            boarded = any(fact.startswith('(boarded {})'.format(p)) for fact in state)
            if boarded:
                # Get destination
                dest = self.passenger_dest.get(p, None)
                if dest is not None:
                    dest_floor = int(dest[1:-1])
                    distance = abs(dest_floor - current_lift)
                    cost += distance + 1  # Move to destination and depart
            else:
                # Get origin
                origin_facts = [fact for fact in state if fact.startswith('(origin {})'.format(p))]
                if origin_facts:
                    origin = origin_facts[0][1:-1].split()[2]
                    origin_floor = int(origin[1:-1])
                    dest = self.passenger_dest.get(p, None)
                    if dest is not None:
                        dest_floor = int(dest[1:-1])
                        distance_to_origin = abs(origin_floor - current_lift)
                        distance_origin_to_dest = abs(dest_floor - origin_floor)
                        cost += distance_to_origin + distance_origin_to_dest + 2  # Move to origin, board, move to dest, depart

        return cost
