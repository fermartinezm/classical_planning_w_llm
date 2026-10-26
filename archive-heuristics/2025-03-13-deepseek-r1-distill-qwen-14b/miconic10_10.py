from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departing actions.

    # Assumptions:
    - The lift can move one floor at a time.
    - Each boarding and departing action counts as one step.
    - Passengers not yet boarded require the lift to move to their origin floor before being served.

    # Heuristic Initialization
    - Extracts static facts about the hierarchy of floors using 'above' relationships.
    - Maps each floor to its level based on the hierarchy.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger, check if they are already served. If all are served, return 0.
    2. For each passenger not served:
       a. If not boarded, calculate the distance from the lift's current floor to their origin, add boarding action, then the distance from origin to destination, and add departing action.
       b. If boarded, calculate the distance from the lift's current floor to their destination and add departing action.
    3. Sum all the calculated actions for each passenger to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static floor information."""
        self.static_facts = task.static
        self.above = self._build_floor_hierarchy()
        self.floor_levels = self._assign_floor_levels()

    def _build_floor_hierarchy(self):
        """Build a dictionary representing the 'above' relationships between floors."""
        above = {}
        for fact in self.static_facts:
            if fnmatch(fact, '(above * *)'):
                parts = fact[1:-1].split()
                lower_floor = parts[1]
                upper_floor = parts[2]
                if lower_floor not in above:
                    above[lower_floor] = []
                above[lower_floor].append(upper_floor)
        return above

    def _assign_floor_levels(self):
        """Assign a level to each floor based on the hierarchy, with the top floor as 0."""
        # Find the top floor (one with no floors above it)
        all_floors = {f for fact in self.static_facts for f in fact[1:-1].split()[1:]}
        top_floors = [f for f in all_floors if f not in self.above]
        top_floor = top_floors[0] if top_floors else None

        level = {top_floor: 0} if top_floor else {}
        visited = set()

        def _assign_level(current_floor, current_level):
            level[current_floor] = current_level
            visited.add(current_floor)
            for upper_floor in self.above.get(current_floor, []):
                if upper_floor not in level:
                    _assign_level(upper_floor, current_level + 1)

        if top_floor:
            _assign_level(top_floor, 0)
            for floor in all_floors:
                if floor not in visited:
                    # This should not happen in a well-formed domain
                    level[ floor ] = 0  # arbitrary, but shouldn't affect heuristic

        return level

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state
        current_lift_floor = None
        served_passengers = set()
        passengers = {}

        # Extract current lift position
        for fact in state:
            if fnmatch(fact, '(lift-at *)'):
                current_lift_floor = fact[1:-1].split()[1]

        # Extract passenger information
        for fact in state:
            if fnmatch(fact, '(origin * *)'):
                passenger = fact[1:-1].split()[1]
                origin = fact[1:-1].split()[2]
                passengers[passenger] = {'origin': origin, 'destination': None, 'boarded': False}
            elif fnmatch(fact, '(destin * *)'):
                passenger = fact[1:-1].split()[1]
                destination = fact[1:-1].split()[2]
                passengers[passenger]['destination'] = destination
            elif fnmatch(fact, '(boarded *)'):
                passenger = fact[1:-1].split()[1]
                passengers[passenger]['boarded'] = True
            elif fnmatch(fact, '(served *)'):
                passenger = fact[1:-1].split()[1]
                served_passengers.add(passenger)

        # Check if all passengers are served
        if all(p in served_passengers for p in passengers):
            return 0

        total_actions = 0

        for passenger, info in passengers.items():
            if passenger in served_passengers:
                continue

            origin = info['origin']
            destination = info['destination']
            boarded = info['boarded']

            # Calculate distance between floors
            def distance(f1, f2):
                return abs(self.floor_levels[f1] - self.floor_levels[f2])

            if not boarded:
                # Need to move to origin, board, move to destination, depart
                if current_lift_floor is None:
                    # This should not happen in a valid state
                    return float('inf')
                actions = distance(current_lift_floor, origin) + 1 + distance(origin, destination) + 1
                total_actions += actions
                # Update current lift position to destination after moving
                current_lift_floor = destination
            else:
                # Already boarded, just need to move to destination and depart
                actions = distance(current_lift_floor, destination) + 1
                total_actions += actions
                current_lift_floor = destination

        return total_actions
