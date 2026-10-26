from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departing actions for each passenger.

    # Assumptions:
    - The lift can move up or down one floor at a time.
    - Each passenger must board and depart once.
    - The heuristic sums the individual costs for each passenger, assuming the lift can handle each passenger's trip independently.

    # Heuristic Initialization
    - Extract the static facts to determine the floor hierarchy and compute distances between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. For each passenger, determine their origin and destination floors.
    3. For each passenger, calculate:
       a. The distance from the lift's current floor to their origin.
       b. The distance from their origin to their destination.
       c. Add 2 actions for boarding and departing.
    4. Sum these values for all passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about floor hierarchy."""
        self.static_facts = task.static
        self.floor_level = self.build_floor_hierarchy()

    def build_floor_hierarchy(self):
        """Build a dictionary mapping each floor to its level based on 'above' relations."""
        floor_level = {}
        floors = set()
        for fact in self.static_facts:
            if fact.startswith('(above '):
                parts = fact[6:-1].split()
                if len(parts) == 3:
                    f1 = parts[1]
                    f2 = parts[2]
                    floors.add(f1)
                    floors.add(f2)
        # Sort floors by their numeric value
        floors = sorted(floors, key=lambda x: int(x[1:]))
        for i, f in enumerate(floors, 1):
            floor_level[f] = i
        return floor_level

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        total_cost = 0

        # Extract current lift floor
        lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact[8:-1].split()
                if len(parts) == 2 and parts[0] == '?f':
                    lift_floor = parts[1]
                    break
        if lift_floor is None:
            return 0

        # Extract passenger origins and destinations
        passenger_origin = {}
        passenger_dest = {}
        served_passengers = set()
        for fact in state:
            if fact.startswith('(origin '):
                parts = fact[7:-1].split()
                if len(parts) == 3 and parts[0] == '?p' and parts[1] == '?f':
                    p = parts[1]
                    f = parts[2]
                    passenger_origin[p] = f
            elif fact.startswith('(destin '):
                parts = fact[7:-1].split()
                if len(parts) == 3 and parts[0] == '?p' and parts[1] == '?f':
                    p = parts[1]
                    f = parts[2]
                    passenger_dest[p] = f
            elif fact.startswith('(served '):
                parts = fact[7:-1].split()
                if len(parts) == 2 and parts[0] == '?p':
                    p = parts[1]
                    served_passengers.add(p)

        # For each passenger, calculate the cost
        for p in passenger_origin:
            if p in served_passengers:
                continue
            origin = passenger_origin[p]
            dest = passenger_dest[p]
            # Get floor levels
            level_lift = self.floor_level.get(lift_floor, 0)
            level_origin = self.floor_level.get(origin, 0)
            level_dest = self.floor_level.get(dest, 0)
            # Calculate distances
            dist_lift_to_origin = abs(level_lift - level_origin)
            dist_origin_to_dest = abs(level_origin - level_dest)
            # Add to total cost
            total_cost += dist_lift_to_origin + dist_origin_to_dest + 2

        return total_cost
