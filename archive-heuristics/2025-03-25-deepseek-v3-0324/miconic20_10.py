from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    by considering:
    1. The current position of the elevator
    2. The passengers that still need to be boarded
    3. The passengers that are boarded but not yet served
    4. The floor relationships (above) for movement costs

    # Assumptions:
    - The elevator can only move between floors connected by the 'above' relation
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The 'above' relation forms a complete ordering of floors (no disconnected floors)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Extract the floor hierarchy from 'above' relations
    - Build a mapping from floors to their positions in the hierarchy

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded:
          - Need to move to origin floor (if not already there)
          - Board action
       b) If boarded:
          - Need to move to destination floor (if not already there)
          - Depart action
    2. Calculate movement costs between floors using the floor hierarchy
    3. Sum all required actions:
       - Board actions for unboarded passengers
       - Depart actions for boarded passengers
       - Movement actions between floors
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract destination floors for each passenger
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor hierarchy from 'above' relations
        self.floor_above = {}
        self.floor_below = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                self.floor_above[floor1] = floor2
                self.floor_below[floor2] = floor1

        # Build floor ordering (from bottom to top)
        self.floor_order = []
        current = None
        # Find bottom floor (not below any other floor)
        for floor in set(self.floor_above.keys()) | set(self.floor_above.values()):
            if floor not in self.floor_below:
                current = floor
                break
        # Build ordered list from bottom to top
        while current in self.floor_above:
            self.floor_order.append(current)
            current = self.floor_above[current]
        self.floor_order.append(current)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Get current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Initialize counters
        boarded_passengers = set()
        unboarded_passengers = set()
        served_passengers = set()

        # Categorize passengers
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "boarded", "*"):
                boarded_passengers.add(parts[1])
            elif match(fact, "served", "*"):
                served_passengers.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                passenger = parts[1]
                if passenger not in served_passengers and passenger not in boarded_passengers:
                    unboarded_passengers.add(passenger)

        total_cost = 0
        current_pos = current_floor

        # Process boarded passengers first (they're already in the elevator)
        for passenger in boarded_passengers:
            if passenger not in served_passengers:
                dest_floor = self.destinations[passenger]
                # Add movement cost from current position to destination
                if current_pos != dest_floor:
                    total_cost += self._get_move_cost(current_pos, dest_floor)
                    current_pos = dest_floor
                # Add depart action
                total_cost += 1

        # Process unboarded passengers
        for passenger in unboarded_passengers:
            # Find origin floor
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break
            
            # Add movement cost to origin floor
            if current_pos != origin_floor:
                total_cost += self._get_move_cost(current_pos, origin_floor)
                current_pos = origin_floor
            # Add board action
            total_cost += 1
            
            # Add movement to destination and depart action
            dest_floor = self.destinations[passenger]
            if current_pos != dest_floor:
                total_cost += self._get_move_cost(current_pos, dest_floor)
                current_pos = dest_floor
            total_cost += 1

        return total_cost

    def _get_move_cost(self, from_floor, to_floor):
        """Calculate the number of move actions needed between two floors."""
        if from_floor == to_floor:
            return 0
        
        # Get positions in floor ordering
        try:
            from_idx = self.floor_order.index(from_floor)
            to_idx = self.floor_order.index(to_floor)
        except ValueError:
            # Fallback if floor ordering isn't complete
            return abs(from_idx - to_idx) if 'from_idx' in locals() else 1
        
        return abs(from_idx - to_idx)
