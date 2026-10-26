from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by moving the lift to their origin and destination floors.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must board the lift at their origin floor and depart at their destination floor.
    - The heuristic assumes the shortest path for the lift between floors.

    # Heuristic Initialization
    - Extracts goal conditions and static facts (floor hierarchy) from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the lift.
    2. For each passenger, determine if they have been served.
    3. For unserved passengers, calculate the number of moves required for the lift to reach their origin and destination floors.
    4. Sum the required actions for all passengers, ensuring efficient computation by minimizing redundant moves.
    """

    def __init__(self, task):
        """Initialize the heuristic with task information."""
        self.goals = task.goals
        self.static = task.static
        self.above_map = self._build_above_map()

    def _build_above_map(self):
        """Construct a map of which floors are above others using static facts."""
        above_map = {}
        for fact in self.static:
            if fnmatch(fact, '(above * *)'):
                parts = fact[1:-1].split()
                floor1, floor2 = parts[1], parts[2]
                if floor1 not in above_map:
                    above_map[floor1] = []
                above_map[floor1].append(floor2)
        return above_map

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        lift_location = self._get_lift_location(state)
        if not lift_location:
            return 0  # No lift position, can't compute

        passengers = self._get_passengers(state)
        total_actions = 0

        for p in passengers:
            origin = self._get_origin(p, state)
            destin = self._get_destin(p, state)
            if self._is_served(p, state):
                continue

            # Calculate moves needed
            moves = self._calculate_floor_moves(lift_location, origin, state)
            moves += self._calculate_floor_moves(origin, destin, state)
            total_actions += 2 + moves  # 2 for boarding and departing

        return total_actions

    def _get_lift_location(self, state):
        """Find the current location of the lift."""
        for fact in state:
            if fnmatch(fact, '(lift-at *)'):
                return fact[9:-1]
        return None

    def _get_passengers(self, state):
        """Extract all passenger objects from the state."""
        passengers = set()
        for fact in state:
            if fnmatch(fact, '(origin * *)'):
                passengers.add(fact[7])
            if fnmatch(fact, '(destin * *)'):
                passengers.add(fact[7])
        return passengers

    def _get_origin(self, p, state):
        """Find the origin floor of passenger p."""
        for fact in state:
            if fnmatch(fact, f'(origin {p} *)'):
                return fact[7:-1].split()[1]

    def _get_destin(self, p, state):
        """Find the destination floor of passenger p."""
        for fact in state:
            if fnmatch(fact, f'(destin {p} *)'):
                return fact[7:-1].split()[1]

    def _is_served(self, p, state):
        """Check if passenger p has been served."""
        return f'(served {p})' in state

    def _calculate_floor_moves(self, current_floor, target_floor, state):
        """Estimate the number of moves needed between floors."""
        if current_floor == target_floor:
            return 0

        # Use static facts to determine the shortest path
        above_map = self.above_map.get(current_floor, [])
        if target_floor in above_map:
            return 1
        else:
            # If target is not directly above, assume moving up/down one floor at a time
            return abs(int(target_floor) - int(current_floor))
