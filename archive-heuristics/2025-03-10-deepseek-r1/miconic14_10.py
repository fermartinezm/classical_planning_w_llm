from heuristics.heuristic_base import Heuristic

class miconic14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the steps needed for each passenger to board and depart, based on the current elevator position.

    # Assumptions:
    - Each passenger requires boarding at their origin and departing at their destination.
    - The elevator can move between any two floors in one action.
    - Passengers are handled sequentially, which may overcount movement steps but provides a fast estimate.

    # Heuristic Initialization
    - Extract the destination of each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current elevator floor from the state.
    2. For each passenger:
        a. If served, skip.
        b. If boarded, add steps to move to destination (if needed) and depart.
        c. If not boarded, add steps to move to origin (if needed), board, move to destination (if needed), and depart.
    3. Sum all steps for a total heuristic value.
    """

    def __init__(self, task):
        """Extract passenger destinations from static facts."""
        self.destin = {}
        for fact in task.static:
            if fact.startswith('(destin '):
                parts = fact[1:-1].split()
                if parts[0] == 'destin':
                    passenger = parts[1]
                    floor = parts[2]
                    self.destin[passenger] = floor

    def __call__(self, node):
        """Calculate the heuristic value for the given state."""
        state = node.state
        current_floor = None
        served = set()
        boarded = set()
        origins = {}

        # Extract current elevator floor, served, boarded, and origins from state
        for fact in state:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'lift-at':
                current_floor = parts[1]
            elif parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'boarded':
                boarded.add(parts[1])
            elif parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                origins[passenger] = floor

        if current_floor is None:
            return 0  # Invalid state, no elevator position

        total = 0
        for passenger in self.destin:
            if passenger in served:
                continue
            if passenger in boarded:
                # Boarded passenger: move to destination and depart
                destin = self.destin[passenger]
                steps = 0
                if current_floor != destin:
                    steps += 1  # Move to destination
                steps += 1  # Depart
                total += steps
            else:
                # Unboarded passenger: check origin
                if passenger not in origins:
                    continue  # No origin in state, invalid case
                origin = origins[passenger]
                destin = self.destin[passenger]
                steps = 0
                if current_floor != origin:
                    steps += 1  # Move to origin
                steps += 1  # Board
                if origin != destin:
                    steps += 1  # Move to destination
                steps += 1  # Depart
                total += steps
        return total
