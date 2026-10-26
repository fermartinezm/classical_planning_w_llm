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
    4. The floor relationships for efficient movement

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The 'above' relations form a complete ordering of floors (no disconnected floors)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a mapping of floor relationships from 'above' predicates
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a) If not boarded:
            - Need to move elevator to origin floor (1 action per floor moved)
            - Board passenger (1 action)
        b) If boarded:
            - Need to move elevator to destination floor (1 action per floor moved)
            - Depart passenger (1 action)
    2. Optimize movement by:
        - Grouping passengers with nearby origin/destination floors
        - Considering the current elevator position
    3. The heuristic sums:
        - Floor movements (distance between current and target floors)
        - Boarding actions (1 per unboarded passenger)
        - Departing actions (1 per boarded passenger not yet served)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destinations[passenger] = floor

        # Build floor ordering from 'above' relations
        self.floor_above = {}
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                self.floor_above[floor1] = floor2

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        total_cost = 0

        # Get current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None:
            return float('inf')  # Invalid state

        # Track passengers that still need service
        unserved_passengers = set()
        boarded_passengers = set()
        origin_floors = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                if passenger in self.passenger_destinations:
                    unserved_passengers.add(passenger)
                    origin_floors[passenger] = floor
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                if passenger in self.passenger_destinations:
                    boarded_passengers.add(passenger)
                    unserved_passengers.add(passenger)
            elif match(fact, "served", "*"):
                passenger = parts[1]
                if passenger in unserved_passengers:
                    unserved_passengers.remove(passenger)

        # If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # Calculate floor movements and required actions
        current_pos = current_floor
        remaining_passengers = list(unserved_passengers)
        
        # Process boarded passengers first (need to be departed)
        boarded_to_serve = [p for p in remaining_passengers if p in boarded_passengers]
        unboarded_to_serve = [p for p in remaining_passengers if p not in boarded_passengers]

        # Process boarded passengers
        for passenger in boarded_to_serve:
            dest_floor = self.passenger_destinations[passenger]
            # Add floor movement cost
            total_cost += self._floor_distance(current_pos, dest_floor)
            current_pos = dest_floor
            # Add depart action
            total_cost += 1

        # Process unboarded passengers
        for passenger in unboarded_to_serve:
            origin_floor = origin_floors[passenger]
            dest_floor = self.passenger_destinations[passenger]
            
            # Move to origin floor
            total_cost += self._floor_distance(current_pos, origin_floor)
            current_pos = origin_floor
            # Board action
            total_cost += 1
            
            # Move to destination floor
            total_cost += self._floor_distance(current_pos, dest_floor)
            current_pos = dest_floor
            # Depart action
            total_cost += 1

        return total_cost

    def _floor_distance(self, floor1, floor2):
        """Calculate the minimum number of moves between two floors."""
        if floor1 == floor2:
            return 0

        # Count steps upwards from floor1 to floor2
        current = floor1
        up_steps = 0
        while current in self.floor_above:
            up_steps += 1
            current = self.floor_above[current]
            if current == floor2:
                return up_steps

        # Count steps downwards from floor1 to floor2
        # Since 'above' relations are complete, we can find the path
        # by going up to the top and then down
        current = floor1
        down_steps = 0
        while current in self.floor_above:
            down_steps += 1
            current = self.floor_above[current]
        
        current = floor2
        down_steps2 = 0
        while current in self.floor_above:
            down_steps2 += 1
            current = self.floor_above[current]
        
        return down_steps + down_steps2
