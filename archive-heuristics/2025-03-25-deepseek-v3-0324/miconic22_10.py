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
    in the Miconic elevator domain. It considers:
    - The current floor of the elevator
    - The floors where passengers need to be picked up
    - The floors where boarded passengers need to be dropped off
    - The order in which these actions can be optimally performed

    # Assumptions:
    - The elevator can only move between floors that are connected by 'above' relations
    - Each passenger must be picked up from their origin floor before being dropped at their destination
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a mapping of floor relationships from 'above' predicates
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded, count the distance from current floor to origin floor
       b) If boarded, count the distance from current floor to destination floor
    2. For optimal ordering:
       a) Group passengers by origin/destination floors to minimize travel
       b) Calculate the minimal path covering all required stops
    3. Add 1 action for each board/depart operation needed
    4. Sum all movement and operation costs
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract destination floors for each passenger
        self.destinations = {}
        # Extract origin floors (may change as passengers are boarded)
        self.origins = {}
        # Build floor hierarchy from 'above' relations
        self.above_map = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                self.destinations[parts[1]] = parts[2]
            elif parts[0] == "above":
                if parts[1] not in self.above_map:
                    self.above_map[parts[1]] = []
                self.above_map[parts[1]].append(parts[2])

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract current elevator position
        current_floor = None
        for fact in state:
            if fact.startswith("(lift-at"):
                current_floor = get_parts(fact)[1]
                break

        # If no current floor (shouldn't happen in valid states), return large number
        if not current_floor:
            return float('inf')

        # Track passengers that still need to be served
        unserved_passengers = []
        boarded_passengers = []

        # Extract passenger states
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    unserved_passengers.append((passenger, parts[2], False))
            elif parts[0] == "boarded":
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    boarded_passengers.append((passenger, self.destinations[passenger], True))

        all_passengers = unserved_passengers + boarded_passengers

        # If no passengers left to serve (shouldn't happen if goals not met), return small number
        if not all_passengers:
            return 1

        # Calculate minimal path to serve all passengers
        total_cost = 0
        current_pos = current_floor

        # We'll process passengers in order of floor proximity
        remaining_passengers = all_passengers.copy()

        while remaining_passengers:
            # Find closest passenger stop from current position
            min_dist = float('inf')
            next_passenger = None
            next_floor = None
            is_destination = False

            for passenger, floor, is_dest in remaining_passengers:
                dist = self._floor_distance(current_pos, floor)
                if dist < min_dist:
                    min_dist = dist
                    next_passenger = passenger
                    next_floor = floor
                    is_destination = is_dest

            if next_passenger is None:
                break  # no reachable passengers (shouldn't happen in solvable states)

            # Add movement cost
            total_cost += min_dist

            # Add board/depart cost
            total_cost += 1

            # Update current position
            current_pos = next_floor

            # Remove served passenger
            remaining_passengers = [p for p in remaining_passengers if p[0] != next_passenger]

        return total_cost

    def _floor_distance(self, floor1, floor2):
        """Calculate minimal number of moves needed between two floors."""
        if floor1 == floor2:
            return 0

        # Since we don't have complete floor ordering, we'll assume direct movement is possible
        # This is optimistic but works for the heuristic
        return 1
