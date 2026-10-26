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
    in an elevator system. It considers:
    - The current floor of the elevator
    - The origin and destination floors of unserved passengers
    - Whether passengers are already boarded
    - The floor relationships (which floors are above others)

    # Assumptions:
    - The elevator can only move between adjacent floors (up/down one at a time)
    - Each passenger must be picked up from their origin floor and dropped at their destination
    - Boarding and departing each take one action
    - The 'above' relationships form a complete ordering of floors

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Extract floor ordering from 'above' relationships in static facts
    - Build a mapping of each floor to its position in the ordering

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a. If not boarded:
          - Add cost to move elevator from current position to origin floor
          - Add 1 action for boarding
       b. Add cost to move from origin to destination floor
       c. Add 1 action for departing
    2. For passengers already boarded:
       - Add cost to move from current position to destination floor
       - Add 1 action for departing
    3. The total heuristic is the sum of all these costs
    4. Movement cost between floors is the absolute difference in their positions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor ordering from 'above' relationships
        self.floor_order = {}  # Maps floor to its position in the ordering
        self.floors = set()
        
        # First collect all floor pairs from 'above' relationships
        above_pairs = []
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                above_pairs.append((floor1, floor2))
                self.floors.update([floor1, floor2])

        # Determine floor ordering (assuming floors are properly ordered)
        # Find the bottom floor (not appearing as second in any 'above' relation)
        bottom = None
        all_seconds = {f2 for _, f2 in above_pairs}
        for floor in self.floors:
            if floor not in all_seconds:
                bottom = floor
                break

        # Build ordering by following 'above' relationships
        if bottom:
            current = bottom
            position = 0
            self.floor_order[current] = position
            while True:
                next_floors = [f2 for f1, f2 in above_pairs if f1 == current]
                if not next_floors:
                    break
                current = next_floors[0]  # Assuming linear ordering
                position += 1
                self.floor_order[current] = position

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_floor = None
        unserved_passengers = set()
        boarded_passengers = set()

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "lift-at", "*"):
                current_floor = parts[1]
            elif match(fact, "origin", "*", "*"):
                passenger = parts[1]
                if not any(match(f, "served", passenger) for f in state):
                    unserved_passengers.add(passenger)
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                if not any(match(f, "served", passenger) for f in state):
                    boarded_passengers.add(passenger)

        total_cost = 0

        # Handle boarded passengers first (they're already in the elevator)
        for passenger in boarded_passengers:
            dest_floor = self.destinations[passenger]
            if current_floor != dest_floor:
                # Cost to move to destination floor
                total_cost += abs(self.floor_order[current_floor] - self.floor_order[dest_floor])
            # Cost to depart
            total_cost += 1

        # Handle unserved passengers not yet boarded
        for passenger in unserved_passengers - boarded_passengers:
            # Find origin floor
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break

            if origin_floor is None:
                continue  # Shouldn't happen for valid states

            # Cost to move to origin floor
            if current_floor != origin_floor:
                total_cost += abs(self.floor_order[current_floor] - self.floor_order[origin_floor])
            # Cost to board
            total_cost += 1
            # Cost to move to destination floor
            dest_floor = self.destinations[passenger]
            total_cost += abs(self.floor_order[origin_floor] - self.floor_order[dest_floor])
            # Cost to depart
            total_cost += 1

        return total_cost
