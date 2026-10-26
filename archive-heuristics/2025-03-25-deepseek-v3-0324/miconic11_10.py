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
    A domain-dependent heuristic for the Miconic domain (elevator scheduling).

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    by considering:
    1. The current position of the elevator
    2. The passengers waiting to board
    3. The passengers already boarded who need to be served
    4. The floor relationships (above/below) for movement planning

    # Assumptions:
    - The elevator can only move between floors with direct "above" relationships
    - Each passenger must be boarded from their origin floor before being served at destination
    - The "above" relationships form a complete ordering of floors
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a mapping of floor relationships from "above" predicates
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded yet:
          - Add cost to move elevator to passenger's origin floor
          - Add cost for boarding action
       b) If boarded:
          - Add cost to move elevator to passenger's destination floor
          - Add cost for depart action
    2. For movement between floors:
       - Calculate the minimum number of up/down actions needed based on floor relationships
    3. Sum all these costs to get the total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract destination floors for each passenger
        self.destinations = {}
        # Extract origin floors for each passenger (from static facts)
        self.origins = {}
        # Build floor ordering from "above" relationships
        self.above_relations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.origins[passenger] = floor
            elif match(fact, "above", "*", "*"):
                floor1, floor2 = parts[1], parts[2]
                self.above_relations.add((floor1, floor2))

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        if self.goals <= state:
            return 0

        # Get current elevator position
        lift_at = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_at = get_parts(fact)[1]
                break

        total_cost = 0
        boarded_passengers = set()
        unserved_passengers = set()

        # Process all passengers in the state
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "boarded", "*"):
                passenger = parts[1]
                boarded_passengers.add(passenger)
            elif match(fact, "origin", "*", "*"):
                passenger = parts[1]
                if not any(match(f, "served", passenger) for f in state):
                    unserved_passengers.add(passenger)

        # For each unserved passenger, calculate required actions
        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                # Passenger is boarded - need to go to destination
                dest_floor = self.destinations[passenger]
                if lift_at != dest_floor:
                    # Add cost for moving to destination floor
                    total_cost += self._get_move_cost(lift_at, dest_floor)
                    lift_at = dest_floor
                # Add cost for depart action
                total_cost += 1
            else:
                # Passenger not boarded - need to go to origin and board
                origin_floor = self.origins[passenger]
                if lift_at != origin_floor:
                    # Add cost for moving to origin floor
                    total_cost += self._get_move_cost(lift_at, origin_floor)
                    lift_at = origin_floor
                # Add cost for board action
                total_cost += 1

        return total_cost

    def _get_move_cost(self, current_floor, target_floor):
        """
        Estimate the number of up/down actions needed to move between floors.
        Uses the "above" relationships to determine the minimum path length.
        """
        if current_floor == target_floor:
            return 0

        # Check if we need to go up or down
        if (current_floor, target_floor) in self.above_relations:
            return 1
        elif (target_floor, current_floor) in self.above_relations:
            return 1
        else:
            # For floors not directly connected, estimate based on floor numbering
            # This assumes floors are named consistently (f1, f2, etc.)
            try:
                current_num = int(current_floor[1:])
                target_num = int(target_floor[1:])
                return abs(current_num - target_num)
            except (ValueError, IndexError):
                # Fallback: assume maximum possible distance
                return len(self.above_relations)
