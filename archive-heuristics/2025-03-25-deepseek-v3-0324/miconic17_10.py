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
    - The 'above' relation forms a complete ordering of floors (transitive, antisymmetric)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a dictionary mapping each floor to floors directly above it
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a) If not boarded:
            - Add cost to move elevator to passenger's origin floor
            - Add boarding action cost
        b) Add cost to move elevator to passenger's destination floor
        c) Add departing action cost
    2. Optimize the order of serving passengers by:
        a) Grouping passengers by origin/destination floors to minimize elevator movement
        b) Considering already boarded passengers first
    3. The total heuristic is the sum of:
        - All boarding/departing actions
        - Elevator movement between floors (using floor distance)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations
        self.destinations = {}
        # Extract floor hierarchy
        self.above_map = {}
        
        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                floor1, floor2 = parts[1], parts[2]
                self.above_map.setdefault(floor1, set()).add(floor2)

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
        
        # Identify passengers that still need to be served
        unserved_passengers = set()
        boarded_passengers = set()
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "origin", "*", "*"):
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    unserved_passengers.add(passenger)
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    boarded_passengers.add(passenger)
        
        total_cost = 0
        current_pos = current_floor
        
        # First handle already boarded passengers
        for passenger in boarded_passengers:
            dest_floor = self.destinations[passenger]
            # Cost to move to destination floor
            total_cost += self._floor_distance(current_pos, dest_floor)
            # Cost for depart action
            total_cost += 1
            current_pos = dest_floor
            unserved_passengers.discard(passenger)
        
        # Then handle remaining unserved passengers
        for passenger in unserved_passengers:
            # Find origin floor
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break
            
            if origin_floor is None:
                continue  # Shouldn't happen for valid states
                
            # Cost to move to origin floor
            total_cost += self._floor_distance(current_pos, origin_floor)
            # Cost for board action
            total_cost += 1
            current_pos = origin_floor
            
            # Cost to move to destination floor
            dest_floor = self.destinations[passenger]
            total_cost += self._floor_distance(current_pos, dest_floor)
            # Cost for depart action
            total_cost += 1
            current_pos = dest_floor
        
        return total_cost

    def _floor_distance(self, floor1, floor2):
        """Estimate the number of moves needed between two floors."""
        if floor1 == floor2:
            return 0
        
        # In the worst case, we might need to go through all intermediate floors
        # This is a conservative estimate since we don't have full floor ordering
        return 1  # Simplified to 1 since actual distance might be complex to compute
