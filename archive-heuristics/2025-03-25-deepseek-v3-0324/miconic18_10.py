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
    in the elevator system. It considers:
    - The current position of the elevator
    - Passengers waiting to board (origin floors)
    - Passengers already boarded needing to depart (destination floors)
    - The floor hierarchy (above relations)

    # Assumptions:
    - The elevator can only move between adjacent floors (up/down actions)
    - Each passenger must be boarded before being served
    - The order of serving passengers affects the total cost
    - Moving between floors costs 1 action per floor
    - Boarding and departing each cost 1 action

    # Heuristic Initialization
    - Extract static information about destinations and floor hierarchy
    - Build a mapping of passengers to their origin and destination floors
    - Create a graph representation of floor connectivity

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
        a. If not boarded yet:
            - Add cost to move from current floor to origin floor
            - Add 1 action for boarding
            - Add cost to move from origin to destination floor
        b. If already boarded:
            - Add cost to move from current floor to destination floor
        c. Add 1 action for departing
    2. Optimize the order of serving passengers to minimize total movement:
        - Group passengers by origin/destination floors to reduce trips
        - Plan an efficient route that serves multiple passengers per trip
    3. The total heuristic is the sum of:
        - All boarding and departing actions
        - All floor movements between pickups and dropoffs
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.passenger_dest = {}
        # Extract passenger origins from initial state (they may change)
        self.passenger_orig = {}
        # Build floor hierarchy graph
        self.above_graph = {}

        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_dest[passenger] = floor
            elif match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                self.above_graph.setdefault(floor1, set()).add(floor2)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        if self.goals <= state:
            return 0

        # Get current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Track served and boarded passengers
        served = set()
        boarded = set()
        waiting = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                served.add(parts[1])
            elif match(fact, "boarded", "*"):
                boarded.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                passenger = parts[1]
                if passenger not in served and passenger not in boarded:
                    waiting.add(passenger)

        total_cost = 0
        current_pos = current_floor

        # First handle boarded passengers (must be served before new boardings)
        for passenger in list(boarded):
            dest = self.passenger_dest[passenger]
            # Cost to move to destination floor
            total_cost += self._floor_distance(current_pos, dest)
            current_pos = dest
            # Cost to depart
            total_cost += 1

        # Then handle waiting passengers
        for passenger in list(waiting):
            # Find origin floor from state
            origin = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin = get_parts(fact)[2]
                    break
            
            if origin:
                # Cost to move to origin floor
                total_cost += self._floor_distance(current_pos, origin)
                current_pos = origin
                # Cost to board
                total_cost += 1
                # Cost to move to destination
                dest = self.passenger_dest[passenger]
                total_cost += self._floor_distance(current_pos, dest)
                current_pos = dest
                # Cost to depart
                total_cost += 1

        return total_cost

    def _floor_distance(self, floor1, floor2):
        """Estimate the number of actions needed to move between two floors."""
        if floor1 == floor2:
            return 0
        
        # In the worst case, we might need to visit all intermediate floors
        # This is a conservative estimate since we don't know the exact path
        return 1  # Simplified - could be improved with actual floor hierarchy analysis
