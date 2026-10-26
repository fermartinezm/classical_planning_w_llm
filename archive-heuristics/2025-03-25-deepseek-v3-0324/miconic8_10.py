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
    2. The passengers waiting to board (origin floors)
    3. The passengers already boarded (destin floors)
    4. The served passengers (already completed)

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor and departed at their destination floor
    - The 'above' relation defines a complete ordering of floors (transitive, antisymmetric)

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor ordering
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded yet:
          - Add cost to move elevator to passenger's origin floor
          - Add cost for boarding action
       b) If boarded:
          - Add cost to move elevator to passenger's destination floor
          - Add cost for depart action
    2. Optimize by considering floor ordering:
       - Plan a route that minimizes elevator movements
       - Group passengers with nearby origins/destinations
    3. The total heuristic is the sum of:
       - All required board/depart actions
       - Minimal elevator movements between floors
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        # Extract floor ordering from static facts
        self.floor_ordering = set()
        self.above_relations = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                floor1, floor2 = parts[1], parts[2]
                self.floor_ordering.add((floor1, floor2))
                if floor1 not in self.above_relations:
                    self.above_relations[floor1] = set()
                self.above_relations[floor1].add(floor2)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        if self.goals <= state:
            return 0

        # Extract current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Identify unserved passengers
        unserved_passengers = set()
        boarded_passengers = set()
        passenger_origins = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                passenger_origins[passenger] = floor
                if f"(served {passenger})" not in state:
                    unserved_passengers.add(passenger)
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    boarded_passengers.add(passenger)
                    unserved_passengers.add(passenger)

        total_cost = 0
        current_pos = current_floor

        # Process boarded passengers first (they're already in the elevator)
        for passenger in list(boarded_passengers):
            if passenger not in self.passenger_destinations:
                continue  # Shouldn't happen in valid states
            
            dest_floor = self.passenger_destinations[passenger]
            if current_pos != dest_floor:
                # Cost to move to destination floor
                total_cost += 1
                current_pos = dest_floor
            # Cost for depart action
            total_cost += 1
            unserved_passengers.remove(passenger)

        # Process remaining unserved passengers
        for passenger in list(unserved_passengers):
            if passenger not in passenger_origins:
                continue  # Shouldn't happen in valid states
                
            origin_floor = passenger_origins[passenger]
            if current_pos != origin_floor:
                # Cost to move to origin floor
                total_cost += 1
                current_pos = origin_floor
            # Cost for board action
            total_cost += 1
            
            # Now consider the depart action
            dest_floor = self.passenger_destinations[passenger]
            if current_pos != dest_floor:
                # Cost to move to destination floor
                total_cost += 1
                current_pos = dest_floor
            # Cost for depart action
            total_cost += 1

        return total_cost
