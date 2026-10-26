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
    This heuristic estimates the number of actions required to serve all passengers
    by considering:
    - The current position of the lift
    - The origin and destination floors of unserved passengers
    - Whether passengers are already boarded
    - The floor hierarchy (above relation)

    # Assumptions:
    - The lift can only move between floors connected by the 'above' relation.
    - Each passenger must be boarded from their origin floor before being served at their destination.
    - The heuristic does not need to be admissible (can overestimate).

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor hierarchy.
    - Store goal conditions (all passengers must be served).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
        a. If not boarded:
            - Add cost to move lift to origin floor (distance from current lift position)
            - Add cost to board (1 action)
        b. If boarded:
            - Add cost to move lift to destination floor (distance from current lift position)
            - Add cost to depart (1 action)
    2. Sum all costs for all unserved passengers.
    3. The heuristic value is the total estimated actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        # Extract floor hierarchy (above relation)
        self.above = set()
        
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif parts[0] == "above":
                floor1, floor2 = parts[1], parts[2]
                self.above.add((floor1, floor2))

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        total_cost = 0

        # Check if we're already in a goal state
        if self.goals <= state:
            return 0

        # Get current lift position
        lift_at = None
        for fact in state:
            if fact.startswith("(lift-at"):
                lift_at = get_parts(fact)[1]
                break

        # If lift position not found (shouldn't happen in valid states)
        if not lift_at:
            return float("inf")

        # Process each passenger
        served_passengers = {get_parts(fact)[1] for fact in state if fact.startswith("(served")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if fact.startswith("(boarded")}
        
        # Get origin floors from state (may change as passengers are boarded)
        origins = {}
        for fact in state:
            if fact.startswith("(origin"):
                parts = get_parts(fact)
                passenger, floor = parts[1], parts[2]
                origins[passenger] = floor

        for passenger, dest_floor in self.destinations.items():
            if passenger in served_passengers:
                continue  # Already served, no cost

            if passenger in boarded_passengers:
                # Passenger is boarded, need to go to destination
                current_floor = lift_at
                target_floor = dest_floor
                # Add cost to move to destination (1 per floor)
                total_cost += self._get_floor_distance(current_floor, target_floor)
                # Add cost for depart action
                total_cost += 1
            else:
                # Passenger not boarded, need to go to origin first
                origin_floor = origins.get(passenger)
                if not origin_floor:
                    continue  # Shouldn't happen for valid states
                
                current_floor = lift_at
                target_floor = origin_floor
                # Add cost to move to origin (1 per floor)
                total_cost += self._get_floor_distance(current_floor, target_floor)
                # Add cost for board action
                total_cost += 1

        return total_cost

    def _get_floor_distance(self, floor1, floor2):
        """Estimate the number of up/down actions needed to move between floors."""
        if floor1 == floor2:
            return 0

        # We don't have complete floor numbering, so we'll use a simple approach:
        # Count the minimum number of 'above' relations needed to connect the floors
        # This is a conservative estimate since we don't have full floor ordering
        
        # If we can find a direct path in the 'above' relation, use that
        if (floor1, floor2) in self.above:
            return 1
        if (floor2, floor1) in self.above:
            return 1

        # Otherwise, assume they're connected by at least one intermediate floor
        # This is a rough estimate since we don't have complete floor numbering
        return 2
