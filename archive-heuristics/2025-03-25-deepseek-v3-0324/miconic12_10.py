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
    - Passengers that still need to be picked up
    - Passengers that are boarded but need to be delivered
    - The elevator's current position and required movements

    # Assumptions:
    - The elevator can only move between floors connected by the 'above' relation
    - Each passenger must be picked up from their origin floor and delivered to their destination floor
    - The 'above' relation defines a total ordering of floors (no branching)

    # Heuristic Initialization
    - Extract passenger destinations from static facts
    - Build a floor hierarchy from the 'above' relations
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a) If not boarded:
            - Add cost for moving elevator to origin floor (if not already there)
            - Add cost for boarding action
        b) If boarded:
            - Add cost for moving elevator to destination floor (if not already there)
            - Add cost for depart action
    2. For movement costs:
        - Calculate the minimum number of 'up' or 'down' actions needed between floors
        - Use the floor hierarchy to determine direction and distance
    3. Sum all these costs to get the total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor hierarchy
        self.above_map = {}
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                if floor1 not in self.above_map:
                    self.above_map[floor1] = []
                self.above_map[floor1].append(floor2)

        # Precompute floor ordering for distance calculation
        self.floor_order = self._compute_floor_order()

    def _compute_floor_order(self):
        """Compute a total ordering of floors based on the 'above' relation."""
        if not self.above_map:
            return []

        # Find the top floor (floor that is not above any other floor)
        all_floors = set(self.above_map.keys())
        for floors in self.above_map.values():
            all_floors.update(floors)

        # Build the ordering from bottom to top
        floor_order = []
        current = None

        # Find the bottom floor (not in any above list)
        for floor in all_floors:
            is_bottom = True
            for above_list in self.above_map.values():
                if floor in above_list:
                    is_bottom = False
                    break
            if is_bottom:
                current = floor
                break

        while current in self.above_map:
            floor_order.append(current)
            # Assuming single successor in linear hierarchy
            current = self.above_map[current][0]
        floor_order.append(current)

        return floor_order

    def _floor_distance(self, floor1, floor2):
        """Calculate the minimum number of moves between two floors."""
        if not self.floor_order or floor1 == floor2:
            return 0

        try:
            idx1 = self.floor_order.index(floor1)
            idx2 = self.floor_order.index(floor2)
            return abs(idx1 - idx2)
        except ValueError:
            # Fallback if floor ordering isn't properly computed
            return 1

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Get current elevator position
        lift_at = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, floor = get_parts(fact)
                lift_at = floor
                break

        total_cost = 0

        # Process all passengers
        for passenger in self.destinations:
            # Skip already served passengers
            if f"(served {passenger})" in state:
                continue

            # Check if passenger is boarded
            boarded = f"(boarded {passenger})" in state

            if boarded:
                # Passenger needs to be delivered to destination
                dest_floor = self.destinations[passenger]
                if lift_at != dest_floor:
                    total_cost += self._floor_distance(lift_at, dest_floor)
                total_cost += 1  # depart action
            else:
                # Passenger needs to be picked up
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, _, floor = get_parts(fact)
                        origin_floor = floor
                        break

                if origin_floor and lift_at != origin_floor:
                    total_cost += self._floor_distance(lift_at, origin_floor)
                total_cost += 1  # board action

        return total_cost
