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
    A domain-dependent heuristic for the Miconic elevator domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    by considering:
    1. The current position of the elevator
    2. The origin and destination floors of unserved passengers
    3. Whether passengers are already boarded
    4. The floor hierarchy (which floors are above others)

    # Assumptions:
    - The elevator can only move between floors that are directly connected via 'above' relations
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The 'above' relations form a complete ordering of floors (no disconnected floors)

    # Heuristic Initialization
    - Extract goal conditions (all passengers must be served)
    - Extract static information about floor hierarchy ('above' relations)
    - Build a mapping of passenger origins and destinations from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded:
          - Add cost to move elevator from current position to passenger's origin floor
          - Add cost for boarding action
       b) If boarded:
          - Add cost to move elevator from current position to passenger's destination floor
          - Add cost for depart action
    2. For the movement costs between floors:
       - Calculate the minimum number of up/down actions needed to move between floors
       using the 'above' hierarchy
    3. Optimize the order of serving passengers to minimize total movement:
       - Group passengers by origin/destination floors when possible
       - Prioritize passengers that are already boarded
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        # Extract passenger origins from static facts (may change as passengers are boarded)
        self.origins = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor
            elif match(fact, "origin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.origins[passenger] = floor

        # Build floor hierarchy from 'above' relations
        self.above_map = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, upper, lower = get_parts(fact)
                if upper not in self.above_map:
                    self.above_map[upper] = []
                self.above_map[upper].append(lower)

        # Precompute floor ordering for distance calculation
        self.floor_order = self._compute_floor_order()

    def _compute_floor_order(self):
        """Compute a linear ordering of floors based on 'above' relations."""
        # This assumes floors form a total order
        if not self.above_map:
            return []

        # Find the top floor (floor that is above all others but no one is above it)
        all_floors = set()
        higher_floors = set()
        for upper, lowers in self.above_map.items():
            all_floors.add(upper)
            all_floors.update(lowers)
            higher_floors.add(upper)

        top_floors = [f for f in higher_floors if f not in self.above_map.values()]
        if not top_floors:
            # No clear top floor, just pick one
            top_floor = next(iter(higher_floors))
        else:
            top_floor = top_floors[0]

        # Build ordered list from top down
        ordered_floors = [top_floor]
        current = top_floor
        while current in self.above_map:
            # Assuming single linear hierarchy
            next_floors = self.above_map[current]
            if len(next_floors) > 1:
                # If multiple floors below, just pick one (heuristic simplification)
                current = next_floors[0]
            else:
                current = next_floors[0]
            ordered_floors.append(current)

        return ordered_floors

    def _floor_distance(self, floor1, floor2):
        """Calculate the minimum number of moves needed between two floors."""
        if floor1 == floor2:
            return 0

        if not self.floor_order:
            # Fallback if floor ordering couldn't be determined
            return 1

        try:
            idx1 = self.floor_order.index(floor1)
            idx2 = self.floor_order.index(floor2)
            return abs(idx1 - idx2)
        except ValueError:
            # One of the floors not in ordering
            return 1

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Find current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if not current_floor:
            # Elevator position unknown (shouldn't happen in valid states)
            return float('inf')

        total_cost = 0
        remaining_passengers = set()

        # First pass: handle already boarded passengers
        for fact in state:
            if match(fact, "boarded", "*"):
                passenger = get_parts(fact)[1]
                remaining_passengers.add(passenger)
                dest_floor = self.destinations[passenger]
                # Cost to move to destination and depart
                total_cost += self._floor_distance(current_floor, dest_floor) + 1
                current_floor = dest_floor  # Elevator now at destination

        # Second pass: handle passengers not yet boarded
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, origin_floor = get_parts(fact)
                if passenger not in remaining_passengers and passenger not in self.goals:
                    # Cost to move to origin, board, then to destination, and depart
                    move_to_origin = self._floor_distance(current_floor, origin_floor)
                    dest_floor = self.destinations[passenger]
                    move_to_dest = self._floor_distance(origin_floor, dest_floor)
                    total_cost += move_to_origin + 1 + move_to_dest + 1
                    current_floor = dest_floor  # Elevator now at destination

        return total_cost
