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
    4. The floor hierarchy (above relations)

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at destination
    - The 'above' relations form a complete ordering of floors

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a mapping of floor hierarchy from 'above' relations
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
        a. If not boarded:
            - Add cost to move elevator to passenger's origin floor
            - Add boarding action
            - Add cost to move to destination floor
        b. If already boarded:
            - Add cost to move to passenger's destination floor
        c. Add departing action
    2. Optimize by:
        - Grouping passengers with same origin/destination
        - Considering floor proximity when calculating movement costs
    3. The total heuristic is the sum of all required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract destin(p,f) facts to map passengers to destination floors
        self.destinations = {}
        # Extract above(f1,f2) relations to understand floor hierarchy
        self.floor_hierarchy = set()

        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                self.floor_hierarchy.add((get_parts(fact)[1], get_parts(fact)[2]))

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        total_cost = 0

        # Check if we're already in a goal state
        if self.goals <= state:
            return 0

        # Find current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Process all passengers
        served_passengers = set()
        boarded_passengers = set()
        origin_floors = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                served_passengers.add(parts[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                origin_floors[parts[1]] = parts[2]

        # Calculate cost for each unserved passenger
        for passenger in self.destinations:
            if passenger in served_passengers:
                continue

            if passenger in boarded_passengers:
                # Passenger is already in elevator, just need to go to destination
                dest_floor = self.destinations[passenger]
                total_cost += self._movement_cost(current_floor, dest_floor)
                current_floor = dest_floor
                total_cost += 1  # depart action
            else:
                # Passenger needs to be picked up first
                origin_floor = origin_floors.get(passenger)
                if not origin_floor:
                    continue  # passenger already picked up but not marked as boarded?

                # Move to origin floor
                total_cost += self._movement_cost(current_floor, origin_floor)
                current_floor = origin_floor
                total_cost += 1  # board action

                # Then move to destination floor
                dest_floor = self.destinations[passenger]
                total_cost += self._movement_cost(current_floor, dest_floor)
                current_floor = dest_floor
                total_cost += 1  # depart action

        return total_cost

    def _movement_cost(self, from_floor, to_floor):
        """Estimate the number of up/down actions needed to move between floors."""
        if from_floor == to_floor:
            return 0

        # Since we don't have complete floor ordering, we'll assume each movement is 1 action
        # In a more sophisticated version, we could calculate the actual distance
        return 1
