from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(above f1 f2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconic5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It considers the number of passengers who are waiting to board, currently boarded,
    and the number of floors the lift needs to travel to pick up and drop off passengers.

    # Assumptions
    - Each passenger requires at least one board and one depart action.
    - The lift needs to move between floors to pick up and drop off passengers.
    - The heuristic does not perfectly account for optimal floor visits, but estimates
      based on the number of unique origin and destination floors.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the static facts.
    - Store the 'above' relationships between floors to estimate travel costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of passengers who are not yet served.
    2. For each unserved passenger, determine their origin and destination floors.
    3. Estimate the number of board actions needed by counting passengers at their origin.
    4. Estimate the number of depart actions needed by counting boarded passengers at their destination.
    5. Estimate the number of up/down actions by counting the number of unique origin and destination floors
       that need to be visited.  This is a simplification, as it doesn't perfectly account for the order
       in which floors are visited, but provides a reasonable estimate.
    6. Sum the estimated number of board, depart, and up/down actions to get the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting origin and destination floors
        for each passenger from the static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above = {}

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if parts[1] not in self.above:
                    self.above[parts[1]] = []
                self.above[parts[1]].append(parts[2])

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger = get_parts(fact)[1]
                destination = get_parts(fact)[2]
                self.passenger_destinations[passenger] = destination

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                passenger = get_parts(fact)[1]
                origin = get_parts(fact)[2]
                self.passenger_origins[passenger] = origin

    def __call__(self, node):
        """Estimate the number of actions required to serve all passengers."""
        state = node.state
        if task.goal_reached(node.state):
            return 0

        served_passengers = set()
        boarded_passengers = set()
        origin_floors = set()
        destination_floors = set()

        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])
            elif match(fact, "origin", "*", "*"):
                origin_floors.add(get_parts(fact)[2])

        unserved_passengers = set(self.passenger_destinations.keys()) - served_passengers

        board_actions = 0
        depart_actions = 0
        move_actions = 0

        for passenger in unserved_passengers:
            if passenger in self.passenger_origins and f"(origin {passenger} {self.passenger_origins[passenger]})" in state:
                board_actions += 1
                origin_floors.add(self.passenger_origins[passenger])
            if passenger in boarded_passengers:
                depart_actions += 1
                destination_floors.add(self.passenger_destinations[passenger])

        # Count unique origin and destination floors to estimate move actions
        lift_at_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_at_floor = get_parts(fact)[1]
                break

        if lift_at_floor:
            if lift_at_floor in origin_floors:
                origin_floors.remove(lift_at_floor)
            if lift_at_floor in destination_floors:
                destination_floors.remove(lift_at_floor)

        move_actions = len(origin_floors) + len(destination_floors)

        return board_actions + depart_actions + move_actions
