from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by efficiently calculating the required movements of the lift and the boarding/departing actions.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must board the lift at their origin floor and depart at their destination floor.
    - The heuristic assumes that the lift can serve multiple passengers in a single trip if possible.

    # Heuristic Initialization
    - Extracts static facts to build a floor hierarchy and compute distances between floors.
    - Maps each passenger to their origin and destination floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Static Information**: Parse the static facts to determine the hierarchy of floors and compute the distance between any two floors.
    2. **Identify Passenger Goals**: For each passenger, determine their origin and destination floors.
    3. **Check Current State**: For each passenger, check if they are already served. If not, determine if they are boarded or need to be boarded.
    4. **Calculate Required Actions**:
       - For passengers not yet boarded, calculate the number of moves needed for the lift to reach their origin floor.
       - Once boarded, calculate the number of moves needed to reach their destination floor.
       - Sum the boarding, moving, and departing actions for all passengers.
    5. **Handle Special Cases**: If the lift is already at the correct floor or if passengers are already served, adjust the action count accordingly.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal information."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor hierarchy and compute distances
        self.floor_above = {}
        self.floor_distance = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                lower_floor = parts[1]
                upper_floor = parts[2]
                if lower_floor not in self.floor_above:
                    self.floor_above[lower_floor] = []
                self.floor_above[lower_floor].append(upper_floor)

        # Precompute floor distances
        self.distances = {}
        for floor in self.floor_above:
            self.distances[floor] = 1
            for upper in self.floor_above[floor]:
                self.distances[upper] = self.distances[floor] + 1

        # Map each passenger to their origin and destination
        self.passenger_info = {}
        for goal in self.goals:
            if match(goal, "destin", "*", "*"):
                passenger = get_parts(goal)[1]
                dest_floor = get_parts(goal)[2]
                self.passenger_info[passenger] = (None, dest_floor)

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                passenger = get_parts(fact)[1]
                origin_floor = get_parts(fact)[2]
                if passenger in self.passenger_info:
                    self.passenger_info[passenger] = (origin_floor, self.passenger_info[passenger][1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        def get_parts(fact):
            return fact[1:-1].split()

        def match(fact, *args):
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Track where passengers are and whether they are served
        passengers = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)
                passengers[p] = (f, None)
            elif match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)
                if p in passengers:
                    passengers[p] = (passengers[p][0], f)
            elif match(fact, "served", "*"):
                p = get_parts(fact)[1]
                passengers[p] = (passengers[p][0], passengers[p][1], True)

        total_actions = 0
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]

        for p in self.passenger_info:
            if p not in passengers:
                continue
            origin, dest, served = passengers[p]
            if served:
                continue

            if origin is None or dest is None:
                continue

            if current_lift_floor is None:
                continue

            if origin == current_lift_floor:
                total_actions += 1  # Board action
            else:
                if origin > current_lift_floor:
                    # Need to move up
                    distance = self.distances[origin] - self.distances[current_lift_floor]
                else:
                    # Need to move down
                    distance = self.distances[current_lift_floor] - self.distances[origin]
                total_actions += distance  # Move actions
                total_actions += 1  # Board action

            if dest == origin:
                continue  # Already at destination
            if dest == current_lift_floor:
                total_actions += 1  # Depart action
            else:
                if dest > current_lift_floor:
                    # Need to move up
                    distance = self.distances[dest] - self.distances[current_lift_floor]
                else:
                    # Need to move down
                    distance = self.distances[current_lift_floor] - self.distances[dest]
                total_actions += distance  # Move actions
                total_actions += 1  # Depart action

        return total_actions
