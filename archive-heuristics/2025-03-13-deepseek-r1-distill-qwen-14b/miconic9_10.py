from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal distance the lift needs to travel to reach each passenger's origin and then their destination, plus the necessary boarding and departing actions.

    # Assumptions:
    - The lift can move one floor at a time.
    - Each boarding and departing action counts as one action per passenger.
    - Passengers are served one at a time.

    # Heuristic Initialization
    - Extract static facts to determine the hierarchy and levels of floors.
    - Build a mapping from each floor to its level.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current floor of the lift.
    2. For each unserved passenger, determine their origin and destination.
    3. Calculate the distance from the lift's current floor to the passenger's origin.
    4. Calculate the distance from the passenger's origin to their destination.
    5. Sum all these distances for all unserved passengers.
    6. Add two actions for each unserved passenger (boarding and departing).
    7. The total heuristic value is the sum of the distances and the actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about floor hierarchy."""
        self.goals = task.goals
        static_facts = task.static

        # Build parent and level dictionaries
        self.parent = {}
        self.level = {}

        # Collect all floors from 'above' facts
        floors = set()
        for fact in static_facts:
            if fact.startswith('(above'):
                parts = fact[1:-1].split()
                above_floor = parts[1]
                below_floor = parts[2]
                self.parent[below_floor] = above_floor
                floors.add(above_floor)
                floors.add(below_floor)

        # Find the bottom floor (has no parent)
        bottom_floor = None
        for floor in floors:
            if floor not in self.parent:
                bottom_floor = floor
                break

        # Assign levels starting from the bottom floor
        if bottom_floor is not None:
            queue = [bottom_floor]
            self.level[bottom_floor] = 0
            while queue:
                current = queue.pop(0)
                if current in self.parent:
                    parent = self.parent[current]
                    self.level[parent] = self.level[current] + 1
                    queue.append(parent)

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state

        # Extract current lift floor
        lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at'):
                parts = fact[1:-1].split()
                lift_floor = parts[1]
                break

        if lift_floor is None:
            return 0  # If lift floor is not found, assume 0 cost

        # Extract unserved passengers
        unserved_passengers = []
        passenger Origins = {}
        passenger Destinations = {}
        for fact in state:
            if fact.startswith('(origin'):
                parts = fact[1:-1].split()
                p, origin = parts[1], parts[2]
                passenger_Origins[p] = origin
            elif fact.startswith('(destin'):
                parts = fact[1:-1].split()
                p, dest = parts[1], parts[2]
                passenger_Destinations[p] = dest

        for p in passenger_Origins:
            if not any(fact.startswith('(served ' + p + ')') for fact in state):
                origin = passenger_Origins[p]
                dest = passenger_Destinations.get(p, None)
                if dest is not None:
                    unserved_passengers.append((p, origin, dest))

        # Calculate heuristic
        total_distance = 0
        num_passengers = 0
        for p, origin, dest in unserved_passengers:
            # Distance from lift to origin
            if origin in self.level and lift_floor in self.level:
                distance_lift_to_origin = abs(self.level[origin] - self.level[lift_floor])
            else:
                distance_lift_to_origin = 0
            # Distance from origin to destination
            if origin in self.level and dest in self.level:
                distance_origin_to_dest = abs(self.level[dest] - self.level[origin])
            else:
                distance_origin_to_dest = 0
            total_distance += distance_lift_to_origin + distance_origin_to_dest
            num_passengers += 1

        # Add boarding and departing actions
        total_actions = total_distance + 2 * num_passengers

        return total_actions
