from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal number of lift movements and service actions required.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger requires exactly two actions: boarding and departing.
    - The heuristic calculates the minimal number of lift movements based on the current state.

    # Heuristic Initialization
    - Extract the static 'above' facts to build a floor hierarchy and compute the depth of each floor.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current lift floor from the state.
    2. For each passenger, determine their origin and destination floors.
    3. Calculate the distance from the current lift floor to the passenger's origin.
    4. Calculate the distance from the origin to the destination.
    5. Sum these distances for all passengers.
    6. Add two actions per passenger (boarding and departing).
    7. The total is the estimated number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about floor hierarchies."""
        self.task = task
        static_facts = task.static

        # Build parent map for floors based on 'above' facts
        self.parent = {}
        for fact in static_facts:
            if fact.startswith('(above'):
                parts = fact[1:-1].split()
                if len(parts) == 3 and parts[0] == 'above':
                    floor1 = parts[1]
                    floor2 = parts[2]
                    self.parent[floor2] = floor1

        # Collect all floors mentioned in static facts
        floors = set()
        for fact in static_facts:
            if fact.startswith('(above'):
                parts = fact[1:-1].split()
                if len(parts) == 3 and parts[0] == 'above':
                    floors.add(parts[1])
                    floors.add(parts[2])
        # Also collect floors from other static facts if any
        for fact in static_facts:
            if fact.startswith('(lift-at ') or fact.startswith('(origin ') or fact.startswith('(destin '):
                parts = fact[1:-1].split()
                if len(parts) >= 2:
                    floor = parts[1]
                    floors.add(floor)
        # Compute depth for each floor
        self.depth = {}
        for floor in floors:
            current = floor
            d = 0
            while current in self.parent:
                current = self.parent[current]
                d += 1
            self.depth[floor] = d

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all passengers."""
        state = node.state

        # Extract current lift floor
        lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact[1:-1].split()
                if len(parts) == 2 and parts[0] == 'lift-at':
                    lift_floor = parts[1]
                    break
        if lift_floor is None:
            return 0  # Should not happen in valid states

        # Extract passengers' origin and destination
        passengers = {}
        for fact in state:
            if fact.startswith('(origin '):
                parts = fact[1:-1].split()
                if len(parts) == 3 and parts[0] == 'origin':
                    p = parts[1]
                    origin = parts[2]
                    passengers[p] = {'origin': origin, 'destin': None}
            if fact.startswith('(destin '):
                parts = fact[1:-1].split()
                if len(parts) == 3 and parts[0] == 'destin':
                    p = parts[1]
                    destin = parts[2]
                    passengers[p]['destin'] = destin

        total_actions = 0
        num_passengers = 0
        for p, data in passengers.items():
            origin = data['origin']
            destin = data['destin']
            # Distance from lift to origin
            d_lift_origin = abs(self.depth[lift_floor] - self.depth[origin])
            # Distance from origin to destination
            d_origin_destin = abs(self.depth[origin] - self.depth[destin])
            total_actions += d_lift_origin + d_origin_destin
            num_passengers += 1
        # Add boarding and departing actions
        total_actions += 2 * num_passengers

        return total_actions
