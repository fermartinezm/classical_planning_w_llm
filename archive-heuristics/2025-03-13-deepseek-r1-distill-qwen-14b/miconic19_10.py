from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by moving the lift to their destination floors.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger requires two actions: boarding and departing.
    - The heuristic assumes the optimal order of operations to minimize the number of actions.

    # Heuristic Initialization
    - Extracts static information about floor hierarchies to compute distances efficiently.
    - Processes the current state to determine the status of each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current floor of the lift.
    2. For each passenger, determine if they are served, boarded, their origin, and destination.
    3. Identify unserved passengers and separate them into two groups:
       - Group 1: Passengers not boarded (need to be boarded).
       - Group 2: Passengers boarded but not served (need to be departed).
    4. For Group 1, find the farthest origin from the current lift floor.
    5. For Group 2, find the farthest destination from the current lift floor.
    6. Calculate the movement cost as the sum of distances from the current floor to the farthest origin and from there to the farthest destination.
    7. The total heuristic value is the movement cost plus twice the number of unserved passengers (boarding and departing actions).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about floor hierarchies."""
        # Build parent and depth maps for floors based on static facts
        self.parent = {}
        self.depth = {}

        for fact in task.static:
            if fact.startswith('(above '):
                parts = fact[1:-1].split()
                f1, f2 = parts[1], parts[2]
                self.parent[f2] = f1  # f2 is above f1

        # Find top floor (the one with no parent)
        top_floors = [f for f in self.parent.keys() if f not in self.parent.values()]
        if top_floors:
            self.top_floor = top_floors[0]
        else:
            self.top_floor = None

        # BFS to assign depths
        from collections import deque
        queue = deque()
        if self.top_floor:
            queue.append(self.top_floor)
            self.depth[self.top_floor] = 0

        while queue:
            current = queue.popleft()
            for child in [f for f, p in self.parent.items() if p == current]:
                self.depth[child] = self.depth[current] + 1
                queue.append(child)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current lift floor
        current_lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_lift_floor = fact[1:-1].split()[1]
                break

        if not current_lift_floor:
            return 0  # No lift position found, should not happen in valid state

        # Extract passenger information from the current state
        passengers = {}
        for fact in state:
            if fact.startswith('(boarded '):
                p = fact[1:-1].split()[1]
                passengers[p] = passengers.get(p, {})
                passengers[p]['boarded'] = True
            elif fact.startswith('(served '):
                p = fact[1:-1].split()[1]
                passengers[p] = passengers.get(p, {})
                passengers[p]['served'] = True
            elif fact.startswith('(origin '):
                p, f = fact[1:-1].split()[1], fact[1:-1].split()[2]
                passengers[p] = passengers.get(p, {})
                passengers[p]['origin'] = f
            elif fact.startswith('(destin '):
                p, f = fact[1:-1].split()[1], fact[1:-1].split()[2]
                passengers[p] = passengers.get(p, {})
                passengers[p]['destination'] = f

        # Identify unserved passengers and categorize them
        unserved = []
        group1_origins = []  # Passengers not boarded
        group2_destinations = []  # Passengers boarded but not served

        for p in passengers:
            if 'served' not in passengers[p] or not passengers[p]['served']:
                unserved.append(p)
                if 'boarded' not in passengers[p] or not passengers[p]['boarded']:
                    group1_origins.append(passengers[p]['origin'])
                else:
                    group2_destinations.append(passengers[p]['destination'])

        num_unserved = len(unserved)
        if num_unserved == 0:
            return 0  # Goal state

        # Determine farthest origin and destination
        farthest_origin = None
        if group1_origins:
            farthest_origin = max(group1_origins, key=lambda x: self.depth[x])

        farthest_destination = None
        if group2_destinations:
            farthest_destination = max(group2_destinations, key=lambda x: self.depth[x])

        # Calculate movement cost
        movement_cost = 0
        if farthest_origin and farthest_destination:
            distance_current_origin = abs(self.depth[current_lift_floor] - self.depth[farthest_origin])
            distance_origin_destination = abs(self.depth[farthest_origin] - self.depth[farthest_destination])
            movement_cost = distance_current_origin + distance_origin_destination
        elif farthest_origin:
            movement_cost = abs(self.depth[current_lift_floor] - self.depth[farthest_origin])
        elif farthest_destination:
            movement_cost = abs(self.depth[current_lift_floor] - self.depth[farthest_destination])
        else:
            movement_cost = 0

        # Total actions: movement cost plus boarding and departing for each unserved passenger
        total_actions = movement_cost + 2 * num_unserved

        return total_actions
