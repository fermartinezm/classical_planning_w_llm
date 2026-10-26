from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all passengers
    from their origin floors to their destination floors using the lift.

    # Assumptions:
    - The lift can move one floor at a time using the up and down actions.
    - Each passenger requires one board and one depart action.
    - Multiple passengers on the same origin floor can be handled in a single trip.

    # Heuristic Initialization
    - Extracts static facts to determine floor hierarchy.
    - Tracks current state of passengers and the lift.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger, calculate the number of floors the lift needs to move to reach their origin floor.
    2. Add the boarding and departing actions for each passenger.
    3. If multiple passengers are on the same origin floor, batch their transportation to minimize steps.
    4. Subtract any steps saved if the lift is already at the required origin floor.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor hierarchy from static facts
        self.floor_above = {}
        for fact in static_facts:
            if fnmatch(fact, '(above * * *)'):
                parts = fact[1:-1].split()
                floor1, floor2 = parts[1], parts[2]
                if floor1 not in self.floor_above:
                    self.floor_above[floor1] = []
                self.floor_above[floor1].append(floor2)

        # Precompute floor distances using BFS
        self.floor_distance = {}
        for fact in static_facts:
            if fnmatch(fact, '(above * * *)'):
                start = fact[1:-1].split()[1]
                self._compute_distances(start)

    def _compute_distances(self, start_floor):
        """Compute distances from start_floor to all other floors using BFS."""
        visited = {start_floor: 0}
        queue = [start_floor]

        while queue:
            current = queue.pop(0)
            if current in self.floor_above:
                for floor in self.floor_above[current]:
                    if floor not in visited:
                        visited[floor] = visited[current] + 1
                        queue.append(floor)
        for floor, dist in visited.items():
            self.floor_distance[(start_floor, floor)] = dist

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state
        current_floors = {}
        passengers = {}

        # Extract current state information
        for fact in state:
            if fnmatch(fact, '(lift-at *)'):
                current_floors['lift'] = fact[1:-1].split()[1]
            if fnmatch(fact, '(origin * *)'):
                p, f = fact[1:-1].split()[1], fact[1:-1].split()[2]
                passengers[p] = {'origin': f, 'destin': None, 'served': False}
            if fnmatch(fact, '(destin * *)'):
                p, f = fact[1:-1].split()[1], fact[1:-1].split()[2]
                passengers[p]['destin'] = f
            if fnmatch(fact, '(served *)'):
                p = fact[1:-1].split()[1]
                passengers[p]['served'] = True

        # Calculate required actions
        total_actions = 0
        origin_count = {}

        for p, data in passengers.items():
            if data['served']:
                continue
            origin = data['origin']
            destin = data['destin']
            if origin == destin:
                continue

            # Calculate floor distance
            if origin not in self.floor_distance or destin not in self.floor_distance[origin]:
                distance = 0
            else:
                distance = self.floor_distance[(origin, destin)]

            # Add actions for moving to origin, boarding, moving to destin, and departing
            total_actions += distance + 2  # Move to origin, board, move to destin, depart

            # Track how many passengers are on each origin floor
            if origin in origin_count:
                origin_count[origin] += 1
            else:
                origin_count[origin] = 1

        # Subtract redundant moves if multiple passengers on the same origin floor
        for count in origin_count.values():
            if count > 1:
                total_actions -= (count - 1) * 2  # Save one move and one reverse move per additional passenger

        # If the lift is already at an origin floor, save the distance
        if 'lift' in current_floors:
            for p, data in passengers.items():
                if data['served']:
                    continue
                origin = data['origin']
                if (origin, origin) in self.floor_distance:
                    distance = self.floor_distance[(origin, origin)]
                else:
                    distance = 0
                if current_floors['lift'] == origin:
                    total_actions -= distance

        return total_actions
