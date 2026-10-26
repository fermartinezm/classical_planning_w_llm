from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed for the lift to serve all passengers by moving to their origin and destination floors.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must be boarded at their origin floor and served at their destination floor.
    - The lift can only serve one passenger at a time.

    # Heuristic Initialization
    - Extract static facts to determine floor relationships.
    - Extract goal conditions to identify destination floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. For each passenger:
       - If already served, do nothing.
       - If boarded but not served, calculate the distance to their destination.
       - If not boarded, calculate the distance to their origin.
    3. Determine the furthest origin floor the lift needs to visit.
    4. Determine the furthest destination floor the lift needs to visit.
    5. Calculate the total number of moves required:
       - Moves to reach the furthest origin.
       - Moves between origins and destinations.
       - Moves to reach the furthest destination.
    6. Sum the required board, depart, and move actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract floor relationships
        self.above = {}
        for fact in static_facts:
            if fnmatch(fact, '(above * * *)'):
                parts = fact[1:-1].split()
                floor1, floor2 = parts[1], parts[2]
                if floor1 not in self.above:
                    self.above[floor1] = []
                self.above[floor1].append(floor2)

        # Build a map from each floor to all floors above it
        self.floor_map = {}
        for fact in static_facts:
            if fnmatch(fact, '(above * * *)'):
                f1, f2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if f1 not in self.floor_map:
                    self.floor_map[f1] = set()
                self.floor_map[f1].add(f2)

        # Precompute all floors for easy access
        self.floors = set()
        for fact in static_facts:
            if fnmatch(fact, '(above * * *)'):
                self.floors.add(fact[1:-1].split()[1])
                self.floors.add(fact[1:-1].split()[2])

    def __call__(self, node):
        """Estimate the minimal number of actions needed to serve all passengers."""
        state = node.state

        # Function to extract components from a fact string
        def get_parts(fact):
            return fact[1:-1].split()

        # Check if all passengers are already served
        all_served = True
        for goal in self.goals:
            if fnmatch(goal, '(served *)'):
                all_served = all_served and (goal in state)
        if all_served:
            return 0

        # Extract current lift position
        lift_pos = None
        for fact in state:
            if fnmatch(fact, '(lift-at *)'):
                lift_pos = get_parts(fact)[1]
                break

        # Extract passenger information
        passengers = {}
        origins = {}
        destinations = {}
        boarded = set()
        served = set()

        for fact in state:
            if fnmatch(fact, '(origin * *)'):
                p, f = get_parts(fact)
                origins[p] = f
            elif fnmatch(fact, '(destin * *)'):
                p, f = get_parts(fact)
                destinations[p] = f
            elif fnmatch(fact, '(boarded *)'):
                p = get_parts(fact)[1]
                boarded.add(p)
            elif fnmatch(fact, '(served *)'):
                p = get_parts(fact)[1]
                served.add(p)

        # List of passengers that need attention
        to_serve = []
        for p in origins:
            if p not in served and p not in boarded:
                to_serve.append(('board', origins[p]))
            elif p not in served:
                to_serve.append(('serve', destinations[p]))

        if not to_serve:
            return 0

        # Function to calculate the minimal moves between floors
        def get_distance(f1, f2):
            floors = sorted(self.floors, key=lambda x: int(x[1:]))
            f1_idx = floors.index(f1)
            f2_idx = floors.index(f2)
            return abs(f2_idx - f1_idx)

        # Determine the furthest origin and destination
        furthest_origin = None
        max_origin_dist = -1
        furthest_dest = None
        max_dest_dist = -1

        for action, floor in to_serve:
            if action == 'board':
                if floor not in self.floor_map.get(lift_pos, []):
                    continue
                dist = get_distance(lift_pos, floor)
                if dist > max_origin_dist:
                    max_origin_dist = dist
                    furthest_origin = floor
            else:
                if floor not in self.floor_map.get(lift_pos, []):
                    continue
                dist = get_distance(lift_pos, floor)
                if dist > max_dest_dist:
                    max_dest_dist = dist
                    furthest_dest = floor

        # Calculate total moves
        total_moves = 0
        if furthest_origin:
            total_moves += max_origin_dist
            if furthest_dest and furthest_dest != furthest_origin:
                total_moves += get_distance(furthest_origin, furthest_dest)
                total_moves += max_dest_dist
            else:
                total_moves += max_dest_dist
        elif furthest_dest:
            total_moves += max_dest_dist

        # Each move action counts as one, each board/depart as one
        num_actions = total_moves + len(to_serve)

        return num_actions
