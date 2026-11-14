from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(origin p1 f2)" -> ["origin", "p1", "f2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the required number of actions to reach a goal state
    in the Miconic elevator domain, where passengers must be boarded at their origin floors,
    transported via an elevator (lift) and served at their destination floors.
    The heuristic guides a greedy best-first search by providing an estimate of how many 
    actions (board, depart, and lift moves) remain until all passengers are served.

    Assumptions:
    - Each passenger must be boarded at their origin floor before departing at their destination floor.
    - The elevator is at exactly one floor at a time.
    - A lift move (up or down) changes the elevator's floor by one step (floor adjacency defined by "above").
    - Moving between non-adjacent floors requires multiple moves.
    - Serving the passenger requires boarding and departing actions.
    - Multiple passengers can be boarded at the same floor in any order without extra move cost.
    - The heuristic is not necessarily admissible but aims to minimize node expansions.

    Heuristic Initialization:
    - Extract the list of floors and establish floor adjacency and relative ordering from static facts "(above fX fY)".
    - Compute and store floor distances (shortest number of lift moves) between every pair of floors.
    - Extract for each passenger the origin floor and destination floor (from static facts "origin" and "destin").
    - Initialize necessary internal maps for quick lookup:
       * floor_distances: dict of dict with minimal floor moves distances
       * origins and destinations per passenger

    Step-By-Step Thinking for Computing Heuristic:
    Given a state:
    1. Identify the elevator's current floor.
    2. Identify which passengers are:
       - not yet boarded (have an "origin" fact),
       - currently boarded ("boarded"),
       - already served ("served").
    3. For passengers not served:
       - For those not boarded, the elevator must visit their origin floor, perform a "board" action.
       - For boarded passengers, the elevator must move to their destination floor and perform a "depart" action.
    4. Compute the minimal elevator travel distance (in moves) to cover all needed boarding and departing floors in some order, considering that each floor visited can fulfill multiple actions for passengers at that floor.
       Since exact TSP is costly, we approximate by summing:
       - The minimum distance from current elevator floor to the closest relevant floor (origin or destination)
       - Plus the sum of minimal distances needed for all distinct floors where boarding or departing must happen (i.e., count number of distinct floors to visit)
         multiplied by an approximate average move cost (e.g., 1 move per intermediate floor)
       This approximation leverages that elevators must travel between floors anyhow.
    5. Add the count of board and depart actions still required.
    6. Return the sum as the heuristic estimate.

    This heuristic balances computational efficiency and informativeness by combining 
    precomputed floor distances with straightforward counting of needed passenger actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic:
        - Compute a floor distance matrix from static facts.
        - Collect origin and destination floors per passenger.
        """
        self.goals = task.goals
        self.static = task.static

        # Collect all floors and passengers reported in static facts
        floors = set()
        self.passengers = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above":  # adjacency info
                floors.add(parts[1])
                floors.add(parts[2])
            elif parts[0] == "destin" or parts[0] == "origin":
                self.passengers.add(parts[1])
                floors.add(parts[2])

        self.floors = sorted(floors)  # sorted list of floors for indexing

        # Build adjacency graph of floors using above facts (directed graph: f1 above f2 means f1 > f2)
        # We'll build an undirected adjacency to allow calculating minimal moves up/down
        adjacency = {floor: set() for floor in self.floors}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above":
                f1, f2 = parts[1], parts[2]
                adjacency[f1].add(f2)
                adjacency[f2].add(f1)  # also add reverse edge for bidirectional moves

        # Precompute shortest distances between floors using BFS
        self.floor_dist = {f: {} for f in self.floors}
        for f in self.floors:
            # BFS from floor f
            visited = {f: 0}
            queue = [f]
            while queue:
                current = queue.pop(0)
                dist = visited[current]
                for neigh in adjacency[current]:
                    if neigh not in visited:
                        visited[neigh] = dist + 1
                        queue.append(neigh)
            self.floor_dist[f] = visited

        # Extract origin and destination floors for each passenger from static facts:
        # origin_map: passenger -> floor
        # destination_map: passenger -> floor
        self.origin_map = {}
        self.destination_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger, floor = parts[1], parts[2]
                self.origin_map[passenger] = floor
            elif parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.destination_map[passenger] = floor

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        Returns 0 if the state satisfies all goal conditions (all passengers served).
        Otherwise, estimates the minimal number of moves, board, and depart actions
        needed to serve all passengers.
        """

        state = node.state

        # Check if goal reached
        if self.goals <= state:
            return 0  # no actions needed

        # Find current lift floor (there must be exactly one lift-at fact)
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                lift_floor = parts[1]
                break
        if lift_floor is None:
            # Defensive fallback - no lift floor found => heuristic high
            return 10000

        # Passengers served, boarded, or waiting to board
        served_passengers = {get_parts(f)[1] for f in state if get_parts(f)[0] == "served"}
        boarded_passengers = {get_parts(f)[1] for f in state if get_parts(f)[0] == "boarded"}

        # Passengers not yet served
        not_served = self.passengers - served_passengers

        # For passengers that still need attention:
        # Identify passengers to board and to depart
        to_board = set()
        to_depart = set()
        # A passenger is to board if:
        #  - not served,
        #  - and not boarded
        # A passenger is to depart if:
        #  - boarded
        #  - not served
        for p in not_served:
            if p in boarded_passengers:
                to_depart.add(p)
            else:
                to_board.add(p)

        # Floors where board and depart actions must happen
        boarding_floors = {self.origin_map[p] for p in to_board}
        departing_floors = {self.destination_map[p] for p in to_depart}

        # Floors that must be visited to perform board or depart actions
        floors_to_visit = boarding_floors.union(departing_floors)

        if not floors_to_visit:
            # No more boarding or departing needed but goal not reached: heuristic 0 for safety
            return 0

        # Calculate minimal elevator travel estimate to cover all needed floors
        # Strategy:
        # 1) Distance from current lift floor to closest needed floor
        distance_to_closest = min(
            self.floor_dist[lift_floor][f] for f in floors_to_visit if f in self.floor_dist[lift_floor]
        )

        # 2) Number of distinct floors to visit for actions (board and depart),
        #    visiting each floor can be approximated as cost 1 between floors
        #    (real cost is captured by floor distances, but exact TSP is costly)
        num_floors_to_visit = len(floors_to_visit)

        # A simple approximation for elevator travel cost:
        # minimal distance to closest needed floor + (number_of_floors_to_visit - 1) moves
        # (assuming we can order visits optimally)
        elevator_moves_estimate = distance_to_closest + (num_floors_to_visit - 1)

        # Count total passenger actions left: each passenger to board and/or depart needs 1 action each
        num_board_actions = len(to_board)
        num_depart_actions = len(to_depart)
        total_actions = num_board_actions + num_depart_actions

        # Sum up total heuristic estimate
        heuristic_value = elevator_moves_estimate + total_actions

        # Defensive floor distances might sometimes be incomplete (if a floor missing in dist)
        # In that rare case, clamp to a large number so heuristic is finite but large
        if heuristic_value < 0 or heuristic_value is None:
            heuristic_value = 10000

        return heuristic_value
