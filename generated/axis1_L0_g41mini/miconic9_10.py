from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded p1)" or "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary:
    --------
    This heuristic estimates the number of actions required to serve all remaining passengers in the Miconic elevator domain.
    It computes an approximate minimum cost in terms of elevator moves, boardings, and departures needed to bring all passengers
    to their destination floors, starting from the current lift position.

    Assumptions:
    ------------
    - The elevator can move only one floor up or down per action, moves are counted as single actions.
    - Each passenger must be boarded at their origin floor, then departed at their destination floor.
    - Passengers that are already served add zero cost.
    - Passengers that are boarded but not yet served require only the transport to the destination floor and departure.
    - The heuristic is not necessarily admissible but aims to be accurate and efficiently computable.
    - The "above" relation is used to determine floors order and to compute floor distances.
    - The elevator is always on exactly one floor.
    - We ignore optimization of multiple passengers served simultaneously and assume sequential boarding/departure per passenger.

    Heuristic Initialization:
    -------------------------
    - Extract all floors and build a floor-order graph using "above" facts to compute distances between floors.
    - Extract the goal served facts for passengers.
    - Extract static passenger origin and destination facts.
    - Store floors in a list and create a mapping for quick floor-to-floor shortest distance computation.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify which passengers are already served - they require no further actions.
    2. Identify which passengers are currently boarded.
    3. Obtain the current lift floor.
    4. For each passenger:
       - If served: cost = 0.
       - If boarded: cost = distance from current lift floor to passenger destination + 1 (depart action).
       - If not boarded: 
         cost = distance from current lift floor to origin floor 
              + 1 (board action) 
              + distance from origin floor to destination floor 
              + 1 (depart action).
    5. Since the lift can only be on one floor at a time, the heuristic sums:
       - The minimum moves to get to the first needed boarding or destination floor from current lift location,
       - Plus the sum of boarding and departure actions for all passengers,
       - Plus an approximation of the move cost between passengers assuming elevator moves are serialized and at minimum required.
       
       To keep it efficient, we approximate the cost by:
       - Summing all board + depart actions (2 actions per unserved passenger),
       - Adding the max of distances among floors involved (lift floor, all origin floors of unboarded passengers, all destination floors),
         which approximates the minimal move cost for the elevator travelling through all floors needed.
       
       This heuristic avoids expensive path computations and provides a practical guidance for greedy search.

    Note:
    -----
    For the "above" relation, it is assumed floors form a linear order. Distances are computed as
    the number of moves up or down needed to go between floors.
    """

    def __init__(self, task):
        # Save goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Extract floors from static facts
        floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above":
                floors.add(parts[1])
                floors.add(parts[2])
        self.floors = sorted(floors)

        # Create a map floor -> index for easy numeric distance calculation
        self.floor_indices = {floor: idx for idx, floor in enumerate(self.floors)}

        # Build a floor distance function via index difference because floors form a linear order
        # (based on "above" it is safe to assume floors in sorted order)
        def floor_distance(f1, f2):
            return abs(self.floor_indices[f1] - self.floor_indices[f2])

        self.floor_distance = floor_distance

        # Extract origin and destination mapping for passengers from static facts
        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "origin":
                _, p, f = parts
                self.passenger_origins[p] = f
            elif parts[0] == "destin":
                _, p, f = parts
                self.passenger_destinations[p] = f

        # Extract all passengers known in static (from origin/destin)
        self.passengers = set(self.passenger_origins.keys()) | set(self.passenger_destinations.keys())

    def __call__(self, node):
        state = node.state

        if self.goals <= state:
            # Goal reached -> cost 0
            return 0

        # Identify passengers in various states
        served_passengers = set()
        boarded_passengers = set()
        for fact in state:
            if fact.startswith("(served "):
                parts = get_parts(fact)
                served_passengers.add(parts[1])
            elif fact.startswith("(boarded "):
                parts = get_parts(fact)
                boarded_passengers.add(parts[1])

        # Passengers not served and not boarded still waiting at origin
        waiting_passengers = self.passengers - served_passengers - boarded_passengers

        # Identify current elevator floor from state: "(lift-at fX)"
        lift_floor = None
        for fact in state:
            if fact.startswith("(lift-at "):
                parts = get_parts(fact)
                lift_floor = parts[1]
                break
        if lift_floor is None:
            # Elevator position unknown (unlikely but handle gracefully)
            # We assign some floor arbitrarily (lowest floor) to prevent failure
            lift_floor = self.floors[0]

        fd = self.floor_distance

        # Actions count: board and depart are always exactly 1 action each
        board_cost_per_passenger = 1
        depart_cost_per_passenger = 1

        # Calculate cost estimate

        # For boarded passengers:
        # cost = move from lift floor to destination + 1 depart
        boarded_costs = []
        for p in boarded_passengers:
            dest_floor = self.passenger_destinations[p]
            dist = fd(lift_floor, dest_floor)
            # The lift will move from its current floor to the destination floor AND depart
            boarded_costs.append(dist + depart_cost_per_passenger)

        # For waiting passengers:
        # cost = move from lift floor to origin + 1 board + move origin to destination + 1 depart
        waiting_costs = []
        for p in waiting_passengers:
            origin_floor = self.passenger_origins[p]
            dest_floor = self.passenger_destinations[p]
            dist_lift_to_origin = fd(lift_floor, origin_floor)
            dist_origin_to_dest = fd(origin_floor, dest_floor)
            cost = dist_lift_to_origin + board_cost_per_passenger + dist_origin_to_dest + depart_cost_per_passenger
            waiting_costs.append(cost)

        # Total count of board and depart actions remains fixed:
        total_board_depart = (len(waiting_passengers) + len(boarded_passengers)) * (board_cost_per_passenger + depart_cost_per_passenger)

        # Approximate elevator moves needed:
        # Floors involved to visit:
        floors_to_visit = set()
        # Include current lift floor
        floors_to_visit.add(lift_floor)
        # Include origin floors of waiting passengers
        for p in waiting_passengers:
            floors_to_visit.add(self.passenger_origins[p])
        # Include destination floors of boarded passengers and waiting passengers
        for p in boarded_passengers:
            floors_to_visit.add(self.passenger_destinations[p])
        for p in waiting_passengers:
            floors_to_visit.add(self.passenger_destinations[p])

        # If only the current lift floor is in set, moves = 0
        if len(floors_to_visit) <= 1:
            approx_move_cost = 0
        else:
            # Movement cost approximated by max distance between floors in floors_to_visit,
            # treating floors as points on a line (the elevator must travel at least this distance)
            # This heuristic tends to be optimistic, but is efficient to compute.
            indices = [self.floor_indices[floor] for floor in floors_to_visit]
            approx_move_cost = max(indices) - min(indices)

        # The total heuristic: total board+depart cost + approximate move cost minus double counting handled:
        # Since board/depart included, and distances from lift to origins/destinations included,
        # Use the bigger of approx_move_cost or max individual distances to avoid underestimating moves.

        # Max distance among all segments considered in waiting_costs and boarded_costs to avoid underestimating
        max_individual_distance = 0
        if boarded_costs:
            max_individual_distance = max(max_individual_distance, max(c - depart_cost_per_passenger for c in boarded_costs))
        if waiting_costs:
            max_distances = []
            for p in waiting_passengers:
                dist_1 = fd(lift_floor, self.passenger_origins[p])
                dist_2 = fd(self.passenger_origins[p], self.passenger_destinations[p])
                max_distances.append(dist_1 + dist_2)
            max_individual_distance = max(max_individual_distance, max(max_distances))

        # Use max between computed distances
        move_component = max(approx_move_cost, max_individual_distance)

        heuristic_value = total_board_depart + move_component

        return heuristic_value
