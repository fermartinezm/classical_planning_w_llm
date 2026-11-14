from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded p1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the number of actions required for the elevator to serve all remaining passengers,
    assuming the elevator must pick up each passenger at their origin floor and then deliver them to their destination floor.
    The heuristic is not guaranteed to be admissible but aims to be an informed and efficiently computable estimate for guiding greedy best-first search.

    Assumptions:
    - Each passenger must be boarded at their origin floor and then departed at their destination floor.
    - The elevator can carry multiple passengers simultaneously, but boarding and departing each passenger require separate actions.
    - Moving the elevator between floors takes one action per floor movement.
    - The "above" predicate denotes a strict total ordering of floors from lowest to highest.
    - The heuristic only counts estimated necessary boarding, departing, and elevator movements based on current passenger locations.
    - Lift can only move one floor per move action (up or down).

    Heuristic Initialization:
    - Parse static facts to extract:
      - Set of floors and their relative vertical ordering.
      - For each floor, compute its absolute floor height number (0-based), by building a total order from the "above" facts.
      - Record for each passenger their fixed destination floor (from static facts).
    - Floors are ordered numerically from lowest (0) to highest (max_floor) to facilitate floor distance computations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current floor where the elevator is located.
    2. Determine three passenger sets:
       - Not yet boarded passengers (with known origin floors).
       - Currently boarded passengers (on elevator but not served yet).
       - Already served passengers do not affect the heuristic.
    3. For passengers not yet boarded, estimate actions:
       - One board action per passenger.
       - Elevator movement cost to get to their origin floors (assumed optimistically combined).
    4. For passengers boarded, estimate actions:
       - One depart action per boarded passenger.
    5. Calculate the minimal number of elevator moves estimated as:
       - The cost of moving to cover all relevant floors: 
         floors where boarding is needed plus floors where departing is needed plus current floor,
       - Using the minimal "span" of floors that the elevator must visit (max_floor - min_floor),
       - Plus the distance from the current floor to that range.
    6. Sum all boarding and departing actions (one per passenger respectively) plus the estimated elevator moves.
    7. If the goal is reached (all served), heuristic value is zero.

    Note:
    This heuristic does not consider complex routing or carrying capacity constraints but balances accuracy and efficiency.
    """

    def __init__(self, task):
        # Extract floor ordering and map floors to integers for numerical distance
        self.static = task.static
        self.goals = task.goals

        # Extract floors set
        self.floors = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                self.floors.add(parts[1])
                self.floors.add(parts[2])
        # Also floors mentioned in destin facts (in case "above" is incomplete)
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                self.floors.add(get_parts(fact)[2])
        for fact in self.static:
            if match(fact, "origin", "*", "*"):
                self.floors.add(get_parts(fact)[2])

        # Build a graph of floor orderings (directed edges: above = floor1 above floor2)
        above_graph = {floor: set() for floor in self.floors}
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                floor1, floor2 = parts[1], parts[2]
                # floor1 is above floor2, so edge from floor1 -> floor2
                above_graph[floor1].add(floor2)

        # We want a total order of floors to assign integer heights,
        # i.e., floor f1 < f2 if f1 is below f2.
        # We can invert the relation: floor A is below floor B if B is above A.
        # Build below relations by reversing above edges:
        below_graph = {floor: set() for floor in self.floors}
        for floor, above_set in above_graph.items():
            for f2 in above_set:
                below_graph[f2].add(floor)

        # Floors with no floor below them are the lowest floors.
        # We topologically sort floors by below relations to get increasing floor order.

        # Kahn's algorithm for topological sorting:
        in_degree = {floor: 0 for floor in self.floors}
        for floor in self.floors:
            for below_floor in below_graph[floor]:
                in_degree[below_floor] += 1

        queue = [floor for floor in self.floors if in_degree[floor] == 0]
        topo_order = []
        while queue:
            floor = queue.pop(0)
            topo_order.append(floor)
            for below_floor in below_graph[floor]:
                in_degree[below_floor] -= 1
                if in_degree[below_floor] == 0:
                    queue.append(below_floor)

        # If topo_order misses floors, add them arbitrarily to keep consistency
        if len(topo_order) < len(self.floors):
            remaining = self.floors.difference(topo_order)
            topo_order.extend(sorted(remaining))

        # Map floors to their numeric heights (0-based)
        self.floor_to_height = {floor: idx for idx, floor in enumerate(topo_order)}
        self.num_floors = len(topo_order)

        # Extract destination floors for each passenger from static facts
        self.passenger_to_dest = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger, floor = parts[1], parts[2]
                self.passenger_to_dest[passenger] = floor

    def __call__(self, node):
        state = node.state

        # If goal reached (all served), heuristic is zero
        if self.goals <= state:
            return 0

        # Helper sets to track passengers and floors
        boarded_passengers = set()
        served_passengers = set()
        origin_of_not_boarded = {}
        destination_of_boarded = {}

        # Current elevator floor (exactly one lift-at fact)
        elevator_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_floor = get_parts(fact)[1]
                break
        assert elevator_floor is not None, "No elevator position found in state."

        elevator_floor_num = self.floor_to_height[elevator_floor]

        # Parse passenger status:
        # boarded(p), served(p), origin(p, f)
        for fact in state:
            if match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                boarded_passengers.add(p)
            elif match(fact, "served", "*"):
                p = get_parts(fact)[1]
                served_passengers.add(p)
            elif match(fact, "origin", "*", "*"):
                p, org = get_parts(fact)[1], get_parts(fact)[2]
                origin_of_not_boarded[p] = org

        # From boarded passengers get their destinations from static facts record.
        for p in boarded_passengers:
            if p in self.passenger_to_dest:
                destination_of_boarded[p] = self.passenger_to_dest[p]

        # Passengers not boarded and not served
        not_boarded_passengers = set(origin_of_not_boarded.keys()) - served_passengers

        # Count boarding and departing actions needed (= one per passenger not served)
        boarding_actions = len(not_boarded_passengers)  # must board each not boarded passenger
        departing_actions = len(boarded_passengers)    # must depart each boarded passenger

        # Floors to visit for boarding (origins of not boarded)
        origin_floors = {self.floor_to_height[org] for org in origin_of_not_boarded.values() if org}

        # Floors to visit for departing (destinations of boarded)
        depart_floors = {self.floor_to_height[dest] for dest in destination_of_boarded.values() if dest}

        # If no boarding or departing floors, only elevator at floor counts, no moves needed
        if not origin_floors and not depart_floors:
            # No passengers to board or depart, but goals not reached could mean served not complete
            # Return sum of board + depart which is zero here so 0 heuristic
            return 0

        # All relevant floors that must be visited (for boarding and departing)
        floors_to_visit = origin_floors | depart_floors

        # Compute minimal elevator moves to cover all floors needed plus current floor
        min_floor = min(floors_to_visit)
        max_floor = max(floors_to_visit)

        # The elevator must at least cover span of floors where passengers boarded or departed
        floors_span = max_floor - min_floor

        # Distance from current elevator floor to the closest end of the floors span
        dist_to_span = min(abs(elevator_floor_num - min_floor), abs(elevator_floor_num - max_floor))

        # Total estimated elevator moves: coverage of floors span + distance to reach the span
        elevator_moves_estimate = floors_span + dist_to_span

        # Total heuristic estimate:
        # sum of boarding + departing actions + elevator moves estimate
        heuristic_value = boarding_actions + departing_actions + elevator_moves_estimate
        return heuristic_value
