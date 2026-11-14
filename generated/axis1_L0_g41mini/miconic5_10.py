from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary
    -------
    This heuristic estimates the minimum number of actions needed to serve all passengers
    that are not yet served from the current state. It accounts for:
    - Elevator moves between floors,
    - Boarding and departing passengers,
    by considering the current elevator position, passengers waiting to board, those on-board needing to be served,
    and floors of origin and destination for each passenger.

    Assumptions
    -----------
    - The elevator can move only one floor up or down per action.
    - Boarding a passenger requires the elevator to be at their origin floor and an action 'board'.
    - Departing a passenger requires the elevator at their destination floor and an action 'depart'.
    - All passengers must be served (goal is served ?p for all passengers).
    - The predicate "above ?f1 ?f2" is transitive and defines the total order between floors.
    - The heuristic is not necessarily admissible but should be informative to reduce node expansions.

    Heuristic Initialization
    ------------------------
    - Extract all floors from static facts using above-relations.
    - Build mapping between floors and indices to compute floor distances.
    - Store for each passenger their origin and destination floors from static facts.
      Note: origin and destin predicates are static, but boarded and served become true dynamically.
    - Construct a floors list sorted by height and a quick lookup for floor indices.
    - Precompute floor distances as absolute difference of indices in the order.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state:

    1. Identify the current elevator floor.
    2. From the state, categorize passengers into:
       - Waiting passengers: those with (origin p f) in state (not boarded or served).
       - Boarded passengers: those with (boarded p) true but not (served p).
       - Served passengers: those with (served p).
    3. For each waiting passenger, retrieve their origin and destination floors.
    4. For each boarded, retrieve their destination floor.
    5. Compute the minimum number of elevator moves needed:
       - Elevator must at least move to each distinct floor where passengers must board or depart.
       - Approximate minimal route as:
         distance from elevator's current floor to the closest floor among boarding or departing floors
         plus the distance that covers all involved floors.
         Since exact TSP is expensive, use the span between min and max involved floors and distance 
         from current floor to closest involved floor.
    6. Add one action per passenger to board and one to depart that passenger.
    7. Sum the elevator movements and boarding/departing actions for total heuristic cost.
    8. Return 0 if the goal is already reached (all passengers served).

    This heuristic enables an efficient, informative estimate guiding greedy best-first search.
    """

    def __init__(self, task):
        """Initialize static data and preprocess floors and passengers."""
        self.goals = task.goals
        static = task.static

        # Extract all floors from "above" predicates (to build floor order)
        floors_set = set()
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) == 3:
                floors_set.add(parts[1])
                floors_set.add(parts[2])
        self.floors = sorted(floors_set, key=lambda f: int(f[1:]) if f[1:].isdigit() else f)
        # Build floor index mapping
        self.floor_to_index = {f: i for i, f in enumerate(self.floors)}

        # Passengers and their origin/destination floors
        self.passengers = set()
        self.origin_of = {}
        self.destin_of = {}

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "origin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.origin_of[p] = f
                self.passengers.add(p)
            elif parts[0] == "destin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.destin_of[p] = f
                self.passengers.add(p)

    def floor_distance(self, f1, f2):
        """Return the distance (number of floors) between two floors based on indices."""
        return abs(self.floor_to_index[f1] - self.floor_to_index[f2])

    def __call__(self, node):
        """Compute heuristic estimate from the given node's state."""
        state = node.state

        # Quick check: if goal is reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Identify elevator current floor
        elevator_floors = [get_parts(f)[1] for f in state if f.startswith("(lift-at ")]
        if not elevator_floors:
            # Should always be exactly one, but just in case:
            return 0
        elevator_floor = elevator_floors[0]

        # Track passengers by status
        waiting = set()
        boarded = set()
        served = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin" and len(parts) == 3:
                # passenger at origin floor and not boarded yet
                p = parts[1]
                if p in self.passengers:
                    waiting.add(p)
            elif parts[0] == "boarded" and len(parts) == 2:
                boarded.add(parts[1])
            elif parts[0] == "served" and len(parts) == 2:
                served.add(parts[1])

        # Remove served passengers from waiting and boarded sets (fresh categorization)
        waiting -= served
        boarded -= served

        # Floors where boarding or departing actions are needed
        boarding_floors = {self.origin_of[p] for p in waiting}
        departing_floors = {self.destin_of[p] for p in boarded}

        # All floors where the elevator must stop for actions
        floors_to_visit = boarding_floors | departing_floors

        if not floors_to_visit:
            # Nothing left to do except maybe waiting passengers already served by error -> heuristic 0
            return 0

        # Calculate elevator moves:
        # Distance from current floor to the closest needed floor:
        dist_to_closest = min(self.floor_distance(elevator_floor, f) for f in floors_to_visit)
        # Elevator must cover at least the span from min to max floors involved:
        indices = [self.floor_to_index[f] for f in floors_to_visit]
        floor_span = max(indices) - min(indices)

        # Each passenger requires one board + one depart action (if not yet served)
        num_passengers_to_serve = len(waiting) + len(boarded)
        # Boarding and departing actions:
        actions_board_depart = 2 * num_passengers_to_serve

        # The total cost estimate:
        # Elevator moves needed + boarding/departing actions
        total_cost = dist_to_closest + floor_span + actions_board_depart

        return total_cost
