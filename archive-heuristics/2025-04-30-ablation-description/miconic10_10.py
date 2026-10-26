from fnmatch import fnmatch
from task import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for predicate name match and arity
    if not parts or len(parts) != len(args):
        return False
    # Check remaining arguments, allowing for wildcards
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all passengers by summing:
    1. The estimated vertical movement cost for the lift to visit all necessary floors (origins for waiting unserved passengers, destinations for boarded unserved passengers).
    2. The estimated action cost (board and depart actions) for all unserved passengers.

    # Assumptions
    - Floors are ordered linearly based on the `above` predicate.
    - The cost of moving between adjacent floors is 1.
    - The cost of board and depart actions is 1.
    - The lift can carry multiple passengers.
    - The heuristic calculates a lower bound on movement by considering the range of required floors and the distance to reach that range.
    - The heuristic counts board actions for waiting unserved passengers and depart actions for all unserved passengers.
    - All passengers in the problem have a defined destination in the static facts.
    - Unserved passengers are always either waiting at their origin or boarded in the lift.

    # Heuristic Initialization
    - Parses static facts to:
        - Map each passenger to their destination floor (`self.destinations`).
        - Determine the rank (level) of each floor based on the `above` predicate (`self.floor_rank`). The lowest floor is assigned rank 0.
        - Identify all passengers in the problem (`self.all_passengers`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all unserved passengers by checking which passengers from the problem's set do not have the `(served ?p)` fact in the current state.
    3. If there are no unserved passengers, the heuristic is 0.
    4. Identify the set of required floors and count waiting unserved passengers:
       - Iterate through the current state facts.
       - For each `(origin ?p ?f)` fact where `?p` is unserved, add `?f` and `self.destinations[?p]` to required floors and increment waiting count.
       - For each `(boarded ?p)` fact where `?p` is unserved, add `self.destinations[?p]` to required floors.
    5. Calculate the estimated movement cost:
       - Find the minimum and maximum ranks among the required floors using `self.floor_rank`.
       - If there are required floors, the movement cost is estimated as the distance needed to travel from the current floor to the nearest required floor, plus the distance needed to traverse the entire range of required floors. This is calculated as `(max_required_rank - min_required_rank) + min(abs(current_lift_rank - min_required_rank), abs(current_lift_rank - max_required_rank))`.
       - If there are no required floors (which implies all unserved passengers are in an unexpected state or the goal is reached), the movement cost is 0.
    6. Calculate the estimated action cost:
       - Count the total number of unserved passengers (`num_unserved`). Each needs a 'depart' action eventually.
       - The count of waiting unserved passengers was determined in step 4 (`num_waiting_unserved`). Each needs a 'board' action first.
       - The total action cost is estimated as `num_unserved + num_waiting_unserved`.
    7. The total heuristic value is the sum of the estimated movement cost and the estimated action cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Map passengers to their destination floors and collect all passengers
        self.destinations = {}
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # 2. Determine floor ranks based on the 'above' predicate
        # Find all unique floors mentioned in 'above' facts
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                all_floors.add(f1)
                all_floors.add(f2)

        # Calculate rank for each floor: count how many floors are below it
        # A floor f_below is below f if (above f f_below) is true
        self.floor_rank = {}
        for floor in all_floors:
            rank = sum(1 for fact in static_facts if match(fact, "above", floor, "*"))
            self.floor_rank[floor] = rank

        # Handle cases with no floors or only one floor (e.g., minimal problems)
        if not self.floor_rank and all_floors:
             # If no 'above' facts but multiple floors exist, assign arbitrary ranks (e.g., alphabetical)
             sorted_floors = sorted(list(all_floors))
             self.floor_rank = {f: i for i, f in enumerate(sorted_floors)}
        elif not self.floor_rank and not all_floors:
             # If no floors defined at all (shouldn't happen in valid PDDL with lift/passengers)
             pass # Heuristic will likely be 0 or encounter errors if floors are needed


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if f"(served {p})" not in state}

        # If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # Identify required floors and count waiting unserved passengers
        required_floors = set()
        num_waiting_unserved = 0

        # Iterate through state facts to find unserved passengers' locations
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, origin_floor = get_parts(fact)
                if passenger in unserved_passengers:
                    num_waiting_unserved += 1
                    required_floors.add(origin_floor)
                    # Destination floor is also required after boarding
                    if passenger in self.destinations:
                         required_floors.add(self.destinations[passenger])
            elif match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                if passenger in unserved_passengers:
                    # Destination floor is required for boarded passengers
                    if passenger in self.destinations:
                        required_floors.add(self.destinations[passenger])
            # Note: Passengers already served are not in origin/boarded facts
            # and are implicitly handled by checking against unserved_passengers set.

        # Calculate movement cost
        movement_cost = 0
        # Ensure current_lift_floor is valid and required_floors is not empty
        if current_lift_floor in self.floor_rank and required_floors:
            required_ranks = {self.floor_rank[f] for f in required_floors if f in self.floor_rank}
            if required_ranks: # Ensure required_ranks is not empty after filtering
                min_r = min(required_ranks)
                max_r = max(required_ranks)
                current_lift_rank = self.floor_rank[current_lift_floor]

                # Estimate movement: distance to nearest required floor + distance to cover the range
                movement_cost = (max_r - min_r) + min(abs(current_lift_rank - min_r), abs(current_lift_rank - max_r))
            # else: required_floors contained floors not in self.floor_rank (shouldn't happen in valid PDDL)
            # movement_cost remains 0

        # Calculate action cost
        # Each unserved passenger needs a depart action (1)
        # Each unserved passenger who is waiting needs a board action (1)
        action_cost = len(unserved_passengers) + num_waiting_unserved

        # Total heuristic is the sum of movement and action costs
        total_cost = movement_cost + action_cost

        return total_cost
