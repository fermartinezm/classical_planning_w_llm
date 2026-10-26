from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state by summing three components: the number of passengers waiting to be
    boarded, the number of passengers currently boarded, and an estimate of
    the lift movement actions needed to visit all necessary floors.

    # Assumptions
    - Each passenger needs one 'board' action and one 'depart' action.
    - Lift movement actions are required to reach the floors where passengers
      are waiting or need to depart.
    - The cost of moving the lift is estimated based on the range of floors
      that need to be visited.

    # Heuristic Initialization
    - The heuristic pre-processes the static facts to determine the order of
      floors and create a mapping from floor names to numerical indices. This
      is done by analyzing the `(above ?f1 ?f2)` facts, which define the
      immediate adjacency and relative height of floors.
    - It also stores the destination floor for each passenger from the static
      `(destin ?p ?f)` facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)`
       in the state.
    2. Identify all passengers who have not yet been served by checking which
       `(served ?p)` facts from the goal are not present in the current state.
    3. Initialize counters for passengers waiting at their origin (`num_waiting`)
       and passengers currently boarded (`num_boarded`).
    4. Initialize a set of required stop floors (`F_stops`).
    5. Iterate through the unserved passengers:
       - If a passenger `p` has the fact `(origin p f_origin)` in the state,
         increment `num_waiting` and add `f_origin` to `F_stops`.
       - If a passenger `p` has the fact `(boarded p)` in the state,
         increment `num_boarded` and add their destination floor (looked up
         from initialization data) to `F_stops`.
    6. Estimate the lift movement cost:
       - If `F_stops` is empty, the movement cost is 0.
       - If `F_stops` is not empty:
         - Get the numerical index for the current lift floor (`curr_idx`)
           using the pre-computed floor index map.
         - Get the numerical indices for all floors in `F_stops` (`stop_indices`).
         - Find the minimum (`min_idx`) and maximum (`max_idx`) indices among
           `stop_indices`.
         - The estimated moves are calculated as:
           `(max_idx - min_idx) + min(abs(curr_idx - min_idx), abs(curr_idx - max_idx))`.
           This estimates the moves needed to traverse the range of required
           floors plus the moves needed to reach the closest end of that range
           from the current position.
    7. The total heuristic value is the sum of `num_waiting`, `num_boarded`,
       and the estimated lift moves.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions (served passengers)
        static_facts = task.static  # Static facts (above, destin)

        # Build floor order and index map from (above f1 f2) facts
        # (above f1 f2) means f1 is immediately above f2.
        # We want a map from floor name to index (0 for lowest, N-1 for highest).
        above_to_below = {}
        all_floors = set()
        floors_above = set()
        floors_below = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                above_to_below[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_above.add(f_above)
                floors_below.add(f_below)

        # Find the highest floor (appears as f_above but not f_below)
        highest_floor = None
        for f in floors_above:
            if f not in floors_below:
                highest_floor = f
                break

        # Build floor_to_index map by traversing down from the highest floor
        self.floor_to_index = {}
        current_floor = highest_floor
        index = len(all_floors) - 1 # Highest floor gets the highest index

        while current_floor is not None:
            self.floor_to_index[current_floor] = index
            index -= 1
            current_floor = above_to_below.get(current_floor)

        # Store passenger destinations for quick lookup
        self.passenger_destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor

        # Store the set of all passengers
        self.all_passengers = set(self.passenger_destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state facts

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             return float('inf') # Or some large value indicating an invalid state

        # Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if '(served {})'.format(p) not in state}

        num_waiting = 0
        num_boarded = 0
        required_stops = set()

        # Iterate through unserved passengers to find required actions and stops
        for passenger in unserved_passengers:
            is_waiting = False
            is_boarded = False
            origin_floor = None

            # Check if waiting at origin
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'origin' and parts[1] == passenger:
                    is_waiting = True
                    origin_floor = parts[2]
                    break

            # Check if boarded
            if not is_waiting: # A passenger cannot be both waiting and boarded
                 for fact in state:
                     parts = get_parts(fact)
                     if parts[0] == 'boarded' and parts[1] == passenger:
                         is_boarded = True
                         break

            if is_waiting:
                num_waiting += 1
                required_stops.add(origin_floor)
            elif is_boarded:
                num_boarded += 1
                # Passenger must have a destination if boarded
                destin_floor = self.passenger_destinations.get(passenger)
                if destin_floor:
                    required_stops.add(destin_floor)
                # else: This state is likely invalid if a boarded passenger has no destination.

        # Estimate lift movement cost
        moves = 0
        if required_stops:
            curr_idx = self.floor_to_index[current_lift_floor]
            stop_indices = {self.floor_to_index[f] for f in required_stops}
            min_idx = min(stop_indices)
            max_idx = max(stop_indices)

            if curr_idx < min_idx:
                # Must go down to min_idx, then up to max_idx
                moves = (min_idx - curr_idx) + (max_idx - min_idx)
            elif curr_idx > max_idx:
                # Must go up to max_idx, then down to min_idx
                 moves = (curr_idx - max_idx) + (max_idx - min_idx)
            else: # min_idx <= curr_idx <= max_idx
                # Must go to one extreme (min or max) and then cover the range
                moves = (max_idx - min_idx) + min(curr_idx - min_idx, max_idx - curr_idx)

        # Total heuristic is sum of actions needed
        # Each waiting passenger needs a board action (1)
        # Each boarded passenger needs a depart action (1)
        # Plus the estimated lift movements
        total_cost = num_waiting + num_boarded + moves

        return total_cost

