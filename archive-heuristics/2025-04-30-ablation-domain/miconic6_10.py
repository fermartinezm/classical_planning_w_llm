from fnmatch import fnmatch
# Assuming the base class is located at heuristics/heuristic_base.py
from heuristics.heuristic_base import Heuristic

# Helper functions (copied from Logistics example, slightly adapted)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Miconic Heuristic Class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of passengers waiting at their origin floor, the number
    of passengers currently boarded in the lift, and an estimate of the minimum
    number of lift movements needed to visit all floors where pickups or dropoffs
    are required.

    # Assumptions
    - Each waiting passenger requires at least one 'board' action.
    - Each boarded passenger requires at least one 'depart' action.
    - Lift movements are required to reach floors where passengers are waiting
      or need to be dropped off.
    - The floors are linearly ordered, named like 'f1', 'f2', etc., and
      '(above f_i f_j)' implies floor i is physically above floor j if i < j
      numerically. This means f1 is the highest, fN is the lowest.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts
      (using the 'destin' predicate).
    - Builds a mapping from floor names to integer indices based on the 'above'
      static facts and the assumed naming/ordering convention. Index 1 is assigned
      to the lowest floor, 2 to the next, etc.
    - Stores the set of all passengers in the problem for efficient goal checking.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state (all passengers are served). If yes, return 0.
    2. Identify the current floor of the lift.
    3. Identify all passengers who are waiting at their origin floor. Count them (`num_waiting`).
    4. Identify all passengers who are currently boarded in the lift. Count them (`num_boarded`).
    5. Determine the set of floors that the lift *must* visit (`F_needed`):
       - All origin floors of waiting passengers.
       - All destination floors of boarded passengers (retrieved from initialized destinations).
    6. Calculate the estimated movement cost (`movement_cost`):
       - If `F_needed` is empty, the movement cost is 0.
       - If `F_needed` is not empty:
         - Get the integer indices for all floors in `F_needed` using the pre-calculated floor mapping.
         - Find the minimum (`min_idx`) and maximum (`max_idx`) floor indices among these needed floors.
         - Get the index of the lift's current floor (`current_idx`).
         - The estimated movement cost is the minimum of the distance from the current floor to the lowest needed floor index and the distance from the current floor to the highest needed floor index, plus the total vertical distance between the lowest and highest needed floor indices. This estimates the moves needed for a single sweep covering all necessary floors.
         - Formula: `min(abs(current_idx - min_idx), abs(current_idx - max_idx)) + (max_idx - min_idx)`
    7. The total heuristic value is the sum of `num_waiting`, `num_boarded`, and `movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build floor mapping (name -> index)
        self.floor_mapping = self._build_floor_mapping(static_facts)

        # Store passenger destinations (passenger name -> floor name) from static facts
        self.passenger_destinations = {}
        for fact in static_facts:
             predicate, *args = get_parts(fact)
             if predicate == "destin":
                 passenger, floor = args
                 self.passenger_destinations[passenger] = floor

        # Store the set of all passengers for efficient goal checking
        self.all_passengers = set(self.passenger_destinations.keys())


    def _build_floor_mapping(self, static_facts):
        """
        Builds a mapping from floor names to integer indices based on 'above' facts.
        Assumes floor names are like 'f1', 'f2', etc., and (above f_i f_j) implies
        floor i is physically above floor j if i < j numerically.
        The lowest floor will have the highest number (e.g., f20 in example 2),
        and the highest floor will have the lowest number (e.g., f1).
        The mapping assigns index 1 to the lowest floor, 2 to the next, etc.
        """
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        if not all_floors:
             return {} # No floors defined

        # Sort floors numerically based on the number part (e.g., f1, f2, ..., f20)
        # This list is ordered from highest floor (f1) to lowest floor (f20) based on the problem examples.
        floor_names_sorted_numerically = sorted(list(all_floors), key=lambda f: int(f[1:]))

        # The order from lowest to highest is the reverse of the numerical sort.
        ordered_floors_lowest_to_highest = list(reversed(floor_names_sorted_numerically))

        # Assign indices starting from 1 for the lowest floor
        floor_mapping = {floor: i + 1 for i, floor in enumerate(ordered_floors_lowest_to_highest)}
        return floor_mapping


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal state is reached (all passengers served)
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        if self.all_passengers == served_passengers_in_state:
             return 0 # Goal state reached

        # 1. Identify the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # Assuming current_lift_floor is always found in a valid state and is in the floor_mapping

        # 2. Identify waiting passengers and their origin floors. Count them.
        # 3. Identify boarded passengers. Count them.
        waiting_passengers = set()
        boarded_passengers = set()
        pickup_floors = set() # Origin floors of waiting passengers

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                p, f = parts[1], parts[2]
                waiting_passengers.add(p)
                pickup_floors.add(f)
            elif parts[0] == "boarded":
                p = parts[1]
                boarded_passengers.add(p)

        num_waiting = len(waiting_passengers)
        num_boarded = len(boarded_passengers)

        # 4. Determine the set of floors that the lift *must* visit.
        dropoff_floors = set() # Destination floors of boarded passengers
        for p in boarded_passengers:
            # Get destination from pre-calculated map
            # Assuming passenger destinations are always defined in static facts
            dropoff_floors.add(self.passenger_destinations[p])

        needed_floors = pickup_floors.union(dropoff_floors)

        # 5. Calculate the estimated movement cost.
        movement_cost = 0
        if needed_floors:
            # Get indices for needed floors, assuming all needed floors are in mapping
            needed_indices = {self.floor_mapping[f] for f in needed_floors}
            min_idx = min(needed_indices)
            max_idx = max(needed_indices)
            current_idx = self.floor_mapping[current_lift_floor] # Assuming current floor is in mapping

            # Movement cost: distance to closest extreme + distance between extremes
            # This estimates the minimum moves to visit all floors in F_needed
            movement_cost = min(abs(current_idx - min_idx), abs(current_idx - max_idx)) + (max_idx - min_idx)

        # 6. Total heuristic value.
        # This sums the number of board actions needed, depart actions needed, and estimated move actions.
        h = num_waiting + num_boarded + movement_cost

        return h
