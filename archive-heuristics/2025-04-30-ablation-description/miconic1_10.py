from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It sums the estimated cost for each unserved passenger,
    considering their current location (waiting at origin or boarded) and the
    lift's current location.

    # Assumptions
    - Floors are linearly ordered, and the `(above f_above f_below)` facts
      define this order such that `f_above` is immediately above `f_below`.
    - Passengers are either waiting at their origin, boarded, or served.
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The `destin` facts are static.
    - All floors relevant to passenger origins/destinations and lift location
      are defined in the static `above` facts or at least mentioned in the
      initial state facts (like `lift-at`).

    # Heuristic Initialization
    - Parses static facts to determine the floor ordering and create a mapping
      from floor names to integer levels.
    - Parses static facts to determine the destination floor for each passenger.
    - Identifies all passengers that need to be served from the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift. If the lift location is unknown
       (should not happen in valid states), return a fallback heuristic value
       (e.g., number of unserved passengers).
    2. Get the integer level for the lift's current floor. If the floor is
       unknown (should not happen), return the fallback heuristic.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each passenger identified in the goal.
    5. For the current passenger:
       a. Check if the passenger is already served. If yes, skip this passenger.
       b. Get the passenger's destination floor and its level (from initialization).
          If destination or its level is unknown (should not happen), skip passenger.
       c. Check if the passenger is boarded.
       d. If the passenger is boarded:
          - Calculate the estimated cost: absolute difference in floor levels
            between the lift's current floor and the passenger's destination floor
            (cost to move lift), plus 1 (for the depart action).
          - Add this cost to the total.
       e. If the passenger is not boarded:
          - Find the passenger's current origin floor from the state. If origin
            is unknown (should not happen for unserved, unboarded passengers),
            skip passenger.
          - Get the integer level for the origin floor. If level is unknown, skip.
          - Calculate the estimated cost: absolute difference in floor levels
            between the lift's current floor and the passenger's origin floor
            (cost to reach origin), plus 1 (for the board action), plus absolute
            difference in floor levels between the passenger's origin floor and
            their destination floor (cost to travel with passenger), plus 1
            (for the depart action).
          - Add this cost to the total.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations
        from static facts.
        """
        super().__init__(task) # Call the base class constructor

        # Build floor level mapping
        self.floor_levels = {}
        floor_immediately_above_map = {}
        all_floors = set()

        # Collect floors and build the map from 'above' facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                floor_immediately_above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor (the one not immediately below any other floor)
        lowest_floor = None
        floors_below = set(floor_immediately_above_map.keys())
        potential_lowest = all_floors - floors_below

        if len(potential_lowest) == 1:
             lowest_floor = potential_lowest.pop()
        elif len(all_floors) == 1:
             # Single floor problem, no 'above' facts
             lowest_floor = list(all_floors)[0]
        elif len(potential_lowest) > 1:
             # Multiple potential lowest floors - implies non-linear structure
             # Attempt to sort them if they look like f<number>
             try:
                 sorted_potential = sorted(list(potential_lowest), key=lambda f: int(f[1:]))
                 lowest_floor = sorted_potential[0]
             except (ValueError, IndexError):
                 # Cannot sort or unexpected format, heuristic might be unreliable
                 # Leave lowest_floor as None, will use fallback heuristic
                 pass
        # else: len(potential_lowest) == 0 and len(all_floors) > 1, implies cycle or malformed above facts.
        # Leave lowest_floor as None, will use fallback heuristic.


        if lowest_floor:
            self.floor_levels[lowest_floor] = 1
            current_floor = lowest_floor
            level = 1
            # Build levels upwards using the map f_below -> f_above
            while current_floor in floor_immediately_above_map:
                next_floor = floor_immediately_above_map[current_floor]
                level += 1
                self.floor_levels[next_floor] = level
                current_floor = next_floor
        # else: floor_levels remains empty, fallback heuristic will be used in __call__


        # Build passenger destination mapping from static facts
        self.passenger_destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination_floor

        # Get the set of all passengers that need to be served from the goal facts
        self.all_passengers_in_goal = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served":
                 self.all_passengers_in_goal.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift location
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # Fallback heuristic if essential information is missing or structure is unexpected
        # This happens if floor_levels is empty or lift_floor is not found/mapped.
        # The fallback is the number of unserved passengers.
        unserved_count = sum(1 for p in self.all_passengers_in_goal if f"(served {p})" not in state)

        if not self.floor_levels:
             # Floor structure could not be determined from static facts
             return unserved_count

        lift_level = self.floor_levels.get(lift_floor)
        if lift_level is None:
             # Lift is at a floor not found in the determined floor levels
             return unserved_count


        total_cost = 0  # Initialize action cost counter.

        # Track boarded passengers for quick lookup
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        # Track passengers at origin for quick lookup
        passengers_at_origin = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}


        for passenger in self.all_passengers_in_goal:
            # Check if passenger is served
            if f"(served {passenger})" in state:
                continue # This passenger is already served

            # Get destination floor and its level
            destin_floor = self.passenger_destinations.get(passenger)
            if destin_floor is None:
                 # Passenger has no destination defined in static facts? Malformed problem?
                 continue

            destin_level = self.floor_levels.get(destin_floor)
            if destin_level is None:
                 # Destination floor '{destin_floor}' for passenger '{passenger}' not in floor levels. Malformed problem?
                 continue


            # Estimate cost based on passenger state
            if passenger in boarded_passengers:
                # Passenger is boarded, needs to reach destination and depart
                # Cost = move_to_destin + depart
                cost = abs(lift_level - destin_level) + 1
                total_cost += cost
            elif passenger in passengers_at_origin:
                # Passenger is waiting at origin, needs pickup, travel, and depart
                # Cost = move_to_origin + board + move_to_destin + depart
                origin_floor = passengers_at_origin[passenger]
                origin_level = self.floor_levels.get(origin_floor) # Use .get for safety

                if origin_level is None:
                     # Origin floor '{origin_floor}' for passenger '{passenger}' not in floor levels. Malformed problem?
                     continue

                cost = abs(lift_level - origin_level) + 1 + abs(origin_level - destin_level) + 1
                total_cost += cost
            else:
                 # Passenger is not served, not boarded, and not at a known origin floor.
                 # This state shouldn't be reachable in a valid plan sequence for this domain.
                 pass # Skip passengers in unexpected states


        return total_cost
