from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers
    by summing the estimated minimum actions required for each unserved
    passenger independently. It considers the lift's current position,
    the passenger's origin (if waiting) or current location (if boarded),
    and their destination.

    # Assumptions
    - Passengers are either waiting at their origin floor, boarded in the lift,
      or have been served at their destination floor.
    - The floor structure is a linear sequence defined by 'above' predicates.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding is 1.
    - The cost of departing is 1.
    - The lift is always at some floor (lift-at predicate is always true).
    - Unserved passengers are either at their origin or boarded.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static 'destin' facts.
    - Determines the ordering of floors and creates a mapping from floor name
      to its numerical level based on the 'above' predicates in the static facts.
      This mapping is used to calculate the distance (number of moves) between floors.
    - Identifies all passenger names from the 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)`.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all passengers identified during initialization.
    4. For each passenger `p`:
       a. Check if the passenger has already been served by looking for the fact `(served p)` in the current state. If yes, their contribution to the heuristic is 0.
       b. If the passenger is not served:
          i. Find the passenger's destination floor `dest_f` using the precomputed destination map.
          ii. Determine the passenger's current status by checking the state facts:
              - If the fact `(boarded p)` is in the state:
                The passenger is in the lift. The estimated cost for this passenger is the distance from the lift's current floor to `dest_f` plus 1 (for the depart action). Add this cost to the total.
              - Otherwise (assuming the passenger must be waiting at their origin):
                Find the passenger's origin floor `origin_f` by looking for the fact `(origin p ?f)` in the state.
                The estimated cost for this passenger is the distance from the lift's current floor to `origin_f` (to pick them up) plus 1 (for the board action) plus the distance from `origin_f` to `dest_f` (to drop them off) plus 1 (for the depart action). Add this cost to the total.
    5. The total heuristic value is the sum of the contributions from all unserved passengers.

    This heuristic is non-admissible as it counts travel costs and actions for
    each passenger independently, potentially double-counting lift movements
    and assuming optimal sequencing for each passenger in isolation. However,
    it provides a more informed estimate than simpler heuristics and aims to
    guide the search effectively in this domain.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and goal locations.
        """
        super().__init__(task)

        # Extract destin facts from static information
        self.passenger_destinations = {}
        self.all_passengers = set()
        for fact in task.static:
             predicate, *args = get_parts(fact)
             if predicate == "destin":
                 person, floor = args
                 self.passenger_destinations[person] = floor
                 self.all_passengers.add(person)

        # Build floor level mapping from 'above' facts
        self.floor_levels = {}
        above_map = {} # f_low -> f_high
        all_floors = set()

        for fact in task.static:
            predicate, *args = get_parts(fact)
            if predicate == "above":
                f_high, f_low = args
                above_map[f_low] = f_high
                all_floors.add(f_high)
                all_floors.add(f_low)

        # Find the lowest floor (a floor that is not a value in above_map)
        lowest_floor = None
        above_values = set(above_map.values())
        for floor in all_floors:
            if floor not in above_values:
                lowest_floor = floor
                break

        # Build the level map by traversing upwards
        current_floor = lowest_floor
        level = 1
        while current_floor is not None:
            self.floor_levels[current_floor] = level
            current_floor = above_map.get(current_floor)
            level += 1

    def get_floor_level(self, floor):
        """Helper to get the numerical level of a floor."""
        # Return 0 or handle error if floor not found (shouldn't happen with valid PDDL)
        return self.floor_levels.get(floor, 0)

    def dist(self, floor1, floor2):
        """Calculate the distance (number of moves) between two floors."""
        level1 = self.get_floor_level(floor1)
        level2 = self.get_floor_level(floor2)
        return abs(level1 - level2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # Iterate through all passengers
        for passenger in self.all_passengers:
            # Check if passenger is served
            if f"(served {passenger})" in state:
                continue # Passenger is served, no cost contribution

            # Passenger is not served, calculate their contribution
            dest_floor = self.passenger_destinations[passenger]

            # Check if passenger is boarded
            if f"(boarded {passenger})" in state:
                # Passenger is in the lift, needs to go to destination and depart
                cost_to_dest = self.dist(current_lift_floor, dest_floor)
                cost_depart = 1
                total_cost += cost_to_dest + cost_depart
            else:
                # Passenger is waiting at origin floor
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break

                # Assuming unserved passengers are either boarded or at their origin
                if origin_floor is not None:
                    # Needs to be picked up at origin, then go to destination and depart
                    cost_to_origin = self.dist(current_lift_floor, origin_floor)
                    cost_board = 1
                    cost_origin_to_dest = self.dist(origin_floor, dest_floor)
                    cost_depart = 1
                    total_cost += cost_to_origin + cost_board + cost_origin_to_dest + cost_depart
                # else: Invalid state if passenger is unserved, not boarded, and not at origin

        return total_cost
