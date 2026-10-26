from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and contains a Heuristic class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It combines a count of pending boarding/departing actions for each unserved
    passenger with an estimate of the lift movement cost needed to visit all
    relevant floors.

    # Assumptions
    - The floor structure is a linear sequence defined by `above` predicates.
    - The cost of moving one floor up or down is 1 action.
    - The cost of boarding is 1 action.
    - The cost of departing is 1 action.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Parses the `above` predicates from the static facts to determine the ordered sequence
      of floors and create a mapping from floor name to its numerical index.
    - Extracts the origin and destination floor for each passenger from the initial state.
      Assumes all passengers are introduced via `origin` and `destin` facts in the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify the current floor of the lift.
    3. Identify all unserved passengers by checking which passengers are not `served`.
    4. Categorize unserved passengers into `waiting` (at origin) and `boarded`.
       (A passenger is assumed waiting if unserved and not boarded).
    5. Calculate a base action cost: 2 for each waiting passenger (board + depart)
       and 1 for each boarded passenger (depart).
    6. Identify all floors that require a stop:
       - The origin floor for each waiting passenger.
       - The destination floor for each boarded passenger.
    7. Collect the numerical indices for the current lift floor and all required stop floors.
    8. If there are no required stop floors (meaning no waiting or boarded passengers),
       the movement cost is 0. This case should only occur if all passengers are served,
       which is handled by step 1.
    9. If there are required stop floors, find the minimum and maximum floor indices
       among all relevant floors (current floor and required stops).
    10. Estimate the movement cost as the total vertical span covered by these relevant floors:
        `max(relevant_indices) - min(relevant_indices)`. This represents the minimum
        distance the lift must travel to cover the range containing all necessary stops.
    11. The total heuristic value is the sum of the base action cost and the estimated
        movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger info.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for origins/destinations

        # 1. Determine floor order and create floor_to_index mapping
        self.floor_to_index = {}
        self.index_to_floor = []

        # Build map: floor_below -> floor_immediately_above
        immediately_above_map = {}
        all_floors = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                immediately_above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor (a floor that is not the second argument of any 'above' fact)
        floors_below_something = set(get_parts(fact)[2] for fact in self.static if match(fact, "above", "*", "*"))
        potential_lowest_floors = all_floors - floors_below_something

        lowest_floor = None
        if len(potential_lowest_floors) == 1:
            lowest_floor = list(potential_lowest_floors)[0]
        elif len(all_floors) == 1:
             # Case with only one floor
             lowest_floor = list(all_floors)[0]
        else:
            # Fallback: If we can't find a unique lowest floor, sort alphabetically.
            # This might happen with malformed PDDL or complex above relationships.
            # print(f"Warning: Could not determine strict floor order. Using alphabetical sort.")
            sorted_floors = sorted(list(all_floors))
            self.index_to_floor = sorted_floors
            self.floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
            lowest_floor = None # Indicate fallback was used

        if lowest_floor is not None:
            # Build the ordered list starting from the lowest floor
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                self.index_to_floor.append(current_floor)
                index += 1
                # Find the floor immediately above the current_floor
                current_floor = immediately_above_map.get(current_floor)

        # 2. Extract passenger origins and destinations from the initial state facts
        self.passenger_destinations = {}
        self.passenger_origins = {}
        self.all_passengers = set() # Keep track of all passengers mentioned

        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "destin":
                 _, passenger, destination_floor = parts
                 self.passenger_destinations[passenger] = destination_floor
                 self.all_passengers.add(passenger)
             elif parts[0] == "origin":
                 _, passenger, origin_floor = parts
                 self.passenger_origins[passenger] = origin_floor
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # If lift-at fact is missing, state is likely malformed or not reachable
        if current_lift_floor is None:
             return float('inf')

        current_lift_floor_idx = self.floor_to_index.get(current_lift_floor)
        # If current floor is not in our floor map, something is wrong
        if current_lift_floor_idx is None:
             return float('inf')


        # 2. Identify unserved passengers and required stops
        unserved_passengers = set()
        waiting_passengers = set()
        boarded_passengers = set()

        for passenger in self.all_passengers:
            # Check if served
            if f"(served {passenger})" not in state:
                unserved_passengers.add(passenger)
                # Check if boarded
                if f"(boarded {passenger})" in state:
                    boarded_passengers.add(passenger)
                else:
                    # If not served and not boarded, they must be waiting at origin
                    waiting_passengers.add(passenger)

        # 3. Calculate base action cost (board/depart)
        # Each waiting passenger needs a board and a depart action (2 actions)
        # Each boarded passenger needs a depart action (1 action)
        base_action_cost = len(waiting_passengers) * 2 + len(boarded_passengers) * 1

        # 4. Identify required stop floors (indices)
        required_stop_indices = set()

        for passenger in waiting_passengers:
            origin_floor = self.passenger_origins.get(passenger)
            if origin_floor:
                origin_idx = self.floor_to_index.get(origin_floor)
                if origin_idx is not None:
                    required_stop_indices.add(origin_idx)
            # else: passenger origin not found in initial state - malformed problem?

        for passenger in boarded_passengers:
            destination_floor = self.passenger_destinations.get(passenger)
            if destination_floor:
                dest_idx = self.floor_to_index.get(destination_floor)
                if dest_idx is not None:
                    required_stop_indices.add(dest_idx)
            # else: passenger destination not found in initial state - malformed problem?

        # 5. Calculate movement cost
        # If there are no required stops, movement cost is 0.
        # This should only happen if all passengers are served (handled by goal check).
        if not required_stop_indices:
             movement_cost = 0
        else:
            # The lift must cover the vertical range from the lowest required stop
            # to the highest required stop, starting from the current floor.
            # The set of floors relevant for movement calculation includes the current floor
            # and all required stop floors.
            all_relevant_floor_indices = required_stop_indices | {current_lift_floor_idx}

            min_relevant_idx = min(all_relevant_floor_indices)
            max_relevant_idx = max(all_relevant_floor_indices)

            # Movement cost is the total vertical span of relevant floors.
            # This is a non-admissible estimate of the travel needed to visit all stops.
            movement_cost = max_relevant_idx - min_relevant_idx

        # Total heuristic = base actions + movement estimate
        total_heuristic = base_action_cost + movement_cost

        return total_heuristic
