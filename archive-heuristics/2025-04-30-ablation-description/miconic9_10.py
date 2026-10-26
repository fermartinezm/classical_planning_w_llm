from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic # Uncomment this line if Heuristic base class is imported

# Helper functions (as provided in examples)
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
# class miconicHeuristic(Heuristic): # Use this line if Heuristic base class is imported
class miconicHeuristic: # Use this line if Heuristic base class is NOT imported (as per "Provide only the Python code")
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all passengers by summing up
    the estimated minimum actions required for each unserved passenger
    independently. The cost for a passenger includes elevator movement
    to their location (origin or current), boarding (if waiting),
    elevator movement to their destination, and departing.

    # Assumptions
    - The floor structure is a linear sequence defined by `above` facts.
    - The cost of moving the elevator one floor is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic sums individual passenger costs, potentially overestimating
      by not accounting for shared elevator trips.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Build a mapping from passenger names to their destination floor names
      from the static `destin` facts.
    - Build a mapping from floor names to numerical floor levels based on
      the static `above` facts, assuming `(above f_below f_above)` means
      `f_below` is immediately below `f_above`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the elevator's current floor from the `lift-at` fact in the state.
    3. Identify the set of passengers who are already `served` in the current state.
    4. Iterate through all facts in the current state:
       a. If the fact is `(origin ?p ?f_origin)` and passenger `?p` is NOT in the set of served passengers:
          i. Get the destination floor `f_destin` for `?p` from the pre-calculated destinations map.
          ii. Calculate the estimated cost for this passenger:
             - Moves from current elevator floor to origin floor: `abs(floor_number(current_floor) - floor_number(f_origin))`
             - Boarding action: +1
             - Moves from origin floor to destination floor: `abs(floor_number(f_origin) - floor_number(f_destin))`
             - Departing action: +1
             - Total for this passenger: `abs(floor_number(current_floor) - floor_number(f_origin)) + 1 + abs(floor_number(f_origin) - floor_number(f_destin)) + 1`
          iii. Add this cost to the total heuristic.
       b. If the fact is `(boarded ?p)` and passenger `?p` is NOT in the set of served passengers:
          i. Get the destination floor `f_destin` for `?p` from the pre-calculated destinations map.
          ii. Calculate the estimated cost for this passenger:
             - Moves from current elevator floor to destination floor: `abs(floor_number(current_floor) - floor_number(f_destin))`
             - Departing action: +1
             - Total for this passenger: `abs(floor_number(current_floor) - floor_number(f_destin)) + 1`
          iii. Add this cost to the total heuristic.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, passenger destinations,
        and floor numbering from the task's static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each passenger.
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.destinations[passenger] = destination_floor

        # Build the floor numbering map.
        # We assume (above f_below f_above) means f_below is immediately below f_above.
        # Build a map from floor_below to floor_above.
        below_map = {}
        all_floors = set()
        # Collect all floors and the 'below' relationships
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_below, f_above = get_parts(fact)
                below_map[f_below] = f_above
                all_floors.add(f_below)
                all_floors.add(f_above)

        self.floor_numbers = {}
        if not all_floors:
             # No floors defined, heuristic will be 0.
             return

        # Find the lowest floor: the floor that is not the 'above' floor for any 'below' floor.
        # In the below_map, this is a floor name that is not a VALUE.
        floors_that_are_above = set(below_map.values())
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_that_are_above:
                 lowest_floor = floor
                 break

        if lowest_floor:
            current_floor = lowest_floor
            level = 1
            self.floor_numbers[current_floor] = level
            # Traverse up the floors using the below_map
            while current_floor in below_map:
                current_floor = below_map[current_floor]
                level += 1
                self.floor_numbers[current_floor] = level
        # If lowest_floor is None, floor_numbers remains empty. This indicates an issue
        # with the 'above' facts structure (e.g., cycle, disconnected, or single floor not handled).
        # For a single floor 'f1', there are no 'above' facts. all_floors={f1}, below_map={},
        # floors_that_are_above={}. lowest_floor=f1. floor_numbers={f1: 1}. This works.
        # For a cycle, lowest_floor might be None. For disconnected floors, only one chain is numbered.
        # Assuming valid linear structure for miconic.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If floor numbering failed, return 0 (or potentially a high value).
        # Returning 0 makes states with parsing errors seem like goal states.
        # Let's return 0, assuming valid problems.
        if not self.floor_numbers:
             return 0

        total_cost = 0  # Initialize action cost counter.

        # Find the elevator's current floor.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If elevator location is unknown, return 0 (shouldn't happen in valid state).
        if current_lift_floor is None or current_lift_floor not in self.floor_numbers:
             return 0

        # Get the set of served passengers.
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through the state to find unserved passengers and their status.
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "origin" and len(parts) == 3:
                _, passenger, origin_floor = parts
                if passenger not in served_passengers:
                    # Passenger is waiting at origin_floor.
                    destination_floor = self.destinations.get(passenger)
                    # Ensure origin and destination floors are valid and numbered
                    if destination_floor and origin_floor in self.floor_numbers and destination_floor in self.floor_numbers:
                        # Cost to get elevator to origin + board + cost to get elevator to destin + depart
                        cost = abs(self.floor_numbers[current_lift_floor] - self.floor_numbers[origin_floor]) + 1 + \
                               abs(self.floor_numbers[origin_floor] - self.floor_numbers[destination_floor]) + 1
                        total_cost += cost
                    # else: Invalid state or static info, ignore or add penalty? Assume valid problem.

            elif predicate == "boarded" and len(parts) == 2:
                _, passenger = parts
                if passenger not in served_passengers:
                    # Passenger is boarded.
                    destination_floor = self.destinations.get(passenger)
                     # Ensure current lift floor and destination floor are valid and numbered
                    if destination_floor and current_lift_floor in self.floor_numbers and destination_floor in self.floor_numbers:
                        # Cost to get elevator to destin + depart
                        cost = abs(self.floor_numbers[current_lift_floor] - self.floor_numbers[destination_floor]) + 1
                        total_cost += cost
                    # else: Invalid state or static info, ignore or add penalty? Assume valid problem.

        return total_cost
