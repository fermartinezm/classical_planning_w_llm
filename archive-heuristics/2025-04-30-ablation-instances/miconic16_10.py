import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args for a valid match attempt
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (board, depart, and lift movement)
    required to transport all unserved passengers to their destination floors. It sums
    the individual board/depart actions needed for each unserved passenger and adds
    an estimate of the minimum lift movement required to visit all floors where
    passengers need to be picked up or dropped off.

    # Assumptions
    - Each unserved passenger currently at their origin needs one 'board' action and one 'depart' action.
    - Each unserved passenger currently 'boarded' needs one 'depart' action.
    - Lift movement actions are required to move the lift between floors. Moving between adjacent floors costs 1 action.
    - The minimum number of lift movements to visit a set of floors on a line, starting from the current floor, is estimated as the distance from the current floor to the closest extreme floor in the required set, plus the total span (distance between min and max floors) of the required set. This assumes the lift travels to one end of the required range and then sweeps across the range.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Identify all passenger names from initial state facts (specifically 'origin' facts).
    - Determine the ordered list of floors by finding all terms matching the floor name pattern ('f' followed by digits), sorting them numerically, and creating mappings between floor names and integer indices.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Find the current floor of the lift by looking for the `(lift-at ?f)` fact in the state.
    3. Initialize the total heuristic cost to 0.
    4. Identify the set of floors the lift must visit (`required_floors`). This set includes:
       - The origin floor for every unserved passenger who is currently at their origin.
       - The destination floor for every unserved passenger who is currently boarded.
    5. Iterate through all known passengers (identified during initialization):
       - Check if the passenger is `(served ?p)` in the current state.
       - If the passenger is NOT served:
         - Check if the passenger is `(boarded ?p)` in the current state.
         - If the passenger is NOT boarded (and not served), they must be at their origin floor (due to domain effects). Find their current origin floor `f_origin` from the state (`(origin ?p f_origin)`). Add 2 to the total cost (for board and depart actions). Add `f_origin` and the passenger's destination floor to `required_floors`.
         - If the passenger IS boarded (and not served), add 1 to the total cost (for the depart action). Add the passenger's destination floor to `required_floors`.
    6. Calculate the estimated lift movement cost:
       - If `required_floors` is empty, the movement cost is 0.
       - If `required_floors` is not empty and the lift's current floor is known and valid:
         - Map the required floor names to their integer indices using the floor-to-integer mapping.
         - Find the minimum (`min_req_int`) and maximum (`max_req_int`) integer indices among the required floors.
         - Get the integer index of the current lift floor (`lift_floor_int`).
         - Calculate the movement cost as `min(abs(lift_floor_int - min_req_int), abs(lift_floor_int - max_req_int)) + (max_req_int - min_req_int)`.
       - If `required_floors` is not empty but the lift's current floor is unknown or invalid, the movement cost cannot be calculated accurately. A robust heuristic might return a large value or just the board/depart costs; here we calculate based on available info and add to total cost.
    7. Add the calculated lift movement cost to the total cost.
    8. Return the total cost as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - Set of all passenger names.
        - Ordered list of floors and floor-to-integer mapping.
        """
        self.goals = task.goals  # Goal conditions (all passengers served).
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to identify all passengers

        # Store destination floors for each passenger.
        self.destinations = {}
        # Identify all passenger names involved in the problem
        self.all_passengers = set()

        # Extract passenger destinations from static facts
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "destin":
                passenger, floor = args
                self.destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # Identify passengers from initial state facts (e.g., origin)
        for fact in initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "origin":
                 passenger, floor = args
                 # We don't store initial origin floor here, just the passenger name
                 self.all_passengers.add(passenger)
             elif predicate == "boarded": # Passengers can start boarded
                 passenger = args[0]
                 self.all_passengers.add(passenger)
             elif predicate == "served": # Passengers can start served
                 passenger = args[0]
                 self.all_passengers.add(passenger)


        # Determine the ordered list of floors and create floor-to-integer mapping.
        # Collect all terms that look like floor names (e.g., 'f1', 'f10')
        floor_names_set = set()
        # Check static facts
        for fact in task.static:
            for part in get_parts(fact):
                if re.fullmatch(r'f\d+', part):
                    floor_names_set.add(part)
        # Check initial state facts
        for fact in task.initial_state:
             for part in get_parts(fact):
                if re.fullmatch(r'f\d+', part):
                    floor_names_set.add(part)

        if not floor_names_set:
             # Handle case with no floors found (should not happen in valid problems)
             self.floors = []
             self.floor_to_int = {}
             self.int_to_floor = {}
             return

        # Convert floor names to integers and find min/max number
        try:
            floor_numbers = sorted([int(name[1:]) for name in floor_names_set])
            min_floor_num = floor_numbers[0]
            max_floor_num = floor_numbers[-1]

            # Assume all floors from min_floor_num to max_floor_num exist and are ordered numerically
            self.floors = [f'f{i}' for i in range(min_floor_num, max_floor_num + 1)]

            # Create floor-to-integer and integer-to-floor mappings
            self.floor_to_int = {floor: i for i, floor in enumerate(self.floors)}
            self.int_to_floor = {i: floor for i, floor in enumerate(self.floors)}
        except ValueError:
             # Handle cases where floor names are not in 'f\d+' format
             print("Warning: Floor names not in 'f\\d+' format. Cannot create numerical floor mapping.")
             self.floors = sorted(list(floor_names_set)) # Fallback to alphabetical sort
             self.floor_to_int = {floor: i for i, floor in enumerate(self.floors)}
             self.int_to_floor = {i: floor for i, floor in enumerate(self.floors)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_cost = 0  # Initialize action cost counter.

        # Find current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # Identify required floors for lift movement
        required_floors = set()

        # Iterate through all known passengers
        for passenger in self.all_passengers:
            is_served = f"(served {passenger})" in state
            if not is_served:
                is_boarded = f"(boarded {passenger})" in state

                if not is_boarded:
                    # Passenger is not served and not boarded, must be at origin
                    # Find the current origin floor for this passenger in the state
                    current_origin_floor = None
                    for fact in state:
                        if match(fact, "origin", passenger, "*"):
                            current_origin_floor = get_parts(fact)[2]
                            break

                    # If passenger is not served, not boarded, and has an origin fact
                    if current_origin_floor:
                        total_cost += 2 # Needs board (1) and depart (1)
                        required_floors.add(current_origin_floor)
                        # Add destination floor if it exists for this passenger
                        if passenger in self.destinations:
                            required_floors.add(self.destinations[passenger])
                        # else: problem is malformed, passenger has no destination

                elif is_boarded:
                    # Passenger is boarded
                    total_cost += 1 # Needs depart (1)
                    # Add destination floor if it exists for this passenger
                    if passenger in self.destinations:
                        required_floors.add(self.destinations[passenger])
                    # else: problem is malformed, passenger has no destination


        # Calculate lift movement cost
        lift_movement_cost = 0
        # Ensure lift_floor is found and is a valid floor name
        if lift_floor in self.floor_to_int:
            # Filter out any required floors that were not found in our floor list (e.g., malformed problem)
            valid_required_floors = {f for f in required_floors if f in self.floor_to_int}

            if valid_required_floors:
                req_floor_ints = {self.floor_to_int[f] for f in valid_required_floors}
                min_req_int = min(req_floor_ints)
                max_req_int = max(req_floor_ints)
                lift_floor_int = self.floor_to_int[lift_floor]

                # Calculate movement cost: distance to closest extreme + span
                dist_to_min = abs(lift_floor_int - min_req_int)
                dist_to_max = abs(lift_floor_int - max_req_int)
                span = max_req_int - min_req_int

                lift_movement_cost = min(dist_to_min, dist_to_max) + span
        # else: lift_floor is missing or invalid, movement cost cannot be calculated accurately.
        # The heuristic will just be the sum of board/depart costs in this case.

        total_cost += lift_movement_cost

        return total_cost

