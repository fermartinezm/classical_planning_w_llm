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

def get_floor_order(static_facts):
    """
    Maps floor names to integer indices based on 'above' facts.
    Assumes a linear ordering of floors defined by 'above' facts.
    """
    above_map = {} # floor -> floor immediately above it
    below_map = {} # floor -> floor immediately below it
    all_floors = set()

    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'above':
            f_above, f_below = parts[1], parts[2]
            above_map[f_below] = f_above
            below_map[f_above] = f_below
            all_floors.add(f_above)
            all_floors.add(f_below)

    if not all_floors:
        return {} # Should not happen in valid miconic problems

    # Find the lowest floor (a floor f such that there is no (above ?x f))
    lowest_floor = None
    # A floor is the lowest if no other floor is immediately below it.
    floors_with_floor_below = set(below_map.values())
    for floor in all_floors:
        if floor not in floors_with_floor_below:
             lowest_floor = floor
             break

    if lowest_floor is None and all_floors:
         # This case indicates a potential issue like a cycle or single floor
         # without explicit above/below. If there's only one floor, it's the lowest.
         if len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
         else:
             # Handle error or return empty if floor structure is invalid
             print("Warning: Could not determine lowest floor. Invalid 'above' facts?")
             return {}


    # Build the ordered list and mapping
    floor_to_int = {}
    current_floor = lowest_floor
    index = 0
    while current_floor is not None:
        floor_to_int[current_floor] = index
        index += 1
        current_floor = above_map.get(current_floor)

    return floor_to_int

def get_floor_distance(floor_to_int, f1, f2):
    """Calculates the number of moves between two floors."""
    if f1 not in floor_to_int or f2 not in floor_to_int:
        # This should not happen in valid miconic problems if floors are from the instance
        # Return a large value to indicate impossibility or error
        print(f"Warning: Floor {f1} or {f2} not found in floor mapping.")
        return float('inf')
    return abs(floor_to_int[f1] - floor_to_int[f2])


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions (board, depart, move)
    required to serve all passengers by summing the estimated cost for each
    unserved passenger independently. It is non-admissible but aims to guide
    a greedy best-first search effectively by providing a strong estimate
    of the remaining "work" per passenger.

    # Assumptions
    - Standard Miconic domain rules apply (lift moves between floors, passengers
      board at origin, depart at destination).
    - Each board, depart, and single-floor move action costs 1.
    - The cost for each passenger can be estimated independently and summed.
      This overcounts shared lift travel but provides a strong signal
      about the total amount of movement and actions required across all passengers.

    # Heuristic Initialization
    - Extracts the floor ordering from 'above' static facts to map floor names
      to integer indices for distance calculation.
    - Extracts the destination floor for each passenger from 'destin' static facts.
    - Identifies all passenger names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current floor of the lift by finding the fact `(lift-at ?f)` in the state.
    3. Check if the goal state is reached (all passengers served). If so, the heuristic is 0.
    4. Iterate through all passengers identified during initialization.
    5. For each passenger:
       a. Check if the passenger is already 'served' by looking for the fact `(served passenger)` in the state. If served, this passenger contributes 0 to the heuristic.
       b. If the passenger is not 'served':
          i. Check if the passenger is 'boarded' by looking for the fact `(boarded passenger)` in the state.
             - If boarded: This passenger needs to be transported from the current lift floor to their destination floor and then depart. The estimated cost for this passenger is 1 (for the depart action) plus the floor distance between the current lift floor and the passenger's destination floor. Add this cost to the total.
          ii. If not boarded (meaning they are waiting at their origin floor):
              - This passenger needs the lift to come to their origin floor, board, travel to their destination floor, and depart.
              - Find the passenger's origin floor by looking for the fact `(origin passenger floor)` in the state.
              - Find the passenger's destination floor using the pre-computed map from initialization.
              - The estimated cost for this passenger is 1 (for the board action) + 1 (for the depart action) + the floor distance between the current lift floor and the passenger's origin floor + the floor distance between the passenger's origin floor and their destination floor. Add this cost to the total.
    6. Return the total calculated cost, which is the sum of the estimated costs for all unserved passengers.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor order and destinations."""
        self.goals = task.goals # Goals are needed to check if a passenger is served
        static_facts = task.static

        # Map floors to integer indices based on 'above' facts
        self.floor_to_int = get_floor_order(static_facts)

        # Store passenger destinations
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, destination_floor = parts[1], parts[2]
                self.destinations[passenger] = destination_floor

        # Get all passenger names from destinations (assuming all passengers have a destination)
        self.all_passengers = set(self.destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached (all passengers served)
        # This is important for the heuristic to be 0 at the goal
        if self.goals <= state:
             return 0

        total_cost = 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This state is likely invalid or terminal without lift location
             # Return a high value to indicate it's likely not on a path to goal
             return float('inf') # Or a large constant

        # Pre-process state facts for faster lookups
        # Convert frozenset to set for O(1) average time complexity lookups
        # for 'in' operator, which is used repeatedly in the loop.
        state_set = set(state)

        for passenger in self.all_passengers:
            # Check if passenger is served
            if f'(served {passenger})' in state_set:
                continue # Passenger is served, contributes 0 cost

            # Check if passenger is boarded
            if f'(boarded {passenger})' in state_set:
                # Passenger is boarded, needs to travel to destination and depart
                destination_floor = self.destinations.get(passenger)
                if destination_floor is None:
                     # Invalid state: boarded passenger without destination
                     return float('inf') # Or handle error

                cost = 1 # Depart action
                cost += get_floor_distance(self.floor_to_int, current_lift_floor, destination_floor) # Travel to destination
                total_cost += cost
            else:
                # Passenger is waiting at origin, needs pickup, travel, and dropoff
                origin_floor = None
                # Find the origin floor for this specific passenger
                for fact in state_set:
                    parts = get_parts(fact)
                    if parts[0] == 'origin' and parts[1] == passenger:
                        origin_floor = parts[2]
                        break

                if origin_floor is None:
                    # Invalid state: unserved, not boarded passenger without origin
                    return float('inf') # Or handle error

                destination_floor = self.destinations.get(passenger)
                if destination_floor is None:
                     # Invalid state: waiting passenger without destination
                     return float('inf') # Or handle error

                cost = 1 # Board action
                cost += 1 # Depart action
                cost += get_floor_distance(self.floor_to_int, current_lift_floor, origin_floor) # Travel to origin
                cost += get_floor_distance(self.floor_to_int, origin_floor, destination_floor) # Travel from origin to destination
                total_cost += cost

        return total_cost
