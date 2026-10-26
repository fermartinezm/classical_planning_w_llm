# Add necessary imports
from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # print(f"Warning: get_parts received unexpected fact format: {fact}")
        return [] # Or raise an error

    # Remove outer parentheses and split by spaces
    parts = fact[1:-1].split()
    return parts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the required number of actions (moves, boards, departs)
    to serve all passengers. It calculates the cost for each unserved passenger
    independently and sums these costs. The cost for a passenger includes the
    estimated lift travel to their origin (if waiting), the board action, the
    estimated lift travel to their destination, and the depart action.

    # Assumptions
    - Floors are linearly ordered, defined by the `above` predicate.
    - The cost of moving the lift one floor is 1.
    - The cost of a `board` action is 1.
    - The cost of a `depart` action is 1.
    - The heuristic sums the costs for each passenger as if they were served
      sequentially and independently, which makes it non-admissible but potentially
      informative for greedy search.

    # Heuristic Initialization
    - Parses the `above` static facts to build a mapping from floor names to
      numerical indices, establishing the floor order.
    - Identifies the destination floor for each passenger from the `destin`
      static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Identify all passengers who are not yet `served`.
    3. For each unserved passenger:
       a. Determine their destination floor using the pre-calculated destination map.
       b. Check if the passenger is waiting at their origin floor or is already boarded.
       c. If the passenger is waiting at their origin floor:
          - Calculate the estimated move cost for the lift to go from its current
            floor to the passenger's origin floor (absolute difference in floor indices).
          - Add 1 for the `board` action.
          - Calculate the estimated move cost for the lift to go from the passenger's
            origin floor to their destination floor (absolute difference in floor indices).
          - Add 1 for the `depart` action.
          - Sum these four components to get the cost for this passenger.
       d. If the passenger is boarded:
          - Calculate the estimated move cost for the lift to go from its current
            floor to the passenger's destination floor (absolute difference in floor indices).
          - Add 1 for the `depart` action.
          - Sum these two components to get the cost for this passenger.
    4. The total heuristic value is the sum of the costs calculated for each
       unserved passenger.
    5. If all passengers are served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        super().__init__(task) # Call base class constructor

        # Build floor index mapping from static 'above' facts
        self.floor_indices = {}
        below_to_above = {}
        all_f1_in_above = set()
        all_f2_in_above = set()
        all_floors_mentioned_in_above = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                below_to_above[f1] = f2
                all_f1_in_above.add(f1)
                all_f2_in_above.add(f2)
                all_floors_mentioned_in_above.add(f1)
                all_floors_mentioned_in_above.add(f2)

        # Find the lowest floor: it's a floor that is a first argument in 'above'
        # but never a second argument.
        lowest_floor = None
        possible_lowest_floors = list(all_f1_in_above - all_f2_in_above)

        if len(possible_lowest_floors) == 1:
             lowest_floor = possible_lowest_floors[0]
        elif len(all_floors_mentioned_in_above) == 1: # Case with only one floor mentioned in above
             lowest_floor = list(all_floors_mentioned_in_above)[0]
        elif not all_floors_mentioned_in_above:
             # No 'above' facts found. Try to find any floor mentioned in static facts or goals.
             # This is a fallback for unusual domain definitions.
             all_objects_mentioned = set()
             for fact in self.static:
                 all_objects_mentioned.update(get_parts(fact)[1:])
             for goal in self.goals:
                 all_objects_mentioned.update(get_parts(goal)[1:])

             # Heuristic guess: objects starting with 'f' are floors.
             all_floor_objects = sorted([obj for obj in all_objects_mentioned if obj.startswith('f')])

             if all_floor_objects:
                 # Assume alphabetically first floor is the lowest if no 'above' facts define order
                 lowest_floor = all_floor_objects[0]
                 # If there's only one floor, its index is 0
                 if len(all_floor_objects) == 1:
                     self.floor_indices[lowest_floor] = 0
                 else:
                     # Cannot determine order without 'above' facts. Heuristic will be inaccurate.
                     # For robustness, assign arbitrary indices 0, 1, 2...
                     for i, floor in enumerate(all_floor_objects):
                         self.floor_indices[floor] = i
             # else: lowest_floor remains None


        if lowest_floor and not self.floor_indices: # Only build index map if we found a lowest floor and it wasn't already built in the fallback
            current_floor = lowest_floor
            index = 0
            self.floor_indices[current_floor] = index
            # Build the chain upwards
            while current_floor in below_to_above:
                next_floor = below_to_above[current_floor]
                index += 1
                self.floor_indices[next_floor] = index
                current_floor = next_floor

        # Extract passenger destinations from static 'destin' facts
        self.destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # Get all passenger names from destinations
        self.all_passengers = set(self.destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached (all passengers served)
        # The goal is (and (served p1) (served p2) ...)
        # We can check if the number of served passengers equals the total number of passengers
        served_passengers_count = sum(1 for fact in state if match(fact, "served", "*"))
        if served_passengers_count == len(self.all_passengers):
             return 0 # Goal state

        # Find current lift location
        current_floor_str = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_floor_str = parts[1]
                break

        if current_floor_str is None:
             # This state is likely invalid or terminal (e.g., no lift-at fact)
             # Return infinity or a very large number
             return float('inf') # Cannot proceed without lift location

        # Get floor index for current location
        current_floor_idx = self.floor_indices.get(current_floor_str)
        if current_floor_idx is None:
             # Current floor not found in floor index map - problem with parsing or domain
             # This might happen with the fallback floor indexing if 'lift-at' floor wasn't found
             return float('inf')


        total_cost = 0

        # Identify served passengers for quick lookup
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify waiting passengers and their origins
        waiting_passengers = {} # {passenger: origin_floor_str}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'origin' and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 waiting_passengers[p] = f

        # Identify boarded passengers
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}


        # Calculate cost for each unserved passenger
        for passenger in self.all_passengers:
            if passenger in served_passengers:
                continue # Passenger is already served

            # Passenger is unserved, find their destination
            destin_floor_str = self.destinations.get(passenger)
            if destin_floor_str is None:
                 # Destination not found for this passenger - problem with static facts
                 # This passenger cannot be served, state is likely unsolvable relative to this goal
                 # print(f"Warning: Destination not found for passenger {passenger}")
                 # Returning inf might be better here if a passenger cannot be served
                 return float('inf')

            destin_floor_idx = self.floor_indices.get(destin_floor_str)
            if destin_floor_idx is None:
                 # Destination floor not found in floor index map
                 # print(f"Warning: Destination floor {destin_floor_str} not found in index map for passenger {passenger}")
                 # Returning inf might be better here if a passenger cannot be served
                 return float('inf')


            if passenger in waiting_passengers:
                # Passenger is waiting at origin
                origin_floor_str = waiting_passengers[passenger]
                origin_floor_idx = self.floor_indices.get(origin_floor_str)
                if origin_floor_idx is None:
                     # Origin floor not found in floor index map
                     # print(f"Warning: Origin floor {origin_floor_str} not found in index map for passenger {passenger}")
                     # Returning inf might be better here if a passenger cannot be served
                     return float('inf')

                # Cost = move to origin + board + move to destin + depart
                # abs(current_floor_idx - origin_floor_idx) : moves to pick up
                # 1 : board action
                # abs(origin_floor_idx - destin_floor_idx) : moves to drop off
                # 1 : depart action
                cost_p = abs(current_floor_idx - origin_floor_idx) + 1 + abs(origin_floor_idx - destin_floor_idx) + 1
                total_cost += cost_p

            elif passenger in boarded_passengers:
                # Passenger is boarded
                # Cost = move to destin + depart
                # abs(current_floor_idx - destin_floor_idx) : moves to drop off
                # 1 : depart action
                cost_p = abs(current_floor_idx - destin_floor_idx) + 1
                total_cost += cost_p
            # Else: Passenger is not served, not waiting, not boarded. This indicates an invalid state.
            # Returning inf is appropriate here.
            else:
                 # print(f"Warning: Passenger {passenger} is unserved but not waiting or boarded.")
                 return float('inf')


        return total_cost
