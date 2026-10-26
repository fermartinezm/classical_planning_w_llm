from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Splits a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each unserved passenger independently. The estimated cost for a
    passenger includes the travel cost for the lift to reach their origin
    (if waiting) and destination, plus the cost of the board and depart
    actions. Travel cost between floors is estimated using the Manhattan
    distance based on a pre-calculated floor index mapping. This heuristic
    is not admissible as it overestimates travel by not accounting for
    shared lift movements, but aims to guide a greedy best-first search
    effectively by prioritizing states where passengers are closer to being
    served.

    Assumptions:
    - Floors are linearly ordered and defined by `(above f_lower f_higher)`
      facts in the static information, where `f_higher` is immediately above
      `f_lower`. These facts form a single chain from the lowest to the highest floor.
    - Passenger origins and destinations are fixed and provided in the
      static information via `(origin ?p ?f)` and `(destin ?p ?f)` facts
      in the initial state (which are treated as static for this purpose).
    - Action costs are uniform (implicitly 1).

    Heuristic Initialization:
    The constructor processes the static facts (including initial state facts
    that define origins and destinations) to build the following data structures:
    - `floor_to_index`: A dictionary mapping floor object names (e.g., 'f1')
      to integer indices representing their position in the floor order (e.g., 1).
      This mapping is built by identifying the lowest floor (one that is not
      the higher floor in any `(above f_lower f_higher)` fact) and following
      the chain of `above` facts upwards.
    - `passenger_origins`: A dictionary mapping passenger object names (e.g., 'p1')
      to their origin floor object name.
    - `passenger_destinations`: A dictionary mapping passenger object names
      to their destination floor object name.
    - `all_passengers`: A set of all passenger object names mentioned in
      origin or destination facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic cost to 0.
    2. Identify the current floor of the lift from the state fact `(lift-at ?f)`.
    3. Get the integer index of the current lift floor using the pre-calculated
       `floor_to_index` map. If the floor is not found in the map (e.g., if
       there were no `above` facts), use index 0.
    4. Iterate through each passenger identified during initialization.
    5. For the current passenger `p`:
       a. Check if the fact `(served p)` is present in the current state. If yes,
          this passenger is served and contributes 0 to the remaining cost; continue
          to the next passenger.
       b. If the passenger is not served, check if the fact `(boarded p)` is
          present in the current state.
       c. If `(boarded p)` is present:
          i. This passenger is inside the lift and needs to be dropped off.
          ii. Retrieve the passenger's destination floor `d` from the
              pre-calculated `passenger_destinations` map.
          iii. Get the integer index of the destination floor `d_idx` using
              `floor_to_index`. If the floor is not found, use index 0.
          iv. The estimated cost for this passenger is the Manhattan distance
              between the current lift floor index and the destination floor index
              (`abs(current_idx - d_idx)`) representing travel, plus 1 for the
              `depart` action. Add this cost to the total.
       d. If `(boarded p)` is not present (and `(served p)` is not present),
          the passenger must be waiting at their origin floor.
          i. Retrieve the passenger's origin floor `o` from the pre-calculated
              `passenger_origins` map.
          ii. Retrieve the passenger's destination floor `d` from the
              pre-calculated `passenger_destinations` map.
          iii. Get the integer indices `o_idx` and `d_idx` using `floor_to_index`.
              If a floor is not found, use index 0.
          iv. The estimated cost for this passenger is the Manhattan distance
              from the current lift floor index to the origin floor index
              (`abs(current_idx - o_idx)`) for the first travel leg, plus 1
              for the `board` action, plus the Manhattan distance from the
              origin floor index to the destination floor index
              (`abs(o_idx - d_idx)`) for the second travel leg, plus 1 for the
              `depart` action. Add this cost to the total.
    6. After iterating through all passengers, the total accumulated cost is the
       heuristic value for the given state.
    """
    def __init__(self, task):
        self.goals = task.goals
        # static_facts includes initial state facts for origins/destinations
        # and also the 'above' facts.
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Extract passenger origins and destinations from static facts (initial state)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'origin':
                p, f = parts[1], parts[2]
                self.passenger_origins[p] = f
                self.all_passengers.add(p)
            elif parts[0] == 'destin':
                p, f = parts[1], parts[2]
                self.passenger_destinations[p] = f
                self.all_passengers.add(p)

        # Build floor order and index mapping from 'above' facts
        self.floor_to_index = {}
        floors = set()
        # Floors that appear as the second argument in 'above' (the higher floor)
        floors_that_are_higher = set()
        # Map from lower floor to the floor immediately above it
        floor_lower_to_floor_higher = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_lower, f_higher = parts[1], parts[2]
                floors.add(f_lower)
                floors.add(f_higher)
                floors_that_are_higher.add(f_higher)
                floor_lower_to_floor_higher[f_lower] = f_higher

        # Find the bottom floor: a floor that is mentioned but is not the higher floor in any 'above' fact
        bottom_floor = None
        for f in floors:
             if f not in floors_that_are_higher:
                 bottom_floor = f
                 break

        # If bottom_floor is found, build the index map by following the 'higher' chain
        if bottom_floor:
            current_floor = bottom_floor
            index = 1
            while current_floor is not None:
                 self.floor_to_index[current_floor] = index
                 current_floor = floor_lower_to_floor_higher.get(current_floor)
                 index += 1

        # Handle the case where there's only one floor or no 'above' facts
        # If floor_to_index is still empty but there are floors mentioned (e.g., in origin/destin)
        # this means the 'above' structure was not a simple chain or was missing.
        # In this case, floor indices default to 0 in __call__, making travel cost 0.
        # If there's exactly one floor mentioned anywhere, assign it index 1.
        if not self.floor_to_index and floors:
             if len(floors) == 1:
                  self.floor_to_index[list(floors)[0]] = 1
             # Otherwise, floor_to_index remains empty, get(floor, 0) will be used.


    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # If lift location is not found, heuristic is infinite (or a large value)
        # as the state is likely invalid or unsolvable from here.
        # Assuming valid states always have lift-at.
        if current_lift_floor is None:
             # This case indicates a serious problem state, likely unsolvable.
             # Returning a large value guides search away from such states.
             return 1000000 # Use a large finite number instead of inf

        # Get index of current lift floor. If floor not in map (unexpected), use 0.
        current_idx = self.floor_to_index.get(current_lift_floor, 0)

        for passenger in self.all_passengers:
            is_served = f'(served {passenger})' in state
            is_boarded = f'(boarded {passenger})' in state

            if not is_served:
                if is_boarded:
                    # Passenger is boarded, needs to go to destination and depart
                    dest_floor = self.passenger_destinations.get(passenger)
                    if dest_floor: # Should always exist for passengers in all_passengers
                        dest_idx = self.floor_to_index.get(dest_floor, 0) # Default 0 if floor not in map
                        # Cost = travel from current lift floor to destination + depart action
                        total_cost += abs(current_idx - dest_idx) + 1
                    # else: Destination not found, problem definition issue.
                    #      If destination is missing for an unserved passenger, problem is likely unsolvable.
                    #      The .get() returning None is handled by the outer 'if dest_floor:'
                    #      If dest_floor exists but is not in floor_to_index, dest_idx becomes 0.
                    #      This is a potential inaccuracy if floor structure is weird, but safe.

                else: # Passenger is waiting at origin
                    origin_floor = self.passenger_origins.get(passenger)
                    dest_floor = self.passenger_destinations.get(passenger)

                    if origin_floor and dest_floor: # Should always exist
                        origin_idx = self.floor_to_index.get(origin_floor, 0) # Default 0 if floor not in map
                        dest_idx = self.floor_to_index.get(dest_floor, 0) # Default 0 if floor not in map

                        # Cost = travel from current lift floor to origin + board action
                        #      + travel from origin to destination + depart action
                        total_cost += abs(current_idx - origin_idx) + 1 + abs(origin_idx - dest_idx) + 1
                    # else: Origin/Destination not found, problem definition issue.
                    #      If origin/destination is missing for an unserved passenger, problem is likely unsolvable.
                    #      Handled by the 'if origin_floor and dest_floor:' check.

        return total_cost
