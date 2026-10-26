from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It calculates the estimated cost for each unserved passenger
    independently and sums these costs. The cost for a single passenger includes
    movement actions for the lift to reach the pickup and dropoff floors, plus
    the board and depart actions.

    # Assumptions
    - Floors are ordered linearly, defined by the 'above' predicate chain.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates the cost for each passenger as if they were
      transported individually, ignoring potential efficiencies from batching
      pickups and dropoffs or shared movements. This makes it non-admissible
      but potentially effective for greedy search by prioritizing states where
      passengers are closer to being served.

    # Heuristic Initialization
    - Parses the 'above' facts from the static information to build a map
      from floor names to numerical indices, representing their order. It assumes
      a linear sequence of floors where `(above f_lower f_upper)` means `f_upper`
      is immediately above `f_lower`.
    - Parses the 'destin' facts from the static information to map each
      passenger to their destination floor.
    - Identifies all passenger objects that need to be served based on the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the goal state is reached. If yes, the heuristic is 0.
    2. Find the current floor of the lift. If the lift location is unknown, return infinity.
    3. Get the numerical index for the current lift floor. If the floor is not indexed, return infinity.
    4. Initialize the total heuristic cost to 0.
    5. Iterate through all passenger objects identified during initialization (those that need to be served).
    6. For each passenger `p`:
       a. Check if `(served p)` is true in the current state. If yes, the cost for this passenger is 0, continue to the next passenger.
       b. If `(served p)` is false:
          i. Find the destination floor `f_destin` for `p` using the pre-calculated map. If destination is unknown, return infinity.
          ii. Get the numerical index for `f_destin`. If the floor is not indexed, return infinity.
          iii. Check if `(origin p f_origin)` is true in the current state.
              - If yes (passenger is waiting at origin `f_origin`):
                - Get the numerical index for `f_origin`. If the floor is not indexed, return infinity.
                - Estimated cost for `p` = (moves from current lift floor to `f_origin`) + 1 (board) + (moves from `f_origin` to `f_destin`) + 1 (depart).
                - Moves are estimated as the absolute difference in floor indices: `abs(index(current_lift_floor) - index(f_origin))` and `abs(index(f_origin) - index(f_destin))`.
              - If no (passenger must be boarded, `(boarded p)` is true):
                - Estimated cost for `p` = (moves from current lift floor to `f_destin`) + 1 (depart).
                - Moves are estimated as `abs(index(current_lift_floor) - index(f_destin))`.
              - If the passenger is not served, not at origin, and not boarded, this indicates an invalid state; return infinity.
          iv. Add the estimated cost for passenger `p` to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions (used to identify all passengers that need serving)
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor index map from 'above' facts
        # Map f_lower -> f_upper if (above f_lower f_upper)
        above_map = {}
        is_below_of = set() # Floors that are the second argument in 'above' (i.e., have a floor above them)
        all_floors_set = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_lower, f_upper = get_parts(fact)[1:] # Correct order based on action logic (f_upper is above f_lower)
                above_map[f_lower] = f_upper
                is_below_of.add(f_upper) # f_upper is above f_lower, so f_upper is 'below_of' some floor higher up (or is the top)
                all_floors_set.add(f_upper)
                all_floors_set.add(f_lower)

        # Find the lowest floor (a floor in all_floors_set that is not the second argument in any 'above' fact)
        # This means no floor is immediately below it.
        lowest_floor = None
        floors_that_are_above_something = set(above_map.keys())
        for floor in all_floors_set:
             # A floor is the lowest if it's mentioned, but no other floor is immediately below it.
             # In the mapping f_lower -> f_upper, f_lower is below f_upper.
             # The lowest floor is one that is not a value (f_upper) in the above_map.
             if floor not in above_map.values():
                 lowest_floor = floor
                 break


        if lowest_floor is None and all_floors_set:
             # This might indicate a cycle or no clear lowest floor among those mentioned in 'above'.
             print("Warning: Could not determine lowest floor from 'above' facts. Heuristic may be inaccurate.")
             self.floor_to_index = {}
        elif lowest_floor is None and not all_floors_set:
             # No floors defined, likely a trivial problem or error. Empty map is fine.
             self.floor_to_index = {}
        else:
            # Found the lowest floor, build the index map by following the chain upwards
            self.floor_to_index = {}
            current_index = 1
            current_floor = lowest_floor
            indexed_count = 0

            # Build a reverse map: f_upper -> f_lower
            below_map = {v: k for k, v in above_map.items()}

            # Start from the lowest floor and go up
            current = lowest_floor
            while current is not None:
                 if current in self.floor_to_index:
                      # Cycle detected? Should not happen in valid miconic.
                      print(f"Warning: Cycle detected in floor 'above' relations at {current}. Heuristic may be inaccurate.")
                      self.floor_to_index = {} # Invalidate map if structure is unexpected
                      break
                 self.floor_to_index[current] = current_index
                 indexed_count += 1
                 current_index += 1
                 # Find the floor immediately above the current one
                 current = above_map.get(current)


            if indexed_count != len(all_floors_set):
                 print(f"Warning: Indexed {indexed_count} floors but found {len(all_floors_set)} total floors mentioned in 'above'. Problem structure might be non-linear or disconnected.")
                 # The heuristic will only work for floors that were successfully indexed.
                 # If a required floor isn't indexed, we'll return infinity in __call__.


        # 2. Map passengers to their destination floors
        self.passenger_destin = {}
        # Passengers that need to be served are those mentioned in the goal
        self.passengers_to_serve = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                p = get_parts(goal)[1]
                self.passengers_to_serve.add(p)

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, f_destin = get_parts(fact)[1:]
                # Only store destinations for passengers we care about serving
                if p in self.passengers_to_serve:
                    self.passenger_destin[p] = f_destin


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Check if floor indexing failed during init
        if not self.floor_to_index:
             # Cannot compute heuristic without floor order
             return float('inf')

        total_cost = 0  # Initialize action cost counter.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This state is likely invalid or terminal without lift.
             return float('inf')

        current_lift_floor_idx = self.floor_to_index.get(current_lift_floor)
        if current_lift_floor_idx is None:
             # Current lift floor not found in floor map - problem with floor parsing or instance.
             print(f"Warning: Current lift floor '{current_lift_floor}' not found in floor index map.")
             return float('inf')


        # Track passenger states
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        for p in self.passengers_to_serve:
            if p in served_passengers:
                # Passenger is already served, no cost.
                continue

            # Passenger is not served. Calculate cost for this passenger.
            p_destin_floor = self.passenger_destin.get(p)
            if p_destin_floor is None:
                 # Passenger needs serving but has no destination defined. Invalid problem?
                 print(f"Warning: Passenger '{p}' needs serving but has no destination.")
                 return float('inf')

            p_destin_floor_idx = self.floor_to_index.get(p_destin_floor)
            if p_destin_floor_idx is None:
                 # Destination floor not found in floor map.
                 print(f"Warning: Destination floor '{p_destin_floor}' for passenger '{p}' not found in floor index map.")
                 return float('inf')

            if p in boarded_passengers:
                # Passenger is boarded, needs to be dropped off at destination.
                # Cost = moves from current lift floor to destination + 1 (depart)
                moves_to_destin = abs(current_lift_floor_idx - p_destin_floor_idx)
                total_cost += moves_to_destin + 1

            elif p in origin_locations:
                # Passenger is waiting at origin, needs pickup and dropoff.
                p_origin_floor = origin_locations[p]
                p_origin_floor_idx = self.floor_to_index.get(p_origin_floor)

                if p_origin_floor_idx is None:
                    # Origin floor not found in floor map.
                    print(f"Warning: Origin floor '{p_origin_floor}' for passenger '{p}' not found in floor index map.")
                    return float('inf')

                # Cost = moves from current lift floor to origin + 1 (board)
                #      + moves from origin to destination + 1 (depart)
                moves_to_origin = abs(current_lift_floor_idx - p_origin_floor_idx)
                moves_origin_to_destin = abs(p_origin_floor_idx - p_destin_floor_idx)
                total_cost += moves_to_origin + 1 + moves_origin_to_destin + 1
            else:
                # Passenger is not served, not boarded, and not at origin.
                # This state should not be reachable in a valid miconic problem.
                # If a passenger is not served, they must be either at origin or boarded.
                print(f"Warning: Passenger '{p}' is not served, not boarded, and not at origin.")
                return float('inf') # Indicate invalid state

        return total_cost

