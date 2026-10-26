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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # If args is shorter than parts, ensure remaining parts match wildcards if args ends with *
    if len(args) < len(parts) and args[-1] != '*':
         return False
    return True


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    For each unserved passenger, it estimates the cost to:
    1. Move the lift to their origin floor (if not boarded).
    2. Board the passenger (if not boarded).
    3. Move the lift to their destination floor.
    4. Depart the passenger.
    The total heuristic is the sum of these estimated costs for all unserved passengers.

    # Assumptions
    - Floors are linearly ordered by the `(above f_higher f_lower)` predicate.
    - The cost of moving between adjacent floors is 1.
    - Boarding a passenger costs 1.
    - Departing a passenger costs 1.
    - The heuristic sums individual passenger costs, which might overestimate
      the true cost as lift movements can serve multiple passengers. This is
      acceptable for a non-admissible heuristic aiming to guide greedy search.
    - A passenger is either waiting at their origin, boarded, or served.

    # Heuristic Initialization
    - Parses the `(above f1 f2)` facts from static information to determine
      the linear order of floors and create a mapping from floor name to an
      integer index (0 for the lowest floor).
    - Extracts the destination floor for each passenger from the static
      `(destin p f)` facts.
    - Extracts the origin floor for each passenger from the static
      `(origin p f)` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state (set of facts).
    2. Find the current floor of the lift by locating the `(lift-at ?f)` fact.
       Convert the floor name to its integer index using the pre-calculated map.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through all passengers identified during initialization (from `destin` facts).
    5. For each passenger:
       - Check if the passenger is already served by looking for the fact `(served passenger)` in the current state. If found, this passenger requires 0 further actions, so continue to the next passenger.
       - If the passenger is not served, check if they are boarded by looking for the fact `(boarded passenger)` in the current state.
         - If boarded:
           - Get the passenger's destination floor using the pre-calculated map.
           - Calculate the number of floor moves required from the current lift floor to the destination floor (absolute difference in floor indices).
           - Add 1 for the `depart` action. This is the estimated cost for this passenger. Add this cost to the total.
         - If not boarded (and not served):
           - The passenger must be waiting at their origin floor. Get the origin floor using the pre-calculated map.
           - Get the passenger's destination floor using the pre-calculated map.
           - Calculate the number of floor moves required from the current lift floor to the origin floor.
           - Add 1 for the `board` action.
           - Calculate the number of floor moves required from the origin floor to the destination floor.
           - Add 1 for the `depart` action. This is the estimated cost for this passenger. Add this cost to the total.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger info.
        """
        self.goals = task.goals # Goal conditions (used implicitly by checking served)
        static_facts = task.static # Facts that are not affected by actions

        # 1. Determine floor order and create floor_to_index map
        lower_to_higher = {}
        all_floors = set()
        lower_floors_in_above = set()
        higher_floors_in_above = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                lower_to_higher[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                higher_floors_in_above.add(f_higher)
                lower_floors_in_above.add(f_lower)

        ordered_floors_asc = []
        if all_floors:
            # Find the lowest floor: the one that is a 'lower' floor but never a 'higher' floor
            # in any (above f_higher f_lower) fact.
            # Or, if there's only one floor, it's the lowest.
            potential_lowest = lower_floors_in_above - higher_floors_in_above
            if not potential_lowest and all_floors: # Case with only one floor or complex above structure
                 # If no clear lowest from the difference, find a floor that is not a 'higher' floor
                 # This should be the lowest in a linear chain
                 potential_lowest = all_floors - higher_floors_in_above
                 if not potential_lowest: # Handle cases where all floors are 'higher' (e.g., cyclic or single floor with no above)
                     # If still no clear lowest, just pick one (e.g., the first one found)
                     lowest_floor = next(iter(all_floors))
                 else:
                     lowest_floor = potential_lowest.pop()
            else:
                 lowest_floor = potential_lowest.pop()


            # Build ordered list from lowest to highest
            current_floor = lowest_floor
            while current_floor in lower_to_higher:
                 ordered_floors_asc.append(current_floor)
                 current_floor = lower_to_higher[current_floor]
            ordered_floors_asc.append(current_floor) # Add the highest floor

        # Create floor_to_index map
        self.floor_to_index = {floor: i for i, floor in enumerate(ordered_floors_asc)}

        # 2. Store passenger destinations and origins from static facts
        self.passenger_destinations = {}
        self.passenger_origins = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor
            elif match(fact, "origin", "*", "*"):
                 _, passenger, origin_floor = get_parts(fact)
                 self.passenger_origins[passenger] = origin_floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is not found (shouldn't happen in valid states), return infinity or a large number
        if current_lift_floor is None:
             return float('inf') # Should not occur in a valid state

        current_lift_floor_index = self.floor_to_index[current_lift_floor]

        total_cost = 0

        # Iterate through all passengers identified during initialization
        for passenger in self.passenger_destinations.keys():
            # Check if passenger is served
            if f"(served {passenger})" in state:
                continue # Passenger is served, cost is 0 for this passenger

            # Check if passenger is boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to go to destination and depart
                dest_f = self.passenger_destinations[passenger]
                dest_f_index = self.floor_to_index[dest_f]

                # Cost = moves to destination + depart
                cost = abs(current_lift_floor_index - dest_f_index) + 1
                total_cost += cost
                # continue # No need for continue, the next checks will fail anyway

            # If not served and not boarded, the passenger must be waiting at origin
            # We don't need to explicitly check for (origin p f) in state here
            # because if they are not served and not boarded, they must be at their origin
            # according to the domain's state transitions.
            # We use the origin floor stored from static facts.
            elif passenger in self.passenger_origins: # Ensure we know the origin for this passenger
                 origin_f = self.passenger_origins[passenger]
                 dest_f = self.passenger_destinations[passenger]

                 # Check if the passenger is actually at the origin floor in the current state
                 # This makes the heuristic more robust to potentially unusual states
                 if f"(origin {passenger} {origin_f})" in state:
                    orig_f_index = self.floor_to_index[origin_f]
                    dest_f_index = self.floor_to_index[dest_f]

                    # Cost = moves to origin + board + moves to destination + depart
                    cost = abs(current_lift_floor_index - orig_f_index) + 1 + abs(orig_f_index - dest_f_index) + 1
                    total_cost += cost
                    # continue # No need for continue

            # Note: If a passenger is unserved, not boarded, and not at their origin
            # (which shouldn't happen in a valid state sequence), they won't contribute
            # to the heuristic cost with this logic. This is acceptable for a non-admissible
            # heuristic focused on the standard flow.

        return total_cost

