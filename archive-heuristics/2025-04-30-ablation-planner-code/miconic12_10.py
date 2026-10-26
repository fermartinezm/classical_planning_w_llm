# Assuming Heuristic base class is available in a planning framework
# from heuristics.heuristic_base import Heuristic
# If running standalone, use the dummy class defined below
import sys
import os

# Add the parent directory to the path to import heuristic_base
# This is a common pattern in planning frameworks
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback for standalone testing or different project structure
    # Define a dummy Heuristic class if heuristic_base is not found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError
        def __str__(self):
            return self.__class__.__name__


from fnmatch import fnmatch

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
    # Ensure the number of parts matches the number of args if args are not wildcards
    if len(parts) != len(args) and '*' not in args:
        return False
    # Use zip to handle cases where parts might be shorter than args (shouldn't happen with valid facts/patterns)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers by summing
    the estimated minimum actions required for each individual unserved passenger,
    ignoring potential synergies from serving multiple passengers in one trip.

    # Assumptions
    - Floors are linearly ordered. The predicate `(above f_higher f_lower)` means
      `f_higher` is immediately above `f_lower`.
    - Floors are named `f1, f2, ..., fn`. `f1` is the highest floor, `fn` is the lowest.
    - The actions `up` and `down` move the lift one floor level at a time.
    - The cost of each action (board, depart, up, down) is 1.
    - An unserved passenger is either waiting at their origin floor or is boarded.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Determines the linear order of floors based on naming (`f1` highest, `fn` lowest)
      and maps each floor object to a numerical level (f1=1, f2=2, ..., fn=n).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift and its corresponding level.
    2. Initialize the total heuristic value to 0.
    3. Get the set of all passengers with defined destinations (from static facts).
    4. Identify passengers who are currently served.
    5. Identify passengers who are currently boarded.
    6. Identify passengers who are currently waiting at their origin floors.
    7. For each passenger who is not served:
       a. If the passenger is boarded:
          - Get the passenger's destination floor and its level.
          - Estimate the cost for this passenger: `abs(current_lift_level - destination_level) + 1` (move to destination + depart action).
          - Add this cost to the total heuristic.
       b. If the passenger is waiting at their origin floor:
          - Get the passenger's origin floor and its level.
          - Get the passenger's destination floor and its level.
          - Estimate the cost for this passenger: `abs(current_lift_level - origin_level) + 1 + abs(origin_level - destination_level) + 1` (move to origin + board action + move to destination + depart action).
          - Add this cost to the total heuristic.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        static_facts = task.static

        # 1. Map floors to levels
        # Collect all floor objects mentioned in static facts
        all_floors = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] in ["above", "destin"]:
                 for part in parts[1:]:
                     # Simple check if it looks like a floor name (starts with 'f')
                     if isinstance(part, str) and part.startswith('f'):
                         all_floors.add(part)

        # Sort floors based on the number part (f1, f2, ...)
        # Assuming floor names are like 'f' followed by a number
        # This establishes the linear order f1, f2, ..., fn
        floor_names = sorted(list(all_floors), key=lambda f: int(f[1:]))

        # Map f_i to level i (f1=1, f2=2, ..., fn=n)
        # This mapping implies f1 is the highest level, fn is the lowest level
        self.floor_to_level = {f_name: i + 1 for i, f_name in enumerate(floor_names)}

        # 2. Store passenger destinations from static facts
        self.passenger_to_destin = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_to_destin[passenger] = floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is not in state (should not happen in valid states)
        if current_lift_floor is None:
             # Cannot compute heuristic without lift location
             # Returning infinity signals this state is likely unreachable or undesirable.
             return float('inf')

        # If current_lift_floor is not in floor_to_level, something is wrong
        if current_lift_floor not in self.floor_to_level:
             # This could happen if the initial state floor wasn't mentioned in static facts (e.g., above/destin)
             # A more robust parser would get objects from task.objects
             # For this problem, assuming floors in state are always in static facts that define levels.
             return float('inf')


        level_current = self.floor_to_level[current_lift_floor]

        total_heuristic = 0

        # Get sets of passengers based on their current state predicates
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_passengers_dict = {} # {passenger: floor}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                origin_passengers_dict[p] = f

        # Consider all passengers whose destinations are known (from static facts)
        all_passengers_with_dest = set(self.passenger_to_destin.keys())

        for passenger in all_passengers_with_dest:
            # If passenger is served, they contribute 0 to the heuristic
            if passenger in served_passengers:
                continue

            # Passenger is unserved. Check their current state.
            destin_floor = self.passenger_to_destin.get(passenger)
            if destin_floor is None:
                 # Should not happen if passenger_to_destin is built correctly from static
                 # and all unserved passengers have destinations.
                 continue # Cannot compute heuristic for passenger without destination

            # If destination floor is not in floor_to_level, something is wrong
            if destin_floor not in self.floor_to_level:
                 return float('inf')

            level_destin = self.floor_to_level[destin_floor]

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to depart at destination
                # Cost: move from current lift floor to destination floor + depart action
                h_p = abs(level_current - level_destin) + 1
                total_heuristic += h_p

            elif passenger in origin_passengers_dict:
                # Passenger is at origin, needs board and depart
                origin_floor = origin_passengers_dict[passenger]

                # If origin floor is not in floor_to_level, something is wrong
                if origin_floor not in self.floor_to_level:
                     return float('inf')

                level_origin = self.floor_to_level[origin_floor]

                # Cost: move from current lift floor to origin floor + board action
                #       + move from origin floor to destination floor + depart action
                h_p = abs(level_current - level_origin) + 1 + abs(level_origin - level_destin) + 1
                total_heuristic += h_p

            # If a passenger is unserved but neither boarded nor at origin,
            # this state is likely invalid or represents a passenger who was
            # dropped off at a non-destination floor. The current heuristic
            # doesn't explicitly handle this, assuming valid state transitions.
            # In a valid state, an unserved passenger is either at origin or boarded.
            # Such passengers implicitly contribute 0 if they don't match boarded or origin.
            # This is acceptable for a non-admissible heuristic on valid states.


        return total_heuristic
