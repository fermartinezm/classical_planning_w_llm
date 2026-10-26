from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Note: The Heuristic base class is assumed to be provided by the environment.
# A dummy class is not included in the final output.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Handle cases where fact might already be a list/tuple (e.g., from internal representation)
    # Or return None or raise error for invalid input
    return [] # Return empty list for invalid format


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the estimated cost for each unserved passenger
    independently. The estimated cost for a passenger includes the travel
    cost for the lift to reach the passenger's origin (if unboarded),
    the board action, the travel cost from origin to destination, and
    the depart action. For boarded passengers, it includes travel from
    the current lift location to the destination and the depart action.
    Travel cost is estimated as the absolute difference in floor levels.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < ...).
    - The 'above' predicates define this linear ordering, confirming the numerical order.
    - Each action (move, board, depart) costs 1.
    - The heuristic calculation for each passenger is independent,
      overestimating travel when multiple passengers share lift movement.

    # Heuristic Initialization
    - Parses static facts to determine the level (integer) for each floor
      based on the 'above' relationships and floor names. Assumes floor names
      like 'fN' allow numerical sorting.
    - Stores the origin and destination floor for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)` in the state.
    2. Convert the current lift floor name to its corresponding integer level using the pre-calculated floor mapping.
    3. Initialize the total heuristic value `h` to 0.
    4. Iterate through all passengers whose origin and destination were identified during initialization.
    5. For each passenger:
       a. Check if the passenger is already 'served' by looking for the fact `(served ?p)` in the current state. If the passenger is served, they contribute 0 to the heuristic.
       b. If the passenger is not served, check if they are 'boarded' by looking for the fact `(boarded ?p)` in the current state.
       c. If the passenger is not boarded (meaning they are still at their origin floor):
          - Retrieve their origin and destination floor names and convert them to integer levels.
          - Estimate the minimum actions needed *for this passenger alone*, starting from the current lift position:
            - Travel from the current lift floor to the passenger's origin floor: `abs(current_level - origin_level)` move actions.
            - Board the passenger: 1 board action.
            - Travel from the origin floor to the destination floor: `abs(destin_level - origin_level)` move actions.
            - Depart the passenger: 1 depart action.
          - Add the sum of these actions (`abs(current_level - origin_level) + 1 + abs(destin_level - origin_level) + 1`) to the total heuristic value `h`.
       d. If the passenger is boarded:
          - Retrieve their destination floor name and convert it to an integer level.
          - Estimate the minimum actions needed *for this passenger alone*, starting from the current lift position:
            - Travel from the current lift floor to the passenger's destination floor: `abs(current_level - destin_level)` move actions.
            - Depart the passenger: 1 depart action.
          - Add the sum of these actions (`abs(current_level - destin_level) + 1`) to the total heuristic value `h`.
    6. Return the total accumulated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        super().__init__(task) # Call base class constructor

        self.floor_levels = {}
        self.passenger_info = {} # {passenger_name: {'origin': floor_name, 'destin': floor_name}}
        all_floors = set()
        all_passengers = set()

        # Collect all floors and passengers from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'origin':
                p, f = parts[1:]
                all_passengers.add(p)
                all_floors.add(f)
                if p not in self.passenger_info: self.passenger_info[p] = {}
                self.passenger_info[p]['origin'] = f
            elif parts[0] == 'destin':
                p, f = parts[1:]
                all_passengers.add(p)
                all_floors.add(f)
                if p not in self.passenger_info: self.passenger_info[p] = {}
                self.passenger_info[p]['destin'] = f
            elif parts[0] == 'above':
                 f1, f2 = parts[1:]
                 all_floors.add(f1)
                 all_floors.add(f2)

        # Assuming floor names are like 'fN' and can be sorted numerically
        # This is a common convention in Miconic benchmarks.
        # A more general approach would build a graph from 'above' facts
        # and perform a topological sort or level assignment.
        try:
            # Sort floors based on the integer part of their name (e.g., 'f1' -> 1)
            sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
            # Assign levels (1-based indexing)
            for i, f in enumerate(sorted_floors):
                self.floor_levels[f] = i + 1
        except (ValueError, IndexError):
             # Fallback or error handling if floor names are not in the expected 'fN' format.
             # For this problem, we proceed assuming the 'fN' format is standard.
             # In a real-world scenario, robust parsing of 'above' facts is needed.
             print(f"Warning: Floor names {all_floors} not in expected 'fN' format. Floor leveling might be incorrect.")
             # Assign levels based on alphabetical sort as a fallback (less ideal)
             sorted_floors = sorted(list(all_floors))
             for i, f in enumerate(sorted_floors):
                 self.floor_levels[f] = i + 1


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of facts)

        h = 0

        # Find current lift floor
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # current_floor should always be found in a valid state
        if current_floor is None:
             # This indicates an unexpected state. Return a high value.
             return float('inf') # Or handle as an error

        current_level = self.floor_levels[current_floor]

        # Iterate through all passengers and sum their estimated costs
        for p, info in self.passenger_info.items():
            f_origin = info['origin']
            f_destin = info['destin']
            origin_level = self.floor_levels[f_origin]
            destin_level = self.floor_levels[f_destin]

            # Check if passenger is served
            if '(served ' + p + ')' not in state:
                # Passenger is unserved
                if '(boarded ' + p + ')' not in state:
                    # Passenger is at origin, not boarded
                    # Needs: travel to origin + board + travel origin to destin + depart
                    # Estimated cost for this passenger:
                    # Travel to origin: abs(current_level - origin_level)
                    # Board: 1
                    # Travel origin to destin: abs(destin_level - origin_level)
                    # Depart: 1
                    h += abs(current_level - origin_level) + 1 + abs(destin_level - origin_level) + 1
                else:
                    # Passenger is boarded
                    # Needs: travel to destin + depart
                    # Estimated cost for this passenger:
                    # Travel to destin: abs(current_level - destin_level)
                    # Depart: 1
                    h += abs(current_level - destin_level) + 1

        return h
