from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists


# Helper functions from Logistics example
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board and depart actions and adds an estimate
    of the minimum lift movement required to visit all relevant floors.

    # Assumptions
    - Floors are named 'f' followed by a number (e.g., f1, f2, f10) and are
      ordered numerically (f1 is the lowest, f2 is the next, etc.).
    - Every unserved passenger is either waiting at their origin or boarded,
      and has a defined destination in the static facts.
    - The cost of board, depart, up, and down actions is 1.

    # Heuristic Initialization
    - Parses static facts to map passengers to their destination floors.
    - Parses floor objects to create a mapping from floor name to its numerical level.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor and its numerical level.
    2. Identify all unserved passengers. A passenger is unserved if the fact
       '(served <passenger>)' is not in the current state.
    3. If there are no unserved passengers, the heuristic value is 0 (goal state).
    4. For each unserved passenger:
       - Determine if they are waiting at their origin ('(origin p o)' in state)
         or are boarded ('(boarded p)' in state).
       - Identify their destination floor (from pre-parsed static facts).
       - Collect the set of all origin floors of waiting passengers (`pickup_floors`).
       - Collect the set of all destination floors of unserved passengers (`dropoff_floors`).
       - Count the total number of waiting passengers (`num_waiting`).
       - Count the total number of unserved passengers (`num_unserved`).
    5. The base action cost is estimated as `num_waiting` (for board actions)
       plus `num_unserved` (for depart actions).
    6. Calculate the movement cost estimate:
       - Identify all floors that need to be visited: `required_floors = pickup_floors | dropoff_floors`.
       - If `required_floors` is empty (should not happen if there are unserved passengers), movement cost is 0.
       - Otherwise, find the minimum and maximum numerical levels among `required_floors`.
       - Estimate the minimum moves required for the lift to travel from its current floor
         to the lowest required floor, and then sweep up to the highest required floor.
         Movement cost = `abs(current_lift_level - min_required_level) + (max_required_level - min_required_level)`.
    7. The total heuristic value is the sum of the base action cost and the movement cost estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        super().__init__(task) # Call base class constructor
        static_facts = task.static

        # 1. Parse destination facts
        self.destinations = {}
        # Store static destin facts for easy lookup
        self.static_destins = {fact for fact in static_facts if match(fact, "destin", "*", "*")}
        for fact in self.static_destins:
            _, passenger, destination = get_parts(fact)
            self.destinations[passenger] = destination

        # 2. Parse floor levels
        self.floor_levels = {}
        floor_names = set()
        # Find all floor objects mentioned in static facts (above)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                floor_names.add(parts[1])
                floor_names.add(parts[2])

        # Assuming floor names are f<number> and ordered numerically
        # Extract number, sort, and assign levels
        # Use a try-except block for robustness, although assuming format for miconic
        try:
            sorted_floors = sorted(list(floor_names), key=lambda f: int(f[1:]))
            for i, floor_name in enumerate(sorted_floors):
                self.floor_levels[floor_name] = i + 1 # Levels start from 1
        except ValueError:
             # If floor names are not in f<number> format, this heuristic's movement
             # calculation based on levels will be incorrect.
             # For this problem, we assume the format holds.
             # In a real-world scenario, a more robust floor ordering parser would be needed.
             pass # Proceed, but level-based movement might be wrong


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Find current lift floor and level
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor name is not recognized, return infinity.
        if current_floor is None or current_floor not in self.floor_levels:
             return float('inf') # Indicate an invalid or unreachable state

        current_floor_level = self.floor_levels[current_floor]


        # 2. Identify unserved passengers and relevant floors/counts
        unserved_passengers = set()
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Build a temporary map for current passenger states
        current_passenger_states = {} # {p: 'waiting'/'boarded'/'served'}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'origin':
                current_passenger_states[parts[1]] = 'waiting'
            elif parts[0] == 'boarded':
                current_passenger_states[parts[1]] = 'boarded'
            elif parts[0] == 'served':
                current_passenger_states[parts[1]] = 'served'

        # Iterate through all passengers known to have destinations (from static facts)
        # These are the only passengers we care about serving.
        for p, destin_p in self.destinations.items():
            state_p = current_passenger_states.get(p, 'served') # Assume served if not in state facts

            if state_p != 'served':
                unserved_passengers.add(p)
                # Add destination to dropoff floors only if it's a known floor
                if destin_p in self.floor_levels:
                    dropoff_floors.add(destin_p)
                # else: Destination floor is unknown, this passenger cannot be served by this heuristic logic.
                #      They will remain in unserved_passengers but won't contribute to required_floors.
                #      This is acceptable for a non-admissible heuristic on potentially malformed problems.


                if state_p == 'waiting':
                    waiting_passengers_count += 1
                    # Find origin from state facts
                    origin_p = None
                    for fact in state:
                         if match(fact, "origin", p, "*"):
                             origin_p = get_parts(fact)[2]
                             break
                    if origin_p and origin_p in self.floor_levels: # Should always be found if state_p is 'waiting' and origin floor is known
                        pickup_floors.add(origin_p)
                    # else: Passenger waiting but origin fact missing or origin floor unknown.
                    #      This passenger will contribute to waiting_passengers_count but not pickup_floors.
                    #      This makes the heuristic less accurate for this specific passenger but avoids crashing.

                elif state_p == 'boarded':
                    boarded_passengers_count += 1
                    # Origin is not relevant anymore, they are already boarded.

        # If no unserved passengers, goal is reached.
        if not unserved_passengers:
            return 0

        # 3. Base action cost: 1 board per waiting, 1 depart per unserved.
        # Total unserved = waiting + boarded
        h = waiting_passengers_count + len(unserved_passengers)

        # 4. Movement cost estimate
        required_floors = pickup_floors | dropoff_floors

        if not required_floors:
             # This can happen if all unserved passengers have unknown origin/destination floors.
             # In this case, we can't calculate movement based on floors, so just return base action cost.
             movement_estimate = 0
        else:
            required_levels = {self.floor_levels[f] for f in required_floors}
            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # Estimate moves: go to the lowest required floor, then sweep up to the highest.
            # This is a simple non-admissible strategy estimate.
            movement_estimate = abs(current_floor_level - min_req_level) + (max_req_level - min_req_level)

        h += movement_estimate

        return h
