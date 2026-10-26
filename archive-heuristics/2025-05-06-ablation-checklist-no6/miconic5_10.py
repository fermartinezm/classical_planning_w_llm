from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assuming miconicHeuristic inherits from a base class named Heuristic
# class miconicHeuristic(Heuristic):
class miconicHeuristic: # Using this line if the base class is not provided directly
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions (board, depart, move) required to serve
    all passengers who are not yet served.

    Heuristic components:
    1. Number of board actions needed: One for each unboarded passenger.
    2. Number of depart actions needed: One for each unserved passenger.
    3. Estimated travel actions: Minimum moves to visit all floors where
       pickups are needed (origin floors of unboarded passengers) and
       dropoffs are needed (destination floors of boarded passengers).
       This is estimated as the span of these floors plus the distance
       from the current lift floor to the closer end of the span.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        # If inheriting from Heuristic base class, call its init:
        # super().__init__(task)

        self.goals = task.goals
        self.static = task.static

        # Extract floor order from static facts
        self.floor_to_index = {}
        self.index_to_floor = []
        lower_to_higher = {} # Map from a floor to the floor directly above it
        all_floors = set()
        floors_that_are_higher = set() # Floors that appear as the second arg in (above ?f_lower ?f_higher)

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, f_lower, f_higher = parts
                    lower_to_higher[f_lower] = f_higher
                    all_floors.add(f_lower)
                    all_floors.add(f_higher)
                    floors_that_are_higher.add(f_higher)

        # Find the lowest floor (a floor that is never the second argument in an 'above' fact)
        lowest_floor = None
        potential_lowest_floors = all_floors - floors_that_are_higher

        if len(potential_lowest_floors) == 1:
             lowest_floor = list(potential_lowest_floors)[0]
        elif len(potential_lowest_floors) > 1:
             # Multiple potential lowest floors, indicates disconnected components or error
             # Fallback: assume alphabetical order for all floors found
             sorted_floors = sorted(list(all_floors))
             self.index_to_floor = sorted_floors
             self.floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
             lowest_floor = None # Indicate fallback was used
        elif not all_floors:
             # No floors found at all
             lowest_floor = None
        # else: len(potential_lowest_floors) == 0, indicates cycle or single floor not in 'above'
        # Fallback: assume alphabetical order for all floors found if all_floors is not empty
        if lowest_floor is None and all_floors:
             sorted_floors = sorted(list(all_floors))
             self.index_to_floor = sorted_floors
             self.floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
             lowest_floor = None # Still indicate fallback was used


        if lowest_floor: # If a unique lowest floor was found and we didn't use fallback
            # Build the ordered list starting from the lowest floor
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                if current_floor in self.floor_to_index: # Avoid infinite loops if 'above' facts form cycle
                     # print(f"Warning: Cycle detected or duplicate floor in sequence starting from {lowest_floor}. Stopping.")
                     break # Stop building sequence if we hit a floor already indexed
                self.index_to_floor.append(current_floor)
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = lower_to_higher.get(current_floor)

        # Extract passenger destinations and identify all passengers
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Destinations are static facts
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, passenger, destination = parts
                    self.passenger_destinations[passenger] = destination
                    self.all_passengers.add(passenger)

        # Also get origins from static facts to know all passengers initially
        for fact in self.static:
             if match(fact, "origin", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, passenger, origin = parts
                     self.all_passengers.add(passenger)

        # Also get passengers from goal facts (served)
        for fact in self.goals:
             if match(fact, "served", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2:
                     _, passenger = parts
                     self.all_passengers.add(passenger)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # If there are no passengers defined in static/goals, the goal is trivially reached (h=0)
        if not self.all_passengers:
             return 0

        # Check if goal is reached (all passengers served)
        all_served = True
        for passenger in self.all_passengers:
            if f"(served {passenger})" not in state:
                all_served = False
                break
        if all_served:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, current_lift_floor = parts
                    break

        # If lift location is unknown or floor indexing failed, return infinity
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # print(f"Warning: Could not determine current lift floor or floor index for '{current_lift_floor}'.")
             return float('inf')

        current_idx = self.floor_to_index[current_lift_floor]

        unboarded_passengers_origins = {} # passenger -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        # Collect passenger states
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "origin" and len(parts) == 3:
                _, passenger, origin_floor = parts
                # Only consider passengers we know about
                if passenger in self.all_passengers:
                     unboarded_passengers_origins[passenger] = origin_floor
            elif predicate == "boarded" and len(parts) == 2:
                 _, passenger = parts
                 if passenger in self.all_passengers:
                     boarded_passengers.add(passenger)
            elif predicate == "served" and len(parts) == 2:
                 _, passenger = parts
                 if passenger in self.all_passengers:
                     served_passengers.add(passenger)

        # Passengers who are unserved = all_passengers - served_passengers
        unserved_passengers = self.all_passengers - served_passengers

        # Calculate board/depart costs
        # Each unboarded passenger needs 1 board action
        board_cost = len(unboarded_passengers_origins)
        # Each unserved passenger needs 1 depart action eventually
        depart_cost = len(unserved_passengers)

        # Calculate travel cost
        needed_floors = set()
        # Add origin floors for unboarded passengers
        needed_floors.update(unboarded_passengers_origins.values())
        # Add destination floors for boarded passengers
        for passenger in boarded_passengers:
             # Ensure passenger has a known destination (should be in static)
             if passenger in self.passenger_destinations:
                 needed_floors.add(self.passenger_destinations[passenger])
             # else: This boarded passenger doesn't have a goal destination? Ignore for heuristic.

        travel_cost = 0
        valid_needed_indices = [self.floor_to_index[f] for f in needed_floors if f in self.floor_to_index]

        if valid_needed_indices: # Ensure we have valid indices
            min_needed_idx = min(valid_needed_indices)
            max_needed_idx = max(valid_needed_indices)

            span = max_needed_idx - min_needed_idx
            dist_to_min = abs(current_idx - min_needed_idx)
            dist_to_max = abs(current_idx - max_needed_idx)

            # Travel is span + distance from current to the closer end of the span
            travel_cost = span + min(dist_to_min, dist_to_max)

        # Total heuristic is sum of costs
        total_heuristic = board_cost + depart_cost + travel_cost

        return total_heuristic
