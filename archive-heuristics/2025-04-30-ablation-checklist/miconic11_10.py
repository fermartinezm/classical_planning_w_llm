from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def parse_fact(fact_string):
    """Extract predicate and arguments from a PDDL fact string."""
    # Remove parentheses and split by spaces
    parts = fact_string[1:-1].split()
    return parts[0], parts[1:]

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to service all
    passengers. It sums the estimated number of board/depart actions needed
    for unserved passengers and an estimate of the lift movement cost.

    # Assumptions
    - The 'above' predicate defines the immediate adjacency of floors, allowing
      us to assign numerical indices to floors. (e.g., (above f2 f1) means f2 is
      immediately above f1).
    - Passengers' destinations are static and available in the initial state
      or static facts via the 'destin' predicate.

    # Heuristic Initialization
    - Maps floor objects to numerical indices based on the 'above' predicates.
    - Stores passenger destinations from 'destin' facts.

    # Step-by-Step Thinking for Computing Heuristic
    1. Map floors to numerical indices: Parse 'above' facts to determine the
       order of floors and assign an index (0 for the bottom floor, 1 for the
       next, etc.). This is done once during initialization.
    2. Identify unserved passengers: Iterate through all passengers defined
       in the problem (e.g., those with 'destin' facts). If a passenger is
       not marked as 'served' in the current state, they need service.
    3. Calculate Base Cost (Board/Depart Actions): For each unserved passenger:
       - If the passenger is waiting at their origin floor ('origin ?p ?f'),
         they will need a 'board' action and a 'depart' action. Add 2 to the base cost.
       - If the passenger is already boarded ('boarded ?p'), they only need
         a 'depart' action. Add 1 to the base cost.
    4. Identify Required Floors: Determine the set of floors the lift must visit
       to service the unserved passengers. This includes:
       - The origin floor for every unserved passenger who is currently waiting
         at their origin.
       - The destination floor for every unserved passenger who is currently boarded.
    5. Calculate Movement Cost: Estimate the number of 'up'/'down' actions needed
       to visit all required floors starting from the lift's current location.
       - Find the current floor of the lift and its index.
       - Find the minimum and maximum indices among the required floors.
       - A simple, non-admissible estimate for movement cost is the distance
         from the current lift floor to the lowest required floor, plus the
         total span (difference between max and min required floor indices).
         This models a strategy where the lift first goes to the lowest needed
         floor and then sweeps upwards, visiting all required floors.
         Movement Cost = `abs(current_lift_index - min_required_index) + (max_required_index - min_required_index)`
    6. Total Heuristic: Sum the Base Cost and the Movement Cost. If there are
       no unserved passengers, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by mapping floors to indices and storing
        passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build floor_to_index map based on 'above' predicates
        floor_immediately_above_map = {}
        all_floors = set()
        for fact in self.static_facts:
            pred, args = parse_fact(fact)
            if pred == 'above':
                # (above f_higher f_lower) means f_higher is immediately above f_lower
                f_higher, f_lower = args
                floor_immediately_above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the bottom floor (a floor that no other floor is immediately above)
        bottom_floor = None
        floors_with_floor_below = set(floor_immediately_above_map.values())
        for floor in all_floors:
            if floor not in floors_with_floor_below:
                bottom_floor = floor
                break

        # Handle case with no floors or single floor if necessary, though PDDL implies structure
        if bottom_floor is None and all_floors:
             # This might happen if there's only one floor, or an invalid 'above' structure
             if len(all_floors) == 1:
                 bottom_floor = list(all_floors)[0]
             else:
                 # For typical miconic, there's a clear bottom. Raise error for unexpected structure.
                 raise ValueError("Could not determine bottom floor from 'above' facts.")
        elif bottom_floor is None and not all_floors:
             # No floors defined, heuristic will always be 0
             pass


        self.floor_to_index = {}
        if bottom_floor:
            current_floor = bottom_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                next_floor = floor_immediately_above_map.get(current_floor)
                current_floor = next_floor
                index += 1

        # Build passenger_to_dest map from static facts (destin)
        self.passenger_to_dest = {}
        for fact in self.static_facts:
            pred, args = parse_fact(fact)
            if pred == 'destin':
                passenger, floor = args
                self.passenger_to_dest[passenger] = floor

    # Helper method to check if a passenger is served
    def is_served(self, state, passenger):
        return f'(served {passenger})' in state

    # Helper method to find passenger's origin floor
    def get_origin_floor(self, state, passenger):
        for fact in state:
            pred, args = parse_fact(fact)
            if pred == 'origin' and len(args) == 2 and args[0] == passenger:
                return args[1]
        return None # Not waiting at any floor

    # Helper method to check if passenger is boarded
    def is_boarded(self, state, passenger):
         return f'(boarded {passenger})' in state

    # Helper method to get current lift floor
    def get_lift_floor(self, state):
        for fact in state:
            pred, args = parse_fact(fact)
            if pred == 'lift-at' and len(args) == 1:
                return args[0]
        return None # Should always be a lift-at fact in a valid state

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If no floors were indexed (empty domain?), return 0
        if not self.floor_to_index:
            return 0

        current_lift_floor = self.get_lift_floor(state)
        # If lift location is unknown (invalid state?), return infinity or a large value
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This indicates an unexpected state structure
             return float('inf') # Or a large number

        current_lift_index = self.floor_to_index[current_lift_floor]

        required_floor_indices = set()
        base_cost = 0

        # Iterate through all passengers with defined destinations
        all_passengers = set(self.passenger_to_dest.keys())

        for passenger in all_passengers:
            if self.is_served(state, passenger):
                continue # This passenger is done

            origin_floor = self.get_origin_floor(state, passenger)
            is_boarded = self.is_boarded(state, passenger)
            dest_floor = self.passenger_to_dest.get(passenger)

            # Destination must exist for passengers considered
            if dest_floor is None or dest_floor not in self.floor_to_index:
                 # This indicates an invalid problem definition or state
                 return float('inf') # Or a large number

            dest_index = self.floor_to_index[dest_floor]

            if origin_floor: # Passenger is waiting at origin_floor
                # Check if origin floor is valid
                if origin_floor not in self.floor_to_index:
                     return float('inf') # Invalid state

                base_cost += 2 # Needs board + depart
                required_floor_indices.add(self.floor_to_index[origin_floor])
                required_floor_indices.add(dest_index)
            elif is_boarded: # Passenger is boarded
                base_cost += 1 # Needs depart
                required_floor_indices.add(dest_index)
            # else: Passenger is not served, not waiting, not boarded.
            # This shouldn't happen for unserved passengers in a valid state
            # if they are listed in destin facts.

        # If no passengers need service, heuristic is 0
        if not required_floor_indices:
            return 0

        # Calculate movement cost
        min_req_idx = min(required_floor_indices)
        max_req_idx = max(required_floor_indices)

        # Movement cost estimate: distance to the lowest required floor
        # plus the span of required floors. This models going down to the
        # lowest stop and then sweeping up to the highest.
        movement_cost = abs(current_lift_index - min_req_idx) + (max_req_idx - min_req_idx)

        return base_cost + movement_cost

