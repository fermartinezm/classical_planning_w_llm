# from fnmatch import fnmatch # Not needed for this implementation
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    Estimates the remaining cost by summing the individual costs for each
    passenger not yet served. The cost for a passenger depends on their
    current state (waiting or boarded) and the lift's current location.
    It assumes passengers are served one by one, ignoring potential
    optimizations from carrying multiple passengers.

    # Heuristic Initialization
    - Extracts the linear order of floors from `above` facts.
    - Maps each floor name to a numerical index.
    - Stores the destination floor for each passenger from `destin` facts.
    - Identifies all passengers that need to be served from the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. For each passenger that needs to be served:
       - If the passenger is already served, their cost is 0.
       - If the passenger is currently boarded in the lift:
         - Calculate the distance (number of floors to traverse) from the lift's current floor to the passenger's destination floor.
         - Add this distance plus 1 (for the `depart` action) to the total cost.
       - If the passenger is waiting at their origin floor:
         - Calculate the distance from the lift's current floor to the passenger's origin floor.
         - Add this distance plus 1 (for the `board` action).
         - Calculate the distance from the passenger's origin floor to their destination floor.
         - Add this distance plus 1 (for the `depart` action).
         - Sum these costs for the passenger.
    3. The total heuristic value is the sum of costs for all passengers not yet served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of passengers to be served from the static facts and goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor order and floor_to_index map
        # Assuming a linear floor structure based on action definitions and examples
        # floor_above_map maps lower floor -> floor directly above it
        floor_above_map = {}
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                # PDDL fact is (above f_higher f_lower)
                f_higher, f_lower = parts[1], parts[2]
                floor_above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        # Find the lowest floor: a floor that is in all_floors but is not a value in floor_above_map
        lowest_floor = None
        floors_that_are_above_another = set(floor_above_map.values())
        for floor in all_floors:
            if floor not in floors_that_are_above_another:
                 lowest_floor = floor
                 break # Found the unique lowest floor

        # Build the ordered list of floors and the floor_to_index map
        self.floor_to_index = {}
        if lowest_floor is not None:
            current_floor = lowest_floor
            index = 1 # Use 1-based indexing for floor numbers
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = floor_above_map.get(current_floor) # Get the floor directly above
        # If lowest_floor is None but there are floors, it implies a non-linear
        # or disconnected structure, or perhaps just one floor. For this heuristic,
        # we assume a linear structure is intended. If not found, floor_to_index
        # remains empty, leading to infinite costs later.

        # 2. Store passenger destinations and identify passengers to serve
        self.destinations = {}
        self.passengers_to_serve = set()

        # Destinations are static facts
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "destin" and len(parts) == 3:
                 passenger, floor = parts[1], parts[2]
                 self.destinations[passenger] = floor

        # Passengers to serve are those mentioned in the goal (served predicate)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served" and len(parts) == 2:
                passenger = parts[1]
                self.passengers_to_serve.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant information from the current state
        current_lift_floor = None
        passenger_states = {} # Map passenger -> 'served', 'boarded', or ('origin', floor)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
            elif predicate == "served" and len(parts) == 2:
                passenger_states[parts[1]] = 'served'
            elif predicate == "boarded" and len(parts) == 2:
                passenger_states[parts[1]] = 'boarded'
            elif predicate == "origin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                passenger_states[passenger] = ('origin', floor)

        # Check if the goal is reached first
        all_served = True
        for passenger in self.passengers_to_serve:
            if passenger_states.get(passenger) != 'served':
                all_served = False
                break
        if all_served:
            return 0 # Goal state reached

        # If not goal and no lift location, state is invalid/unreachable in this domain
        if current_lift_floor is None:
             return float('inf')

        # If floor mapping failed during init (e.g., invalid above facts structure)
        if not self.floor_to_index:
             return float('inf')

        total_cost = 0

        # Calculate cost for each passenger not yet served
        for passenger in self.passengers_to_serve:
            state_info = passenger_states.get(passenger)

            if state_info == 'served':
                continue # Passenger is already served (redundant check, but safe)

            # Get floor indices, handle potential errors if floor not found (shouldn't happen with correct parsing)
            current_floor_index = self.floor_to_index.get(current_lift_floor)
            # Check for None already done above for current_lift_floor

            if state_info == 'boarded':
                # Passenger is in the lift, needs to go to destination and depart
                destin_floor = self.destinations.get(passenger)
                if destin_floor is None:
                     # Should not happen if destinations are correctly extracted from static facts
                     return float('inf')

                destin_floor_index = self.floor_to_index.get(destin_floor)
                if destin_floor_index is None:
                     # Should not happen if destination floor is part of the floor structure
                     return float('inf')

                # Cost: move from current lift floor to destination + depart
                move_cost = abs(destin_floor_index - current_floor_index)
                depart_cost = 1
                total_cost += move_cost + depart_cost

            elif isinstance(state_info, tuple) and state_info[0] == 'origin':
                # Passenger is waiting at origin
                origin_floor = state_info[1]
                destin_floor = self.destinations.get(passenger)
                if destin_floor is None:
                     # Should not happen if destinations are correctly extracted from static facts
                     return float('inf')

                origin_floor_index = self.floor_to_index.get(origin_floor)
                destin_floor_index = self.floor_to_index.get(destin_floor)
                if origin_floor_index is None or destin_floor_index is None:
                     # Should not happen if origin/destination floors are part of the floor structure
                     return float('inf')

                # Cost: move from current lift floor to origin + board + move from origin to destination + depart
                move_to_origin_cost = abs(origin_floor_index - current_floor_index)
                board_cost = 1
                move_to_destin_cost = abs(destin_floor_index - origin_floor_index)
                depart_cost = 1

                total_cost += move_to_origin_cost + board_cost + move_to_destin_cost + depart_cost
            # else: Passenger is not served, boarded, or at origin. This state is unexpected
            # for a passenger who needs to be served in a valid miconic problem state.
            # We don't add cost for this passenger, assuming valid states are explored.

        return total_cost
