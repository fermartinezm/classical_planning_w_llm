# Assuming heuristics.heuristic_base.Heuristic is available
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
# In the actual environment, this import will be used.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic check for fact format
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the minimum board/depart actions needed for each passenger not yet served
    and adds an estimate of the lift travel cost required to visit all necessary floors
    (origin floors for waiting passengers, destination floors for boarded passengers).

    # Assumptions
    - The floor structure is linear, defined by the 'above' facts.
    - The lift can carry multiple passengers.
    - The heuristic estimates travel cost based on a simplified sweep strategy:
      go to the nearest extreme required floor (lowest or highest), then sweep
      to the other extreme, visiting floors in between.

    # Heuristic Initialization
    - Parses the static 'above' facts to determine the linear order of floors and
      create mappings between floor names and integer indices.
    - Extracts the destination floor for each passenger from the initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the lift's current floor.
    2. Initialize total heuristic cost to 0.
    3. Initialize a set `required_stops` to store floors the lift must visit.
    4. For each passenger whose destination is known (from initialization):
       - Check if the passenger is 'served' in the current state. If yes, they contribute 0 cost.
       - If the passenger is 'boarded' in the current state:
         - Add 1 to the total cost (for the 'depart' action).
         - Add the passenger's destination floor to `required_stops`.
       - If the passenger is at their 'origin' floor in the current state:
         - Add 2 to the total cost (for the 'board' and 'depart' actions).
         - Add the passenger's origin floor to `required_stops`.
         - Add the passenger's destination floor to `required_stops`.
    5. Calculate the estimated travel cost:
       - If `required_stops` is empty, travel cost is 0.
       - Otherwise, map the lift's current floor and all floors in `required_stops` to their integer indices using the floor mapping created during initialization.
       - Find the minimum index (`min_stop_index`) and maximum index (`max_stop_index`) among the required stop floors.
       - Find the index of the current lift floor (`current_floor_index`).
       - Estimate the travel cost as `min(abs(current_floor_index - min_stop_index), abs(current_floor_index - max_stop_index)) + (max_stop_index - min_stop_index)`. This represents the cost of moving from the current floor to the nearest extreme required floor (either the lowest or highest) and then sweeping through all floors up to the other extreme.
    6. Add the estimated travel cost (from step 5) to the total heuristic cost (from step 4).
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        # The Task object is assumed to have attributes: goals, static, initial_state, operators, facts, name
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state # Destin facts are here

        # Build floor ordering from 'above' facts
        # (above f_lower f_upper) means f_upper is immediately above f_lower.
        above_map = {} # maps floor_lower -> floor_upper
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                # (above f_lower f_upper)
                if len(parts) == 3: # Ensure correct number of arguments
                    f_lower, f_upper = parts[1], parts[2]
                    above_map[f_lower] = f_upper
                    all_floors.add(f_lower)
                    all_floors.add(f_upper)
                # else: ignore malformed fact

        # Find the lowest floor: a floor that is in all_floors but is not a value in above_map
        lowest_floor = None
        above_values = set(above_map.values())
        for floor in all_floors:
            if floor not in above_values:
                lowest_floor = floor
                break

        # If no floors found or structure is not linear starting from a unique lowest floor,
        # the mapping might be incomplete or incorrect. Handle gracefully.
        self.floor_to_index = {}
        self.index_to_floor = {}
        if lowest_floor is not None:
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                self.index_to_floor[index] = current_floor
                index += 1
                current_floor = above_map.get(current_floor) # Get the floor immediately above

        # Extract passenger destinations from initial state facts
        self.passenger_destinations = {}
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "destin":
                 if len(parts) == 3: # Ensure correct number of arguments
                     passenger, floor = parts[1], parts[2]
                     self.passenger_destinations[passenger] = floor
                 # else: ignore malformed fact


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If floor mapping failed during init, or no passengers need serving, handle accordingly.
        # If there are passengers defined with destinations but no valid floor structure, it's unsolvable.
        # If there are no passengers defined with destinations, the goal is trivially reached (h=0).
        if not self.passenger_destinations:
             return 0 # No passengers to serve, goal is met

        if not self.floor_to_index:
             # Passengers exist, but floor structure is invalid/missing. Unsolvable.
             return float('inf')


        total_cost = 0
        required_stops = set()
        current_lift_floor = None

        # Find lift location
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                if len(parts) == 2:
                    current_lift_floor = parts[1]
                    break
                # else: ignore malformed fact

        # If lift location is unknown or not a valid floor, state is likely invalid or goal is unreachable
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return float('inf') # Should not happen in reachable states with valid domain/instance


        # Track passenger states
        served_passengers = set()
        boarded_passengers = set()
        waiting_passengers_info = {} # {passenger: origin_floor}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "served" and len(parts) == 2:
                served_passengers.add(parts[1])
            elif predicate == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == "origin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                waiting_passengers_info[passenger] = floor
            # Ignore other facts like 'above', 'destin', 'lift-at' (already processed or not needed here)

        # Passengers whose (served p) is a goal fact and are not yet served
        goal_passengers = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == "served"}
        passengers_not_served = goal_passengers - served_passengers

        for passenger in passengers_not_served:
            # Get destination for this passenger (should exist based on init parsing)
            dest_floor = self.passenger_destinations.get(passenger)
            if dest_floor is None:
                 # This passenger is a goal passenger but has no destination defined.
                 # Problem instance might be inconsistent. Treat as unsolvable.
                 return float('inf')

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to depart at destination
                total_cost += 1 # cost for 'depart' action
                required_stops.add(dest_floor)
            elif passenger in waiting_passengers_info: # Check if passenger is in waiting_passengers_info keys
                # Passenger is waiting, needs to board and then depart
                origin_floor = waiting_passengers_info[passenger]
                total_cost += 2 # cost for 'board' + 'depart' actions
                required_stops.add(origin_floor)
                required_stops.add(dest_floor)
            # else: passenger is a goal passenger but is not served, not boarded, not waiting.
            # This state should not be reachable from a valid initial state where goal passengers
            # start with an (origin ...) fact. Treat as unsolvable.
            else:
                 return float('inf')


        # Calculate travel cost
        if not required_stops:
            travel_cost = 0
        else:
            # Ensure all required stops are valid floors in our mapping
            valid_required_stops = {f for f in required_stops if f in self.floor_to_index}
            if len(valid_required_stops) != len(required_stops):
                 # Some required stops are not valid floors. Unsolvable.
                 return float('inf')

            required_indices = {self.floor_to_index[f] for f in valid_required_stops}
            current_floor_index = self.floor_to_index[current_lift_floor] # Already checked current_lift_floor is valid

            min_stop_index = min(required_indices)
            max_stop_index = max(required_indices)

            # Estimate travel based on sweeping strategy
            dist_to_min = abs(current_floor_index - min_stop_index)
            dist_to_max = abs(current_floor_index - max_stop_index)
            dist_min_to_max = max_stop_index - min_stop_index # Always non-negative

            # Travel cost is distance to nearest extreme + distance covering the range
            travel_cost = min(dist_to_min, dist_to_max) + dist_min_to_max


        total_cost += travel_cost

        return total_cost
