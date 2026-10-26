from heuristics.heuristic_base import Heuristic
# No need for fnmatch if parsing is manual.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    # Basic check for expected format, though robust error handling might be needed in a real system
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Depending on expected input, could raise error or log warning
         # For typical PDDL facts as strings, this is safe.
         pass
    return fact_str[1:-1].split()


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the number of pending 'board' actions, the number
    of pending 'depart' actions, and an estimate of the lift movement cost.

    # Assumptions
    - The goal is to have all passengers served.
    - The lift can carry multiple passengers.
    - Floor ordering is defined by the 'above' predicate in static facts.
    - Action costs are uniform (each action counts as 1).
    - All passengers mentioned in the problem have a destination defined in static facts.
    - The state always contains exactly one 'lift-at' fact.

    # Heuristic Initialization
    - Parses static facts to determine the floor order and create a mapping
      from floor names to numerical indices (lowest floor gets index 1).
    - Parses static facts to determine the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify the current floor of the lift from the state.
    2.  Identify the set of passengers who are currently served based on the state.
    3.  Identify the set of passengers who are currently waiting at their origin floor based on the state, storing their origin floor.
    4.  Identify the set of passengers who are currently boarded in the lift based on the state.
    5.  Determine the set of unserved passengers by taking all passengers with defined goals (from static facts) and removing those who are served. If this set is empty, the heuristic value is 0 (goal state).
    6.  Calculate the 'board' cost: This is the number of unserved passengers who are currently waiting at their origin floor. Each such passenger requires a 'board' action.
    7.  Calculate the 'depart' cost: This is the number of unserved passengers who are currently boarded in the lift. Each such passenger requires a 'depart' action.
    8.  Identify the set of floors the lift *must* visit to serve the unserved
        passengers. This set includes:
        -   The origin floor for each unserved passenger who is waiting.
        -   The destination floor for each unserved passenger who is boarded (retrieved from the pre-computed passenger goals).
    9.  If the set of required stops identified in step 8 is empty, the movement cost is 0.
    10. If there are required stops, map these floor names to their numerical
        indices using the pre-computed floor-to-index mapping.
    11. Find the minimum and maximum floor indices among the required stops.
    12. Calculate the movement cost estimate: This is the minimum of the absolute difference between the current lift floor index and the minimum required floor index, and the absolute difference between the current lift floor index and the maximum required floor index. To this, add the total range covered by the required floors (max_required_idx - min_required_idx). This formula estimates the minimum vertical travel needed to reach the closer extreme of the required floor range and then traverse the entire range once.
        `movement_cost = min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx)`
    13. The total heuristic value is the sum of the 'board' cost, the 'depart' cost, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger goals
        from static facts.
        """
        # self.goals = task.goals # Goal conditions are implicitly handled by checking served passengers
        static_facts = task.static

        # 1. Determine floor order and create floor_to_index map
        above_facts = []
        all_floors = set()
        # Floors that appear as the first arg in (above f_i f_j) are higher floors
        floors_that_are_above_something = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                above_facts.append((f_above, f_below))
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_that_are_above_something.add(f_above)

        # Find the lowest floor (the one not appearing as the first arg in any 'above' fact)
        # Assuming there's exactly one lowest floor in a connected set of floors.
        # If all_floors is empty (no floors defined), this would fail. Assuming valid domain.
        lowest_floor = (all_floors - floors_that_are_above_something).pop()

        # Build the sorted list of floors from lowest to highest
        sorted_floors = [lowest_floor]
        current_floor_in_sequence = lowest_floor
        # Build a map from a floor to the floor immediately above it
        # This map is built from (above f_above f_below) facts, mapping f_below -> f_above
        floor_below_to_above_map = {f_below: f_above for f_above, f_below in above_facts}

        # Iteratively find the next higher floor
        while current_floor_in_sequence in floor_below_to_above_map:
            next_higher_floor = floor_below_to_above_map[current_floor_in_sequence]
            sorted_floors.append(next_higher_floor)
            current_floor_in_sequence = next_higher_floor

        # Create the floor_to_index map (index 1 for lowest floor)
        self.floor_to_index = {f: i + 1 for i, f in enumerate(sorted_floors)}

        # 2. Extract passenger goals (destination floors)
        self.passenger_goals = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                person, floor = parts[1], parts[2]
                self.passenger_goals[person] = floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current lift floor
        current_f = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_f = parts[1]
                break # Assuming exactly one lift-at fact is always true

        # If current_f is None, the state is malformed for this domain.
        # Returning a high value or raising an error would be appropriate,
        # but assuming valid states based on problem description.
        current_f_idx = self.floor_to_index[current_f]

        # 2-4. Identify served, waiting, and boarded passengers
        served_passengers = set()
        waiting_passengers_info = {} # {p: f}
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_passengers.add(parts[1])
            elif parts[0] == 'origin':
                p, f = parts[1], parts[2]
                waiting_passengers_info[p] = f
            elif parts[0] == 'boarded':
                boarded_passengers.add(parts[1])

        # 5. Determine unserved passengers
        # Consider all passengers whose destinations are defined in static facts
        all_passengers_with_goals = set(self.passenger_goals.keys())
        unserved_passengers = all_passengers_with_goals - served_passengers

        if not unserved_passengers:
            return 0 # Goal state reached

        # 6. Calculate board cost
        # Count unserved passengers who are currently waiting
        board_cost = sum(1 for p in unserved_passengers if p in waiting_passengers_info)

        # 7. Calculate depart cost
        # Count unserved passengers who are currently boarded
        depart_cost = sum(1 for p in unserved_passengers if p in boarded_passengers)

        # 8. Identify required stops
        required_stops = set()
        # Origin floors for unserved waiting passengers
        for p in unserved_passengers:
            if p in waiting_passengers_info:
                required_stops.add(waiting_passengers_info[p])
        # Destination floors for unserved boarded passengers
        for p in unserved_passengers:
            if p in boarded_passengers:
                 # Ensure passenger has a destination defined in static facts
                 # This check is mostly for robustness against malformed instances
                 if p in self.passenger_goals:
                    required_stops.add(self.passenger_goals[p])


        # 9-12. Calculate movement cost
        movement_cost = 0
        if required_stops:
            required_indices = {self.floor_to_index[f] for f in required_stops}
            min_idx = min(required_indices)
            max_idx = max(required_indices)

            # Estimate movement: distance to closest extreme + range traversal
            # This assumes the lift goes to one end of the required range and sweeps through.
            movement_cost = min(abs(current_f_idx - min_idx), abs(current_f_idx - max_idx)) + (max_idx - min_idx)

        # 13. Total heuristic value
        total_cost = board_cost + depart_cost + movement_cost

        return total_cost
