from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Ensure fact is treated as a string and remove outer parentheses
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
        # This might happen for objects/types in the initial state,
        # but we only care about predicate facts.
        return [] # Return empty list for non-fact strings

    # Remove outer parentheses and split by spaces
    parts = fact_str[1:-1].split()
    return parts

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (move, board, depart)
    required to get all unserved passengers to their destination floors.
    It considers the lift's current position, the floors where passengers
    are waiting, and the destination floors of all unserved passengers.

    # Assumptions
    - Floor names are in the format 'f<number>', and the index of a floor
      is the number part (e.g., f1 is index 1, f5 is index 5).
    - Each action (move between adjacent floors, board, depart) costs 1.
    - The lift can carry multiple passengers.
    - The minimum moves to visit a set of floors is the distance required
      to traverse the range of those floors, starting from the current
      lift position.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor.
    2. Identify all passengers currently waiting at an origin floor or boarded
       in the lift.
    3. Identify all passengers who have already been served.
    4. Determine the set of all passengers involved in the problem (those
       mentioned in initial state origin/boarded facts or static destination facts).
    5. Identify the set of unserved passengers (all passengers minus served passengers).
    6. Determine the set of floors the lift *must* visit. This includes:
       - The origin floor for every passenger currently waiting at an origin floor.
       - The destination floor for every unserved passenger (both waiting and boarded).
    7. If there are no floors the lift needs to visit (meaning all unserved
       passengers have no required stops, which implies there are no unserved
       passengers), the heuristic value is 0.
    8. If there are floors to visit, find the minimum and maximum floor indices
       among these required floors.
    9. Calculate the estimated number of move actions. This is the minimum
       distance the lift must travel to cover the range of required floors,
       starting from its current floor.
       - If the current floor is below the minimum required floor, the moves
         are the distance from the current floor up to the maximum required floor.
       - If the current floor is above the maximum required floor, the moves
         are the distance from the current floor down to the minimum required floor.
       - If the current floor is within the range of required floors, the moves
         are the distance between the minimum and maximum required floors.
    10. Calculate the estimated number of non-move actions (board and depart).
        - Each passenger currently waiting at an origin floor needs one 'board' action.
        - Each unserved passenger (waiting or boarded) needs one 'depart' action.
        The total non-move actions is the count of waiting passengers plus the
        count of all unserved passengers.
    11. The total heuristic value is the sum of the estimated move actions and
        the estimated non-move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations from static facts.
        """
        self.passenger_destinations = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin' and len(parts) > 2:
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor

    def floor_to_index(self, floor_name):
        """Converts floor name 'f<number>' to integer index <number>."""
        # Assumes floor names are like 'f1', 'f2', etc.
        # Returns 0 or raises error for unexpected formats.
        try:
            # Extract the number part after 'f'
            return int(floor_name[1:])
        except (ValueError, IndexError):
            # This should not happen with standard miconic benchmarks
            # print(f"Error: Unexpected floor name format encountered: {floor_name}")
            # Raising an error is safer for debugging problem files.
            raise ValueError(f"Invalid floor name format: {floor_name}")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_lift_floor = None
        waiting_passengers_at_floor = {} # {passenger: floor}
        boarded_passengers = set()
        served_passengers = set()
        all_passengers_in_state = set() # Passengers mentioned in origin/boarded/served facts in this state

        # First pass: Extract all relevant dynamic info from state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip non-fact strings

            predicate = parts[0]
            if predicate == 'lift-at' and len(parts) > 1:
                current_lift_floor = parts[1]
            elif predicate == 'origin' and len(parts) > 2:
                passenger, floor = parts[1], parts[2]
                waiting_passengers_at_floor[passenger] = floor
                all_passengers_in_state.add(passenger)
            elif predicate == 'boarded' and len(parts) > 1:
                passenger = parts[1]
                boarded_passengers.add(passenger)
                all_passengers_in_state.add(passenger)
            elif predicate == 'served' and len(parts) > 1:
                passenger = parts[1]
                served_passengers.add(passenger)
                all_passengers_in_state.add(passenger)

        # Combine all passengers seen in static destinations and current state
        all_passengers = set(self.passenger_destinations.keys())
        all_passengers.update(all_passengers_in_state)

        # Identify unserved passengers
        unserved_passengers = {p for p in all_passengers if p not in served_passengers}

        # Determine floors the lift must visit
        required_floors = set()
        waiting_passengers = set() # Set of passengers who are currently waiting at an origin

        for p in unserved_passengers:
            # If passenger is waiting at an origin floor
            if p in waiting_passengers_at_floor:
                 origin_floor = waiting_passengers_at_floor[p]
                 required_floors.add(origin_floor)
                 waiting_passengers.add(p) # Add to the set of waiting passengers

            # Passenger needs to be dropped off at destination
            # Ensure passenger has a destination defined in static facts
            if p in self.passenger_destinations:
                 required_floors.add(self.passenger_destinations[p])
            # Note: Passengers not in self.passenger_destinations but in unserved_passengers
            # might indicate an issue with problem definition or parsing, but we proceed
            # assuming valid problems where all passengers have destinations.


        # If no floors need visiting, all unserved passengers must be
        # boarded and their destinations are not in the required_floors set.
        # This implies P_wait is empty, and for all P_unserved, their destination
        # is not in the set of required floors. This can only happen if P_unserved is empty.
        # So, if required_floors is empty, all passengers are served.
        if not required_floors:
             return 0

        # Calculate move cost
        # Need the current lift floor. If not found (e.g., initial state parsing error),
        # this would be an issue. Assuming current_lift_floor is always found.
        if current_lift_floor is None:
             # This indicates a problem parsing the state, or an invalid state.
             # Returning infinity or a large value is appropriate for invalid states.
             # print("Error: Lift location not found in state.") # Keep silent for competitive eval
             return float('inf') # Return infinity for invalid states


        current_idx = self.floor_to_index(current_lift_floor)
        min_required_idx = min(self.floor_to_index(f) for f in required_floors)
        max_required_idx = max(self.floor_to_index(f) for f in required_floors)

        if current_idx < min_required_idx:
            move_cost = max_required_idx - current_idx
        elif current_idx > max_required_idx:
            move_cost = current_idx - min_required_idx
        else: # min_required_idx <= current_idx <= max_required_idx
            move_cost = max_required_idx - min_required_idx


        # Calculate non-move cost (board and depart actions)
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        non_move_cost = len(waiting_passengers) + len(unserved_passengers)

        return move_cost + non_move_cost
