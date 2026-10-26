# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Assuming Heuristic base class is available from this import path
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    unserved passengers. It sums the estimated lift movement cost to visit
    necessary floors (origins for waiting passengers, destinations for boarded
    passengers) and the number of board and depart actions needed.

    # Assumptions
    - Floors are named 'f' followed by a number (e.g., f1, f2, f10) and are
      linearly ordered according to these numbers.
    - All passengers requiring service are listed in the goal state as (served ?p).
    - An unserved passenger is either waiting at their origin floor or boarded
      in the lift.

    # Heuristic Initialization
    - Parses all floor names from static facts and goals, sorts them numerically,
      and creates mappings between floor names and their integer indices.
    - Extracts the destination floor for each passenger from static facts.
    - Identifies all passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify all passengers who are not yet served (i.e., (served ?p) is not in the state).
    3. Find the current floor of the lift.
    4. Determine the set of "required floors" indices that the lift must visit:
       - For each unserved passenger who is waiting at their origin floor, add the index of their origin floor to the set.
       - For each unserved passenger who is currently boarded, add the index of their destination floor to the set.
    5. If there are no required floors, the move cost is 0.
    6. If there are required floors, calculate the minimum vertical movement cost for the lift to travel from its current floor to cover the range of required floors (from the lowest required floor index to the highest). This is calculated as the distance from the current floor index to the closest end of the required floor index range, plus the distance spanning the entire required range.
    7. Count the number of unserved passengers who are currently waiting at their origin. This is the number of 'board' actions needed.
    8. Count the total number of unserved passengers (both waiting and boarded). This is the number of 'depart' actions needed.
    9. The total heuristic value is the sum of the estimated move cost, the number of board actions needed, and the number of depart actions needed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of all passengers.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all floor names and sort them numerically
        floor_names = set()
        # Floors appear in static facts (e.g., above, destin) and sometimes goals (though less common)
        for fact in self.static:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('f') and part[1:].isdigit():
                    floor_names.add(part)
        # Although not strictly necessary based on miconic goals, good practice to check goals too
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                if part.startswith('f') and part[1:].isdigit():
                    floor_names.add(part)


        # Sort floors based on their numerical index (f1 < f2 < ...)
        # This assumes floor names are consistently formatted as 'f' followed by digits.
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))

        # Create floor_name -> index map and index -> floor_name map
        self.floor_to_index = {floor: i for i, floor in enumerate(sorted_floor_names)}
        self.index_to_floor = {i: floor for i, floor in enumerate(sorted_floor_names)}

        # Identify all passengers from goals (assuming all passengers needing service are in goals)
        self.all_passengers = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == "served" and len(get_parts(goal)) == 2}

        # Store destination floors for each passenger by looking at static facts
        self.passenger_destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3 and parts[1] in self.all_passengers:
                self.passenger_destinations[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if the current state is a goal state.
        if self.goals <= state:
            return 0

        # 2. Identify unserved passengers.
        unserved_passengers = {p for p in self.all_passengers if f"(served {p})" not in state}

        # If unserved_passengers is empty but goal is not reached, something is wrong.
        # Based on problem description, goal is just (served p) for all p.
        # So if unserved_passengers is empty, self.goals <= state must be true.
        # We rely on the initial check returning 0 in that case.

        # 3. Find current lift floor.
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        # This should always be found in a valid state.
        if current_lift_floor is None:
             # Should not happen in valid problem states, but return inf for safety.
             return float('inf')

        current_lift_index = self.floor_to_index[current_lift_floor]

        # 4. Determine the set of "required floors" indices.
        required_floor_indices = set()
        num_waiting = 0
        num_boarded = 0

        # Find status (waiting or boarded) for each unserved passenger
        # We iterate through state facts to find passenger locations/status
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             # Check for waiting passengers
             if predicate == "origin" and len(parts) == 3:
                  passenger = parts[1]
                  origin_floor = parts[2]
                  if passenger in unserved_passengers:
                      required_floor_indices.add(self.floor_to_index[origin_floor])
                      num_waiting += 1
             # Check for boarded passengers
             elif predicate == "boarded" and len(parts) == 2:
                  passenger = parts[1]
                  if passenger in unserved_passengers:
                      destination_floor = self.passenger_destinations.get(passenger)
                      if destination_floor: # Ensure destination is known
                          required_floor_indices.add(self.floor_to_index[destination_floor])
                      num_boarded += 1

        # 5. Calculate move cost.
        move_cost = 0
        if required_floor_indices:
            min_req_idx = min(required_floor_indices)
            max_req_idx = max(required_floor_indices)

            if current_lift_index < min_req_idx:
                # Must go up at least to max_req_idx
                move_cost = (min_req_idx - current_lift_index) + (max_req_idx - min_req_idx)
            elif current_lift_index > max_req_idx:
                 # Must go down at least to min_req_idx
                move_cost = (current_lift_index - max_req_idx) + (max_req_idx - min_req_idx)
            else: # current_lift_index is within [min_req_idx, max_req_idx]
                # Must travel from current pos to one end, then traverse the range.
                # Path 1: C -> min -> max. Cost: (C - min) + (max - min)
                # Path 2: C -> max -> min. Cost: (max - C) + (max - min)
                # Min cost = (max - min) + min(current_lift_index - min_req_idx, max_req_idx - current_lift_index)
                move_cost = (max_req_idx - min_req_idx) + min(current_lift_index - min_req_idx, max_req_idx - current_lift_index)


        # 7. Count board actions needed.
        total_board_actions = num_waiting

        # 8. Count depart actions needed.
        total_depart_actions = len(unserved_passengers) # This is num_waiting + num_boarded

        # 9. Calculate total heuristic cost.
        total_heuristic_cost = move_cost + total_board_actions + total_depart_actions

        return total_heuristic_cost
