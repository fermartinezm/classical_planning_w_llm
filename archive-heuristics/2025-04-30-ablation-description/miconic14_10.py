from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The number of passengers who have not yet been served.
    2. An estimate of the minimum vertical travel distance required for the lift
       to visit all floors where passengers need to be picked up or dropped off,
       starting from the current lift location.

    # Assumptions
    - Floors are ordered sequentially, defined by `above` predicates.
    - Each passenger needs to be picked up at their origin and dropped off at their destination.
    - The cost of moving between adjacent floors is 1.
    - The cost of board/depart actions is not explicitly counted in the travel part,
      but implicitly captured by the "number of unserved passengers" component.

    # Heuristic Initialization
    - Build a mapping from floor names to integer indices based on the `above` facts
      to easily calculate vertical distances.
    - Build a mapping from passenger names to their destination floor names using
      the `destin` facts.
    - Identify all passenger names present in the problem (those appearing in `served` goals).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Count the number of passengers who are listed in the goal as needing to be `served` but are not currently in the `served` state. If this count is 0, the heuristic is 0.
    3. Identify the set of "required floors":
       - The current lift floor.
       - For every unserved passenger:
         - If the passenger is currently waiting at their origin (`origin ?p ?f` is in state), add their origin floor `?f` to the set.
         - If the passenger is currently boarded (`boarded ?p` is in state), add their destination floor (looked up from initialization) to the set.
    4. If the set of required floors contains only the current lift floor (meaning all unserved passengers are either waiting at the current floor or boarded and heading to the current floor), the minimum vertical moves is 0.
    5. Otherwise, map the required floors to their integer indices using the pre-calculated floor-to-index mapping.
    6. Find the minimum and maximum index among the required floor indices.
    7. Calculate the current lift floor index.
    8. Estimate the minimum vertical moves required to visit all floors within the range [min_required_index, max_required_index] starting from the current lift index. This is calculated as `(max_required_index - min_required_index) + min(abs(current_lift_index - min_required_index), abs(current_lift_index - max_required_index))`.
    9. The total heuristic value is the sum of the number of unserved passengers and the estimated minimum vertical moves.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and identifying all passengers from the static facts and goal.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor_to_index mapping
        floor_above_map = {}  # Map: floor_lower -> floor_higher
        floor_below_map = {}  # Map: floor_higher -> floor_lower
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_lower, f_higher = parts[1], parts[2]
                floor_above_map[f_lower] = f_higher
                floor_below_map[f_higher] = f_lower
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        # Find the lowest floor (a floor that is not a key in floor_below_map)
        lowest_floor = None
        self.floor_to_index = {}

        if all_floors:
            # Find floors that are not values in floor_below_map (i.e., no floor is below them)
            potential_lowest = all_floors - set(floor_below_map.values())

            # Check if we found a unique lowest floor (standard miconic structure)
            if len(potential_lowest) == 1:
                 lowest_floor = list(potential_lowest)[0]

            if lowest_floor:
                current_floor = lowest_floor
                index = 0
                visited_floors = set()
                # Follow the chain upwards using the floor_above_map
                while current_floor is not None and current_floor not in visited_floors:
                    visited_floors.add(current_floor)
                    self.floor_to_index[current_floor] = index
                    next_floor = floor_above_map.get(current_floor)
                    current_floor = next_floor
                    index += 1
                # If the number of floors indexed doesn't match total floors, structure is malformed
                if len(self.floor_to_index) != len(all_floors):
                     self.floor_to_index = {} # Invalidate map if structure is not a simple chain


        # 2. Build passenger_to_destin mapping and identify all passengers who need serving
        self.passenger_to_destin = {}
        self.passengers_to_serve = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, destin_floor = parts[1], parts[2]
                self.passenger_to_destin[passenger] = destin_floor

        # Passengers to serve are those listed in the goal
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts and parts[0] == "served":
                 self.passengers_to_serve.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or floor mapping failed, heuristic is infinite
        if current_lift_floor is None or not self.floor_to_index:
             return float('inf')

        current_lift_index = self.floor_to_index.get(current_lift_floor)
        # If current lift floor is not in floor_to_index (malformed problem), return inf
        if current_lift_index is None:
             return float('inf')


        # 2. Count unserved passengers and identify their relevant floors
        num_unserved = 0
        required_floors = {current_lift_floor}

        # Keep track of passenger status in the current state for quick lookup
        passenger_current_status = {}
        passenger_origin_floor = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "origin":
                    p, f = parts[1], parts[2]
                    passenger_current_status[p] = 'origin'
                    passenger_origin_floor[p] = f
                elif parts[0] == "boarded":
                    p = parts[1]
                    passenger_current_status[p] = 'boarded'
                elif parts[0] == "served":
                    p = parts[1]
                    passenger_current_status[p] = 'served'


        for p in self.passengers_to_serve:
            status = passenger_current_status.get(p, 'unknown')

            if status != 'served':
                num_unserved += 1
                if status == 'origin':
                    # Passenger is waiting at origin floor
                    origin_floor = passenger_origin_floor.get(p)
                    # Add origin floor if it's a valid floor
                    if origin_floor in self.floor_to_index:
                        required_floors.add(origin_floor)
                elif status == 'boarded':
                    # Passenger is boarded, needs to go to destination floor
                    destin_floor = self.passenger_to_destin.get(p)
                    # Add destination floor if it's a valid floor
                    if destin_floor in self.floor_to_index:
                        required_floors.add(destin_floor)
                # If status is 'unknown' for a passenger who needs serving,
                # it implies a problem definition issue or an unexpected state.
                # We don't add a required floor in this case, but the passenger
                # is still counted as unserved.


        # If all passengers are served, heuristic is 0
        if num_unserved == 0:
            return 0

        # 3. Calculate minimum vertical moves
        # Filter out required floors that are not in our floor_to_index map (malformed problem)
        valid_required_floors = {f for f in required_floors if f in self.floor_to_index}

        # If after filtering, only the current floor remains, no moves needed
        if len(valid_required_floors) == 1 and current_lift_floor in valid_required_floors:
             min_vertical_moves = 0
        elif not valid_required_floors:
             # This case should ideally not happen if current_lift_floor is valid and in floor_to_index,
             # but as a fallback, if no valid required floors other than possibly current,
             # and current is valid, moves is 0. If current was invalid, we returned inf earlier.
             min_vertical_moves = 0
        else:
            required_indices = {self.floor_to_index[f] for f in valid_required_floors}
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # Minimum moves to cover the range [min_req_idx, max_req_idx] starting from current_lift_index
            # This is the distance to one end + the full range traversal
            dist_to_min = abs(current_lift_index - min_req_idx)
            dist_to_max = abs(current_lift_index - max_req_idx)
            range_dist = max_req_idx - min_req_idx

            # Option 1: Go to min_req_idx first, then traverse to max_req_idx
            moves1 = dist_to_min + range_dist
            # Option 2: Go to max_req_idx first, then traverse to min_req_idx
            moves2 = dist_to_max + range_dist

            min_vertical_moves = min(moves1, moves2)


        # 4. Total heuristic = unserved passengers + minimum vertical moves
        total_cost = num_unserved + min_vertical_moves

        return total_cost
