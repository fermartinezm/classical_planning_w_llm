from heuristics.heuristic_base import Heuristic
from task import Task

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into a list of strings."""
    # Remove leading/trailing brackets and split by space
    parts = fact_string[1:-1].split()
    return parts

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum movement cost
    to visit all necessary floors (origins for unboarded passengers, destinations
    for all unserved passengers) and the minimum number of board/depart actions
    required for unserved passengers.

    Assumptions:
    - The domain is Miconic.
    - Floors are linearly ordered, defined by `(above f_i f_j)` facts where
      `f_j` is the floor immediately above `f_i`.
    - Passenger origins and destinations are static.
    - A valid initial state includes a `(lift-at ?f)` fact.
    - All floors mentioned in `origin` or `destin` facts are part of the
      floor structure defined by `above` facts.

    Heuristic Initialization:
    - Parses the static facts (`self.task.static`) to determine the linear
      ordering of floors and creates mappings between floor names and integer
      indices (`floor_to_index`, `index_to_floor`). It handles the case of
      a single floor correctly.
    - Extracts the destination floor for each passenger from static facts
      and stores them in `passenger_destinations`.
    - Extracts the set of all passenger names from the goal facts
      (`self.task.goals`).

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state (`node.state`).
    2. Find the lift's current floor by searching for the fact `(lift-at ?f)`
       in the state. If not found, return infinity (should not happen in valid states).
       Convert the floor name to its integer index (`current_floor_idx`).
    3. Identify all unserved passengers. These are passengers from the
       pre-computed list whose `(served ?p)` fact is not in the current state.
    4. If there are no unserved passengers, the goal is reached, and the
       heuristic value is 0.
    5. Separate unserved passengers into those who are `boarded` and those
       who are not (`unboarded`) by checking for the fact `(boarded ?p)`
       in the state.
    6. Find the origin floor for each unserved, unboarded passenger by searching
       for the fact `(origin ?p ?f)` in the current state.
    7. Determine the set of floors where the lift *must* stop to perform
       a board or depart action for unserved passengers (`stops_needed`).
       - For each unserved, unboarded passenger, add their origin floor
         (if found) and their destination floor (pre-computed) to `stops_needed`.
       - For each unserved, boarded passenger, add their destination floor
         (pre-computed) to `stops_needed`.
    8. Calculate the minimum number of board/depart actions required:
       `2 * (number of unserved, unboarded passengers) + 1 * (number of unserved, boarded passengers)`.
    9. Calculate the minimum movement cost:
       - If `stops_needed` is empty, the movement cost is 0.
       - Otherwise, get the integer indices for all floors in `stops_needed`.
       - Find the minimum (`min_stop_idx`) and maximum (`max_stop_idx`)
         indices among the required stops.
       - The minimum moves required to visit the extreme floors (`min_stop_idx`
         and `max_stop_idx`) starting from the current floor index (`c_idx`)
         is calculated as:
         - If `c_idx < min_stop_idx`: `max_stop_idx - c_idx`
         - If `c_idx > max_stop_idx`: `c_idx - min_stop_idx`
         - If `min_stop_idx <= c_idx <= max_stop_idx`: `(max_stop_idx - min_stop_idx) + min(c_idx - min_stop_idx, max_stop_idx - c_idx)`
         This formula represents the minimum vertical travel distance to cover
         the range of required stops, starting from the current floor. It is
         a lower bound on the actual moves needed to visit all required stops.
    10. The total heuristic value is the sum of the movement cost and the
        board/depart action cost.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # --- Initialize floor mapping ---
        below_to_above = {}
        all_floors_set = set()
        for fact_str in self.task.static:
            parts = parse_fact(fact_str)
            if parts[0] == 'above':
                f_below, f_above = parts[1:]
                below_to_above[f_below] = f_above
                all_floors_set.add(f_below)
                all_floors_set.add(f_above)

        self.floor_to_index = {}
        self.index_to_floor = {}

        if not all_floors_set:
            # Case with only one floor. Find its name from initial state.
            single_floor = None
            for fact_str in self.task.initial_state:
                 parts = parse_fact(fact_str)
                 if parts[0] == 'lift-at':
                     single_floor = parts[1]
                     break
            if single_floor:
                self.floor_to_index[single_floor] = 0
                self.index_to_floor[0] = single_floor
            # If no lift-at in initial state (shouldn't happen in valid PDDL),
            # the maps remain empty, which is handled by checking for empty stops_needed.
        else:
            # Find the lowest floor (appears on left of 'above' but not on right)
            all_above_floors = set(below_to_above.values())
            lowest_floor = None
            # Find any floor that is mentioned but is not the target of an 'above' relation
            potential_lowest = all_floors_set - all_above_floors
            if len(potential_lowest) == 1:
                 lowest_floor = potential_lowest.pop()
            elif len(potential_lowest) > 1:
                 # This indicates a problem with the 'above' facts not forming a single chain
                 # or multiple disconnected chains. Handle defensively.
                 # For simplicity, pick one or return inf later if floors are missing.
                 # Assuming valid miconic, there's a unique lowest floor.
                 lowest_floor = potential_lowest.pop() # Pick one arbitrarily

            if lowest_floor:
                # Build the sequence and mappings
                current_floor = lowest_floor
                index = 0
                while current_floor is not None:
                    self.floor_to_index[current_floor] = index
                    self.index_to_floor[index] = current_floor
                    current_floor = below_to_above.get(current_floor) # Get floor immediately above
            # If lowest_floor is None, floor_to_index/index_to_floor remain empty,
            # handled by checking for floor existence during __call__.


        # --- Initialize passenger destinations and list of all passengers ---
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Destinations are in static facts
        for fact_str in self.task.static:
            parts = parse_fact(fact_str)
            if parts[0] == 'destin':
                p, f_destin = parts[1:]
                self.passenger_destinations[p] = f_destin

        # All passengers are those mentioned in the goal (served)
        for goal_fact_str in self.task.goals:
            parts = parse_fact(goal_fact_str)
            if parts[0] == 'served':
                p = parts[1]
                self.all_passengers.add(p)


    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor
        current_floor = None
        for fact_str in state:
            parts = parse_fact(fact_str)
            if parts[0] == 'lift-at':
                current_floor = parts[1]
                break

        # If lift position is unknown (should not happen in valid states), return infinity
        if current_floor is None:
             return float('inf')

        current_floor_idx = self.floor_to_index.get(current_floor)
        # If current floor not in floor map (problematic state), return infinity
        if current_floor_idx is None:
             return float('inf')


        # 2. Identify unserved passengers and required stops
        unserved_passengers = {p for p in self.all_passengers if '(served ' + p + ')' not in state}

        if not unserved_passengers:
            # Goal state reached
            return 0

        # Separate unserved passengers into boarded and unboarded
        unserved_boarded = {p for p in unserved_passengers if '(boarded ' + p + ')' in state}
        unserved_unboarded = unserved_passengers - unserved_boarded

        # Find origins for unserved_unboarded passengers by scanning state
        passenger_origins = {}
        for fact_str in state:
             parts = parse_fact(fact_str)
             if parts[0] == 'origin' and parts[1] in unserved_unboarded:
                  p, f_origin = parts[1:]
                  passenger_origins[p] = f_origin

        # Determine required stops (floors where board/depart actions are needed)
        stops_needed = set()
        for p in unserved_unboarded:
            if p in passenger_origins: # Ensure origin was found in state
                stops_needed.add(passenger_origins[p])
            # Add destination regardless of whether origin was found;
            # a passenger might be unboarded due to a state change not captured by origin fact removal.
            # However, the domain removes origin on board, so if unboarded, origin should be in state.
            # Let's stick to the domain definition: unboarded means origin is true.
            if p in self.passenger_destinations: # Ensure destination exists (should always)
                 stops_needed.add(self.passenger_destinations[p])

        for p in unserved_boarded:
             if p in self.passenger_destinations: # Ensure destination exists (should always)
                 stops_needed.add(self.passenger_destinations[p])


        # 3. Calculate board/depart action cost
        board_depart_cost = 2 * len(unserved_unboarded) + 1 * len(unserved_boarded)

        # 4. Calculate movement cost
        movement_cost = 0
        if stops_needed:
            # Get indices for required stops, filter out any unknown floors
            stops_needed_indices = {self.floor_to_index[f] for f in stops_needed if f in self.floor_to_index}

            if stops_needed_indices: # Check if set is not empty after filtering
                min_stop_idx = min(stops_needed_indices)
                max_stop_idx = max(stops_needed_indices)

                # Calculate minimum moves to visit the range [min_stop_idx, max_stop_idx]
                # starting from current_floor_idx.
                if current_floor_idx < min_stop_idx:
                    movement_cost = max_stop_idx - current_floor_idx
                elif current_floor_idx > max_stop_idx:
                    movement_cost = current_floor_idx - min_stop_idx
                else: # min_stop_idx <= current_floor_idx <= max_stop_idx
                    # Minimum moves to visit both extremes starting from current_floor_idx
                    # is distance to nearest extreme + distance between extremes.
                    movement_cost = (max_stop_idx - min_stop_idx) + min(current_floor_idx - min_stop_idx, max_stop_idx - current_floor_idx)
            # If stops_needed was not empty, but none of the floors were in floor_to_index,
            # movement_cost remains 0. This indicates an issue with the problem instance
            # or floor parsing, but we proceed defensively.


        # 5. Total heuristic
        h_value = movement_cost + board_depart_cost

        return h_value

