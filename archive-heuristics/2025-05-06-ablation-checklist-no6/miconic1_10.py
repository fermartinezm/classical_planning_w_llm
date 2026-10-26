# Import necessary modules
# from heuristics.heuristic_base import Heuristic # Assuming this import path

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str):
        return []
    return fact.strip('()').split()

class miconicHeuristic: # Inherit from Heuristic if available in the environment
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    Estimates the cost as the sum of:
    1. The number of board/depart actions needed for unserved passengers.
       - 2 actions (board, depart) for passengers waiting at origin.
       - 1 action (depart) for passengers already boarded.
    2. The minimum number of moves required to visit all necessary floors.
       - Necessary floors are origins of waiting passengers and destinations of boarded passengers.
       - Move cost is estimated as the distance from the current lift floor to the closest extreme (min or max index) of the required floors, plus the span of the required floors.

    # Heuristic Initialization
    - Extracts passenger destinations from static facts.
    - Determines floor ordering based on 'above' facts and creates floor-to-index mapping.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the state is a goal state (all passengers served). If yes, return 0.
    2. Find the current floor of the lift.
    3. Identify all unserved passengers and their current state (at origin or boarded).
    4. Calculate H1: Sum 2 for each unserved passenger at their origin, and 1 for each unserved boarded passenger.
    5. Identify the set of floors the lift must visit: origins of waiting passengers and destinations of boarded passengers.
    6. If no floors need visiting (H1 is 0), H2 is 0.
    7. If floors need visiting, find the minimum and maximum indices among these floors.
    8. Calculate H2: Minimum moves from the current lift floor to reach the range of required floors and traverse the span. This is min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx).
    9. The total heuristic is H1 + H2.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Passenger destinations.
        - Floor ordering and indexing.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.passenger_destinations = {}
        all_passengers = set()
        all_floors = set()

        # Extract info from static facts
        floor_above_map = {} # Maps floor_below -> floor_above
        floors_that_are_second_arg = set() # Floors that have something immediately above them

        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]

             if predicate == 'destin':
                 if len(parts) == 3:
                     passenger, dest_floor = parts[1], parts[2]
                     self.passenger_destinations[passenger] = dest_floor
                     all_passengers.add(passenger)
                     all_floors.add(dest_floor)
             elif predicate == 'above':
                 if len(parts) == 3:
                     f_above, f_below = parts[1], parts[2]
                     floor_above_map[f_below] = f_above # Map f_below to f_above
                     floors_that_are_second_arg.add(f_below)
                     all_floors.add(f_above)
                     all_floors.add(f_below)

        # Extract floors/passengers from initial state (origins, lift-at, boarded, served)
        for fact in initial_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'origin':
                 if len(parts) == 3:
                    all_passengers.add(parts[1])
                    all_floors.add(parts[2])
            elif predicate == 'lift-at':
                 if len(parts) == 2:
                    all_floors.add(parts[1])
            elif predicate == 'boarded':
                 if len(parts) == 2:
                    all_passengers.add(parts[1])
            elif predicate == 'served':
                 if len(parts) == 2:
                    all_passengers.add(parts[1])


        # 2. Determine floor ordering and indexing
        self.floor_to_index = {}
        self.index_to_floor = []

        if not all_floors:
             pass # No floors in the problem
        else:
            # Find the lowest floor: a floor that is in all_floors but is not the second argument
            # of any 'above' fact.
            lowest_floor = None
            potential_lowest_floors = [f for f in all_floors if f not in floors_that_are_second_arg]

            if len(potential_lowest_floors) == 1:
                lowest_floor = potential_lowest_floors[0]
            elif len(potential_lowest_floors) > 1:
                 # Multiple potential lowest floors, try finding the highest instead.
                 floors_that_are_first_arg = {get_parts(fact)[1] for fact in static_facts if get_parts(fact) and get_parts(fact)[0] == 'above' and len(get_parts(fact))==3}
                 potential_highest_floors = [f for f in all_floors if f not in floors_that_are_first_arg]

                 if len(potential_highest_floors) == 1:
                     highest_floor = potential_highest_floors[0]
                     # Build map downwards from highest
                     floor_below_map = {v: k for k, v in floor_above_map.items()}
                     current_floor = highest_floor
                     current_index = len(all_floors) - 1
                     temp_floor_to_index = {}
                     temp_index_to_floor = [None] * len(all_floors)
                     visited_floors = set()
                     while current_floor is not None and current_index >= 0 and current_floor not in visited_floors:
                         visited_floors.add(current_floor)
                         temp_floor_to_index[current_floor] = current_index
                         temp_index_to_floor[current_index] = current_floor
                         current_floor = floor_below_map.get(current_floor)
                         current_index -= 1

                     if len(temp_floor_to_index) == len(all_floors):
                         self.floor_to_index = temp_floor_to_index
                         self.index_to_floor = temp_index_to_floor
                 # else: Cannot determine unique highest floor either, floor_to_index remains empty

            elif len(potential_lowest_floors) == 0 and all_floors:
                 # This might happen with a single floor not involved in 'above', or a cycle.
                 if len(all_floors) == 1:
                      lowest_floor = list(all_floors)[0]
                 # else: Cannot determine lowest floor, floor_to_index remains empty


            # If lowest_floor was found successfully and index map is not already built by fallback
            if lowest_floor is not None and not self.floor_to_index:
                current_floor = lowest_floor
                current_index = 0
                visited_floors = set()
                # Traverse upwards from the lowest floor
                while current_floor is not None and current_floor not in visited_floors:
                    visited_floors.add(current_floor)
                    self.floor_to_index[current_floor] = current_index
                    self.index_to_floor.append(current_floor)
                    current_floor = floor_above_map.get(current_floor)
                    current_index += 1

            # Final check: ensure all floors found are in the index map
            if len(self.floor_to_index) != len(all_floors):
                 # If not all floors could be indexed, clear the map as it's unreliable
                 self.floor_to_index = {}
                 self.index_to_floor = []


        # Store all passenger names for easy iteration
        self.all_passengers = list(all_passengers)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Check if floor indexing was successful during init
        if not self.floor_to_index or len(self.floor_to_index) != len(self.index_to_floor) or len(self.floor_to_index) != len(set(self.floor_to_index.values())):
             # Floor indexing failed during init or resulted in duplicates/incomplete map
             return float('inf')


        # 1. Find current lift floor
        curr_f = None
        # Convert state frozenset to set of strings for efficient lookup
        state_facts_str = {str(fact) for fact in state}

        for fact_str in state_facts_str:
            parts = get_parts(fact_str)
            if not parts: continue
            if parts[0] == 'lift-at' and len(parts) == 2:
                curr_f = parts[1]
                break

        if curr_f is None or curr_f not in self.floor_to_index:
             # Should not happen in a valid miconic state, but handle defensively
             return float('inf')

        curr_idx = self.floor_to_index[curr_f]

        # 2. Identify unserved passengers and their state
        unserved_passengers = []
        passenger_state = {} # Map passenger -> 'origin' or 'boarded'

        # Find served passengers
        served_passengers = {get_parts(fact_str)[1] for fact_str in state_facts_str if get_parts(fact_str) and get_parts(fact_str)[0] == 'served' and len(get_parts(fact_str)) == 2}

        # Find unserved passengers and their state
        for p in self.all_passengers:
            if p not in served_passengers:
                unserved_passengers.append(p)
                # Check if boarded
                if f'(boarded {p})' in state_facts_str:
                    passenger_state[p] = 'boarded'
                else:
                    # Check if at origin
                    found_origin = False
                    for fact_str in state_facts_str:
                        parts = get_parts(fact_str)
                        if parts and parts[0] == 'origin' and len(parts) == 3 and parts[1] == p:
                            passenger_state[p] = 'origin'
                            found_origin = True
                            break
                    if not found_origin:
                         # Unserved, not boarded, not at origin? Invalid state.
                         return float('inf')


        # If no unserved passengers, goal is reached (already checked, but defensive)
        if not unserved_passengers:
             return 0

        # 3. Calculate H1 (actions) and identify required floors
        H1 = 0
        pickup_floors = set()
        dropoff_floors = set()

        for p in unserved_passengers:
            state_p = passenger_state.get(p)
            dest_f = self.passenger_destinations.get(p)

            if dest_f is None or dest_f not in self.floor_to_index:
                 # Should not happen in valid problem, passenger must have destination and it must be a known floor
                 return float('inf')

            if state_p == 'origin':
                # Find origin floor from state
                orig_f = None
                for fact_str in state_facts_str:
                    parts = get_parts(fact_str)
                    if parts and parts[0] == 'origin' and len(parts) == 3 and parts[1] == p:
                        orig_f = parts[2]
                        break
                if orig_f is None or orig_f not in self.floor_to_index:
                     # Should not happen for unserved passenger state 'origin'
                     return float('inf')

                H1 += 2 # Need board and depart
                pickup_floors.add(orig_f)
                dropoff_floors.add(dest_f)

            elif state_p == 'boarded':
                H1 += 1 # Need depart
                dropoff_floors.add(dest_f)

            # else: invalid state handled above

        # 4. Calculate H2 (moves)
        required_floors = pickup_floors | dropoff_floors

        if not required_floors:
             # This case should be covered by H1 == 0 check, but defensive
             H2 = 0
        else:
            required_indices = {self.floor_to_index[f] for f in required_floors}

            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # Minimum moves to visit all required floors starting from curr_f
            # Go from curr_f to one extreme, then sweep across the range.
            dist_to_min = abs(curr_idx - min_req_idx)
            dist_to_max = abs(curr_idx - max_req_idx)
            span = max_req_idx - min_req_idx

            H2 = min(dist_to_min, dist_to_max) + span

        # Total heuristic
        return H1 + H2
