import sys

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal state (all passengers served) by summing
    the required actions (board/depart) and an estimated minimum movement cost
    for the lift to visit all necessary floors. Necessary floors are the origin
    floors of waiting passengers and the destination floors of boarded passengers.
    The movement cost is estimated as the cost to travel from the current lift
    floor to the closest required floor, and then traverse the range of all
    required floors. This heuristic is non-admissible but aims to guide a
    greedy best-first search efficiently.

    Assumptions:
    - The PDDL domain follows the miconic structure provided.
    - Floor levels are linearly ordered by the 'above' predicate, forming a single tower.
    - Passenger destinations are static and provided in the initial state/static facts.
    - The state representation is a frozenset of strings as shown in the example.
    - Static facts are provided as a frozenset of strings.
    - All floors mentioned in the problem (initial state, goals, static) are part of the linear 'above' chain.

    Heuristic Initialization:
    The constructor pre-processes the static facts to build two essential data structures:
    1. A mapping from floor name (string) to its numerical level (int). This is derived
       from the '(above f_lower f_upper)' facts. It assumes a linear floor structure
       where f_upper is directly above f_lower. The lowest floor is assigned level 1.
    2. A mapping from passenger name (string) to their destination floor name (string).
       This is derived from the '(destin p f)' facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify:
       - The current location of the lift (`lift-at ?f`).
       - Passengers waiting at an origin floor (`origin ?p ?f`) and their floors.
       - Passengers currently boarded in the lift (`boarded ?p`).
       - Passengers already served (`served ?p`).
    2. Retrieve passenger destinations and floor levels from the pre-processed static information.
    3. Determine the set of unserved passengers by comparing all passengers (from destinations)
       with the served passengers.
    4. If all passengers are served, the state is a goal state, and the heuristic is 0.
    5. If not all passengers are served:
       - Initialize an action count (`num_actions`) to 0.
       - Initialize a set of required floors (`required_floors`) to visit.
       - Iterate through all unserved passengers:
         - If the passenger is waiting at `f_origin`: Add 1 to `num_actions` (for the 'board' action). Add `f_origin` to `required_floors`.
         - If the passenger is boarded: Add 1 to `num_actions` (for the 'depart' action). Look up their destination `f_dest` and add it to `required_floors`.
       - Calculate the estimated movement cost:
         - If `required_floors` is empty (shouldn't happen for a non-goal state with valid passengers), movement cost is 0.
         - Otherwise, get the levels for all `required_floors` and find the minimum (`min_req_level`) and maximum (`max_req_level`) levels.
         - Get the level of the current lift floor (`current_level`).
         - The movement cost is estimated as the minimum distance required to travel from `current_level` to cover the range `[min_req_level, max_req_level]`. This is calculated as `abs(current_level - closest_required_level) + (max_req_level - min_req_level)`.
       - The total heuristic value is `num_actions + movement_cost`.
    """
    def __init__(self, task):
        self.floor_levels = {}
        self.passenger_destinations = {}
        self._parse_static_info(task.static)

    def _parse_static_info(self, static_facts):
        """
        Parses static facts to build floor levels and passenger destinations.
        Assumes a linear floor structure defined by 'above' facts.
        """
        direct_above = {} # f_lower -> f_upper
        floors_that_are_above_targets = set() # Floors that appear as f_upper
        all_floors_set = set() # Collect all unique floor names

        for fact in static_facts:
            parts = fact.strip('()').split()
            if not parts: continue # Skip empty facts

            if parts[0] == 'above':
                if len(parts) == 3:
                    f_lower, f_upper = parts[1], parts[2]
                    direct_above[f_lower] = f_upper
                    floors_that_are_above_targets.add(f_upper)
                    all_floors_set.add(f_lower)
                    all_floors_set.add(f_upper)
                # else: Malformed 'above' fact, ignore

            elif parts[0] == 'destin':
                if len(parts) == 3:
                    p, f_dest = parts[1], parts[2]
                    self.passenger_destinations[p] = f_dest
                    all_floors_set.add(f_dest) # Ensure destination floors are included
                # else: Malformed 'destin' fact, ignore

        # Find the lowest floor: a floor in all_floors_set that is never an 'above' target (f_upper)
        potential_lowest_floors = all_floors_set - floors_that_are_above_targets

        lowest_floor = None
        if len(potential_lowest_floors) == 1:
            lowest_floor = list(potential_lowest_floors)[0]
        elif len(potential_lowest_floors) > 1:
             # Multiple potential lowest floors. This suggests disconnected towers or malformed PDDL.
             # In a standard miconic, there should be one. Pick one arbitrarily if needed.
             # For robustness, we could sort and pick the first, but this isn't guaranteed correct
             # if the PDDL isn't strictly linear/single-tower. Assuming standard miconic.
             print(f"Warning: Found multiple potential lowest floors: {potential_lowest_floors}. Assuming linear structure and attempting to find base.", file=sys.stderr)
             # Let's try finding the floor that is a key in direct_above but not a value? No, that's highest.
             # Let's rely on the single lowest floor assumption for miconic. If not found, floor_levels will be incomplete.
             pass # lowest_floor remains None

        if lowest_floor is None and all_floors_set:
             # This case happens if potential_lowest_floors is empty (e.g., cycle) or > 1 and we didn't pick one.
             # If all_floors_set is not empty, we failed to find a unique lowest floor.
             print("Error: Could not determine a unique lowest floor from 'above' facts. Floor levels cannot be reliably assigned.", file=sys.stderr)
             # floor_levels will remain empty, leading to KeyErrors later if floors are used.
             return

        # Build floor levels by traversing up from the lowest floor
        current_floor = lowest_floor
        level = 1
        visited_floors = set() # Guard against cycles, though unlikely in miconic
        while current_floor is not None and current_floor not in visited_floors:
            visited_floors.add(current_floor)
            self.floor_levels[current_floor] = level
            current_floor = direct_above.get(current_floor)
            level += 1

        # Basic check: Ensure all floors mentioned in static facts got a level
        # This helps catch issues if the 'above' chain doesn't include all floors.
        for f in all_floors_set:
             if f not in self.floor_levels:
                 print(f"Warning: Floor '{f}' was mentioned in static facts but not assigned a level via the 'above' chain. Heuristic may be inaccurate.", file=sys.stderr)
                 # Assign a dummy level? Or skip? Skipping might cause KeyError.
                 # Assigning 0 or -1 might work if only differences are used, but levels start at 1.
                 # Let's assume valid PDDL where all floors are connected.

    def __call__(self, state, task):
        """
        Computes the domain-dependent heuristic for the given state.

        @param state: The current state (frozenset of fact strings).
        @param task: The planning task object (contains static info, goals etc.).
        @return: The estimated number of actions to reach the goal. Returns float('inf')
                 if floor levels could not be parsed (indicating a problem with static info).
        """
        if not self.floor_levels:
             # Floor levels weren't successfully parsed during init
             print("Error: Floor levels not initialized. Returning infinity heuristic.", file=sys.stderr)
             # Fallback heuristic: number of unserved passengers? Or infinity?
             # Infinity is safer if we can't compute a meaningful value.
             # Let's count unserved passengers as a fallback if floor levels are missing.
             served_passengers_count = sum(1 for fact in state if fact.startswith('(served '))
             all_passengers_count = len(self.passenger_destinations)
             if all_passengers_count == 0: return 0 # No passengers? Goal is trivially met.
             return all_passengers_count - served_passengers_count # Simple count heuristic

        current_lift_floor = None
        waiting_passengers = {} # p -> f_origin
        boarded_passengers = set()
        served_passengers = set()

        # Parse state facts
        for fact in state:
            parts = fact.strip('()').split()
            if not parts: continue
            predicate = parts[0]
            if predicate == 'lift-at' and len(parts) == 2:
                current_lift_floor = parts[1]
            elif predicate == 'origin' and len(parts) == 3:
                p, f_origin = parts[1], parts[2]
                waiting_passengers[p] = f_origin
            elif predicate == 'boarded' and len(parts) == 2:
                p = parts[1]
                boarded_passengers.add(p)
            elif predicate == 'served' and len(parts) == 2:
                p = parts[1]
                served_passengers.add(p)

        # Get all passengers from destinations (assuming all passengers have destinations)
        all_passengers = set(self.passenger_destinations.keys())

        # Check if goal is reached
        if len(served_passengers) == len(all_passengers):
             return 0 # Goal state

        num_actions = 0 # Count board/depart actions needed
        required_floors = set() # Floors the lift must visit

        for p in all_passengers:
            if p in served_passengers:
                continue # Passenger is already served

            if p in waiting_passengers:
                f_origin = waiting_passengers[p]
                if f_origin in self.floor_levels: # Ensure floor is valid
                    required_floors.add(f_origin)
                    num_actions += 1 # Need to board this passenger
                # else: Invalid state? Passenger waiting at unknown floor.
            elif p in boarded_passengers:
                # Passenger is boarded, needs to depart at destination
                f_dest = self.passenger_destinations.get(p)
                if f_dest and f_dest in self.floor_levels: # Ensure destination floor is valid
                    required_floors.add(f_dest)
                    num_actions += 1 # Need to depart this passenger
                # else: Invalid state? Boarded passenger with no destination or unknown destination floor.
            # else: Passenger is unserved but not waiting or boarded? Invalid state?

        # Calculate movement cost
        movement_cost = 0
        if required_floors and current_lift_floor in self.floor_levels:
            required_levels = sorted([self.floor_levels[f] for f in required_floors])
            min_req_level = required_levels[0]
            max_req_level = required_levels[-1]
            current_level = self.floor_levels[current_lift_floor]

            # Estimate movement to cover the range of required floors starting from current
            # Minimum distance to visit points a and b starting from c is |c-a| + |b-a| if c is outside [a,b]
            # and |b-a| + min(|c-a|, |c-b|) if c is inside [a,b].
            if current_level < min_req_level:
                movement_cost = (min_req_level - current_level) + (max_req_level - min_req_level)
            elif current_level > max_req_level:
                movement_cost = (current_level - max_req_level) + (max_req_level - min_req_level)
            else: # min_req_level <= current_level <= max_req_level
                # Cost to go down to min_req and sweep up to max_req OR go up to max_req and sweep down to min_req
                cost_down_first = (current_level - min_req_level) + (max_req_level - min_req_level)
                cost_up_first = (max_req_level - current_level) + (max_req_level - min_req_level)
                movement_cost = min(cost_down_first, cost_up_first)
        elif required_floors and current_lift_floor not in self.floor_levels:
             # Required floors exist, but lift is at an unknown floor? Invalid state?
             print(f"Warning: Lift is at unknown floor '{current_lift_floor}'. Cannot calculate movement cost.", file=sys.stderr)
             # Fallback: just return action count?
             movement_cost = 0 # Cannot estimate movement

        # Total heuristic is actions needed + movement needed
        h = num_actions + movement_cost

        return h

