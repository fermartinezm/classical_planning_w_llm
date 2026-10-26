from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the minimum required actions for each unserved passenger (board/depart)
        and the estimated minimum movement cost for the lift to visit all
        necessary floors (origins of waiting passengers and destinations of
        boarded passengers).

    Assumptions:
        - The lift has unlimited capacity.
        - All actions have a unit cost of 1.
        - The 'above' predicates define a total linear order on floors, forming a single tower.
        - The state representation includes all relevant facts about passenger
          locations (origin/boarded/served) and lift location.
        - All passengers mentioned in the problem have a destination defined
          in the static facts.
        - The PDDL input describes a valid miconic problem instance conforming
          to the standard structure (single tower, consistent facts).

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic pre-processes the static
        information from the task and initial state:
        1. It collects all passenger names mentioned in static ('destin') and
           initial state ('origin', 'boarded', 'served') facts into `self.all_passengers`.
        2. It extracts the destination floor for each passenger from the
           '(destin ?person ?floor)' facts and stores them in a dictionary
           `self.destinations`.
        3. It collects all floor names mentioned in static ('above', 'destin')
           and initial state ('lift-at', 'origin') facts into `all_floors`.
        4. It parses the '(above ?floor1 ?floor2)' facts to determine the
           linear order of floors. It builds a map from each floor to the
           floor directly above it (`above_map`).
        5. It finds the lowest floor (a floor in `all_floors` that is not the
           upper floor in any 'above' fact). It assumes there is exactly one
           such floor, representing the bottom of the single tower. If no such
           floor is found or multiple are found, the floor ordering cannot be
           uniquely determined, and subsequent floor lookups will likely fail.
        6. It constructs an ordered list of all floors starting from the
           lowest floor and following the 'above' relationships using `above_map`.
        7. It creates a dictionary `self.floor_to_index` mapping each floor
           name to its index in the ordered list (0 for the lowest, 1 for
           the next, etc.). This allows easy calculation of distances between
           floors. If the floor structure is invalid or incomplete, this map
           may not contain all floors, leading to errors during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        In the call method (`__call__`), for a given state:
        1. Identify the current floor of the lift by finding the fact
           '(lift-at ?floor)' in the state. If not found or the floor is unknown,
           return infinity as the state is likely invalid or unreachable.
        2. Identify all passengers who have not yet been served by checking
           for the absence of '(served ?person)' facts among `self.all_passengers`.
        3. If there are no unserved passengers, the goal is reached, and the
           heuristic value is 0.
        4. Categorize the unserved passengers based on the state facts:
           - `waiting_passengers`: Passengers for whom an '(origin ?person ?floor)'
             fact exists in the state. Store their origin floor in a dictionary `{p: f_o}`.
           - `boarded_passengers`: Passengers for whom a '(boarded ?person)'
             fact exists in the state. Store them in a set `{p}`.
           (It's assumed unserved passengers are either waiting or boarded in a valid state).
        5. Calculate the base cost representing the minimum number of board/depart
           actions needed. Each waiting passenger needs 1 board and 1 depart action
           (cost 2). Each boarded passenger needs 1 depart action (cost 1).
           Base cost = `len(waiting_passengers) * 2 + len(boarded_passengers) * 1`.
        6. Identify the set of 'target' floors that the lift must visit to
           make progress for the unserved passengers. These are the origin
           floors of waiting passengers and the destination floors of boarded
           passengers (retrieved from `self.destinations`).
        7. If there are no target floors (meaning all unserved passengers are
           boarded and already at their destination - this implies the goal is
           reached, handled in step 3), the movement cost is 0.
        8. If there are target floors, map them to their floor indices using
           `self.floor_to_index` and sort the indices. If any target floor is
           not in the index map, return infinity as the floor structure is
           inconsistent. Let `min_target_idx` and `max_target_idx` be the
           minimum and maximum indices among the target floors. Let `current_idx`
           be the index of the current lift floor.
        9. Calculate the movement cost: This is the minimum number of floor
           movements required to travel from the current floor to visit all
           target floors. This is approximated by the distance needed to cover
           the range of target floors, plus the distance from the current floor
           to the nearest end of that range.
           - If `current_idx` is less than or equal to `min_target_idx`, the cost is
             `max_target_idx - current_idx`. (Go up to the highest target)
           - If `current_idx` is greater than or equal to `max_target_idx`, the cost is
             `current_idx - min_target_idx`. (Go down to the lowest target)
           - If `current_idx` is strictly between `min_target_idx` and `max_target_idx`,
             the cost is `(max_target_idx - min_target_idx)` (to cover the range)
             plus `min(current_idx - min_target_idx, max_target_idx - current_idx)`
             (the minimum travel to reach one end of the range from the current position).
        10. The total heuristic value is the sum of the base cost (pending
            board/depart actions) and the calculated movement cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.destinations = {}
        all_passengers = set()
        above_map = {} # f_lower -> f_higher
        all_floors = set()

        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.destinations[p] = f
                all_passengers.add(p)
                all_floors.add(f)
            elif parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        # Process initial state facts to get floors and passengers
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "lift-at":
                  all_floors.add(parts[1])
             elif parts[0] == "origin":
                  all_passengers.add(parts[1])
                  all_floors.add(parts[2])
             elif parts[0] == "boarded":
                  all_passengers.add(parts[1])
             elif parts[0] == "served":
                  all_passengers.add(parts[1])

        self.all_passengers = all_passengers

        # Find the lowest floor
        lowest_floor = None
        higher_floors = set(above_map.values())
        candidate_lowest = all_floors - higher_floors

        if len(candidate_lowest) == 1:
             lowest_floor = list(candidate_lowest)[0]
        # else: If len(candidate_lowest) != 1, the floor structure is ambiguous or invalid.
        # The heuristic will likely fail later if floors are present but not orderable.


        # Build ordered floor list and index map
        self.ordered_floors = []
        self.floor_to_index = {}
        current = lowest_floor
        index = 0
        # Traverse upwards from the lowest floor
        while current is not None and current in all_floors:
            self.ordered_floors.append(current)
            self.floor_to_index[current] = index
            current = above_map.get(current)
            index += 1

        # If floor structure couldn't be built correctly, self.floor_to_index might be incomplete.
        # Subsequent lookups will raise KeyError or checks will return inf.


    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                lift_floor = parts[1]
                break

        # If lift_floor is None or not a known floor, state is invalid/inconsistent.
        if lift_floor is None or lift_floor not in self.floor_to_index:
             return float('inf')


        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "served"}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        # 3. Goal check
        if not unserved_passengers:
            return 0

        # 4. Categorize unserved passengers
        waiting_passengers = {} # {p: f_o}
        boarded_passengers = set() # {p}

        origin_facts_map = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if get_parts(fact)[0] == "origin"}
        boarded_facts_set = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "boarded"}

        for p in unserved_passengers:
            if p in boarded_facts_set:
                 boarded_passengers.add(p)
            elif p in origin_facts_map:
                 waiting_passengers[p] = origin_facts_map[p]
            # else: Unserved passenger is neither waiting nor boarded. This shouldn't happen in valid states.
            # If a passenger is unserved but not in origin_facts_map or boarded_facts_set,
            # it implies an inconsistent state or a passenger not needing service (but they are unserved).
            # Assuming valid states, this branch is not taken for unserved passengers.


        # 5. Calculate base cost (pending board/depart actions)
        N_waiting = len(waiting_passengers)
        N_boarded = len(boarded_passengers)
        base_cost = N_waiting * 2 + N_boarded # 2 actions for waiting (board, depart), 1 for boarded (depart)

        # 6. Identify target floors
        pickup_floors = set(waiting_passengers.values())
        # Ensure destinations exist for all boarded passengers (should be true based on init)
        dropoff_floors = set()
        for p in boarded_passengers:
             if p in self.destinations:
                  dropoff_floors.add(self.destinations[p])
             # else: Boarded passenger has no destination? Invalid state/problem.

        target_floors = pickup_floors | dropoff_floors

        # 7. Calculate movement cost
        movement_cost = 0
        if target_floors: # Only calculate if there are floors to visit
            # Ensure all target floors are in our floor index map
            if not all(f in self.floor_to_index for f in target_floors):
                 return float('inf') # Target floor is unknown

            target_indices = sorted([self.floor_to_index[f] for f in target_floors])
            min_target_idx = target_indices[0]
            max_target_idx = target_indices[-1]
            current_idx = self.floor_to_index[lift_floor]

            if current_idx <= min_target_idx:
                movement_cost = max_target_idx - current_idx
            elif current_idx >= max_target_idx:
                movement_cost = current_idx - min_target_idx
            else: # min_target_idx < current_idx < max_target_idx
                # Cost to reach one end + cost to sweep to the other
                movement_cost = (max_target_idx - min_target_idx) + min(current_idx - min_target_idx, max_target_idx - current_idx)

        # 10. Total heuristic
        return base_cost + movement_cost
