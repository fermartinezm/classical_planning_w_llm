# from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available elsewhere

def get_parts(fact):
    """Helper function to parse a PDDL fact string into a list of parts."""
    # Example: '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    return fact[1:-1].split()

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state (all passengers served) by summing the estimated minimum actions
    needed for each unserved passenger independently. It considers the lift's
    current position and the passenger's origin and destination floors.

    Assumptions:
    - The 'above' predicates define a linear ordering of floors where
      (above f_i f_j) means f_i is directly above f_j. This allows assigning
      a unique index to each floor from lowest to highest.
    - The heuristic is non-admissible; it sums individual passenger costs,
      overlooking potential efficiencies from batching pickups and dropoffs
      in a single lift trip.
    - All unserved passengers are either waiting at their origin or are boarded.
    - Passenger destinations are static and available in the initial state facts.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic precomputes:
    1. The linear ordering of floors by parsing the `(above f_i f_j)` static facts.
       It identifies the highest floor (one that is not above any other floor)
       and traverses downwards using the 'above' relationships to build the
       ordered list from highest to lowest. This list is then reversed to get
       the order from lowest to highest floor.
    2. A mapping from each floor name (e.g., 'f1') to its corresponding index
       in the ordered list (1-based index).
    3. A mapping from each passenger to their destination floor by parsing the
       `(destin p f_d)` static facts.

    Step-By-Step Thinking for Computing Heuristic:
    In the heuristic function (`__call__`), for a given state:
    1. Find the current floor of the lift by locating the `(lift-at ?f)` fact.
       Get its corresponding floor index using the precomputed map.
    2. Identify all passengers who are currently served by finding the `(served ?p)` facts.
    3. Determine the set of unserved passengers by taking all passengers (whose
       destinations were precomputed) and removing the served ones.
    4. If there are no unserved passengers, the state is a goal state, and the
       heuristic returns 0.
    5. If there are unserved passengers, iterate through them. For each unserved
       passenger `p`:
       a. Determine if the passenger is waiting at their origin or is boarded
          by checking for `(origin p ?f_o)` or `(boarded p)` facts in the current state.
       b. Retrieve the passenger's destination floor `f_d` using the precomputed
          destination map. Get its corresponding floor index.
       c. If the passenger is boarded:
          The estimated cost for this passenger is the number of moves required
          to travel from the current lift floor to the destination floor, plus
          1 action for departing. Cost = `abs(current_lift_floor_index - destination_floor_index) + 1`.
       d. If the passenger is waiting at their origin `f_o`:
          Get the origin floor's index from the state fact. The estimated cost
          for this passenger is the number of moves from the current lift floor
          to the origin floor, plus 1 action for boarding, plus the number of
          moves from the origin floor to the destination floor, plus 1 action
          for departing.
          Cost = `abs(current_lift_floor_index - origin_floor_index) + 1 + abs(origin_floor_index - destination_floor_index) + 1`.
       e. Sum the estimated cost for each unserved passenger to get the total
          heuristic value.
    6. Return the total computed cost.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Precompute floor ordering and index mapping
        above_map = {} # f_above -> f_below means f_above is directly above f_below
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                above_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the highest floor (one that is not a key in above_map)
        # This assumes a linear structure where only the highest floor is not above another.
        floors_above = set(above_map.keys())
        highest_floor = None
        for floor in all_floors:
             if floor not in floors_above:
                 highest_floor = floor
                 break

        # Handle case with a single floor or potential graph issues (though unlikely in valid miconic)
        if highest_floor is None and len(all_floors) > 0:
             # If no floor is found that isn't above another, it might be a single floor
             # or a non-linear 'above' structure. Assume single floor if only one exists.
             if len(all_floors) == 1:
                  highest_floor = list(all_floors)[0]
             else:
                  # Fallback: Find the lowest floor (not a value in above_map) and traverse up
                  floors_below = set(above_map.values())
                  lowest_floor = None
                  for floor in all_floors:
                       if floor not in floors_below:
                            lowest_floor = floor
                            break
                  if lowest_floor:
                       # Build reverse map to traverse upwards
                       below_map = {v: k for k, v in above_map.items()} # f_below -> f_above
                       ordered_floors_asc = []
                       current = lowest_floor
                       while current is not None:
                            ordered_floors_asc.append(current)
                            current = below_map.get(current)
                       # Create floor_to_index map (1-based index)
                       self.floor_to_index = {floor: i + 1 for i, floor in enumerate(ordered_floors_asc)}
                       # Precompute passenger destinations from static facts
                       self.destin_map = {} # p -> f_d
                       for fact in static_facts:
                            parts = get_parts(fact)
                            if parts[0] == 'destin':
                                p, f_d = parts[1], parts[2]
                                self.destin_map[p] = f_d
                       return # Initialization complete via fallback

        # Primary path: Found highest_floor, traverse downwards
        ordered_floors_desc = []
        current = highest_floor
        while current is not None:
            ordered_floors_desc.append(current)
            current = above_map.get(current)

        # Reverse to get lowest to highest
        ordered_floors_asc = ordered_floors_desc[::-1]

        # Create floor_to_index map (1-based index)
        self.floor_to_index = {floor: i + 1 for i, floor in enumerate(ordered_floors_asc)}

        # Precompute passenger destinations from static facts
        self.destin_map = {} # p -> f_d
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'destin':
                 p, f_d = parts[1], parts[2]
                 self.destin_map[p] = f_d


    def __call__(self, node):
        state = node.state

        # Find current lift location
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                lift_floor = parts[1]
                break

        # If lift location is not found, return a large value (should not happen in valid states)
        if lift_floor is None:
             return 1_000_000 # Indicate an invalid state

        idx_c = self.floor_to_index[lift_floor]

        # Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}

        # Get all passengers from precomputed destinations
        all_passengers = set(self.destin_map.keys())

        # Identify unserved passengers
        unserved_passengers = all_passengers - served_passengers

        # If no unserved passengers, goal is reached
        if not unserved_passengers:
            return 0

        # Collect current status of unserved passengers
        waiting_passengers_details = {} # p -> f_o
        boarded_passengers_set = set() # p

        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is about a passenger and that passenger is unserved
            if len(parts) > 1 and parts[1] in unserved_passengers:
                predicate = parts[0]
                p = parts[1]
                if predicate == 'origin':
                    f_o = parts[2]
                    waiting_passengers_details[p] = f_o
                elif predicate == 'boarded':
                    boarded_passengers_set.add(p)

        # Calculate heuristic cost by summing individual passenger costs
        total_cost = 0

        for p in unserved_passengers:
            # An unserved passenger must be either waiting or boarded.
            # Check boarded status first as it's simpler.
            if p in boarded_passengers_set:
                # Passenger is boarded, needs to go to destination and depart
                f_d = self.destin_map[p]
                idx_d = self.floor_to_index[f_d]
                # Cost: move from current lift floor to destin + depart
                cost = abs(idx_c - idx_d) + 1
                total_cost += cost
            elif p in waiting_passengers_details:
                # Passenger is waiting at origin, needs pickup, travel, and dropoff
                f_o = waiting_passengers_details[p]
                idx_o = self.floor_to_index[f_o]
                f_d = self.destin_map[p]
                idx_d = self.floor_to_index[f_d]
                # Cost: move from current lift floor to origin + board + move from origin to destin + depart
                cost = abs(idx_c - idx_o) + 1 + abs(idx_o - idx_d) + 1
                total_cost += cost
            # else: This case should not be reached for a valid unserved passenger.
            # If it were reached, it would mean an unserved passenger is neither
            # waiting nor boarded, which contradicts the domain state representation.
            # We could add an assertion here: assert False, f"Unserved passenger {p} is neither waiting nor boarded."

        return total_cost
