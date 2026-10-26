from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (board, depart, and lift travel)
    required to serve all passengers. It counts the minimum necessary board and
    depart actions and adds an estimate of the lift travel cost. The travel cost
    is estimated as the distance required to cover the range of floors where
    pickups or dropoffs are needed, starting from the current lift floor.

    # Assumptions
    - The cost of each action (board, depart, up, down) is 1.
    - The lift can carry multiple passengers.
    - The lift can pick up and drop off multiple passengers at the same floor.
    - The lift travel cost between adjacent floors is 1.
    - The floor structure is linear, defined by the 'above' predicate, allowing
      assignment of integer ranks (floor numbers).
    - Passenger destinations are static and available in the task's static facts.

    # Heuristic Initialization
    - Extracts the floor ordering from static facts using the 'above' predicate
      to create a mapping from floor names (e.g., 'f1') to integer floor numbers.
      The lowest floor (not below any other floor) is assigned rank 1, the next
      lowest rank 2, and so on.
    - Extracts passenger destination floors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)`.
    2. Identify all passengers who are not yet 'served' by checking for the
       absence of the fact `(served ?p)`.
    3. If there are no unserved passengers, the state is a goal state, and the
       heuristic value is 0.
    4. Initialize counters for required board and depart actions to 0.
    5. Initialize sets for required pickup floors and required dropoff floors to empty.
    6. Iterate through all unserved passengers:
       - Check if the passenger is waiting at their origin floor by finding a fact
         `(origin ?p ?f_o)`. If found, increment the board action counter, and add
         `?f_o` to the set of required pickup floors.
       - Check if the passenger is currently boarded by finding the fact `(boarded ?p)`.
         If found, increment the depart action counter.
       - Regardless of whether the passenger is waiting or boarded, they need to
         reach their destination. Find their destination floor `?f_d` (from the
         pre-computed map) and add `?f_d` to the set of required dropoff floors.
    7. Determine the set of all floors the lift must stop at (`S_stops`), which is
       the union of required pickup floors and required dropoff floors.
    8. Calculate the estimated lift travel cost:
       - If `S_stops` is empty (this case should be covered by step 3, but as a
         safeguard), the travel cost is 0.
       - Otherwise, find the minimum and maximum integer floor numbers among the
         floors in `S_stops` using the pre-computed floor mapping.
       - Get the integer floor number for the current lift floor.
       - The estimated travel cost is the distance from the current lift floor
         number to the closest end of the required floor range ([min_stop_num, max_stop_num]),
         plus the distance to traverse the entire range.
         Travel cost = `min(abs(current_floor_num - min_stop_num), abs(current_floor_num - max_stop_num)) + (max_stop_num - min_stop_num)`.
    9. The total heuristic value is the sum of the required board actions,
       required depart actions, and the estimated lift travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor mapping and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static

        # Build floor mapping: floor_name -> integer rank
        self.floor_to_int = {}
        self.int_to_floor = {}
        # Map f_higher -> set of f_lower floors it is directly above
        above_to_below = {}
        # Set of all floors mentioned in 'above' or 'destin' facts
        all_floors = set()
        # Set of floors that are below some other floor
        floors_that_are_below = set()

        # First pass: Collect all floors and build the above_to_below map
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                if f_higher not in above_to_below:
                    above_to_below[f_higher] = set()
                above_to_below[f_higher].add(f_lower)
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_that_are_below.add(f_lower)
            # Also collect floors from destination facts
            if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 all_floors.add(floor)


        # Find the lowest floor(s) - those not appearing as f_lower in any (above f_higher f_lower)
        current_rank = 1
        ranked_floors = set()
        # Floors that are not below anything else are the lowest rank
        current_level_floors = all_floors - floors_that_are_below

        # Handle cases with no 'above' facts or a single floor
        if not current_level_floors and all_floors:
             if len(all_floors) == 1:
                 f = list(all_floors)[0]
                 self.floor_to_int[f] = 1
                 self.int_to_floor[1] = f
                 ranked_floors.add(f)
             else:
                 # Problematic 'above' definition (e.g., cycle or disconnected)
                 # Cannot determine linear floor order. Heuristic will likely be inf.
                 pass


        while current_level_floors:
            next_level_floors = set()
            for floor in current_level_floors:
                self.floor_to_int[floor] = current_rank
                self.int_to_floor[current_rank] = floor
                ranked_floors.add(floor)

            # Find floors that are directly above floors in the current level
            # A floor 'f_above' is in the next level if it is above some floor in the current level
            # AND all floors 'f_below' that 'f_above' is above are already ranked.
            for f_above in all_floors:
                 if f_above in ranked_floors: # Already ranked
                      continue

                 is_ready_for_next_rank = False
                 if f_above in above_to_below:
                      # Check if all floors f_below that f_above is above are ranked
                      is_ready_for_next_rank = True
                      for f_below in above_to_below[f_above]:
                           if f_below not in ranked_floors:
                                is_ready_for_next_rank = False
                                break

                 if is_ready_for_next_rank:
                      next_level_floors.add(f_above)


            current_rank += 1
            current_level_floors = next_level_floors # next_level_floors already excludes ranked_floors by the check above

        # Check if all floors were ranked
        if len(ranked_floors) != len(all_floors):
             # This indicates a problem with the 'above' facts (e.g., disconnected components, cycles)
             # The heuristic might be inaccurate or return inf if unranked floors are relevant.
             pass


        # Store passenger destinations
        self.passenger_destin = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destin[passenger] = floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If floor mapping failed or lift is at an unranked floor, return infinity
        if not self.floor_to_int or current_lift_floor not in self.floor_to_int:
             # This state is likely invalid or involves floors not defined by 'above'
             return float('inf')

        current_floor_num = self.floor_to_int[current_lift_floor]

        # 2. Identify unserved passengers
        all_passengers = set(self.passenger_destin.keys())
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state reached

        # 3. Identify waiting and boarded unserved passengers and required stops
        required_pickup_floors = set()
        required_dropoff_floors = set()
        num_board_actions = 0
        num_depart_actions = 0

        state_facts_str = set(state) # Convert frozenset to set for faster lookup

        for p in unserved_passengers:
            # Check if waiting at origin
            origin_fact_pattern = f"(origin {p} *)"
            origin_floor = None
            for fact in state_facts_str:
                 if fnmatch(fact, origin_fact_pattern):
                      origin_floor = get_parts(fact)[2]
                      break

            if origin_floor:
                num_board_actions += 1
                required_pickup_floors.add(origin_floor)
            elif f"(boarded {p})" in state_facts_str:
                num_depart_actions += 1
            # else:
                # Passenger is unserved but neither waiting nor boarded.
                # This shouldn't happen in a valid state. Ignore or return inf?
                # Let's ignore for now, assuming valid states.

            # Destination floor is needed for dropoff for ALL unserved passengers
            destin_floor = self.passenger_destin.get(p)
            if destin_floor:
                required_dropoff_floors.add(destin_floor)
            # else: Destination not found for unserved passenger (problematic instance)


        # 7. Determine all floors the lift must stop at
        S_stops = required_pickup_floors.union(required_dropoff_floors)

        # 8. Calculate estimated lift travel cost
        travel_cost = 0
        stop_floor_nums = [self.floor_to_int[f] for f in S_stops if f in self.floor_to_int]

        if stop_floor_nums: # Ensure there are valid floors in S_stops
            min_stop_num = min(stop_floor_nums)
            max_stop_num = max(stop_floor_nums)

            # Travel from current floor to the range [min_stop_num, max_stop_num]
            # plus travel to cover the range.
            # This is the distance to the closest end + the range size.
            dist_to_min = abs(current_floor_num - min_stop_num)
            dist_to_max = abs(current_floor_num - max_stop_num)
            range_dist = max_stop_num - min_stop_num

            travel_cost = min(dist_to_min, dist_to_max) + range_dist

        # 9. Total heuristic value
        total_cost = num_board_actions + num_depart_actions + travel_cost

        return total_cost

