from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of arguments, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the fixed costs for boarding and departing each unserved passenger
    and adds an estimated travel cost for the lift to visit all necessary floors.

    # Assumptions
    - The goal is to have all passengers in the 'served' state.
    - Each unboarded passenger requires one 'board' action.
    - Each unserved passenger requires one 'depart' action.
    - The lift must visit the origin floor of each unboarded passenger.
    - The lift must visit the destination floor of each unserved passenger.
    - The travel cost is estimated based on the lift's current position and the
      range of floors that need to be visited for pickups and dropoffs.

    # Heuristic Initialization
    - Parses the static facts to determine the floor order (mapping floor names
      like 'f1', 'f2' to numerical values) using the 'above' predicate.
    - Parses the static facts to store the destination floor for each passenger
      using the 'destin' predicate.
    - Identifies all passengers from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet 'served'.
    3. If there are no unserved passengers, the state is a goal state, return 0.
    4. Separate unserved passengers into those who are 'boarded' and those who
       are waiting at their 'origin' floor (unboarded).
    5. Calculate the fixed cost for boarding and departing:
       - Add 1 for each unboarded unserved passenger (for the 'board' action).
       - Add 1 for each unserved passenger (for the 'depart' action).
    6. Determine the set of floors the lift *must* visit:
       - Include the origin floor for every unboarded unserved passenger.
       - Include the destination floor for every unserved passenger.
    7. If the set of required floors is empty (this should only happen if there are no unserved passengers, which is handled in step 3), the travel cost is 0.
    8. If the set of required floors is not empty, calculate the estimated travel cost:
       - Convert required floor names to their numerical values.
       - Find the minimum and maximum floor numbers among the required floors.
       - Get the numerical value of the lift's current floor.
       - The travel cost is estimated as the span of the required floors
         (`max_req_num - min_req_num`) plus the minimum distance from the
         current floor to either the minimum or maximum required floor.
         Travel = `(max_req_num - min_req_num) + min(abs(current_floor_num - min_req_num), abs(current_floor_num - max_req_num))`.
    9. The total heuristic value is the sum of the fixed board/depart cost and the estimated travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract floor order and create floor name <-> number mapping
        # Find all floors mentioned in 'above' predicates
        all_floors = set()
        above_map = {} # Maps floor_below -> floor_above
        below_map = {} # Maps floor_above -> floor_below
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_above, f_below = get_parts(fact)
                all_floors.add(f_above)
                all_floors.add(f_below)
                above_map[f_below] = f_above
                below_map[f_above] = f_below

        if not all_floors:
             # Handle case with no floors (shouldn't happen in valid miconic)
             self.floor_to_num = {}
             self.num_to_floor = {}
        else:
            # Find the lowest floor (a floor that is not above any other floor)
            # In this domain definition, (above f1 f2) means f1 is higher than f2.
            # So, the floor f where no (above ?x f) exists is the highest floor.
            # The floor f where no (above f ?x) exists is the lowest floor.
            lowest_floor = None
            for floor in all_floors:
                if floor not in above_map: # This floor is not below any other floor
                    lowest_floor = floor
                    break

            # Build the ordered list of floors from lowest to highest
            ordered_floors = []
            current = lowest_floor
            num = 1
            self.floor_to_num = {}
            self.num_to_floor = {}
            while current is not None:
                ordered_floors.append(current)
                self.floor_to_num[current] = num
                self.num_to_floor[num] = current
                num += 1
                current = below_map.get(current) # Find the floor above the current one

        # Extract passenger destinations
        self.destinations = {}
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # Ensure all passengers from goals are included (in case some aren't in destin facts)
        for goal in self.goals:
             if match(goal, "served", "*"):
                  _, passenger = get_parts(goal)
                  self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # 1. Identify current lift floor
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None:
             # Should not happen in a valid state, but handle defensively
             # If lift location is unknown, cannot estimate travel.
             # Fallback: sum of board/depart for all unserved?
             # Or a large value to indicate an invalid state for this heuristic?
             # Assuming valid states always have (lift-at ?f)
             return float('inf') # Or raise an error

        current_floor_num = self.floor_to_num.get(current_floor)
        if current_floor_num is None:
             # Handle unknown floor (should not happen)
             return float('inf')


        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        # 3. If no unserved passengers, it's a goal state
        if not unserved_passengers:
            return 0

        # 4. Separate unserved passengers
        boarded_unserved = {p for p in unserved_passengers if '(boarded ' + p + ')' in state}
        unboarded_unserved = {p for p in unserved_passengers if '(origin ' + p + ' ' + '*' + ')' in state}
        # Note: A passenger is either origin or boarded if not served.
        # We can verify this assumption if needed, but it holds for miconic.

        # 5. Calculate fixed board/depart cost
        total_cost += len(unboarded_unserved) # 1 board action per unboarded passenger
        total_cost += len(unserved_passengers) # 1 depart action per unserved passenger

        # 6. Determine required floors
        required_floors = set()
        # Origin floors for unboarded passengers
        for fact in state:
             if match(fact, "origin", "*", "*"):
                  _, passenger, floor = get_parts(fact)
                  if passenger in unboarded_unserved:
                       required_floors.add(floor)

        # Destination floors for all unserved passengers
        for passenger in unserved_passengers:
             dest_floor = self.destinations.get(passenger)
             if dest_floor: # Ensure destination is known
                  required_floors.add(dest_floor)
             # else: Handle error/unknown destination? Assuming valid problem.


        # 7. Calculate estimated travel cost
        travel_cost = 0
        if required_floors:
            required_floor_nums = sorted([self.floor_to_num[f] for f in required_floors if f in self.floor_to_num])

            if required_floor_nums: # Ensure we have valid floor numbers
                min_req_num = required_floor_nums[0]
                max_req_num = required_floor_nums[-1]

                # Travel cost = span + min distance to an extreme
                span = max_req_num - min_req_num
                dist_to_min = abs(current_floor_num - min_req_num)
                dist_to_max = abs(current_floor_num - max_req_num)

                travel_cost = span + min(dist_to_min, dist_to_max)

        total_cost += travel_cost

        # 9. Return total heuristic value
        return total_cost

