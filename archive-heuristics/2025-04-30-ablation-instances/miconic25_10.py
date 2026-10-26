from fnmatch import fnmatch
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
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

# Assuming Heuristic base class is available via import
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not available, you might need to define a simple one
# or adjust the class definition depending on the environment.
# For this response, we assume the environment provides the base class structure
# expected by the __init__ and __call__ methods.
# If inheriting, uncomment the line below:
# class miconicHeuristic(Heuristic):
class miconicHeuristic: # Use this if Heuristic base class is not provided
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the number of 'board' actions needed for unboarded passengers,
    the number of 'depart' actions needed for unserved passengers, and
    an estimate of the minimum number of 'up'/'down' actions required for the lift
    to visit all necessary floors (origins of unboarded, destinations of unserved).

    # Assumptions
    - Passengers are either waiting at their origin floor, boarded in the lift, or served.
    - The lift can carry multiple passengers.
    - The floor names are structured as 'fN' where N is the floor number.
    - The 'above' predicate defines the floor order from bottom (f1) to top (fN_max).
    - The minimum travel cost to visit a set of floors on a line is the distance
      from the current floor to the nearest extreme of the required floor range,
      plus the distance covering the entire required floor range.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from static facts.
    - Build a mapping from floor names (e.g., 'f1') to integer floor numbers (e.g., 1)
      by parsing static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers in the problem instance (e.g., from destination facts).
    3. Classify passengers as 'served', 'boarded', or 'unboarded' based on the current state.
       - 'Served': `(served p)` is in the state.
       - 'Boarded': `(boarded p)` is in the state and not served.
       - 'Unboarded': `(origin p of)` is in the state for some floor `of` and not served.
    4. Count the number of 'board' actions needed: This is the count of unboarded passengers.
    5. Count the number of 'depart' actions needed: This is the count of unserved passengers (unboarded + boarded).
    6. Determine the set of floors the lift must visit:
       - Include the origin floor for each unboarded passenger.
       - Include the destination floor for each unserved passenger.
    7. Calculate the estimated travel cost:
       - If there are no required floors to visit, travel cost is 0.
       - Otherwise, find the minimum and maximum floor numbers among the required floors.
       - The travel cost is the minimum of the distance from the current lift floor
         to the minimum required floor and the distance from the current lift floor
         to the maximum required floor, plus the distance covering the entire required floor range. This estimates the cost of one trip covering the range.
    8. Sum the counts from steps 4, 5, and 7 to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - Mapping from floor names to integer numbers.
        """
        # task.goals and task.static are available from the base class or task object

        # Extract destination floors for each passenger
        self.destinations = {}
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor mapping (e.g., 'f1' -> 1, 'f2' -> 2)
        # Find all unique floor names by looking for terms starting with 'f' followed by digits
        floor_names = set()
        for fact in task.static:
            parts = get_parts(fact)
            for part in parts:
                 # Use regex to find floor names like f1, f10, etc.
                 if re.fullmatch(r'f\d+', part):
                     floor_names.add(part)

        # Create the map assuming 'fN' means floor number N
        self.floor_map = {floor_name: int(floor_name[1:]) for floor_name in floor_names}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the current floor of the lift.
        current_lift_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor_name = get_parts(fact)
                break # Assuming only one lift-at fact

        # current_lift_floor_num will be None if current_lift_floor_name is None
        current_lift_floor_num = self.floor_map.get(current_lift_floor_name, None)


        # 2. Identify all passengers.
        # We get all passengers from the destinations found in static facts.
        all_passengers = set(self.destinations.keys())

        # 3. Classify passengers.
        served_passengers = {p for p in all_passengers if f'(served {p})' in state}
        unserved_passengers = all_passengers - served_passengers

        # Unboarded passengers are those unserved and not boarded
        unboarded_passengers = set()
        boarded_passengers = set()
        for p in unserved_passengers:
            if f'(boarded {p})' in state:
                boarded_passengers.add(p)
            # If not boarded and not served, they must be at their origin floor
            else:
                 unboarded_passengers.add(p)


        # 4. Count board actions needed.
        num_board_actions = len(unboarded_passengers)

        # 5. Count depart actions needed.
        num_depart_actions = len(unserved_passengers)

        # 6. Determine the set of floors the lift must visit.
        required_floor_nums = set()

        # Origin floors for unboarded passengers
        # We need to find the origin floor for each unboarded passenger from the state
        for p in unboarded_passengers:
            origin_floor_name = None
            for fact in state:
                 if match(fact, "origin", p, "*"):
                     _, _, origin_floor_name = get_parts(fact)
                     break
            if origin_floor_name:
                 origin_floor_num = self.floor_map.get(origin_floor_name)
                 if origin_floor_num is not None: # Check if floor name is valid
                     required_floor_nums.add(origin_floor_num)
            # else: Unboarded passenger must have an origin fact in a valid state.


        # Destination floors for unserved passengers
        for p in unserved_passengers:
            dest_floor_name = self.destinations.get(p)
            if dest_floor_name:
                dest_floor_num = self.floor_map.get(dest_floor_name)
                if dest_floor_num is not None: # Check if floor name is valid
                    required_floor_nums.add(dest_floor_num)
            # else: Unserved passenger must have a destination in a valid state.


        # 7. Calculate the estimated travel cost.
        travel_cost = 0
        # Only calculate travel if there are floors to visit and we know the lift's location
        if required_floor_nums and current_lift_floor_num is not None:
            min_floor_num = min(required_floor_nums)
            max_floor_num = max(required_floor_nums)

            # Minimum travel to cover the range [min_floor_num, max_floor_num]
            # starting from current_lift_floor_num
            dist_to_min = abs(current_lift_floor_num - min_floor_num)
            dist_to_max = abs(current_lift_floor_num - max_floor_num)
            range_dist = max_floor_num - min_floor_num

            # The lift must travel from current_floor to one end of the required range,
            # then traverse the range.
            travel_cost = min(dist_to_min, dist_to_max) + range_dist


        # 8. Sum the costs.
        # Total heuristic = board actions + depart actions + travel actions
        total_heuristic = num_board_actions + num_depart_actions + travel_cost

        return total_heuristic
