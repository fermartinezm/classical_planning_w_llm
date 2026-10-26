from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
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

# Define the heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for boarded passengers), and
    adds an estimate for the minimum number of 'up'/'down' movements required
    to visit all floors where passengers need to be picked up or dropped off.

    # Assumptions
    - The floor structure is linear and defined by 'above' facts.
    - Each waiting passenger requires one 'board' and one 'depart' action.
    - Each boarded passenger requires one 'depart' action.
    - Movement cost is estimated based on the range of floors that need servicing (pickup or dropoff).

    # Heuristic Initialization
    - Parses static facts to determine the floor ordering and create a mapping
      from floor names to integer numbers.
    - Parses static facts to store the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet 'served'.
    3. Count the number of unserved passengers who are waiting at their origin floor ('origin' predicate). This is the number of 'board' actions needed.
    4. Count the number of unserved passengers who are currently boarded ('boarded' predicate). This is the number of 'depart' actions needed.
    5. Determine the set of floors that need to be visited:
       - Origin floors of waiting passengers.
       - Destination floors of boarded passengers.
    6. If the set of floors to visit is empty, the movement cost is 0.
    7. If the set of floors to visit is not empty:
       - Map the floor names to their corresponding integer numbers using the pre-calculated mapping.
       - Find the minimum and maximum floor numbers among the floors to visit.
       - Get the integer number for the current lift floor.
       - Estimate the movement cost as the minimum number of moves to travel from the current floor to one end of the range [min_stop_num, max_stop_num] and then sweep across the entire range. This is calculated as `min(abs(current_floor_num - min_stop_num), abs(current_floor_num - max_stop_num)) + (max_stop_num - min_stop_num)`.
    8. The total heuristic value is the sum of the number of waiting passengers, the number of boarded passengers, and the estimated movement cost.
    9. If the state is a goal state (all passengers served), the heuristic should be 0. This is covered by step 2-8, as counts will be 0 and floors to visit will be empty.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals # Goal conditions are used to check if a passenger is served

        # 1. Parse static facts to determine floor ordering and mapping
        above_facts = [get_parts(fact) for fact in task.static if match(fact, "above", "*", "*")]
        all_floor_names = set()
        above_map = {} # floor_below -> floor_above
        below_map = {} # floor_above -> floor_below

        for _, f_above, f_below in above_facts:
            all_floor_names.add(f_above)
            all_floor_names.add(f_below)
            above_map[f_below] = f_above
            below_map[f_above] = f_below

        # Find the lowest floor (a floor that is not 'above' any other floor)
        # It's a floor name in all_floor_names that is not a key in below_map
        # Handle case with only one floor or no floors (though unlikely in miconic)
        if not all_floor_names:
             self.floor_name_to_num = {}
             lowest_floor = None
        else:
            lowest_floors = all_floor_names - set(below_map.keys())
            # In a valid linear ordering, there should be exactly one lowest floor
            lowest_floor = lowest_floors.pop() if lowest_floors else None


        # Build sorted list of floors and create mapping
        sorted_floors = []
        if lowest_floor:
            current = lowest_floor
            while current in above_map:
                sorted_floors.append(current)
                current = above_map[current]
            sorted_floors.append(current) # Add the highest floor

        self.floor_name_to_num = {name: i + 1 for i, name in enumerate(sorted_floors)}

        # 2. Parse static facts to store passenger destinations
        self.passenger_destinations = {}
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        num_waiting = 0
        num_boarded = 0
        pickup_floors_names = set()
        dropoff_floors_names = set()
        current_floor_name = None

        # Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through state facts to find relevant information
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "origin":
                p, of = parts[1], parts[2]
                if p not in served_passengers:
                    num_waiting += 1
                    pickup_floors_names.add(of)
            elif predicate == "boarded":
                p = parts[1]
                if p not in served_passengers:
                    num_boarded += 1
                    # Get destination floor for boarded passenger
                    df = self.passenger_destinations.get(p)
                    if df: # Ensure destination exists
                         dropoff_floors_names.add(df)
            elif predicate == "lift-at":
                current_floor_name = parts[1]

        # If all passengers are served, heuristic is 0
        if num_waiting == 0 and num_boarded == 0:
             return 0

        # Calculate movement cost
        F_stops_names = pickup_floors_names.union(dropoff_floors_names)

        movement_cost = 0
        # Only calculate movement if there are floors to visit AND we know the current floor
        if F_stops_names and current_floor_name is not None:
            # Ensure all stop floors and current floor are in our mapping (should be if problem is valid)
            valid_stops_nums = {self.floor_name_to_num[f] for f in F_stops_names if f in self.floor_name_to_num}
            if current_floor_name in self.floor_name_to_num and valid_stops_nums:
                min_stop_num = min(valid_stops_nums)
                max_stop_num = max(valid_stops_nums)
                current_floor_num = self.floor_name_to_num[current_floor_name]

                # Estimated moves to cover the range [min_stop_num, max_stop_num] starting from current_floor_num
                # This is the distance to the nearest end of the range plus the size of the range.
                movement_cost = min(abs(current_floor_num - min_stop_num), abs(current_floor_num - max_stop_num)) + (max_stop_num - min_stop_num)
            # else: If mapping fails or no valid stops, movement_cost remains 0.

        # Total heuristic = number of board actions needed + number of depart actions needed + estimated movement cost
        total_heuristic = num_waiting + num_boarded + movement_cost

        return total_heuristic
