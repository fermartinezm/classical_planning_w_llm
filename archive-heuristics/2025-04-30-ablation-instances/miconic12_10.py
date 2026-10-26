from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, though for valid PDDL facts this format is expected.
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions and adds an estimate
    of the minimum lift movement required to visit all necessary floors
    (origins for unboarded passengers, destinations for boarded passengers).

    # Assumptions
    - Floors are arranged in a single linear stack, ordered by the 'above' predicate.
    - There is only one lift.
    - Passengers must be boarded at their origin floor and departed at their destination floor.

    # Heuristic Initialization
    - Extracts passenger destination floors from static 'destin' facts.
    - Extracts floor ordering from static 'above' facts and creates a mapping
      from floor names (e.g., 'f1', 'f2') to numerical floor levels (e.g., 1, 2).
    - Identifies all passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift.
    2. Identify all passengers who have not yet been served.
    3. For each unserved passenger, determine their current status:
       - Are they 'boarded' (inside the lift)?
       - Are they at their 'origin' floor (waiting)?
    4. Collect the set of floors the lift *must* visit:
       - The origin floor for every unboarded passenger.
       - The destination floor for every boarded passenger.
       Let this set be `floors_needed`.
    5. Calculate the number of required 'board' actions: This is equal to the number of unboarded passengers.
    6. Calculate the number of required 'depart' actions: This is equal to the number of boarded passengers.
    7. Estimate the minimum number of 'up' or 'down' movement actions:
       - If `floors_needed` is empty (all passengers served or no passengers needing service), movement cost is 0.
       - Otherwise, find the minimum and maximum floor numbers among `floors_needed`.
       - Get the numerical level of the current lift floor.
       - The movement cost is estimated as the vertical range covered by `floors_needed`
         (`max_needed_num - min_needed_num`) plus the minimum distance from the current
         lift floor to either the minimum or maximum needed floor (`min(abs(current_floor_num - min_needed_num), abs(current_floor_num - max_needed_num))`).
    8. The total heuristic value is the sum of the costs from steps 5, 6, and 7.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals # Store goals if needed, though not directly used in this heuristic calculation

        # Extract passenger destinations and identify all passengers
        self.passenger_dest = {}
        self.all_passengers = set()
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, dest_floor = get_parts(fact)
                self.passenger_dest[passenger] = dest_floor
                self.all_passengers.add(passenger)

        # Extract floor ordering and create floor name to number mapping
        self.floor_name_to_num = {}
        above_facts = [get_parts(fact) for fact in task.static if match(fact, "above", "*", "*")]

        # Find all floors mentioned in above facts
        all_floors = set()
        for parts in above_facts:
            if len(parts) == 3: # Ensure it's a valid (above f1 f2) fact
                all_floors.add(parts[1])
                all_floors.add(parts[2])

        # Find floors that are the second argument of 'above' (i.e., have a floor below them)
        floors_with_predecessor = {parts[2] for parts in above_facts if len(parts) == 3}

        # The lowest floor is one that is mentioned but does not have a floor below it
        lowest_floor = None
        potential_lowest = all_floors - floors_with_predecessor
        if len(potential_lowest) == 1:
             lowest_floor = potential_lowest.pop()
        elif len(all_floors) == 1 and not above_facts:
             # Case with only one floor and no above facts
             lowest_floor = all_floors.pop()
        # else: This might indicate multiple towers or an invalid structure.
        # For standard miconic, the above logic should find the single lowest floor.

        if lowest_floor:
            current_floor = lowest_floor
            floor_num = 1
            # Build the mapping by following the 'above' chain
            while current_floor:
                self.floor_name_to_num[current_floor] = floor_num
                floor_num += 1
                next_floor = None
                # Find the floor immediately above current_floor
                for parts in above_facts:
                    if len(parts) == 3 and parts[1] == current_floor:
                        next_floor = parts[2]
                        break
                current_floor = next_floor

        # Note: If floor_name_to_num is still empty, it implies no floors or invalid structure.
        # The heuristic will likely return 0 if no floors/passengers are found or mapped.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Check if goal is reached (all passengers served)
        # This check is slightly redundant as the heuristic calculation will be 0 for goal states,
        # but explicit check is faster.
        if self.goals <= state:
             return 0

        unboarded_passengers = []
        boarded_passengers = []
        O_floors = set() # Origin floors of unboarded passengers
        D_floors = set() # Destination floors of boarded passengers

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # This should not happen in a valid state, but handle defensively
        if current_lift_floor is None or current_lift_floor not in self.floor_name_to_num:
             # Cannot compute heuristic without a valid lift location
             # Return a large value to discourage this state, or 0 if non-admissible is fine.
             # Returning 0 might be misleading. Let's return a value indicating problem.
             # A large constant or sum of max possible actions could work.
             # For simplicity and typical greedy search behavior, returning 0 might work
             # if such states are unreachable or indicate a problem state we don't want to explore.
             # Assuming valid states have a lift location mapped to a number.
             pass # current_lift_floor is guaranteed by problem structure and mapping

        # Find unserved passengers and their status
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                # Check if passenger is boarded
                is_boarded = any(match(fact, "boarded", passenger) for fact in state)

                if is_boarded:
                     boarded_passengers.append(passenger)
                     # Add destination floor to D_floors
                     if passenger in self.passenger_dest and self.passenger_dest[passenger] in self.floor_name_to_num:
                         D_floors.add(self.passenger_dest[passenger])
                     # else: Destination not found or not mapped? Ignore or handle error.
                else:
                     # Passenger is unserved and not boarded, must be at origin
                     unboarded_passengers.append(passenger)
                     # Find origin floor from state facts
                     origin_floor = None
                     for fact in state:
                         if match(fact, "origin", passenger, "*"):
                             origin_floor = get_parts(fact)[2]
                             break
                     # origin_floor should be found if passenger is unserved and not boarded
                     if origin_floor and origin_floor in self.floor_name_to_num:
                         O_floors.add(origin_floor)
                     # else: Passenger lost or origin not mapped? Ignore or handle error.

        # Calculate action costs (board + depart)
        action_cost = len(unboarded_passengers) + len(boarded_passengers)

        # Calculate movement cost
        floors_needed = O_floors.union(D_floors)
        movement_cost = 0

        if floors_needed:
            # Ensure current lift floor is mapped
            if current_lift_floor in self.floor_name_to_num:
                f_curr_num = self.floor_name_to_num[current_lift_floor]

                # Get numerical levels for all needed floors that are mapped
                needed_nums = [self.floor_name_to_num[f] for f in floors_needed if f in self.floor_name_to_num]

                if needed_nums: # Only calculate if there are mapped floors to visit
                    min_needed_num = min(needed_nums)
                    max_needed_num = max(needed_nums)

                    # Movement cost estimate: range + min distance to ends
                    range_cost = max_needed_num - min_needed_num
                    cost_to_min = abs(f_curr_num - min_needed_num)
                    cost_to_max = abs(f_curr_num - max_needed_num)

                    movement_cost = range_cost + min(cost_to_min, cost_to_max)
                # else: floors_needed was not empty, but none were in our mapping. movement_cost remains 0.
            # else: current_lift_floor not in mapping. movement_cost remains 0.

        # Total heuristic value
        total_cost = action_cost + movement_cost

        return total_cost

