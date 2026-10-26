# Required imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re # Although simple split is used, re might be useful for more complex parsing

# Helper functions (copied from example, slightly adapted)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    # Split by spaces, ignoring spaces inside quotes if necessary (not needed for miconic)
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have at least as many parts as the pattern has arguments
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It sums the required non-move actions (board and depart) for
    each non-served passenger and the estimated minimum number of move actions
    the lift needs to make to visit all necessary floors.

    # Assumptions
    - Passengers are either waiting at an origin floor, boarded in the lift,
      or served at their destination floor.
    - The lift has unlimited capacity.
    - The 'above' facts define a total order on floors.
    - Moving the lift between adjacent floors (in the defined order) costs 1 action.
    - Moving the lift between any two floors f_a and f_b costs abs(level(f_a) - level(f_b)) actions,
      where level(f) is the numerical level of floor f based on the 'above' relation.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Determine the total order of floors and assign a numerical level to each floor
      based on the 'above' facts. This is done by counting how many other floors
      each floor is 'above' according to the static facts. Floors that are above
      more other floors are considered higher.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Non-Served Passengers: Determine which passengers are neither
       'boarded' nor 'served'. This is done by checking against the set of
       passengers whose goal is to be 'served'.
    2. Calculate Non-Move Actions: For each non-served passenger:
       - If the passenger is waiting at their origin floor (predicate 'origin' is true),
         they will need a 'board' action and a 'depart' action (2 actions).
       - If the passenger is already boarded (predicate 'boarded' is true),
         they will only need a 'depart' action (1 action).
       Sum these actions for all non-served passengers.
    3. Identify Required Stops: Determine the set of floors the lift must visit.
       This includes the origin floors of all waiting passengers and the
       destination floors of all boarded (but not served) passengers.
    4. Calculate Move Actions:
       - Find the lift's current floor using the 'lift-at' predicate.
       - If there are no required stops, the move cost is 0.
       - If there are required stops, find the minimum and maximum floor levels
         among the required stops using the pre-calculated floor level mapping.
       - The estimated minimum number of move actions is the vertical distance
         the lift must traverse. This is estimated as the distance from the
         current floor level to the closest extreme required floor level (min or max)
         plus the distance between the minimum and maximum required floor levels.
         Specifically, it's (max_required_level - min_required_level) +
         min(abs(current_level - min_required_level), abs(current_level - max_required_level)).
         Each unit of vertical distance corresponds to one move action.
    5. Sum Costs: The total heuristic value is the sum of the non-move actions
       and the estimated move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor levels.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        # Identify all passengers whose goal is to be served
        self.goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Populate passenger destinations from static facts
        for fact in self.static:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.passenger_destinations[passenger] = floor

        # Determine floor order and assign levels
        # We assume (above f_higher f_lower) means f_higher is above f_lower.
        # Count how many floors each floor is above to determine relative height.
        floor_above_counts = {}
        all_floors = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                floor_above_counts[f_higher] = floor_above_counts.get(f_higher, 0) + 1

        # Floors not mentioned as f_higher are the lowest.
        for floor in all_floors:
             if floor not in floor_above_counts:
                 floor_above_counts[floor] = 0 # Lowest floors have 0 count

        # Sort floors by count descending to get order from highest to lowest.
        # Use floor name as a tie-breaker for deterministic ordering.
        sorted_floors = sorted(floor_above_counts.keys(), key=lambda f: (-floor_above_counts[f], f))

        # Assign levels (1-based, 1 being the lowest floor)
        self.floor_levels = {}
        # The sorted list is highest to lowest. We want level 1 for the lowest.
        # So reverse the sorted list and assign levels 1, 2, ...
        for level, floor in enumerate(reversed(sorted_floors), 1):
             self.floor_levels[floor] = level

        # Handle case with only one floor (no above facts)
        if not self.floor_levels and all_floors:
             # If there's only one floor, it gets level 1
             self.floor_levels[list(all_floors)[0]] = 1
        # If all_floors is empty, self.floor_levels remains empty, which is handled in __call__


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_f = None
        waiting_passengers_in_state = set()
        boarded_passengers_in_state = set()
        served_passengers_in_state = set()
        passenger_origins_in_state = {} # Store origins for waiting passengers

        # Parse state to find current lift location and passenger states
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'lift-at' and len(parts) == 2:
                current_f = parts[1]
            elif predicate == 'origin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                waiting_passengers_in_state.add(p)
                passenger_origins_in_state[p] = f
            elif predicate == 'boarded' and len(parts) == 2:
                boarded_passengers_in_state.add(parts[1])
            elif predicate == 'served' and len(parts) == 2:
                served_passengers_in_state.add(parts[1])

        # Identify non-served passengers relevant to the goal
        non_served_passengers = self.goal_passengers - served_passengers_in_state

        num_non_move_actions = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Calculate non-move actions and identify required stops
        for p in non_served_passengers:
            # Check if the passenger is waiting or boarded in the current state
            is_waiting = p in waiting_passengers_in_state
            is_boarded = p in boarded_passengers_in_state

            if is_waiting:
                # Needs board and depart (2 actions)
                num_non_move_actions += 2
                origin_f = passenger_origins_in_state.get(p)
                if origin_f:
                    pickup_floors.add(origin_f)
            elif is_boarded:
                # Needs depart (1 action)
                num_non_move_actions += 1
                destin_f = self.passenger_destinations.get(p)
                if destin_f:
                    dropoff_floors.add(destin_f)
            # Note: Passengers not in waiting_passengers_in_state or boarded_passengers_in_state
            # but in non_served_passengers would indicate an inconsistent state
            # (e.g., passenger disappeared). Assuming valid states.


        required_stops = pickup_floors.union(dropoff_floors)

        move_actions = 0
        # Only calculate move actions if there are stops required and we know the lift's location
        # and the floors involved are in our level mapping.
        if required_stops and current_f in self.floor_levels:
            # Get levels for current floor and required stops
            current_level = self.floor_levels[current_f]
            # Filter required stops to include only floors present in our level mapping
            required_levels = [self.floor_levels[f] for f in required_stops if f in self.floor_levels]

            if required_levels: # Ensure required_levels is not empty after filtering
                min_required_level = min(required_levels)
                max_required_level = max(required_levels)

                # Estimate move actions based on vertical distance
                # The lift must cover the range [min_required_level, max_required_level]
                # starting from current_level.
                # Minimum distance = distance to one end + distance of the range
                dist_to_min = abs(current_level - min_required_level)
                dist_to_max = abs(current_level - max_required_level)
                range_dist = max_required_level - min_required_level

                move_actions = range_dist + min(dist_to_min, dist_to_max)

        # If current_f is not in floor_levels (e.g., initial state malformed),
        # or if required_stops is empty, move_actions remains 0.

        return num_non_move_actions + move_actions
