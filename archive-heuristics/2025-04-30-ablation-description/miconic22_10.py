from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
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
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all passengers by summing:
    1. The minimum number of floor movements required for the lift to visit all floors where passengers need to be picked up or dropped off.
    2. The total number of 'board' actions needed (for passengers waiting at their origin) and 'depart' actions needed (for passengers currently boarded).

    # Assumptions
    - Floors are ordered numerically (e.g., f1 < f2 < f3).
    - The cost of moving one floor up or down is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic assumes a strategy where the lift visits necessary floors to pick up and drop off passengers. The minimum travel cost is estimated based on the range of required floors and the lift's current position.

    # Heuristic Initialization
    - Extracts the floor order and creates a mapping from floor names to integer levels.
    - Extracts the destination floor for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet served.
    3. For each unserved passenger:
       - If the passenger is at their origin floor, note this origin floor as a required pickup stop.
       - If the passenger is boarded, note their destination floor as a required dropoff stop.
    4. Collect the set of unique floor levels that need to be visited (required levels) for pickups and dropoffs.
    5. If there are no required floors and the goal is reached (all passengers served), the heuristic is 0.
    6. If there are required floors:
       - Calculate the minimum and maximum required floor levels.
       - Calculate the minimum travel cost for the lift to go from its current floor level to visit all required levels. This is estimated as the span of the required levels plus the distance from the current level to the closest end of the required level span: `(max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))`.
       - Count the number of passengers waiting at their origin (need 'board') and the number of passengers currently boarded (need 'depart').
       - The heuristic value is the sum of the minimum travel cost and the total count of pending 'board' and 'depart' actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        super().__init__(task)

        # 1. Extract floor order and create floor name -> level mapping
        # Find all floor objects from task.facts (which contains all declared objects)
        floor_objects = sorted([
            get_parts(fact)[1] for fact in self.task.facts if match(fact, "floor", "*")
        ], key=lambda f: int(f[1:])) # Sort floors numerically by the number part

        self.floor_map = {floor_name: i + 1 for i, floor_name in enumerate(floor_objects)}

        # 2. Extract passenger destinations from static facts
        self.destin_map = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.destin_map[passenger] = destination_floor

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Find current lift floor level
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_floor = get_parts(fact)
                break
        # If lift-at fact is not found, something is wrong with the state.
        # For robustness, handle this case, though it shouldn't happen in valid states.
        if current_floor is None:
             # This state is likely invalid or a dead end. Return a large value.
             return float('inf')

        current_level = self.floor_map[current_floor]

        # 2. Identify required floors (pickup and dropoff) and pending actions
        pickup_levels = set()
        dropoff_levels = set()
        unboarded_passengers_at_origin = 0
        boarded_passengers = 0

        # Iterate through state facts to find relevant passenger info
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "origin":
                _, passenger, origin_floor = parts
                # This passenger is unboarded and at origin
                pickup_levels.add(self.floor_map[origin_floor])
                unboarded_passengers_at_origin += 1
            elif predicate == "boarded":
                _, passenger = parts
                # This passenger is boarded
                # Ensure passenger exists in destin_map (should always be true for valid problems)
                if passenger in self.destin_map:
                    destination_floor = self.destin_map[passenger]
                    dropoff_levels.add(self.floor_map[destination_floor])
                    boarded_passengers += 1
                # else: boarded passenger with no destination? Invalid state.

        required_levels = pickup_levels | dropoff_levels

        # If no required stops, but goal not reached, something is wrong or state is terminal non-goal
        # As discussed, assume valid states where this implies goal is reached.
        # If required_levels is empty, the goal check at the start should have returned 0.
        # If we reach here and required_levels is empty, it implies unserved passengers
        # are neither at origin nor boarded, which is impossible in this domain.
        # Return a large value for safety if this unexpected state occurs.
        if not required_levels:
             return float('inf')


        min_required_level = min(required_levels)
        max_required_level = max(required_levels)

        # Calculate minimum travel cost
        # The lift must travel from current_level to cover the range [min_required_level, max_required_level].
        # Minimum travel is distance to closest end + span.
        # This formula covers cases where current_level is below, within, or above the required range.
        travel_cost = (max_required_level - min_required_level) + \
                      min(abs(current_level - min_required_level), abs(current_level - max_required_level))

        # Calculate action cost (board + depart)
        action_cost = unboarded_passengers_at_origin + boarded_passengers

        # Total heuristic is sum of travel and action costs
        heuristic_value = travel_cost + action_cost

        return heuristic_value
