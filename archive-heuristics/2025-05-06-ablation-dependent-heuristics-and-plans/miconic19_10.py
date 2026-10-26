from fnmatch import fnmatch
# Assuming heuristic_base.py is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match a PDDL fact string against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for arity
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the Heuristic base class if not provided in the environment
# This is a minimal definition based on the example usage
class Heuristic:
    def __init__(self, task):
        raise NotImplementedError
    def __call__(self, node):
        raise NotImplementedError


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the required number of actions (board, depart, lift movements)
    to get all passengers to their destinations. It sums the minimum actions per passenger
    (board + depart) and adds an estimate for the lift movement cost required to visit
    all floors where pickups or dropoffs are needed.

    # Assumptions
    - The domain uses standard STRIPS semantics with unit costs for actions.
    - The 'above' predicates define a linear ordering of floors.
    - The lift can carry multiple passengers.
    - The heuristic does not need to be admissible.

    # Heuristic Initialization
    - Parses the 'above' predicates to establish the floor order and create a mapping
      from floor names to numerical indices. Assumes floors are ordered linearly
      where (above f_higher f_lower) means f_lower is immediately below f_higher.
    - Stores the destination floor for each passenger from the static 'destin' facts.
    - Identifies the set of passengers that need to be served based on the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the fact '(lift-at ?f)' in the state.
    2. Determine which passengers are not yet 'served' by checking the goal conditions
       against the 'served' facts in the current state.
    3. If the set of unserved goal passengers is empty, the heuristic is 0 (goal state).
    4. Initialize a count for the minimum actions per passenger. For each unserved passenger,
       they will require at least a 'board' action and a 'depart' action to reach the served state.
       Add 2 to the total heuristic count for each unserved passenger.
    5. Identify the set of unique floors that are relevant for servicing the unserved passengers
       in the current state:
       - For each unserved passenger currently at their origin floor (fact '(origin ?p ?f_o)' in state),
         add their origin floor ?f_o to the set of required floors (for pickup).
       - For each unserved passenger currently 'boarded' (fact '(boarded ?p)' in state),
         add their destination floor (looked up from initialization data) to the set of required floors (for dropoff).
    6. Include the current lift floor in the set of relevant floors used for calculating the travel range.
    7. Using the floor-to-index mapping created during initialization, find the numerical indices
       for all relevant floors.
    8. Determine the minimum and maximum floor indices among these relevant floors.
    9. Calculate the estimated lift movement cost: This is the total span of floors that must be covered
       (maximum relevant index - minimum relevant index) plus the minimum distance from the current
       lift floor's index to either the minimum or maximum relevant index. This estimates the minimum
       travel needed to visit all required floors starting from the current location.
    10. The total heuristic value is the sum of the minimum actions per passenger (step 4) and the
        estimated lift movement cost (step 9).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, floor indices,
        passenger destinations, and goal passengers from the task definition.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor order and create floor_to_index map
        # (above f_higher f_lower) means f_lower is immediately below f_higher
        floor_below_map = {} # Maps higher_floor -> lower_floor
        all_floors = set()
        floors_that_are_higher = set() # Floors that appear as the higher floor in an 'above' fact
        floors_that_are_lower = set()  # Floors that appear as the lower floor in an 'above' fact

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                floor_below_map[f_higher] = f_lower
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                floors_that_are_higher.add(f_higher)
                floors_that_are_lower.add(f_lower)

        # Find the highest floor (is higher than something, but nothing is higher than it)
        # This is the floor that appears as a higher floor but never a lower floor
        highest_floor = None
        # Check floors that are higher than something
        for f in floors_that_are_higher:
            # If this floor is never the lower floor for any other 'above' fact, it's the highest
            if f not in floors_that_are_lower:
                 highest_floor = f
                 break
        # If no 'above' facts, or only one floor, the single floor is the highest (and lowest)
        if highest_floor is None and all_floors:
             highest_floor = list(all_floors)[0]


        # If there are no floors defined, the map is empty
        if not all_floors:
             self.floor_to_index = {}
        else:
            # Build floor_to_index map starting from the highest floor and going down
            self.floor_to_index = {}
            current_floor = highest_floor
            # Assuming floor indices are 1-based
            current_index = len(all_floors) # Start index from the total number of floors

            # Traverse downwards from the highest floor using the floor_below_map
            # The loop continues as long as the current_floor is a key in the map,
            # meaning there is a floor below it.
            while current_floor in floor_below_map:
                self.floor_to_index[current_floor] = current_index
                current_floor = floor_below_map[current_floor]
                current_index -= 1

            # Add the lowest floor which is the last one in the chain (not a key in floor_below_map)
            self.floor_to_index[current_floor] = current_index


        # 2. Store passenger destinations
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # 3. Identify goal passengers
        self.goal_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.goal_passengers.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is unknown, state is likely invalid or not reachable
        if current_lift_floor is None:
             return float('inf')

        # Get the index of the current lift floor
        current_lift_index = self.floor_to_index.get(current_lift_floor)
        # If the current floor is not in our floor index map, something is wrong
        if current_lift_index is None and self.floor_to_index:
             return float('inf')
        # Handle case with only one floor (no 'above' facts) - index is 1
        if not self.floor_to_index:
             current_lift_index = 1


        # 2. Determine unserved passengers and collect required floors
        unserved_passengers = set()
        required_floors = set() # Floors that need a pickup or dropoff

        # Check which goal passengers are not yet served
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for passenger in self.goal_passengers:
            if passenger not in served_passengers_in_state:
                unserved_passengers.add(passenger)

        num_unserved = len(unserved_passengers)

        # 3. If all goal passengers are served, heuristic is 0
        if num_unserved == 0:
            return 0

        # Collect required floors for unserved passengers
        # Need to know the current status (origin or boarded) for each unserved passenger
        passenger_status = {} # Map passenger -> 'origin_floor' or 'boarded'
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "origin":
                  p, f = parts[1], parts[2]
                  if p in unserved_passengers:
                      passenger_status[p] = f # Store origin floor
             elif parts[0] == "boarded":
                  p = parts[1]
                  if p in unserved_passengers:
                      passenger_status[p] = "boarded" # Store status

        for passenger in unserved_passengers:
            status = passenger_status.get(passenger)

            if status and status != "boarded": # Passenger is at an origin floor (status is the floor name)
                 required_floors.add(status) # Need to stop here for pickup
            elif status == "boarded": # Passenger is boarded
                 dest_floor = self.destinations.get(passenger)
                 if dest_floor: # Should always exist for a goal passenger
                      required_floors.add(dest_floor) # Need to stop here for dropoff
            # else: Passenger is unserved but not at origin and not boarded? (e.g. initial state inconsistency)
            # This heuristic assumes valid states reachable from a valid initial state.

        # 4. Calculate estimated lift movement cost
        # Include the current lift floor in the set of relevant floors for range calculation
        all_relevant_floors = required_floors | {current_lift_floor}

        # If for some reason no relevant floors were found but there are unserved passengers,
        # this indicates an issue or a state where passengers are unserved but require no action
        # (e.g., already at destination but not departed - this state should allow depart).
        # In a well-formed problem, unserved passengers imply required floors.
        if not all_relevant_floors:
             movement_cost = 0
        else:
            # Get indices for relevant floors
            # Ensure all relevant floors are in our map. If not, heuristic is infinite.
            try:
                relevant_indices = {self.floor_to_index[f] for f in all_relevant_floors}
            except KeyError:
                 # A relevant floor was not in the parsed floor hierarchy. Invalid state.
                 return float('inf')


            min_idx = min(relevant_indices)
            max_idx = max(relevant_indices)

            # Cost to travel from current floor to cover the range [min_idx, max_idx]
            # Go to one end of the range and sweep to the other
            movement_cost = (max_idx - min_idx) + min(abs(current_lift_index - min_idx), abs(current_lift_index - max_idx))

        # 5. Total heuristic = (actions per passenger) + movement cost
        # Each unserved passenger needs at least 2 actions (board, depart)
        # This is a lower bound on actions *per passenger*, ignoring shared travel.
        # The movement cost is added separately.
        total_heuristic = num_unserved * 2 + movement_cost

        return total_heuristic
