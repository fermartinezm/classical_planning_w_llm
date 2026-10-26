from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the number of necessary board and depart actions
    for unserved passengers and adds an estimate of the minimum lift
    travel needed to reach the next service floor.

    # Assumptions
    - Each unserved passenger needs one board action (if not already boarded)
      and one depart action.
    - The lift must visit floors where passengers are waiting or need to depart.
    - The cost of travel is estimated by the distance from the current lift
      floor to the nearest floor requiring service.
    - The 'above' facts define a linear order of all relevant floors.

    # Heuristic Initialization
    - Parses the 'above' facts from the static information to build a map
      of floor indices, allowing calculation of distances between floors.
    - Stores the destination floor for each passenger by looking at the
      goal facts (to identify passengers) and static facts (to find their destinations).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Find the current floor of the lift using the `(lift-at ?f)` fact.
       - Identify passengers waiting at their origin floors using `(origin ?p ?f)` facts. Store these in a map {passenger: origin_floor}.
       - Identify passengers currently boarded using `(boarded ?p)` facts.
       - Identify passengers already served using `(served ?p)` facts.

    2. Identify Unserved Passengers:
       - Get the set of all passengers from the stored destination map (built during initialization).
       - Subtract the set of served passengers to get the set of unserved passengers.

    3. Categorize Unserved Passengers:
       - Create a set `P_waiting` containing unserved passengers found in the `origin_map_state`.
       - Create a set `P_boarded` containing unserved passengers found in the `boarded_passengers` set.

    4. Calculate Passenger Action Cost:
       - Each passenger in `P_waiting` needs a board action and a depart action (estimated 2 actions).
       - Each passenger in `P_boarded` needs a depart action (estimated 1 action).
       - Sum these costs: `passenger_action_cost = 2 * len(P_waiting) + 1 * len(P_boarded)`.

    5. Identify Floors Requiring Service:
       - Create a set `service_floors` containing the origin floors for all passengers in `P_waiting` and the destination floors for all passengers in `P_boarded`. Use the `destin_map` (built during initialization) for destinations.

    6. Calculate Estimated Travel Cost:
       - If `service_floors` is empty, the travel cost is 0 (all relevant passengers are at or past the boarding stage and either boarded or served). Note: If the goal is not reached but service_floors is empty, it implies an unusual state where unserved passengers are neither waiting nor boarded, which might indicate an unsolvable state or an issue with state representation/domain definition. Assuming valid states, service_floors will be non-empty if the goal is not reached.
       - If `service_floors` is not empty, calculate the minimum distance from the `current_lift_floor` to any floor in `service_floors`. This distance is the estimated travel cost. The distance is calculated using the floor index map.

    7. Compute Total Heuristic Value:
       - The total heuristic value is the sum of `passenger_action_cost` and `travel_cost`.
       - If the goal state is reached, the heuristic is 0. This check is performed first.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build floor order and index map from 'above' facts
        below_map = {}
        all_floors = set()
        is_above_target = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_higher, f_lower = parts[1], parts[2]
                below_map[f_higher] = f_lower
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                is_above_target.add(f_lower)

        self.floors = []
        self.floor_index_map = {}

        if all_floors:
            # Find the highest floor (a floor that is never the target of 'above')
            # Assumes a single chain of floors
            highest_floor = next((f for f in all_floors if f not in is_above_target), None)

            # If highest_floor is None, it implies a cycle or all floors are targets, which shouldn't happen in miconic.
            # If there's only one floor, all_floors has 1 element, is_above_target is empty, highest_floor is that floor.
            # If there are multiple floors, there must be at least one highest.
            # If somehow highest_floor is None but all_floors is not empty, this might indicate an invalid domain structure.
            # We proceed assuming a valid linear floor structure defined by 'above'.

            if highest_floor is not None:
                floor_order_desc = []
                current = highest_floor
                while current is not None:
                    floor_order_desc.append(current)
                    current = below_map.get(current)

                self.floors = list(reversed(floor_order_desc)) # Lowest to Highest
                self.floor_index_map = {floor: index for index, floor in enumerate(self.floors)}
            # else: # Handle case where highest_floor is None but all_floors is not empty?
                   # This implies a non-linear or cyclic 'above' structure, not standard miconic.
                   # The heuristic would likely be ill-defined. We proceed assuming linear.


        # Store goal locations for each passenger
        self.destin_map = {}
        # Passengers are identified by the 'served' goals
        passengers_in_goal = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}

        # Find destinations for these passengers from static facts
        for passenger in passengers_in_goal:
             for static_fact in static_facts:
                 static_parts = get_parts(static_fact)
                 if static_parts[0] == 'destin' and static_parts[1] == passenger:
                     self.destin_map[passenger] = static_parts[2]
                     break # Found destination for this passenger


    def distance(self, f1, f2):
        """Calculate the number of floors between f1 and f2."""
        # Return a large value if floors are not in the map (e.g., invalid state or floor)
        if f1 not in self.floor_index_map or f2 not in self.floor_index_map:
             # This could happen if a floor appears in state/goals but not in 'above' facts.
             # Assuming 'above' facts define all relevant floors for movement.
             # If a floor is isolated, distance is infinite for practical purposes of movement.
             # However, for heuristic calculation, we might need a finite value.
             # Let's assume valid inputs where all floors are in the map.
             # If not, returning a large number makes this path less desirable.
             return float('inf')
        return abs(self.floor_index_map[f1] - self.floor_index_map[f2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        current_lift_floor = None
        origin_map_state = {} # {passenger: origin_floor} for waiting passengers in the current state
        boarded_passengers_state = set() # set of passengers currently boarded
        served_passengers_state = set() # set of passengers currently served

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
            elif parts[0] == 'origin':
                origin_map_state[parts[1]] = parts[2]
            elif parts[0] == 'boarded':
                boarded_passengers_state.add(parts[1])
            elif parts[0] == 'served':
                served_passengers_state.add(parts[1])

        # Identify unserved passengers based on the goal requirements
        # Unserved are those in destin_map whose served goal is not met in the current state
        all_passengers_in_goal = set(self.destin_map.keys())
        unserved_passengers = all_passengers_in_goal - served_passengers_state

        # Categorize unserved passengers based on their current state
        # A passenger is waiting if they are unserved AND their origin fact is in the state
        P_waiting = {p for p in unserved_passengers if p in origin_map_state}
        # A passenger is boarded if they are unserved AND their boarded fact is in the state
        P_boarded = {p for p in unserved_passengers if p in boarded_passengers_state}

        # Calculate passenger action cost (board + depart)
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart = 1 action
        passenger_action_cost = 2 * len(P_waiting) + 1 * len(P_boarded)

        # Identify floors requiring service
        # Origin floors of waiting passengers + Destination floors of boarded passengers
        service_floors = set()
        for p in P_waiting:
            # Add origin floor for waiting passengers
            service_floors.add(origin_map_state[p])
        for p in P_boarded:
            # Add destination floor for boarded passengers
            # Ensure passenger has a destination in the map (should be true for unserved)
            if p in self.destin_map:
                service_floors.add(self.destin_map[p])

        # Calculate estimated travel cost
        travel_cost = 0
        # Travel is needed only if there are floors to visit and the lift's current floor is known and valid
        if service_floors and current_lift_floor in self.floor_index_map:
            # Distance from current lift floor to the nearest service floor
            min_dist = float('inf')
            for f in service_floors:
                # Ensure the service floor is also in the map
                if f in self.floor_index_map:
                    dist = self.distance(current_lift_floor, f)
                    min_dist = min(min_dist, dist)

            if min_dist != float('inf'):
                 travel_cost = min_dist
            # else: # All service floors are not in the map? Indicates invalid problem structure.
                   # travel_cost remains 0 or could be set to inf. Let's keep 0 for now.


        # Total heuristic is sum of passenger actions and estimated travel
        return passenger_action_cost + travel_cost
