# Need to import the base class
from heuristics.heuristic_base import Heuristic
# No other specific modules like task or operator are needed for this heuristic calculation itself,
# as task information is passed to the constructor and state/node to __call__.
# The task object itself provides access to goals and static info.

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the number of remaining actions (board/depart) plus the estimated
    lift travel distance required to serve all unserved passengers. The travel
    distance is estimated as the span of floors that must be visited
    (origin floors for unboarded passengers, destination floors for boarded
    passengers) plus the distance from the current lift floor to the closest
    end of this required floor span.

    Assumptions:
    - The PDDL state representation is consistent with the domain definition.
    - The 'above' predicate defines a total order on floors.
    - Every passenger has a defined origin (if not boarded) and a defined
      destination (static).
    - The state only contains dynamic facts (lift-at, origin, boarded, served).
    - Static facts (destin, above) are provided in task.static.
    - The problem is solvable (heuristic is finite).

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes the static information:
    1. Parses 'destin' facts to create a mapping from passenger names to their
       destination floor names.
    2. Parses 'above' facts to determine the total order of floors. It counts
       for each floor how many other floors are below it based on the 'above'
       relation.
    3. Sorts the floors based on this count to get the ordered list from lowest
       to highest.
    4. Creates a mapping from floor names to their index in the ordered list.
       This allows quick calculation of distances between floors.

    Step-By-Step Thinking for Computing Heuristic:
    In the __call__ method for a given state:
    1. Check if the current state is a goal state (all passengers served). If yes,
       the heuristic value is 0.
    2. Identify the current floor of the lift by finding the '(lift-at ?f)' fact
       in the state and get its index using the pre-calculated floor-to-index map.
    3. Identify all passengers that are not yet served by checking against the
       goal facts (e.g., '(served ?p)').
    4. Categorize the unserved passengers into 'unboarded' (have an '(origin ?p ?f)'
       fact in the state) and 'boarded' (have a '(boarded ?p)' fact in the state).
    5. Collect the indices of the floors that the lift *must* visit:
       - The origin floor index for each unboarded, unserved passenger.
       - The destination floor index for each boarded, unserved passenger
         (using the pre-calculated destin map).
    6. Calculate the 'action cost': This is the number of 'board' actions needed
       (equal to the number of unboarded, unserved passengers) plus the number
       of 'depart' actions needed (equal to the number of boarded, unserved
       passengers).
    7. Calculate the 'travel cost':
       - If there are no required floors to visit (i.e., all unserved passengers
         are already at their destination floors, or there are no unserved
         passengers - handled by step 1), the travel cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required
         floors. The travel cost is estimated as the difference between the maximum
         and minimum required indices (the span) plus the minimum distance from
         the current lift floor index to either the minimum or maximum required index.
    8. The total heuristic value is the sum of the action cost and the travel cost.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.destin_map = {}
        above_pairs = set()
        all_floors = set()

        # Process static facts from task.static
        for fact_str in task.static:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin_map[passenger] = floor
            elif predicate == 'above':
                f_higher = parts[1]
                f_lower = parts[2]
                above_pairs.add((f_higher, f_lower))
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Determine floor order based on 'above' relations
        # Count how many floors are below each floor
        num_below = {f: 0 for f in all_floors}
        for f_higher, f_lower in above_pairs:
            # If f_higher is above f_lower, then f_lower is below f_higher
            # We count how many floors are below a given floor f_higher
            # This count will be higher for higher floors.
            # Sorting by this count ascendingly gives order from lowest to highest.
            num_below[f_higher] += 1

        # Sort floors by the number of floors below them (ascending)
        # This gives the order from lowest to highest floor
        self.floors = sorted(list(all_floors), key=lambda f: num_below[f])
        self.floor_to_index = {f: i for i, f in enumerate(self.floors)}

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove leading '(' and trailing ')'
        cleaned_str = fact_str[1:-1]
        # Split by space
        parts = cleaned_str.split()
        return parts

    def __call__(self, node):
        state = node.state

        # 1. Check for goal state
        # The base Heuristic class might do this, but we ensure it's 0 only for goal states.
        if self.task.goal_reached(state):
             return 0

        # Pre-process state facts for faster lookup
        state_facts_dict = {}
        current_lift_floor = None
        for fact_str in state:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'lift-at':
                current_lift_floor = parts[1]
            # Store facts by predicate for easier lookup
            if predicate not in state_facts_dict:
                state_facts_dict[predicate] = []
            state_facts_dict[predicate].append(parts)

        # Should always find lift-at fact in a valid state
        idx_current = self.floor_to_index[current_lift_floor]

        # 3. Identify unserved passengers
        # Passengers in the goal are the ones that need to be served
        goal_passengers = set()
        for goal_fact_str in self.task.goals:
             if goal_fact_str.startswith('(served '):
                 parts = self._parse_fact(goal_fact_str)
                 goal_passengers.add(parts[1])

        unserved_passengers = set()
        served_facts = {tuple(p) for p in state_facts_dict.get('served', [])}
        for p in goal_passengers:
            if ('served', p) not in served_facts:
                unserved_passengers.add(p)

        # If no unserved passengers, it must be a goal state (already checked)
        # or an intermediate state where all goal passengers are served but other facts differ.
        # However, the goal is defined as ALL goal passengers served.
        # If unserved_passengers is empty, goal_reached must be True.
        if not unserved_passengers:
             return 0 # Should be caught by goal_reached check, but double-check

        # 4. Categorize unserved passengers and collect required floor indices
        num_unboarded = 0
        num_boarded = 0
        required_indices = set()

        boarded_passengers_in_state = {p[1] for p in state_facts_dict.get('boarded', [])}
        origin_facts_in_state = {tuple(p) for p in state_facts_dict.get('origin', [])}

        for p in unserved_passengers:
            if p in boarded_passengers_in_state:
                num_boarded += 1
                # Passenger is boarded, needs to go to destination
                f_dest = self.destin_map[p]
                required_indices.add(self.floor_to_index[f_dest])
            else: # Assume unboarded if not served and not boarded
                num_unboarded += 1
                # Passenger is unboarded, needs to be picked up at origin
                f_origin = None
                # Find origin floor for this unboarded passenger
                for fact_parts in origin_facts_in_state:
                    if fact_parts[1] == p:
                         f_origin = fact_parts[2]
                         break
                # Should always find origin if unboarded and unserved in a valid state
                if f_origin is not None:
                    required_indices.add(self.floor_to_index[f_origin])
                # else: This would indicate an invalid state where an unserved, unboarded
                # passenger is not at any origin floor. Assuming valid states.


        # 6. Calculate heuristic
        # Actions: 1 for each board, 1 for each depart
        # This counts the number of pickup/dropoff events needed
        action_cost = num_unboarded + num_boarded

        # Travel cost
        travel_cost = 0
        if required_indices: # If there are floors the lift must visit
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)
            # Travel is the span of required floors plus distance to closest end
            # This estimates the travel needed to cover the range [min_req_idx, max_req_idx]
            # starting from idx_current.
            travel_cost = (max_req_idx - min_req_idx) + min(abs(idx_current - min_req_idx), abs(idx_current - max_req_idx))

        h_value = action_cost + travel_cost

        return h_value
