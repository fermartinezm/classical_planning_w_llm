from heuristics.heuristic_base import Heuristic
# No need for fnmatch as we parse facts manually
# from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace from parsing
    return fact.strip()[1:-1].split()


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the number of remaining board/depart actions needed
    for unserved passengers and an estimate of the lift movement actions
    required to visit all necessary floors (origins for waiting passengers,
    destinations for boarded passengers).

    # Assumptions
    - Floor names are in the format 'f<number>' (e.g., f1, f2, f10) and are
      ordered numerically. The heuristic relies on this numerical ordering
      to calculate distances between floors.
    - All relevant 'destin' facts for passengers appearing in the goal are
      available in the static facts or initial state.
    - All relevant 'above' facts defining the floor structure are available
      in the static facts.
    - All floors involved in the problem (origins, destinations, lift location)
      can be identified from static facts ('above', 'destin') and the initial
      state ('lift-at', 'origin').

    # Heuristic Initialization
    - Extracts all passenger names that need to be served from the goal conditions.
    - Extracts all relevant floor names from static facts ('above', 'destin') and
      initial state facts ('lift-at', 'origin').
    - Creates a mapping from floor names ('f<number>') to integer indices,
      sorted based on the numerical part of the floor name.
    - Stores passenger destination floors by mapping passenger names to floor
      names, extracted from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Passengers: Iterate through the goal conditions and
       the current state to determine which passengers are not yet in the
       '(served ?p)' state. If the set of unserved passengers is empty, the
       goal is reached, and the heuristic is 0.

    2. Find Current Lift Location: Scan the current state facts to find the
       '(lift-at ?f)' predicate, which indicates the floor where the lift
       is currently located.

    3. Identify Required Floors:
       - Initialize two sets: 'pickup_floors' and 'dropoff_floors'.
       - Iterate through the current state facts:
         - If a fact is '(origin ?p ?f)' and '?p' is an unserved passenger,
           add floor '?f' to 'pickup_floors'.
         - If a fact is '(boarded ?p)' and '?p' is an unserved passenger,
           look up the destination floor for '?p' using the pre-computed
           destination map and add that floor to 'dropoff_floors'.

    4. Determine Target Floors: The set of floors the lift *must* visit to
       make progress towards the goal is the union of 'pickup_floors' and
       'dropoff_floors'. If this combined set is empty, it means all unserved
       passengers are currently boarded and at their destination floors. In
       this specific scenario, the only remaining actions are 'depart' actions.
       The number of such passengers is equal to the total number of unserved
       passengers. The move cost is 0 as the lift is already at the required
       floors. The heuristic value is simply the number of unserved passengers.

    5. Calculate Estimated Move Actions:
       - If the set of target floors is not empty:
         - Map the current lift floor and all target floors to their numerical
           indices using the pre-computed floor-to-index map.
         - Find the minimum index ('min_target_idx') and maximum index
           ('max_target_idx') among the target floor indices.
         - Calculate the estimated minimum number of moves required for the
           lift to travel from its current floor index ('lift_floor_idx') to
           visit all floors within the range defined by 'min_target_idx' and
           'max_target_idx'.
           - If 'lift_floor_idx' is below 'min_target_idx', the moves needed
             are the distance from 'lift_floor_idx' up to 'max_target_idx'.
           - If 'lift_floor_idx' is above 'max_target_idx', the moves needed
             are the distance from 'lift_floor_idx' down to 'min_target_idx'.
           - If 'lift_floor_idx' is within the range [min_target_idx,
             max_target_idx], the moves needed are the distance to traverse the
             entire range ('max_target_idx' - 'min_target_idx') plus the minimum
             distance from 'lift_floor_idx' to either end of the range
             (i.e., min(distance down to min, distance up to max)).

    6. Calculate Total Heuristic Value: The total heuristic value is the sum
       of the number of unserved passengers (which is a lower bound on the
       total number of board and depart actions required) and the estimated
       move actions calculated in the previous step.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal details.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions (e.g., frozenset({'(served p1)', ...}))
        self.static_facts = task.static # Static facts (e.g., frozenset({'(destin p1 f10)', '(above f1 f2)', ...}))
        self.initial_state = task.initial_state # Initial state facts

        # Extract all passengers that need to be served from the goals
        self.goal_passengers = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}

        # Extract all relevant floors and map them to indices
        self.all_floors = set()
        self.passenger_destin = {}

        # Get floors from static facts (above, destin)
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                self.all_floors.add(parts[1])
                self.all_floors.add(parts[2])
            elif parts[0] == 'destin':
                p, f = parts[1], parts[2]
                self.passenger_destin[p] = f
                self.all_floors.add(f)

        # Get floors from initial state facts (lift-at, origin)
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'lift-at':
                 self.all_floors.add(parts[1])
             elif parts[0] == 'origin':
                 self.all_floors.add(parts[2])

        # Create floor name to index mapping (assuming f<number> format)
        # Sort floors based on the numerical part to ensure correct ordering
        try:
            sorted_floors = sorted(list(self.all_floors), key=lambda f: int(f[1:]))
            self.floor_to_idx = {f: i + 1 for i, f in enumerate(sorted_floors)}
        except ValueError:
            # Handle cases where floor names might not follow f<number> format strictly
            # A more complex floor ordering logic based on 'above' predicates would be needed.
            # For this problem, we assume f<number> format based on examples.
            print("Warning: Floor names might not be in 'f<number>' format. Heuristic might be inaccurate.")
            # Fallback: simple alphabetical sort, less reliable for floor order
            sorted_floors = sorted(list(self.all_floors))
            self.floor_to_idx = {f: i + 1 for i, f in enumerate(sorted_floors)}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions from the current state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining actions to reach a goal state.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify Unserved Passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_passengers = self.goal_passengers - served_passengers
        N_unserved = len(unserved_passengers)

        # If all passengers are served, the goal is reached.
        if N_unserved == 0:
            return 0

        # 2. Find Current Lift Location
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                lift_floor = parts[1]
                break
        # Should always find lift_floor in a valid state for the miconic domain
        if lift_floor is None:
             # This indicates an unexpected state representation.
             # Return a high value to discourage exploring this state.
             return float('inf')

        # 3. Identify Required Floors (pickup and dropoff)
        pickup_floors = set()
        dropoff_floors = set()
        boarded_unserved_passengers = set()

        # Find waiting passengers and boarded passengers in the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'origin' and parts[1] in unserved_passengers:
                # Passenger is waiting at origin
                pickup_floors.add(parts[2])
            elif parts[0] == 'boarded' and parts[1] in unserved_passengers:
                # Passenger is boarded but not yet served
                boarded_unserved_passengers.add(parts[1])

        # Add destination floors for boarded unserved passengers
        for p in boarded_unserved_passengers:
            # Destination floor is static information
            dest_floor = self.passenger_destin.get(p)
            if dest_floor: # Ensure destination is known (should be for goal passengers)
                dropoff_floors.add(dest_floor)
            # else: Problem instance might be malformed if destination is missing

        # 4. Determine Target Floors
        target_floors = pickup_floors | dropoff_floors

        # If there are unserved passengers but no target floors, it implies
        # unserved passengers are boarded and at their destination.
        # The remaining cost is just the depart actions.
        if not target_floors:
             # N_unserved = N_waiting + N_boarded_unserved.
             # If target_floors is empty, pickup_floors is empty (N_waiting=0).
             # So N_unserved = N_boarded_unserved.
             # The heuristic is N_unserved (number of depart actions needed).
             return N_unserved


        # 5. Calculate Estimated Move Actions
        # Map floors to indices for distance calculation
        try:
            lift_floor_idx = self.floor_to_idx[lift_floor]
            target_indices = {self.floor_to_idx[f] for f in target_floors}
        except KeyError:
             # This indicates a floor in the state was not found during initialization.
             # Return a high value.
             # print(f"Error: Floor not found in mapping: {lift_floor} or {target_floors}")
             return float('inf')


        min_target_idx = min(target_indices)
        max_target_idx = max(target_indices)

        moves = 0
        if lift_floor_idx < min_target_idx:
            # Lift is below the lowest required floor. Must go up to at least min, then up to max.
            # Minimum moves = distance from current up to max target floor.
            moves = max_target_idx - lift_floor_idx
        elif lift_floor_idx > max_target_idx:
            # Lift is above the highest required floor. Must go down to at least max, then down to min.
            # Minimum moves = distance from current down to min target floor.
            moves = lift_floor_idx - min_target_idx
        else: # lift_floor_idx is within [min_target_idx, max_target_idx]
            # Lift is within the range of required floors. Must go to one end, then sweep the range.
            # Minimum moves = distance to traverse the range + distance to the closer end from current.
            moves = (max_target_idx - min_target_idx) + min(
                lift_floor_idx - min_target_idx,  # Distance down to min
                max_target_idx - lift_floor_idx    # Distance up to max
            )

        # 6. Calculate Total Heuristic Value
        # N_unserved is a lower bound on the number of board/depart actions.
        # Moves is an estimate of the lift travel cost.
        total_cost = N_unserved + moves

        return total_cost
