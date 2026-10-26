# Need to import Heuristic base class if it exists, or define a mock one for standalone testing.
# Assuming Heuristic base class is provided in the environment.
# from heuristics.heuristic_base import Heuristic # Uncomment if needed

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Simple split on space after removing parens is sufficient for STRIPS facts.
    return fact[1:-1].split()


class miconicHeuristic: # Inherit from Heuristic if base class is available
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions for unserved passengers and
    adds an estimate of the travel cost for the lift to visit all necessary floors.

    # Assumptions
    - Floors are linearly ordered, defined by `(above f_i f_j)` facts, forming a single tower.
    - The lift can carry multiple passengers.
    - The goal is to serve all passengers.
    - Passenger origin and destination floors are static properties defined in the initial state.

    # Heuristic Initialization
    - Parses static facts to build a mapping from floor names to numerical levels based on the `above` predicate chain.
    - Parses initial state facts (assumed to be available via `task.initial_state_facts`) to identify all passengers and their destination floors.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located using the `(lift-at ?f)` predicate in the state. Convert this floor name to its numerical level using the pre-calculated floor mapping.
    2.  **Identify Unserved Passengers:** Determine which passengers are not yet served by checking for the absence of `(served ?p)` facts in the state. The set of all passengers is known from the goal conditions and initial state facts.
    3.  **Count Board Actions Needed:** Count the number of unserved passengers who are currently waiting at their origin floor. These passengers need a `board` action. This is found by checking for `(origin ?p ?f)` facts in the state for unserved passengers `?p`.
    4.  **Count Depart Actions Needed:** Count the number of unserved passengers who are currently boarded in the lift. These passengers need a `depart` action at their destination floor. This is found by checking for `(boarded ?p)` facts in the state for unserved passengers `?p`.
    5.  **Identify Required Floors:** Determine the set of floors the lift *must* visit to serve the remaining passengers. This includes:
        - The origin floor for every unserved passenger who is waiting at their origin.
        - The destination floor for every unserved passenger who is currently boarded.
        Map these floor names to their numerical levels.
    6.  **Calculate Travel Cost:**
        - If there are no required floors (all relevant passengers are served or don't need the lift), the travel cost is 0.
        - Otherwise, find the minimum and maximum levels among the required floors.
        - The estimated travel cost is the sum of the distance from the current lift level to the minimum required level and the distance from the current lift level to the maximum required level. This estimates the travel needed to reach one extreme of the required floors and then traverse to the other extreme, covering all floors in between.
    7.  **Sum Costs:** The total heuristic value is the sum of the counts from step 3 (board actions), step 4 (depart actions), and the travel cost from step 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build floor level mapping
        floors = set()
        below_to_above = {}
        # Build the graph: f_i -> f_j if (above f_i f_j)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_below, f_above = parts[1], parts[2]
                floors.add(f_below)
                floors.add(f_above)
                below_to_above[f_below] = f_above

        # Find the bottom floor: a floor that is in the set of all floors
        # but is not the 'above' floor for any other floor.
        all_above_floors = set(below_to_above.values())
        bottom_floor = None
        for f in floors:
            if f not in all_above_floors:
                bottom_floor = f
                break # Assuming a single bottom floor in a linear chain

        self.floor_to_level = {}
        if bottom_floor:
            level = 1
            current_floor = bottom_floor
            # Traverse the chain upwards
            while current_floor in below_to_above:
                self.floor_to_level[current_floor] = level
                current_floor = below_to_above[current_floor]
                level += 1
            # Add the very top floor which is a value but not a key in below_to_above
            self.floor_to_level[current_floor] = level
        else:
             # Handle case with no 'above' facts or complex structure (e.g., single floor, disconnected)
             # For robustness, assign levels alphabetically if no clear bottom found.
             # This might not be accurate but provides a fallback.
             print("Warning: Could not determine linear floor ordering from 'above' facts. Assigning levels alphabetically.")
             sorted_floors = sorted(list(floors))
             self.floor_to_level = {f: i+1 for i, f in enumerate(sorted_floors)}

        # Store passenger destinations (extracted from initial state facts)
        self.destinations = {}
        self.all_passengers = set()
        # Assume task object provides initial state facts via 'initial_state_facts' attribute
        initial_facts = getattr(task, 'initial_state_facts', frozenset())
        if not initial_facts and hasattr(task, 'init'): # Try .init as fallback
             initial_facts = task.init

        for fact in initial_facts:
             parts = get_parts(fact)
             if parts[0] == 'destin':
                 passenger, floor = parts[1], parts[2]
                 self.destinations[passenger] = floor
                 self.all_passengers.add(passenger) # Ensure all passengers are known

        # Ensure all passengers from goals are known (they must have destinations defined in init)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'served':
                 self.all_passengers.add(parts[1])

        # Validate that destinations were found for all passengers identified from goals
        for passenger in self.all_passengers:
             if passenger not in self.destinations:
                  print(f"Warning: Destination not found for passenger '{passenger}'. Heuristic may be inaccurate.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Current Lift Location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # If lift location is not found, state is invalid or goal is reached without lift-at?
        # Assuming valid states have lift-at. If not found, heuristic is undefined.
        # For safety, return a large value or handle as error. Let's assume it's always present.
        current_lift_level = self.floor_to_level.get(current_lift_floor, 0) # Default to 0 if floor not mapped

        # 2. Identify Unserved Passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_passengers = self.all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state reached (all passengers served)

        # Pre-process state facts for quick lookup
        state_predicates = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate not in state_predicates:
                state_predicates[predicate] = []
            state_predicates[predicate].append(parts)

        # 3. Count Board Actions Needed & Identify Pickup Floors
        num_unboarded_unserved = 0
        pickup_levels = set()
        
        origin_facts_in_state = state_predicates.get('origin', [])
        boarded_passengers_in_state = {p for p, in state_predicates.get('boarded', [])}

        for passenger in unserved_passengers:
            # Check if passenger is at origin floor
            origin_floor = None
            for _, p, f in origin_facts_in_state:
                 if p == passenger:
                      origin_floor = f
                      break

            is_boarded = passenger in boarded_passengers_in_state

            if origin_floor and not is_boarded: # Passenger is at origin and not boarded
                 num_unboarded_unserved += 1
                 if origin_floor in self.floor_to_level:
                      pickup_levels.add(self.floor_to_level[origin_floor])
                 # else: origin floor not mapped (warning printed in init)


        # 4. Count Depart Actions Needed & Identify Dropoff Floors
        num_boarded_unserved = 0
        dropoff_levels = set()

        for passenger in unserved_passengers:
             if passenger in boarded_passengers_in_state: # Passenger is boarded
                  num_boarded_unserved += 1
                  # Get destination floor (from self.destinations, populated in init)
                  destin_floor = self.destinations.get(passenger)
                  if destin_floor and destin_floor in self.floor_to_level:
                       dropoff_levels.add(self.floor_to_level[destin_floor])
                  # else: destination floor not found or not mapped (warning printed in init)


        # 5. Identify Required Floors (Levels)
        all_required_levels = pickup_levels.union(dropoff_levels)

        # 6. Calculate Travel Cost
        travel_cost = 0
        # Filter out level 0 if it was used for unmapped floors
        valid_required_levels = {level for level in all_required_levels if level > 0}

        if valid_required_levels:
            min_required_level = min(valid_required_levels)
            max_required_level = max(valid_required_levels)

            # Travel cost estimate: abs(current - min) + abs(current - max)
            # This estimates the travel needed to reach one extreme of the required floors
            # and then traverse to the other extreme, covering all floors in between.
            travel_cost = abs(current_lift_level - min_required_level) + abs(current_lift_level - max_required_level)

        # 7. Sum Costs
        # Heuristic = (Board actions needed) + (Depart actions needed) + Travel cost
        heuristic_value = num_unboarded_unserved + num_boarded_unserved + travel_cost

        return heuristic_value
