import itertools
from fnmatch import fnmatch
# Assuming the planner infrastructure provides Heuristic and Task classes.
# If running standalone, these would need definitions or mocks.
# We expect the following structure based on the provided examples:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError
# class Node: # A node in the search tree
#     def __init__(self, state): self.state = state # state is a frozenset of fact strings
# class Task: # Represents the planning task
#     def __init__(self, name, facts, initial_state, goals, operators, static):
#         self.name = name
#         self.facts = facts # All possible facts (not used here)
#         self.initial_state = initial_state # frozenset of strings
#         self.goals = goals # frozenset of strings
#         self.operators = operators # set of Operator objects (not used here)
#         self.static = static # frozenset of strings for static facts

# Placeholder for the base class if it's not available in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder if the base class is not found
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components (predicate and arguments) of a PDDL fact string.
    Input: fact string like '(predicate arg1 arg2)'
    Output: list of strings ['predicate', 'arg1', 'arg2']
    Handles potential empty strings or malformed facts gracefully by returning an empty list.
    """
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Optionally log a warning for malformed facts
        # print(f"Warning: Malformed fact string encountered: {fact}")
        return []
    # Split the content within the parentheses
    return fact[1:-1].split()

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by transporting them from their origin floors to their destination floors using a single elevator.
    It calculates a cost based on the number of board/depart actions needed for unserved passengers
    and an estimate of the vertical distance the elevator must travel.
    This heuristic is designed for use with Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - The `(above f1 f2)` predicate means floor `f1` is strictly higher than floor `f2`.
    - The `above` relation defines a consistent partial order (likely a total order/chain) over the floors.
    - The cost of moving the elevator between adjacent floors (`up` or `down` actions) is 1.
    - All passengers specified in the goal `(served p)` have a corresponding static `(destin p f)` fact.
    - The elevator's position `(lift-at f)` is always uniquely defined in any valid state.
    - Object names (floors, passengers) do not contain spaces or parentheses.

    # Heuristic Initialization
    - Parses static facts to store passenger destinations `(destin p f)`.
    - Identifies all unique passenger and floor objects from the task definition (static facts, initial state, goals).
    - Computes the 'level' for each floor based on the `(above f1 f2)` relations. The level of a floor `f`
      is defined as the number of distinct floors strictly below it, determined via the transitive closure
      of the `above` relation. The lowest floor(s) in the hierarchy will have level 0. This precomputation
      allows for efficient distance calculation later.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current elevator location (`lift_f`), the set of passengers currently
        boarded (`boarded_passengers`), the set of passengers waiting at their origin (`waiting_passengers` mapped
        to their origin floor), and the set of passengers already served (`served_passengers`).
    2.  **Identify Unserved Passengers:** Determine the set of passengers who still need to reach their destination
        (`unserved_passengers = all_passengers - served_passengers`).
    3.  **Check Goal State:** If `unserved_passengers` is empty, the current state is a goal state, return 0.
    4.  **Calculate Base Action Cost:** Initialize `base_cost = 0`. Iterate through each `p` in `unserved_passengers`:
        - If `p` is waiting at their origin (`p` in `waiting_passengers`): Add 2 to `base_cost` (1 action to `board` + 1 action to `depart` later). Also, record their origin floor as a required `pickup_floor`.
        - If `p` is currently boarded (`p` in `boarded_passengers`): Add 1 to `base_cost` (1 action to `depart` later).
        - Record the destination floor `destin(p)` as a required `dropoff_floor` for all unserved passengers.
    5.  **Calculate Estimated Movement Cost:**
        - Determine the set of all floors the elevator needs to visit: `required_floors = pickup_floors | dropoff_floors`.
        - If `required_floors` is empty (this case should typically only occur if the goal is reached, handled in step 3), the `movement_cost` is 0.
        - Otherwise:
            - Consider all floors relevant to the remaining plan: `involved_floors = required_floors | {lift_f}`.
            - Retrieve the precomputed levels for all `involved_floors`. Handle any unexpected missing floor levels defensively (e.g., assume level 0).
            - Find the minimum level (`min_level`) and maximum level (`max_level`) among these involved floors.
            - Estimate the movement cost as the total vertical span: `movement_cost = max_level - min_level`. This approximates the minimum vertical distance the elevator needs to cover to potentially service all required floors.
    6.  **Calculate Total Heuristic Value:** The heuristic estimate `h` is the sum of the base action cost and the estimated movement cost: `h = base_cost + movement_cost`.
    7.  **Ensure Progress:** If the calculated `h` is 0 but the state is not a goal state (i.e., `unserved_passengers` is not empty), return 1. This ensures that the heuristic value is always positive for non-goal states, which is crucial for greedy search algorithms to guarantee progress.
    """

    def __init__(self, task):
        """Initializes the heuristic by processing static information from the task."""
        # super().__init__(task) # Call base class init if it exists and requires it
        self.goals = task.goals
        static_facts = task.static

        # Store passenger destinations and identify all passengers and floors
        self.destinations = {}
        self.passengers = set()
        all_floors = set()
        above_relations = set() # Store pairs (f_above, f_below)

        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'destin':
                if len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.destinations[passenger] = floor
                    self.passengers.add(passenger)
                    all_floors.add(floor)
                # else: print(f"Warning: Malformed 'destin' fact: {fact}") # Optional logging
            elif predicate == 'above':
                if len(parts) == 3:
                    f_above, f_below = parts[1], parts[2]
                    above_relations.add((f_above, f_below))
                    all_floors.add(f_above)
                    all_floors.add(f_below)
                # else: print(f"Warning: Malformed 'above' fact: {fact}") # Optional logging

        # Add passengers mentioned in goals (might not be in 'destin' if goal is trivial)
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts and parts[0] == 'served' and len(parts) == 2:
                 self.passengers.add(parts[1])

        # Add floors/passengers mentioned in initial state (might not be in static facts)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'lift-at' and len(parts) == 2:
                 all_floors.add(parts[1])
             elif predicate == 'origin' and len(parts) == 3:
                 all_floors.add(parts[2])
                 self.passengers.add(parts[1]) # Also capture passengers from origin

        self.floors = frozenset(all_floors)

        # --- Compute floor levels using transitive closure ---
        self.floor_levels = {}
        # floor_to_below[f] = set of floors strictly below f
        floor_to_below = {f: set() for f in self.floors}

        # Initialize with direct relations from static facts
        for f_above, f_below in above_relations:
            # Ensure both floors are recognized before adding relation
            if f_above in self.floors and f_below in self.floors:
                 floor_to_below[f_above].add(f_below)

        # Compute transitive closure iteratively
        while True:
            added_new = False
            # Iterate over a copy of floors to handle potential modifications safely
            for f1 in list(self.floors):
                # Check if f1 is in the dictionary (it should be)
                if f1 not in floor_to_below: continue

                # Floors directly below f1 (make a copy to iterate)
                direct_below = list(floor_to_below[f1])

                # Collect all floors below the floors that are directly below f1
                newly_discovered_below = set()
                for f2 in direct_below:
                    if f2 in floor_to_below: # Check if f2 exists in our map
                        newly_discovered_below.update(floor_to_below[f2])

                # Add these newly found floors to f1's below set, avoiding self-references
                original_size = len(floor_to_below[f1])
                floor_to_below[f1].update(newly_discovered_below - {f1})

                if len(floor_to_below[f1]) > original_size:
                    added_new = True

            if not added_new:
                break # Closure computation finished when no new relations are found

        # Assign level = number of distinct floors below
        for f in self.floors:
            # Use .get() for safety, though f should be present
            self.floor_levels[f] = len(floor_to_below.get(f, set()))

        # Ensure all identified floors have a level, default to 0 if isolated/not in 'above'
        for f in self.floors:
            if f not in self.floor_levels:
                 self.floor_levels[f] = 0


    def __call__(self, node):
        """
        Calculates the heuristic value for the state represented by the given search node.
        Input: node - A search node containing the state (node.state).
        Output: An integer estimate of the cost to reach the goal.
        """
        state = node.state # state is expected to be a frozenset of fact strings

        # --- State Parsing ---
        lift_f = None
        boarded_passengers = set()
        # Store all current origin facts, filter later based on served status
        current_origins = {} # map passenger -> origin_floor
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            # Extract relevant information based on predicate type
            if predicate == 'lift-at' and len(parts) == 2:
                lift_f = parts[1]
            elif predicate == 'boarded' and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == 'origin' and len(parts) == 3:
                current_origins[parts[1]] = parts[2]
            elif predicate == 'served' and len(parts) == 2:
                served_passengers.add(parts[1])

        # --- Sanity Checks ---
        if lift_f is None:
            # This indicates an invalid state where the lift location is unknown.
            # Return infinity as this state should not be expanded.
            # print(f"CRITICAL WARNING: Lift location ('lift-at') not found in state. State: {state}. Returning infinity.")
            return float('inf')

        # --- Heuristic Calculation ---
        # Identify passengers who still need service
        unserved_passengers = self.passengers - served_passengers

        # If goal is reached (no unserved passengers)
        if not unserved_passengers:
            return 0

        base_cost = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Calculate base cost and required floors for unserved passengers
        for p in unserved_passengers:
            # Check if passenger 'p' has a defined destination (should always be true based on assumptions)
            if p not in self.destinations:
                 # print(f"Warning: Unserved passenger '{p}' has no destination defined. Skipping.")
                 continue # Cannot plan for this passenger if destination is unknown

            destination_floor = self.destinations[p]
            dropoff_floors.add(destination_floor) # Destination is always a required dropoff

            if p in current_origins:
                # Passenger is waiting at their origin floor
                base_cost += 2 # Estimate 1 board + 1 depart action
                origin_floor = current_origins[p]
                pickup_floors.add(origin_floor) # Origin is a required pickup
            elif p in boarded_passengers:
                # Passenger is already inside the lift
                base_cost += 1 # Estimate 1 depart action needed
            # else:
                # This case (unserved, not at origin, not boarded) should ideally not occur in valid states.
                # If it does, we might still count the depart cost as they need to reach the destination.
                # print(f"Warning: Unserved passenger '{p}' is neither at origin nor boarded. State: {state}")
                # base_cost += 1 # Tentatively add depart cost

        # Calculate estimated movement cost based on floor levels
        required_floors = pickup_floors | dropoff_floors
        movement_cost = 0

        if required_floors:
            # Include the current lift floor in the set of relevant locations
            involved_floors = required_floors | {lift_f}
            involved_levels = []

            # Get levels for all involved floors
            for f in involved_floors:
                if f in self.floor_levels:
                    involved_levels.append(self.floor_levels[f])
                else:
                    # Handle case where a floor appears unexpectedly (not in init/static)
                    # print(f"Warning: Floor '{f}' involved in plan but has no computed level. Assuming level 0.")
                    involved_levels.append(0) # Default to level 0 for unknown floors

            # Calculate the vertical span needed
            if involved_levels: # Should always be true if lift_f exists and has a level
                min_level = min(involved_levels)
                max_level = max(involved_levels)
                movement_cost = max_level - min_level
            # else: movement_cost remains 0 (should not happen)

        # Total heuristic value is sum of base actions and movement estimate
        h = base_cost + movement_cost

        # Safeguard: Ensure heuristic is non-zero for non-goal states
        # This prevents greedy search from getting stuck on non-goal states with h=0.
        if h == 0 and unserved_passengers:
            # print(f"Warning: Heuristic calculated 0 for non-goal state. Returning 1. State: {state}")
            return 1

        return h
