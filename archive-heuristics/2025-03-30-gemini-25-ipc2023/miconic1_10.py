import math
from fnmatch import fnmatch
# Assuming Heuristic base class is available in the following path.
# If not, you might need to adjust the import path based on your project structure.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: '(pred obj1 obj2)' -> ['pred', 'obj1', 'obj2']
    Returns an empty list if the format is invalid.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by transporting them from their origin floors to their destination floors using
    a single elevator (lift). It calculates the sum of mandatory passenger actions
    (boarding and departing) and adds an estimate of the lift's movement cost.
    The movement cost estimation includes the distance to the nearest passenger
    requiring service (either pickup or drop-off) and the subsequent travel distances
    needed for passengers who are waiting to be picked up. This heuristic is designed
    for Greedy Best-First Search and prioritizes informativeness over admissibility.

    # Assumptions
    - The predicate `(above f1 f2)` signifies that floor `f1` is directly above floor `f2`.
    - The floors form a single, linear vertical arrangement.
    - The `above` predicates provided in the static facts define the complete adjacency
      relationship between all floors in the building.
    - The cost of moving the lift between adjacent floors (using `up` or `down` actions) is 1.
    - The cost of passengers boarding (`board`) or departing (`depart`) the lift is 1 each.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the static information provided in `task.static`.
    - It extracts and stores the destination floor for each passenger from `(destin p f)` facts.
    - It builds a representation of the floor layout by processing `(above f_higher f_lower)` facts.
      - It identifies the bottom-most floor (the one with no floor below it).
      - It assigns an integer level to each floor, starting from 0 for the bottom floor and
        incrementing upwards.
      - It precomputes the distance `dist(f1, f2)` between all pairs of floors based on the
        absolute difference of their assigned levels. These distances are stored in `self.dist`.
    - It stores the goal conditions (`task.goals`) for reference, although the primary goal check
      relies on tracking served passengers.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse State:** Extract the current state information from `node.state`:
        - Find the current floor of the lift using `(lift-at ?f)`.
        - Identify the set of passengers currently inside the lift using `(boarded ?p)`.
        - Identify passengers waiting at their origin floors using `(origin ?p ?f)`.
        - Identify passengers who have already reached their destination using `(served ?p)`.
    2.  **Identify Passengers:** Determine the set of all passengers involved in the problem
        using the destinations extracted during initialization (`self.destinations`).
    3.  **Identify Unserved Passengers:** Find the set of passengers who are not yet `served`.
    4.  **Identify Waiting Passengers:** Find the subset of unserved passengers who are currently
        at their origin floor (i.e., have an `origin` fact in the state) and are not `boarded`.
    5.  **Identify Boarded Passengers:** Use the set identified directly from `(boarded ?p)` facts.
    6.  **Goal Check:** If the set of unserved passengers is empty, the goal state has been
        reached, and the heuristic value is 0.
    7.  **Calculate Passenger Action Cost (`h_actions`):**
        - Add 1 action cost for the `board` action required for each waiting passenger.
        - Add 1 action cost for the `depart` action required for every unserved passenger
          (as they all eventually need to depart at their destination).
        - `h_actions = count(waiting_passengers) + count(unserved_passengers)`.
    8.  **Calculate Lift Movement Cost (`h_move`):**
        - Get the lift's current floor, `lf`.
        - Determine the origin floors (`O`) for all waiting passengers.
        - Determine the destination floors (`D_boarded`) for all currently boarded passengers.
        - Calculate the distances from `lf` to each floor in `O` (`pickup_dists`).
        - Calculate the distances from `lf` to each floor in `D_boarded` (`dropoff_dists`).
        - Find the minimum *valid* distance among all pickup and drop-off distances. A distance
          is valid if it's not infinity (i.e., the target floor is reachable). This minimum
          distance (`dist_to_nearest_task`) estimates the cost to reach the first point of service.
          If no immediate service is needed or possible, this is 0 or infinity, respectively.
        - Calculate the sum of travel distances required for each waiting passenger *after* they
          theoretically board the lift: `travel_after_boarding = sum(dist(origin(p), destin(p)))`
          for all waiting passengers `p`. If any of these origin-destination paths are impossible
          (distance is infinity), the state is considered a dead end.
        - The total estimated movement cost is `h_move = dist_to_nearest_task + travel_after_boarding`.
    9.  **Final Heuristic Value:** The heuristic estimate is the sum of the action cost and the
        movement cost: `h = h_actions + h_move`. If any calculation indicated that the goal is
        unreachable from the current state (e.g., due to infinite distances), return infinity.
        Ensure the returned value is non-negative.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract passenger destinations
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Ensure parts is not empty and the predicate is 'destin'
            if parts and parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # 2. Determine floor levels and precompute distances
        self.floor_levels = {}
        self.dist = {}
        adj_up = {}   # Map: floor -> floor directly above
        adj_down = {} # Map: floor -> floor directly below
        floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                f_higher, f_lower = parts[1], parts[2]
                floors.add(f_higher)
                floors.add(f_lower)
                # Record adjacency based on 'above f_higher f_lower'
                adj_up[f_lower] = f_higher
                adj_down[f_higher] = f_lower

        if not floors:
             # No floors defined in the problem
             print("Warning: MiconicHeuristic init - No floors found based on 'above' predicates.")
             return # Distances cannot be computed

        # Find the bottom floor (a floor that does not appear as f_higher in any 'above' fact)
        bottom_floor = None
        for f in floors:
            if f not in adj_down: # Check if f has no floor below it
                bottom_floor = f
                break

        # Handle the edge case of a single floor explicitly
        if bottom_floor is None and len(floors) == 1:
            bottom_floor = next(iter(floors))
        elif bottom_floor is None and floors:
             print(f"Warning: MiconicHeuristic init - Could not determine a unique bottom floor among {floors}. Check 'above' predicates for linearity.")
             # Cannot proceed reliably without a bottom floor to start leveling
             return

        # Assign levels starting from the identified bottom floor
        if bottom_floor is not None:
            curr_f = bottom_floor
            level = 0
            visited_floors = set() # To detect cycles or gaps
            while curr_f is not None and curr_f not in visited_floors:
                self.floor_levels[curr_f] = level
                visited_floors.add(curr_f)
                # Move to the floor directly above, if one exists
                next_f = adj_up.get(curr_f)
                curr_f = next_f
                level += 1

            # Verify that all known floors were assigned a level
            if len(self.floor_levels) != len(floors):
                 unleveled = floors - set(self.floor_levels.keys())
                 print(f"Warning: MiconicHeuristic init - Floor level assignment incomplete. Leveled {len(self.floor_levels)}/{len(floors)}. Unleveled: {unleveled}. Check 'above' predicates.")

        # Precompute distances between all pairs of leveled floors
        floor_list = list(self.floor_levels.keys())
        for f1 in floor_list:
            # Distance from a floor to itself is 0
            self.dist[(f1, f1)] = 0
            for f2 in floor_list:
                # Compute distance only once per pair (f1, f2)
                if (f1, f2) not in self.dist:
                    level1 = self.floor_levels.get(f1) # Level should exist
                    level2 = self.floor_levels.get(f2) # Level should exist
                    # Calculate distance if both levels were found
                    if level1 is not None and level2 is not None:
                        distance = abs(level1 - level2)
                        self.dist[(f1, f2)] = distance
                        self.dist[(f2, f1)] = distance # Ensure symmetry
                    else:
                        # Assign infinity if a level was missing (indicates init error)
                        self.dist[(f1, f2)] = float('inf')
                        self.dist[(f2, f1)] = float('inf')


    def get_distance(self, f1, f2):
        """
        Returns the precomputed distance between two floors.
        Returns float('inf') if floors are None, not found, or distance wasn't computed.
        """
        if f1 is None or f2 is None:
            return float('inf')
        # Return precomputed distance, default to infinity if pair is unknown
        return self.dist.get((f1, f2), float('inf'))


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # 1. Parse current state information
        lift_at = None
        boarded_passengers = set()
        passenger_origins = {} # Map: passenger -> origin_floor (from state)
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            pred = parts[0]
            if pred == 'lift-at':
                lift_at = parts[1]
            elif pred == 'boarded':
                boarded_passengers.add(parts[1])
            elif pred == 'origin':
                passenger_origins[parts[1]] = parts[2]
            elif pred == 'served':
                served_passengers.add(parts[1])

        # Essential check: lift location must be known
        if lift_at is None:
             # If there are no floors and no passengers, goal might be met
             if not self.floor_levels and not self.destinations:
                 return 0
             else:
                 # Cannot calculate heuristic without knowing lift location
                 print("Error: MiconicHeuristic call - lift-at predicate missing in state.")
                 return float('inf') # Indicate state is invalid or unreachable

        # 2. Identify passenger sets
        all_passengers = set(self.destinations.keys())
        unserved_passengers = all_passengers - served_passengers

        # 3. Goal check: If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # Determine waiting passengers: unserved, at their origin, and not boarded
        waiting_passengers = set()
        for p, origin_f in passenger_origins.items():
            # Check if passenger 'p' is unserved and not currently boarded
            if p in unserved_passengers and p not in boarded_passengers:
                 waiting_passengers.add(p)

        # 4. Calculate base action cost (board + depart actions)
        h_actions = len(waiting_passengers)      # One 'board' per waiting passenger
        h_actions += len(unserved_passengers)   # One 'depart' per unserved passenger

        # 5. Calculate estimated lift movement cost
        dist_to_nearest_task = float('inf')
        tasks_exist = False
        valid_initial_dists = [] # Store reachable initial distances

        # Calculate distances for picking up waiting passengers
        for p in waiting_passengers:
            origin_floor = passenger_origins.get(p) # Origin must exist for waiting pass.
            if origin_floor:
                dist = self.get_distance(lift_at, origin_floor)
                if dist != float('inf'):
                    valid_initial_dists.append(dist)
                    tasks_exist = True
                else:
                    # If lift cannot reach a waiting passenger's origin, goal is unreachable
                    return float('inf')

        # Calculate distances for dropping off boarded passengers
        for p in boarded_passengers:
            # Ensure the boarded passenger still needs to be served
            if p in unserved_passengers:
                dest_floor = self.destinations.get(p) # Destination must exist
                if dest_floor:
                    dist = self.get_distance(lift_at, dest_floor)
                    if dist != float('inf'):
                        valid_initial_dists.append(dist)
                        tasks_exist = True
                    else:
                        # If lift cannot reach a boarded passenger's destination, goal is unreachable
                        return float('inf')

        # Determine the cost to reach the nearest task requiring lift movement
        if tasks_exist and valid_initial_dists:
             # Minimum distance to initiate either a pickup or a drop-off
             dist_to_nearest_task = min(valid_initial_dists)
        elif tasks_exist and not valid_initial_dists:
             # This case should be caught by the infinity checks above, but as safety:
             return float('inf') # Tasks exist but none are reachable
        else:
             # No immediate tasks requiring lift movement (no waiting, no boarded)
             # If unserved passengers still exist, this state might be unusual.
             # Assume 0 cost for the initial move if no tasks are pending right now.
             dist_to_nearest_task = 0


        # Calculate the sum of travel distances for waiting passengers (origin to destination)
        travel_after_boarding = 0
        for p in waiting_passengers:
            origin_floor = passenger_origins.get(p)
            dest_floor = self.destinations.get(p)
            # Both origin and destination must be known for waiting passengers
            if origin_floor and dest_floor:
                dist = self.get_distance(origin_floor, dest_floor)
                if dist == float('inf'):
                    # If any passenger cannot travel from their origin to destination, goal is unreachable
                    return float('inf')
                travel_after_boarding += dist
            else:
                 # Data inconsistency: waiting passenger missing origin/destination info
                 print(f"Error: MiconicHeuristic call - Missing origin/destination for waiting passenger {p}.")
                 return float('inf')


        # Combine movement costs, ensuring we don't use infinity from initial step if it was reset to 0
        if dist_to_nearest_task == float('inf'):
             # This check might be redundant due to earlier returns, but ensures safety
             return float('inf')

        h_move = dist_to_nearest_task + travel_after_boarding

        # 6. Final heuristic value: sum of actions and movement
        h_total = h_actions + h_move

        # Ensure heuristic value is non-negative
        return max(0, h_total)

