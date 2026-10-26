# Need to import the base class
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided externally, you might need a minimal definition like:
class Heuristic:
    """Base class for heuristics."""
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError("Heuristic must implement __call__")


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # print(f"Warning: Invalid fact format: {fact}") # Optional: for debugging
        return []
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It sums the estimated cost for each unserved passenger,
    considering the actions needed (board, depart) and the lift movement
    required to complete their journey.

    # Assumptions
    - Standard Miconic domain rules apply (lift moves between floors, picks up, drops off).
    - Lift has unlimited capacity.
    - Each action (board, depart, move between adjacent floors) has a cost of 1.
    - Floor levels can be determined from the static `above` predicates, forming a linear structure.
    - Passenger origins and destinations are static and provided in the initial state.

    # Heuristic Initialization
    - Extract floor ordering and map floor names to numerical levels based on `above` facts.
    - Extract passenger origins and destinations from the initial state and identify all passengers.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the lift's current floor.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all passengers identified during initialization.
    4. For each passenger:
       a. Check if the passenger is already `served` in the current state. If yes, this passenger contributes 0 to the heuristic.
       b. If the passenger is not served:
          i. Check if the passenger is `boarded` in the current state.
          ii. If the passenger is `boarded`:
              - This passenger needs to be departed at their destination.
              - Add 1 (for the `depart` action) to the total cost.
              - Add the vertical distance between the lift's current floor and the passenger's destination floor to the total cost.
          iii. If the passenger is not `boarded` (they must be waiting at their origin floor, assuming a valid state):
              - This passenger needs to be boarded at their origin and then departed at their destination.
              - Add 1 (for the `board` action) to the total cost.
              - Add 1 (for the `depart` action) to the total cost.
              - Add the vertical distance between the lift's current floor and the passenger's origin floor to the total cost.
              - Add the vertical distance between the passenger's origin floor and their destination floor to the total cost.
    5. The total heuristic value is the sum accumulated in step 4.

    This heuristic is an overestimate because it sums individual passenger costs and movement, not accounting for shared lift travel or simultaneous pickups/dropoffs. However, it captures the essential work remaining and provides a goal-directed estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        # self.goals = task.goals # Goal conditions (implicitly all served)
        self.initial_state = task.initial_state # Needed to get static origin/destin
        static_facts = task.static # Static facts like 'above'

        # 1. Extract floor ordering and map floor names to numerical levels
        above_map = {} # floor_lower -> floor_higher
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        self.floor_levels = {}
        if not all_floors:
             # Handle case with no floors or a single floor
             lift_fact_parts = next((get_parts(f) for f in self.initial_state if get_parts(f) and get_parts(f)[0] == 'lift-at'), None)
             if lift_fact_parts and len(lift_fact_parts) > 1:
                current_floor = lift_fact_parts[1]
                self.floor_levels[current_floor] = 1
                all_floors.add(current_floor)
             if not all_floors:
                 # print("Warning: No floors found in problem definition.") # Optional: for debugging
                 pass # Cannot proceed without floors, heuristic will handle missing levels later

        # Find the lowest floor (a floor that is in all_floors but is not a value in above_map)
        higher_floors = set(above_map.values())
        lowest_floor_candidates = list(all_floors - higher_floors)

        lowest_floor = None
        if len(lowest_floor_candidates) == 1:
             lowest_floor = lowest_floor_candidates[0]
        elif len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
        else:
             # print("Warning: Could not uniquely determine the lowest floor. Assuming the first floor found is the lowest.") # Optional: for debugging
             # Fallback: Pick an arbitrary floor if available
             if all_floors:
                 lowest_floor = sorted(list(all_floors))[0] # Pick alphabetically first as a stable fallback
             # else: print("Error: No floors available to determine levels.") # Optional: for debugging


        if lowest_floor and lowest_floor in all_floors: # Ensure lowest_floor is actually one of the floors
            # Build floor_levels using BFS starting from the lowest floor
            q = [(lowest_floor, 1)]
            visited = {lowest_floor}

            while q:
                current_f, level = q.pop(0)
                self.floor_levels[current_f] = level

                # Find the floor immediately above current_f
                f_above = above_map.get(current_f)
                if f_above and f_above not in visited:
                     visited.add(f_above)
                     q.append((f_above, level + 1))

            # Ensure all floors found are in floor_levels (handles disconnected components or errors)
            if len(self.floor_levels) != len(all_floors):
                 # print(f"Warning: Mismatch in floor count. Found {len(all_floors)} floors, leveled {len(self.floor_levels)}.") # Optional: for debugging
                 # Add any missing floors with a default level (e.g., 1)
                 for f in all_floors:
                     if f not in self.floor_levels:
                         self.floor_levels[f] = 1
                         # print(f"Warning: Assigned default level 1 to floor {f}") # Optional: for debugging
        # else: print("Error: Could not determine floor levels.") # Optional: for debugging


        # 2. Extract passenger origins and destinations
        self.origin_floors = {}
        self.destin_floors = {}
        self.all_passengers = set()

        # Origin and destin facts are typically in the initial state
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and len(parts) == 3:
                predicate, passenger, floor = parts
                if predicate == 'origin':
                    self.origin_floors[passenger] = floor
                    self.all_passengers.add(passenger)
                elif predicate == 'destin':
                    self.destin_floors[passenger] = floor
                    self.all_passengers.add(passenger)

        # Ensure all passengers have both origin and destination (problem assumption)
        # This check is mostly for debugging problem files, heuristic can still run
        # even if info is missing for some passengers, they just won't contribute
        # correctly to the heuristic.
        # for p in list(self.all_passengers): # Iterate over a copy if modifying set
        #     if p not in self.origin_floors or p not in self.destin_floors:
        #          print(f"Warning: Passenger {p} is missing origin or destination. Removing from consideration.") # Optional: for debugging
        #          self.all_passengers.discard(p)


    def distance(self, floor1, floor2):
        """Calculate the vertical distance between two floors."""
        # Use a default distance if floor levels weren't fully determined or floor is unknown
        if floor1 not in self.floor_levels or floor2 not in self.floor_levels:
             # print(f"Warning: Cannot calculate distance for floors {floor1}, {floor2}. Using default 0.") # Optional: for debugging
             return 0 # Or float('inf') if this indicates an impossible state fragment? 0 is safer for heuristic.
        return abs(self.floor_levels[floor1] - self.floor_levels[floor2])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached (all passengers served)
        all_served = True
        for passenger in self.all_passengers:
            if f'(served {passenger})' not in state:
                all_served = False
                break
        if all_served:
             return 0

        # Find the lift's current floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid state, but handle defensively
             # print("Error: Lift location not found in state.") # Optional: for debugging
             return float('inf') # Indicates an invalid state

        total_cost = 0

        # Iterate through all passengers and sum their estimated costs
        for passenger in self.all_passengers:
            is_served = f'(served {passenger})' in state

            if not is_served:
                is_boarded = f'(boarded {passenger})' in state
                origin_floor = self.origin_floors.get(passenger)
                destin_floor = self.destin_floors.get(passenger)

                if not origin_floor or not destin_floor:
                     # Cannot calculate cost for this passenger if info is missing
                     # print(f"Warning: Skipping passenger {passenger} due to missing origin/destin info.") # Optional: for debugging
                     continue # Skip this passenger

                if is_boarded:
                    # Passenger is in the lift, needs to go to destination and depart
                    total_cost += 1 # Cost of depart action
                    total_cost += self.distance(current_lift_floor, destin_floor) # Movement to destination
                else: # Passenger is not served and not boarded, must be waiting at origin
                    # Verify they are actually waiting at their origin in this state
                    is_waiting_at_origin = f'(origin {passenger} {origin_floor})' in state
                    if is_waiting_at_origin:
                        # Passenger is waiting at origin, needs board, move to destin, depart
                        total_cost += 1 # Cost of board action
                        total_cost += 1 # Cost of depart action
                        total_cost += self.distance(current_lift_floor, origin_floor) # Movement to origin
                        total_cost += self.distance(origin_floor, destin_floor) # Movement from origin to destination
                    # else: # Passenger is unserved, not boarded, and not at origin - invalid state?
                    #     print(f"Warning: Passenger {passenger} unserved, not boarded, not at origin {origin_floor}.") # Optional: for debugging
                    #     total_cost += float('inf') # Indicate potentially invalid state


        return total_cost
