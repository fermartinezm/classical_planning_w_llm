# Import necessary base class
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list for invalid format
         return []
    # Split by space, ignoring leading/trailing whitespace from the stripped string
    return fact[1:-1].strip().split()

# The heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions and adds an estimate of the
    minimum lift movement actions required to visit all necessary floors for
    pickups and dropoffs, considering these in two separate phases.

    # Assumptions
    - Passengers must be picked up at their origin floor and dropped off at their destination floor.
    - The lift can carry multiple passengers.
    - The floor structure is a linear sequence defined by 'above' predicates.
    - All passengers must be served to reach the goal.
    - Unserved passengers are either at their origin floor or boarded.

    # Heuristic Initialization
    - Builds an ordered list of floors and mappings between floor names and their indices based on 'above' facts.
    - Stores the destination floor for each passenger from the static facts.
    - Stores the initial origin floor for each passenger from the initial state (needed to identify unboarded passengers in subsequent states).
    - Identifies all relevant passengers from initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all passengers that are not yet served based on the goal conditions and the current state.
    2. For each unserved passenger, determine their current status: are they unboarded (still at their origin floor) or boarded?
    3. Count the total number of unserved passengers (`num_unserved`) and the number of unboarded passengers (`num_unboarded`).
    4. Estimate the minimum number of 'board' actions needed: This is equal to the number of unboarded passengers (`num_unboarded`).
    5. Estimate the minimum number of 'depart' actions needed: This is equal to the number of unserved passengers (`num_unserved`), as each unserved passenger will eventually need to depart.
    6. The base action cost is the sum of estimated board and depart actions: `num_unboarded + num_unserved`.
    7. Identify the set of floors the lift *must* visit for pickups (`pickup_floors`): These are the origin floors of all currently unboarded passengers.
    8. Identify the set of floors the lift *must* visit for dropoffs (`dropoff_floors`): These are the destination floors of all currently unserved passengers.
    9. Calculate the minimum movement cost for the lift to visit all `pickup_floors` starting from the current lift location. This calculation assumes the lift travels to the closest extreme floor in the `pickup_floors` range and then sweeps to the other extreme. This gives `move_cost1` and the estimated `end_idx1` (index of the floor where the pickup phase ends).
    10. Calculate the minimum movement cost for the lift to visit all `dropoff_floors` starting from `end_idx1`. This calculation uses the same logic as step 9. This gives `move_cost2`.
    11. The total heuristic value is the sum of the base action cost (`num_unboarded + num_unserved`) and the total estimated movement cost (`move_cost1 + move_cost2`).
    12. If the state is a goal state (all passengers served), the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and floor structure."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Build floor order and index mapping
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_map = {} # Map floor -> floor immediately above
        floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                if len(parts) == 3:
                    f_lower, f_higher = parts[1], parts[2]
                    above_map[f_lower] = f_higher
                    floors.add(f_lower)
                    floors.add(f_higher)

        if floors:
            # Find the lowest floor (a floor f such that no (above ?x f) exists)
            floors_that_are_below_something = set(above_map.values())

            lowest_floor = None
            # A floor is lowest if it's in the set of all floors but is never the 'higher' floor in an 'above' fact.
            potential_lowests = floors - floors_that_are_below_something

            if potential_lowests:
                 # Sorting provides a deterministic choice if multiple candidates exist (e.g., disconnected floors)
                 lowest_floor = sorted(list(potential_lowests))[0]
            elif len(floors) == 1:
                 lowest_floor = list(floors)[0]
            else:
                 # Fallback: If no clear lowest floor from 'above' structure, sort all floors alphabetically.
                 # This assumes a total order exists even if not fully specified by 'above'.
                 sorted_floors = sorted(list(floors))
                 if sorted_floors:
                     lowest_floor = sorted_floors[0]
                 else:
                     # No floors found at all
                     lowest_floor = None


            if lowest_floor:
                # Build the ordered list by following the 'above' chain
                current_floor = lowest_floor
                index = 0
                visited_floors_in_chain = set()
                while current_floor is not None and current_floor not in visited_floors_in_chain:
                    visited_floors_in_chain.add(current_floor)
                    self.floor_to_index[current_floor] = index
                    self.index_to_floor[index] = current_floor
                    index += 1
                    current_floor = above_map.get(current_floor)

                # If the chain didn't include all floors found initially, add the remaining ones.
                # This might indicate an invalid problem structure (disconnected floors),
                # but we assign them arbitrary indices to avoid errors.
                for floor in floors:
                    if floor not in self.floor_to_index:
                         self.floor_to_index[floor] = index
                         self.index_to_index[index] = floor # Fix: should be index_to_floor
                         self.index_to_floor[index] = floor
                         index += 1


        # Store goal locations for each passenger
        self.goal_locations = {}
        self.all_passengers = set()

        # Get all passengers mentioned in initial state (origin or boarded) or goals (served)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ['origin', 'boarded'] and len(parts) > 1:
                 self.all_passengers.add(parts[1])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served' and len(parts) > 1:
                 self.all_passengers.add(parts[1])

        # Find destinations for all relevant passengers from static facts
        for passenger in self.all_passengers:
             destin_fact = next((fact for fact in static_facts if get_parts(fact)[:2] == ['destin', passenger]), None)
             if destin_fact:
                 parts = get_parts(destin_fact)
                 if len(parts) == 3:
                     self.goal_locations[passenger] = parts[2]
                 else:
                     self.goal_locations[passenger] = None # Indicate unknown destination
             else:
                 self.goal_locations[passenger] = None # Indicate unknown destination

        # Store initial origin locations for passengers (needed to identify unboarded)
        self.initial_origins = {}
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'origin' and len(parts) == 3:
                 self.initial_origins[parts[1]] = parts[2]


    def calculate_movement_cost(self, start_idx, required_indices):
        """
        Calculate the minimum movement cost to visit a set of floor indices
        starting from a given index. Assumes optimal path visits closest end
        and sweeps to the other.

        Returns: (cost, end_idx)
        """
        if not required_indices:
            return 0, start_idx

        min_idx = min(required_indices)
        max_idx = max(required_indices)

        cost = 0
        end_idx = start_idx # Default end is start if no movement

        if start_idx < min_idx:
            # Must go up to at least min_idx, then sweep up to max_idx
            cost = (min_idx - start_idx) + (max_idx - min_idx)
            end_idx = max_idx
        elif start_idx > max_idx:
            # Must go down to at least max_idx, then sweep down to min_idx
            cost = (start_idx - max_idx) + (max_idx - min_idx)
            end_idx = min_idx
        else: # start_idx is within [min_idx, max_idx]
            # Must traverse the range [min_idx, max_idx]
            cost = (max_idx - min_idx)
            # Determine end_idx based on which end is further from the start
            dist_to_min = abs(start_idx - min_idx)
            dist_to_max = abs(start_idx - max_idx)
            if dist_to_min <= dist_to_max:
                # Closer to min, go down first, end at max
                end_idx = max_idx
            else:
                # Closer to max, go up first, end at min
                end_idx = min_idx

        return cost, end_idx


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift location
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_floor = parts[1]
                break

        if current_floor is None or current_floor not in self.floor_to_index:
             # This state is likely invalid or terminal (e.g., no lift-at fact or unknown floor)
             # Return infinity as it's likely unreachable or an error state
             return float('inf')

        current_idx = self.floor_to_index[current_floor]

        unserved_passengers = set()
        unboarded_passengers = set()
        boarded_passengers = set()

        # Identify passenger states
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == 'served' and len(get_parts(fact)) > 1}
        currently_boarded = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == 'boarded' and len(get_parts(fact)) > 1}
        currently_at_origin = {} # Map passenger -> floor if at origin
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'origin' and len(parts) == 3:
                 currently_at_origin[parts[1]] = parts[2]

        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                unserved_passengers.add(passenger)
                if passenger in currently_boarded:
                    boarded_passengers.add(passenger)
                elif passenger in currently_at_origin:
                    unboarded_passengers.add(passenger)
                # Note: Passengers not in served, boarded, or origin are not explicitly
                # handled here, assuming valid states maintain this property for unserved passengers.


        num_unserved = len(unserved_passengers)
        num_unboarded = len(unboarded_passengers)
        # num_boarded_not_served = len(boarded_passengers) # These are the unserved boarded passengers

        # Base action cost: 1 board per unboarded, 1 depart per unserved
        # Each unboarded passenger needs board (1) + depart (1) = 2 actions
        # Each boarded passenger needs depart (1) = 1 action
        # Total base actions = num_unboarded * 2 + num_boarded_not_served * 1
        # Since num_unserved = num_unboarded + num_boarded_not_served,
        # this simplifies to num_unboarded + num_unserved
        base_action_cost = num_unboarded + num_unserved

        # Identify required floors for pickup and dropoff
        # Pickup floors are origins of unboarded passengers
        pickup_floors = {currently_at_origin[p] for p in unboarded_passengers if p in currently_at_origin}
        # Dropoff floors are destinations of all unserved passengers
        dropoff_floors = {self.goal_locations[p] for p in unserved_passengers if p in self.goal_locations and self.goal_locations[p] is not None}

        # Convert floors to indices, filtering out any floors not in our map
        pickup_indices = {self.floor_to_index[f] for f in pickup_floors if f in self.floor_to_index}
        dropoff_indices = {self.floor_to_index[f] for f in dropoff_floors if f in self.floor_to_index}

        # Calculate movement cost Phase 1 (pickups)
        move_cost1, end_idx1 = self.calculate_movement_cost(current_idx, pickup_indices)

        # Calculate movement cost Phase 2 (dropoffs)
        move_cost2, _ = self.calculate_movement_cost(end_idx1, dropoff_indices)

        total_cost = base_action_cost + move_cost1 + move_cost2

        return total_cost
