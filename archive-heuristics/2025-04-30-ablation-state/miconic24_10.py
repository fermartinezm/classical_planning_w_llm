from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the required number of actions (board, depart, and lift movements)
    to transport all unserved passengers to their destinations. It sums the number of
    necessary board and depart actions with an estimate of the lift movement cost.

    # Assumptions
    - Floors are linearly ordered, defined by the `above` predicate.
    - The lift can carry multiple passengers simultaneously.
    - The cost of moving the lift one floor up or down is 1.
    - The cost of boarding or departing a passenger is 1.
    - The problem instance defines a linear sequence of floors using `above` predicates,
      and all relevant floors (origins, destinations, initial lift location) are part
      of this sequence.

    # Heuristic Initialization
    The heuristic pre-processes the static facts from the task:
    - It identifies the destination floor for each passenger from `destin` facts.
    - It determines the linear order of floors based on `above` facts and creates
      a mapping from floor names to integer indices (starting from 1 for the lowest floor)
      and vice-versa.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift from the `(lift-at ?f)` fact.
    2. Identify all passengers who have not yet been served (`(served ?p)` is false).
    3. If no passengers are unserved, the heuristic is 0 (goal state).
    4. Separate unserved passengers into two groups: those waiting at their origin floor
       (`(origin ?p ?f)`) and those who have boarded the lift (`(boarded ?p)`).
       (Note: Origin facts are dynamic and must be read from the state).
    5. Determine the set of floors the lift *must* visit to pick up or drop off unserved passengers:
       - The origin floor for every unserved passenger waiting at their origin.
       - The destination floor for every unserved passenger who has boarded.
    6. Calculate the Action Cost:
       - Each unserved passenger needs a `depart` action at their destination.
       - Each unserved passenger waiting at their origin needs a `board` action at their origin.
       - Total Action Cost = (Number of unserved passengers) + (Number of unserved passengers at origin).
    7. Calculate the Movement Cost:
       - If there are no required floors to visit (which should not happen in a valid non-goal state), the movement cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors identified in step 5.
       - The estimated movement cost is the distance from the current lift floor to the
         closest required floor (either the min or max required floor index), plus the
         total span of the required floors (max index - min index).
         Movement Cost = `min(abs(current_lift_index - min_req_idx), abs(current_lift_index - max_req_idx)) + (max_req_idx - min_req_idx)`.
    8. The total heuristic value is the sum of the Action Cost and the Movement Cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.destinations = {}
        above_map = {}  # f_lower -> f_higher
        all_floors_from_above = set()
        higher_floors_from_above = set()
        all_floors_from_destin = set()

        # Parse static facts to get destinations and floor relationships
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                p, dest_f = parts[1], parts[2]
                self.destinations[p] = dest_f
                all_floors_from_destin.add(dest_f)
            elif parts[0] == 'above':
                f_lower, f_higher = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors_from_above.add(f_lower)
                all_floors_from_above.add(f_higher)
                higher_floors_from_above.add(f_higher)

        all_floors = all_floors_from_above | all_floors_from_destin

        # Find the lowest floor (the one not appearing as f_higher in 'above' facts)
        # If there are 'above' facts, the lowest is unique.
        # If no 'above' facts, floors might be unordered or just one.
        lowest_floor = None
        if all_floors_from_above:
             # Standard case: find the unique lowest floor from the 'above' chain
             # Assumes a single connected component for floors via 'above'
             lowest_floor = (all_floors_from_above - higher_floors_from_above).pop()
        elif all_floors:
             # No 'above' facts, but floors exist (e.g., from destinations).
             # If there's only one floor total, that's the lowest.
             if len(all_floors) == 1:
                 lowest_floor = all_floors.pop()
             else:
                 # Multiple floors but no 'above' chain. This heuristic based on order is invalid.
                 # Set num_floors to 0 to indicate this.
                 self.num_floors = 0
                 self.floor_name_to_index = {}
                 self.index_to_floor_name = {}
                 return # Cannot build floor map based on order

        # Build floor index maps starting from the lowest floor
        self.floor_name_to_index = {}
        self.index_to_floor_name = {}
        if lowest_floor: # Proceed only if a lowest floor was found
            current_floor = lowest_floor
            index = 1
            # Traverse the 'above' chain
            while current_floor is not None:
                self.floor_name_to_index[current_floor] = index
                self.index_to_floor_name[index] = current_floor
                current_floor = above_map.get(current_floor)
                index += 1
            self.num_floors = index - 1 # Number of floors in the chain

            # Check if all floors mentioned in destinations are in the built map.
            # If not, the problem definition is inconsistent with the assumed structure.
            # For this heuristic, we assume consistency.
            # If a floor is missing, accessing floor_name_to_index will raise KeyError.
            # This is acceptable for a domain-dependent heuristic assuming valid domain instances.

        else:
             # No floors found at all (empty problem or only passengers with no floor facts)
             self.num_floors = 0 # No floors found

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        # If no floors were initialized, the problem is likely malformed or empty.
        # Heuristic cannot be computed meaningfully based on floor movement.
        # Return a value based on passengers if they exist, otherwise 0.
        if self.num_floors == 0:
             all_passengers = set(self.destinations.keys())
             if not all_passengers:
                 return 0 # Empty problem, goal reached

             # Problem has passengers but no floor structure suitable for movement heuristic.
             # Count minimal actions (board + depart) for unserved passengers.
             served_passengers = {p for p in all_passengers if f'(served {p})' in node.state}
             unserved_passengers = all_passengers - served_passengers

             # If all passengers are served, the goal is reached
             if not unserved_passengers:
                 return 0

             unboarded_unserved_count = 0
             # Need to iterate state to find origin facts for unserved passengers
             for fact in node.state:
                 parts = get_parts(fact)
                 if parts[0] == 'origin' and parts[1] in unserved_passengers:
                     unboarded_unserved_count += 1

             # Each unserved needs depart. Each unboarded unserved needs board.
             return len(unserved_passengers) + unboarded_unserved_count # Minimal actions if no movement possible


        state = node.state

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # Assuming current_lift_floor is always found in a valid state and is a known floor
        # If not found or not in floor_name_to_index, the state is likely invalid.
        # Returning a large value might be appropriate in a robust planner.
        # For this exercise, assume valid states reachable from initial state.
        current_lift_index = self.floor_name_to_index[current_lift_floor]

        # Identify unserved, unboarded, boarded passengers
        all_passengers = set(self.destinations.keys())
        served_passengers = {p for p in all_passengers if f'(served {p})' in state}
        unserved_passengers = all_passengers - served_passengers

        # If all passengers are served, the goal is reached
        if not unserved_passengers:
            return 0

        unboarded_unserved = set()
        boarded_unserved = set()
        origins = {} # Origins are dynamic facts

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'origin':
                p, f = parts[1], parts[2]
                origins[p] = f # Store origin for lookup
                if p in unserved_passengers:
                     unboarded_unserved.add(p)
            elif parts[0] == 'boarded':
                p = parts[1]
                if p in unserved_passengers:
                    boarded_unserved.add(p)

        # Identify required floors (origins for unboarded, destins for boarded)
        # In a valid state, an unserved passenger is either at origin or boarded.
        # So origins[p] and self.destinations[p] should exist for relevant p.
        origins_needed = {origins[p] for p in unboarded_unserved}
        destins_needed = {self.destinations[p] for p in boarded_unserved}

        required_floors = origins_needed | destins_needed

        # Calculate Action Cost
        # Each unserved passenger needs a depart action.
        # Each unboarded unserved passenger needs a board action.
        action_cost = len(unserved_passengers) + len(unboarded_unserved)

        # Calculate Movement Cost
        # If unserved > 0, required_floors should not be empty in a valid state.
        # If required_floors is somehow empty (e.g., invalid state), movement cost is 0.
        if not required_floors:
             movement_cost = 0
        else:
            required_indices = {self.floor_name_to_index[f] for f in required_floors}
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # Movement cost: distance to closest required floor + span
            dist_to_min = abs(current_lift_index - min_req_idx)
            dist_to_max = abs(current_lift_index - max_req_idx)
            span = max_req_idx - min_req_idx

            movement_cost = min(dist_to_min, dist_to_max) + span

        # Total heuristic
        total_cost = movement_cost + action_cost

        return total_cost
