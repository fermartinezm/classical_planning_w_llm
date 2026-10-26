from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys # Import sys for handling potential infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions required to serve all unserved passengers.
    Heuristic = (2 * num_waiting_passengers) + (1 * num_boarded_passengers) + MovementCost

    MovementCost is estimated as the vertical span of all floors that need
    to be visited (origin floors for waiting passengers, destination floors
    for all unserved passengers), including the current lift floor.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build floor index map (0-based from lowest floor)
        self.floor_to_index = {}
        floor_below_map = {} # f_higher -> f_lower
        all_floors = set()

        # Collect floors from above facts and build floor_below_map
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                # Ensure fact has enough parts
                if len(parts) > 2:
                    f_higher, f_lower = parts[1], parts[2]
                    floor_below_map[f_higher] = f_lower
                    all_floors.add(f_higher)
                    all_floors.add(f_lower)

        # Collect floors from origin and destin facts in static (initial state)
        # This ensures we get all floors even if only one exists or above facts are minimal
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] in ['origin', 'destin']:
                  # Ensure fact has enough parts before accessing index 2
                  if len(parts) > 2:
                    all_floors.add(parts[2]) # Add the floor object

        if not all_floors:
             # Should not happen in a valid problem, but handle defensively
             # Cannot build floor map, heuristic will return infinity
             self.floor_to_index = {} # Ensure it's initialized
             # print("Warning: No floors found in static facts.") # Keep quiet
             return

        if len(all_floors) == 1:
             # Special case: only one floor
             self.floor_to_index[list(all_floors)[0]] = 0
        else:
            # Find the highest floor (a floor that is in all_floors but not a value in floor_below_map)
            floors_with_floor_above = set(floor_below_map.values())

            highest_floor = None
            potential_highest = all_floors - floors_with_floor_above
            
            if len(potential_highest) == 1:
                 highest_floor = list(potential_highest)[0]
            else:
                 # Fallback: If above facts are missing or malformed, sort floors alphabetically.
                 # This is a weak fallback but prevents crash.
                 # print(f"Warning: Could not determine unique highest floor. Floors: {all_floors}, Potential highest: {potential_highest}") # Keep quiet
                 sorted_floors = sorted(list(all_floors))
                 for i, floor in enumerate(sorted_floors):
                      self.floor_to_index[floor] = i
                 # print("Using alphabetical floor order.") # Keep quiet
                 return # Cannot proceed with chain logic

            # Build the ordered list of floors from highest to lowest
            floor_list_desc = [highest_floor]
            current = highest_floor
            while current in floor_below_map:
                current = floor_below_map[current]
                floor_list_desc.append(current)

            # Reverse the list to get lowest to highest
            floor_list_asc = floor_list_desc[::-1]

            # Assign indices (0-based from lowest)
            for i, floor in enumerate(floor_list_asc):
                self.floor_to_index[floor] = i

        # Store passenger destinations and collect all passenger names
        self.destinations = {}
        self.all_passengers = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                # Ensure fact has enough parts
                if len(parts) > 2:
                    p, d = parts[1], parts[2]
                    self.destinations[p] = d
                    self.all_passengers.add(p)
            elif parts[0] == 'origin':
                 # Ensure fact has enough parts
                 if len(parts) > 2:
                    p, o = parts[1], parts[2]
                    self.all_passengers.add(p)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        # If floor map wasn't built due to init error
        if not self.floor_to_index:
             return sys.maxsize

        state = node.state

        # Check if goal is reached
        # This check is redundant if the search algorithm checks goals,
        # but it ensures h=0 for goal states as required.
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                # Ensure fact has enough parts
                if len(parts) > 1:
                    current_lift_floor = parts[1]
                    break

        # If lift location is unknown or floor is not indexed, heuristic is infinity
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return sys.maxsize # Use a large integer for infinity

        current_lift_floor_idx = self.floor_to_index[current_lift_floor]

        # Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served' and len(get_parts(fact)) > 1}
        unserved_passengers = self.all_passengers - served_passengers

        # If no unserved passengers, but goal not reached, something is wrong or
        # the goal includes more than just served predicates.
        # Assuming goal is only served predicates, this case implies goal is reached.
        # The check `if self.goals <= state:` handles this already.

        U_origin = set() # Names of passengers waiting at origin
        U_boarded = set() # Names of passengers boarded

        # Collect current state of unserved passengers
        for fact in state:
            parts = get_parts(fact)
            if len(parts) > 1 and parts[1] in unserved_passengers:
                if parts[0] == 'origin':
                    U_origin.add(parts[1])
                elif parts[0] == 'boarded':
                    U_boarded.add(parts[1])

        # Calculate non-movement cost
        # Each passenger in U_origin needs board (1) + depart (1) = 2 actions
        # Each passenger in U_boarded needs depart (1) action
        non_movement_cost = 2 * len(U_origin) + len(U_boarded)

        # Calculate movement cost
        required_floor_indices = set()

        # Add origin floor indices for waiting passengers
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'origin' and len(parts) > 2 and parts[1] in U_origin:
                  # Ensure floor exists in index map
                  if parts[2] in self.floor_to_index:
                      required_floor_indices.add(self.floor_to_index[parts[2]])

        # Add destination floor indices for all unserved passengers (waiting or boarded)
        all_unserved_passenger_names = U_origin | U_boarded
        for p in all_unserved_passenger_names:
             dest_floor = self.destinations.get(p) # Use .get for safety
             if dest_floor and dest_floor in self.floor_to_index: # Ensure destination exists and floor is indexed
                 required_floor_indices.add(self.floor_to_index[dest_floor])
             # else: # Passenger has no destination or destination floor is not indexed. Problem definition issue.

        movement_cost = 0
        if required_floor_indices:
             min_req_idx = min(required_floor_indices)
             max_req_idx = max(required_floor_indices)
             # Movement cost is the span of required floors plus distance from current floor to the span
             # A simple estimate is the span including the current floor
             movement_cost = max(max_req_idx, current_lift_floor_idx) - min(min_req_idx, current_lift_floor_idx)

        # Total heuristic value
        heuristic_value = non_movement_cost + movement_cost

        return heuristic_value
