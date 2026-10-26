from heuristics.heuristic_base import Heuristic
import sys # Import sys for float('inf')

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts with no arguments like '(predicate)'
    if fact.strip() == '()':
        return []
    # Remove outer parentheses and split by spaces
    return fact.strip()[1:-1].split()

# Define the heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the minimum number of board and depart actions needed and adds an estimate
    of the minimum lift movement cost to visit all necessary floors (origins for
    unboarded passengers and destinations for boarded passengers).

    # Assumptions
    - Floors are linearly ordered and can be mapped to integers based on their names (e.g., f1, f2, ...).
    - Each board and depart action costs 1.
    - Lift movement between adjacent floors costs 1.
    - The minimum lift movement to visit a set of floors on a line starting from
      the current floor is estimated by traveling to the nearest extreme required
      floor and then sweeping across the range of required floors.

    # Heuristic Initialization
    - Build a mapping from floor names to integer indices based on the assumption
      that floors are named f<number> and f1 < f2 < ...
    - Store the destination floor for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. In the constructor (`__init__`):
       - Parse all floor objects from the initial state and static facts using
         relevant predicates (`lift-at`, `origin`, `destin`, `above`).
       - Create a mapping from floor name (string) to an integer index by sorting
         the collected floors based on the numerical part of their name (assuming f1, f2, ... naming).
       - Parse the static facts to find the destination floor for each passenger
         using the `(destin ?p ?f)` predicate. Store this in a dictionary `passenger_destinations`.

    2. In the heuristic function (`__call__`):
       - Identify the current floor of the lift from the state using the `(lift-at ?f)` predicate.
       - If the lift's location is not defined or not in the floor mapping, return infinity (unreachable state).
       - Get the set of all passengers whose destinations are known (from `passenger_destinations`).
       - Get the set of passengers who are currently served from the state using the `(served ?p)` predicate.
       - Identify the set of unserved passengers (`all_passengers - served_passengers`).
       - If there are no unserved passengers, the heuristic is 0 (goal state).
       - Initialize counters for unserved passengers at origin (`n_origin`) and
         unserved passengers who are boarded (`n_boarded`).
       - Initialize a set to store floors the lift *must* visit (`required_floors`).
       - For each unserved passenger `p`:
         - Check if `(origin p o)` is true in the current state. If yes:
           - Increment `n_origin`.
           - Add floor `o` to `required_floors`.
         - Else, check if `(boarded p)` is true in the current state. If yes:
           - Increment `n_boarded`.
           - Find the destination `d` for passenger `p` using `passenger_destinations`.
           - Add floor `d` to `required_floors`.

       - Calculate the action cost component: `action_cost = n_origin * 2 + n_boarded`.
         (Each unserved passenger at origin needs a board and a depart action.
          Each unserved boarded passenger needs a depart action).

       - Calculate the lift movement cost component:
         - If `required_floors` is empty, `lift_movement_cost = 0`.
         - Otherwise:
           - Map the current lift floor and all floors in `required_floors` to their integer indices using the floor mapping.
           - Ensure all required floors were successfully mapped.
           - If the set of mapped required indices is empty, `lift_movement_cost = 0`.
           - Otherwise:
             - Find the minimum (`min_idx`) and maximum (`max_idx`) indices among the mapped `required_floors`.
             - Get the index of the current lift floor.
             - Calculate the minimum distance to visit all points in required_indices starting from current_idx.
             `current_idx = floor_mapping[current_lift_floor]`
             `dist_to_min = abs(current_idx - min_idx)`
             `dist_to_max = abs(current_idx - max_idx)`
             `sweep_dist = max_idx - min_idx`

             `lift_movement_cost = min(dist_to_min, dist_to_max) + sweep_dist`


       - The total heuristic value is `action_cost + lift_movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the floor mapping and storing
        passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Build floor mapping (name to index)
        self.floor_mapping = self._build_floor_mapping(task)

        # 2. Store passenger destinations
        self.passenger_destinations = self._get_passenger_destinations()

    def _build_floor_mapping(self, task):
        """
        Build a mapping from floor names (e.g., 'f1') to integer indices (e.g., 1).
        Assumes floors are named f<number> and ordered numerically.
        Collects floors from relevant predicates in initial state and static facts.
        """
        all_floors = set()
        # Collect all floor objects from initial state and static facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) > 1:
                all_floors.add(parts[1])
            elif parts and parts[0] in ['origin', 'destin'] and len(parts) > 2:
                 all_floors.add(parts[2])

        for fact in task.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'above' and len(parts) > 2:
                 all_floors.add(parts[1])
                 all_floors.add(parts[2])

        # Sort floors based on the number in their name
        # Assumes floor names are like 'f1', 'f10', 'f2'
        try:
            # Attempt numerical sort based on the number after 'f'
            sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback to alphabetical sort if naming convention is not strictly followed
             # or if floor names are not in the expected format.
             sorted_floors = sorted(list(all_floors))


        # Create mapping
        floor_map = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
        return floor_map

    def _get_passenger_destinations(self):
        """
        Extract destination floor for each passenger from the static facts.
        """
        destinations = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin' and len(parts) > 2:
                passenger, floor = parts[1], parts[2]
                destinations[passenger] = floor
        return destinations


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) > 1:
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or not in mapping, heuristic is infinite (unreachable state).
        # This should not happen in valid states generated by the planner.
        if current_lift_floor is None or current_lift_floor not in self.floor_mapping:
             return float('inf')


        # Identify unserved passengers and required floors
        all_passengers = set(self.passenger_destinations.keys())

        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == 'served' and len(get_parts(fact)) > 1}

        unserved_passengers = all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        passengers_at_origin = set() # Unserved passengers at their origin
        passengers_boarded = set()   # Unserved passengers who are boarded
        required_floors = set()

        # Determine state for each unserved passenger
        for p in unserved_passengers:
            is_at_origin = False
            is_boarded = False
            # Check state facts for passenger location/status
            for fact in state:
                parts = get_parts(fact)
                if parts and len(parts) > 1 and parts[1] == p:
                    if parts[0] == 'origin' and len(parts) > 2:
                        is_at_origin = True
                        origin_floor = parts[2]
                        passengers_at_origin.add(p)
                        required_floors.add(origin_floor)
                        break # Assume a passenger is either at origin OR boarded
                    elif parts[0] == 'boarded':
                        is_boarded = True
                        passengers_boarded.add(p)
                        # Add destination floor to required floors
                        if p in self.passenger_destinations:
                             required_floors.add(self.passenger_destinations[p])
                        break # Assume a passenger is either at origin OR boarded

        n_origin = len(passengers_at_origin)
        n_boarded = len(passengers_boarded)

        # Action cost: Minimum board/depart actions needed.
        # Each unserved passenger at origin needs 1 board + 1 depart = 2 actions.
        # Each unserved boarded passenger needs 1 depart action.
        action_cost = n_origin * 2 + n_boarded

        # Lift movement cost
        lift_movement_cost = 0
        if required_floors:
            # Map required floors to indices, filtering out any not in mapping
            required_indices = {self.floor_mapping[f] for f in required_floors if f in self.floor_mapping}

            if required_indices: # Check if any required floors were successfully mapped
                 min_idx = min(required_indices)
                 max_idx = max(required_indices)
                 current_idx = self.floor_mapping[current_lift_floor]

                 # Minimum distance to visit all points in required_indices starting from current_idx
                 # This is min(dist(curr, min_req) + sweep(min_req, max_req), dist(curr, max_req) + sweep(max_req, min_req))
                 dist_to_min = abs(current_idx - min_idx)
                 dist_to_max = abs(current_idx - max_idx)
                 sweep_dist = max_idx - min_idx

                 lift_movement_cost = min(dist_to_min, dist_to_max) + sweep_dist
            # If required_indices is empty after mapping (e.g., required floor not found), cost remains 0.


        # Total heuristic is the sum of actions and lift movement
        total_heuristic = action_cost + lift_movement_cost

        return total_heuristic
