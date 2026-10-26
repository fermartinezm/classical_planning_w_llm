from fnmatch import fnmatch
# Assuming heuristic_base.py is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided in the execution environment
# This allows the code to be runnable standalone for syntax check, but requires
# the actual base class in the target planning environment.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    Estimates the number of actions (board, depart, move) required to serve all passengers.
    It counts the pending board and depart actions and adds an estimate of the necessary lift movement.

    # Assumptions
    - Each passenger needs one board action (if not already boarded) and one depart action (if not yet served).
    - The lift must visit the origin floor of waiting passengers and the destination floor of all unserved passengers.
    - The movement cost is estimated by the minimum travel distance to cover the range of all required stop floors, starting from the lift's current floor.
    - The `above` predicates define a linear ordering of floors from lowest to highest, where `(above f_i f_{i+1})` means `f_i` is immediately below `f_{i+1}`.

    # Heuristic Initialization
    - Extracts the floor ordering from `above` predicates in static facts to map floor names to numerical indices. Finds the lowest floor and traverses the `above` chain.
    - Stores passenger destination floors by extracting `(destin ?p ?f)` facts from static information (assuming destinations are static).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse floor ordering from `above` predicates in static facts to create a floor-to-index mapping. Find the lowest floor (the one not appearing as the second argument in any `above` fact) and traverse the chain using the `above` relationships.
    2.  Store passenger destination floors by extracting `(destin ?p ?f)` facts from static information.
    3.  Identify the lift's current floor from the state by finding the `(lift-at ?f)` fact.
    4.  Identify unserved passengers by checking which passengers (from the set of all passengers with destinations) are not `(served ?p)` in the current state.
    5.  If there are no unserved passengers, the heuristic is 0 (goal state).
    6.  Categorize unserved passengers into 'waiting' (`(origin ?p ?f)`) and 'boarded' (`(boarded ?p)`) based on the current state facts.
    7.  Count the number of pending `board` actions: This is the number of waiting passengers.
    8.  Count the number of pending `depart` actions: This is the total number of unserved passengers (each needs a final depart action to be served).
    9.  Identify the set of floors the lift *must* visit: These are the origin floors of waiting passengers and the destination floors of all unserved passengers (both waiting and boarded).
    10. Calculate the movement cost:
        - Find the minimum and maximum floor indices among the required stop floors identified in the previous step.
        - The movement cost is estimated as the minimum distance to reach either the min or max required floor from the current lift floor, plus the distance to traverse the entire range between the min and max required floors. This is calculated as `min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)`.
    11. The total heuristic value is the sum of the number of pending board actions, the number of pending depart actions, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor ordering
        # Build adjacency list for floors based on 'above'
        # (above f_i f_{i+1}) means f_i is immediately below f_{i+1}
        below_to_above_map = {}
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_below, f_above = get_parts(fact)[1:]
                below_to_above_map[f_below] = f_above
                all_floors.add(f_below)
                all_floors.add(f_above)

        # Find the lowest floor (the one appearing as the first argument in 'above' but not the second)
        # If there's only one floor, all_floors will contain it, and below_to_above_map will be empty.
        lowest_floor = None
        above_floors = set(below_to_above_map.values())
        for floor in all_floors:
            if floor not in above_floors:
                lowest_floor = floor
                break
        
        # Handle case with a single floor
        if lowest_floor is None and len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
        elif lowest_floor is None and len(all_floors) > 1:
             # This indicates an issue with the above predicates not forming a single chain
             # or not having a clear lowest floor. Handle defensively.
             # For standard miconic, this shouldn't happen.
             pass # Proceeding might lead to errors if floor isn't found later

        # Build floor_to_index map by traversing from the lowest floor
        self.floor_to_index = {}
        current_floor = lowest_floor
        index = 0
        while current_floor is not None:
            self.floor_to_index[current_floor] = index
            current_floor = below_to_above_map.get(current_floor)
            index += 1

        # Store destination floor for each passenger from static facts
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                self.passenger_destinations[p] = f

        # Note: Passengers mentioned in goals might not be the only passengers.
        # The set of all passengers should ideally come from object definitions,
        # but we infer them from destinations in static facts.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify lift's current floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        # Assuming lift-at is always present in a valid state and is a known floor
        if lift_floor not in self.floor_to_index:
             # Should not happen in valid states based on domain definition
             return float('inf')

        lift_idx = self.floor_to_index[lift_floor]

        # Identify unserved passengers and their state
        # Get all passengers known from destinations
        all_passengers = set(self.passenger_destinations.keys())
        # Get served passengers from the current state
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        # Unserved are those with destinations who are not served
        unserved_passengers = all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        waiting_passengers = set()
        boarded_passengers = set()
        passenger_origins = {} # Store origins for waiting passengers

        # Iterate through state facts to find waiting and boarded unserved passengers
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                if p in unserved_passengers:
                     waiting_passengers.add(p)
                     passenger_origins[p] = f
            elif match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                if p in unserved_passengers:
                     boarded_passengers.add(p)

        # Count pending actions (excluding moves)
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action eventually.
        num_board_actions = len(waiting_passengers)
        num_depart_actions = len(unserved_passengers)

        # Identify required stop floors
        required_stop_floors = set()
        # Lift must visit origin floors for waiting passengers
        for p in waiting_passengers:
            # Ensure origin floor is known
            if p in passenger_origins and passenger_origins[p] in self.floor_to_index:
                required_stop_floors.add(passenger_origins[p])
            # else: handle error or invalid state? Assuming valid.

        # Lift must visit destination floors for all unserved passengers (both waiting and boarded)
        for p in unserved_passengers:
             # Ensure destination floor is known
             if p in self.passenger_destinations and self.passenger_destinations[p] in self.floor_to_index:
                required_stop_floors.add(self.passenger_destinations[p])
             # else: handle error or invalid state? Assuming valid.


        # Calculate movement cost
        # If there are unserved passengers, there must be required stops
        # (either origins for waiting or destinations for unserved), unless
        # a passenger is waiting at their destination and the lift is there,
        # and they are the only unserved passenger. Even in that edge case,
        # the origin/destination floor is a required stop.
        # So, required_stop_floors should not be empty if unserved_passengers > 0.
        # Handle defensively just in case.
        if not required_stop_floors:
             movement_cost = 0
        else:
            required_indices = [self.floor_to_index[f] for f in required_stop_floors]
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # Movement cost estimate: min moves to cover the range [min_req_idx, max_req_idx] starting from lift_idx
            # This is distance to closest end + distance to traverse the range
            movement_cost = min(abs(lift_idx - min_req_idx), abs(lift_idx - max_req_idx)) + (max_req_idx - min_req_idx)

        # Total heuristic is sum of actions and movement
        total_heuristic = num_board_actions + num_depart_actions + movement_cost

        return total_heuristic
