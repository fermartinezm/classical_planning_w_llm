# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers.
    It sums the number of required 'board' and 'depart' actions for unserved
    passengers and adds an estimate of the minimum vertical travel distance
    the lift needs to cover to visit all floors where a pickup or dropoff
    is required for unserved passengers, starting from the current lift floor.

    # Assumptions
    - Floors are linearly ordered by the 'above' predicate.
    - The lift can carry multiple passengers simultaneously.
    - The goal is to serve all passengers specified in the problem goal.
    - Passenger origins and destinations are provided in the initial state.

    # Heuristic Initialization
    - Determine the ordered list of floors and create a mapping from floor name to index
      by parsing 'above' facts from static information and identifying all floor objects
      mentioned in relevant initial state facts.
    - Extract the origin and destination floors for each passenger from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Initialize the heuristic value `h` to 0.
    3. Identify the set of floors the lift *must* visit (`required_indices`) and count pending actions:
       - Initialize `required_indices = set()`.
       - Iterate through all passengers `p` known from the initial state.
       - If `(served p)` is NOT true in the current state:
         - Get origin `o` and destination `d` for `p` from the pre-calculated passenger info.
         - If `o` is known and `(origin p o)` is true in the current state:
           - If `o` is a valid floor in the index map, add its index to `required_indices`.
           - Add 1 to `h` (for the 'board' action needed for `p`).
         - Elif `d` is known and `(boarded p)` is true in the current state:
           - If `d` is a valid floor in the index map, add its index to `required_indices`.
           - Add 1 to `h` (for the 'depart' action needed for `p`).
    4. Calculate the estimated movement cost:
       - Get the index of the current lift floor. If the current floor is not in the index map, return a large value (should not happen in valid problems).
       - If `required_indices` is empty (meaning all unserved passengers are either at their destination and boarded, or all passengers are served):
         - Movement cost is 0.
       - If `required_indices` is not empty:
         - Find the minimum required index (`min_req_index`) and maximum required index (`max_req_index`).
         - The movement cost is estimated as the total span covered by the current lift floor and all required floors: `max(current_lift_index, max_req_index) - min(current_lift_index, min_req_index)`.
    5. Add the calculated movement cost to `h`.
    6. Return `h` as the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger info.
        """
        self.task = task
        self.floor_to_index = {}
        self.passenger_info = {}

        # 1. Determine floor order and index map
        floor_above_map = {}
        all_floors = set()

        # Collect floors and above relations from static facts
        for fact in task.static:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                all_floors.add(f_above)
                all_floors.add(f_below)
                floor_above_map[f_below] = f_above # f_above is above f_below

        # Collect floors from initial state facts if not in static
        for fact in task.initial_state:
             if match(fact, "lift-at", "*"):
                 all_floors.add(get_parts(fact)[1])
             elif match(fact, "origin", "*", "*"):
                 all_floors.add(get_parts(fact)[2])
             elif match(fact, "destin", "*", "*"):
                 all_floors.add(get_parts(fact)[2])

        # Find the bottom floor (a floor that is not a value in floor_above_map)
        bottom_floor = None
        if all_floors:
             floors_with_floor_below = set(floor_above_map.keys()) # These floors have a floor below them
             potential_bottoms = all_floors - floors_with_floor_below
             if potential_bottoms:
                 # Assuming there's only one bottom floor in a linear structure
                 bottom_floor = potential_bottoms.pop()
             elif all_floors:
                 # Handle single floor case or circular/complex structures (miconic is linear)
                 # If all floors have a floor below them according to map, pick any to start
                 # This case might indicate an issue with problem definition or parsing
                 # For robustness, if no clear bottom, just list floors alphabetically or by appearance
                 # However, for miconic, a bottom floor should exist if there's > 1 floor
                 # If only one floor, all_floors will have 1 element, floor_above_map is empty, potential_bottoms has 1 element.
                 bottom_floor = next(iter(all_floors)) # Fallback

        ordered_floors = []
        if bottom_floor:
            current = bottom_floor
            # Build the ordered list by following the 'above' chain upwards
            while current is not None and current in all_floors: # Add check for safety
                ordered_floors.append(current)
                current = floor_above_map.get(current)
                # Prevent infinite loop in case of malformed above facts forming a cycle
                if current in ordered_floors:
                     # print(f"Warning: Floor cycle detected involving {current}. Floor ordering may be incorrect.") # Debugging
                     break


        # If ordered_floors is still empty but there are floors (e.g., single floor), just list them
        if not ordered_floors and all_floors:
             ordered_floors = sorted(list(all_floors)) # Use sorted list for consistency

        self.floor_to_index = {f: i for i in range(len(ordered_floors)) for f in [ordered_floors[i]]}


        # 2. Extract passenger origin and destination info from initial state
        # We collect info for any passenger mentioned in origin/destin in initial state
        for fact in task.initial_state:
            if match(fact, "origin", "*", "*"):
                p, o = get_parts(fact)[1:]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['origin'] = o
            elif match(fact, "destin", "*", "*"):
                p, d = get_parts(fact)[1:]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['destin'] = d

        # Passengers mentioned in goals must also be in passenger_info
        # Their origin/destin must be in the initial state facts for a valid problem
        for goal in task.goals:
             if match(goal, "served", "*"):
                 p = get_parts(goal)[1]
                 if p not in self.passenger_info:
                      # This passenger is in the goal but not in initial origin/destin facts.
                      # This shouldn't happen in valid miconic problems according to typical PDDL setup.
                      # Add a placeholder, but heuristic might be inaccurate for this passenger.
                      # print(f"Warning: Passenger {p} in goal but not in initial origin/destin facts.") # Debugging
                      self.passenger_info[p] = {'origin': None, 'destin': None}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached (heuristic is 0 at goal)
        if self.task.goal_reached(state):
            return 0

        # 1. Get current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor not indexed, state is likely invalid or unsolvable
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # print(f"Error: Lift location {current_lift_floor} not found or not indexed.") # Debugging
             return float('inf') # Should not happen in valid states

        current_lift_index = self.floor_to_index[current_lift_floor]


        # 3. Identify required floors and count actions
        required_indices = set()
        h = 0

        # Iterate through all passengers we know about from init/goal
        for p, info in self.passenger_info.items():
            # Check if passenger is served
            served_fact = f"(served {p})"
            if served_fact not in state:
                origin = info.get('origin')
                destin = info.get('destin')

                # Check if passenger is waiting at origin
                # Only add origin if we know it and passenger is at origin
                if origin and f"(origin {p} {origin})" in state:
                    if origin in self.floor_to_index:
                         required_indices.add(self.floor_to_index[origin])
                    h += 1 # Cost for board action

                # Check if passenger is boarded
                # Use elif because a passenger is either at origin OR boarded OR served
                elif destin and f"(boarded {p})" in state:
                    if destin in self.floor_to_index:
                         required_indices.add(self.floor_to_index[destin])
                    h += 1 # Cost for depart action
                # Note: If passenger is unserved but neither at origin nor boarded,
                # this state is likely unreachable or invalid according to domain rules.
                # The heuristic will not count actions/movement for them in this state.


        # 4. Calculate estimated movement cost
        movement_cost = 0
        if required_indices:
            min_req_index = min(required_indices)
            max_req_index = max(required_indices)
            # Estimate movement as the span covering current lift position and required floors
            movement_cost = max(current_lift_index, max_req_index) - min(current_lift_index, min_req_index)

        # 5. Add movement cost to heuristic
        h += movement_cost

        return h
