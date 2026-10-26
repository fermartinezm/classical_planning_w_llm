from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the minimum number of board/depart actions needed for each unserved
    passenger and adds an estimate of the minimum lift travel required to visit
    all necessary floors (origins of unboarded passengers and destinations of
    boarded passengers).

    # Assumptions
    - The floor structure is linear, defined by `(above f_i f_{i+1})` facts.
    - The cost of each action (board, depart, up, down) is 1.
    - The lift can pick up/drop off multiple passengers at a floor.
    - The estimated travel cost assumes the lift makes a single sweep covering
      the range of required floors, starting from its current position.
    - States provided to the heuristic are valid according to the domain (e.g.,
      a passenger is either at their origin, boarded, or served; lift location is specified).

    # Heuristic Initialization
    - Parses `(above ?f1 ?f2)` facts from static information to determine the
      linear order of floors and create mappings between floor names and
      numerical indices.
    - Parses `(destin ?person ?floor)` facts from the initial state to store
      the destination floor for each passenger.
    - Parses `(served ?person)` facts from the goal state to identify all
      passengers that need to be served.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the goal state is reached. If yes, the heuristic is 0.
    2. Find the current floor of the lift. If not found, the state is invalid
       for this heuristic's calculation; handle appropriately (e.g., return a large value).
    3. Initialize `action_cost` to 0 and `required_floors` set to empty.
    4. Create a temporary mapping of passengers to their origin floors from the current state facts.
    5. Iterate through all passengers that need to be served (identified during initialization).
    6. For each unserved passenger `p`:
       - If `p` is currently `(boarded ?p)`:
         - Add 1 to `action_cost` (for the future `depart` action).
         - Add `p`'s destination floor (looked up from initialization data) to the `required_floors` set.
       - If `p` is currently at an origin floor `f_origin` (found in step 4):
         - Add 2 to `action_cost` (for the future `board` and `depart` actions).
         - Add `p`'s origin floor `f_origin` to the `required_floors` set.
         - Add `p`'s destination floor (looked up from initialization data) to the `required_floors` set.
       - (Passengers not in `served`, `boarded`, or `origin` predicates in the state but needing to be served are considered in an invalid state for this heuristic).
    7. Calculate `travel_cost`:
       - If `required_floors` is empty, `travel_cost` is 0.
       - Otherwise, get the indices for the current lift floor and all required floors.
       - Find the minimum (`min_req_idx`) and maximum (`max_req_idx`) floor indices among the `required_floors`.
       - The travel cost is estimated as the distance covering the range of required floors plus the minimum distance from the current floor to either end of that range:
         `travel_cost = (max_required_floor_index - min_required_floor_index) + min(abs(current_lift_index - min_required_floor_index), abs(current_lift_index - max_required_floor_index))`
    8. The total heuristic value is `action_cost + travel_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of passengers to be served.
        """
        self.goals = task.goals # Store goals to check for goal state

        # 1. Determine floor order and create index mappings
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_map_up = {} # Maps floor_below -> floor_above
        all_floors = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                above_map_up[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor (a floor that is not the value in any above_map_up entry)
        lowest_floor = None
        # Check if all_floors is not empty before iterating
        if all_floors:
            for floor in all_floors:
                if floor not in above_map_up.values():
                     lowest_floor = floor
                     break
        # Assuming a valid miconic domain instance with a linear floor structure
        # If lowest_floor is None here, it implies an invalid floor structure (e.g., cycle or disconnected)
        # assert lowest_floor is not None, "Could not find the lowest floor"

        current_floor = lowest_floor
        index = 0
        # Build the ordered list only if a lowest floor was found
        if lowest_floor is not None:
            while current_floor in all_floors: # Loop until we've added all floors
                self.floor_to_index[current_floor] = index
                self.index_to_floor[index] = current_floor
                index += 1
                # Find the floor above the current one
                current_floor = above_map_up.get(current_floor)

        self.num_floors = len(self.floor_to_index) # Number of floors

        # 2. Store passenger destinations from initial state
        self.passenger_destinations = {}
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "destin":
                p, f_destin = parts[1], parts[2]
                self.passenger_destinations[p] = f_destin

        # 3. Identify all passengers that need to be served from goal state
        self.passengers_to_serve = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "served":
                self.passengers_to_serve.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is not found, this is an invalid state for the domain.
        # Cannot compute meaningful heuristic. Return a large value.
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This indicates a problem with the state representation or domain definition
             # where the lift is not at a recognized floor.
             # Returning a large value makes this state unattractive.
             return float('inf') # Or some large integer

        # 3. Calculate action cost and identify required floors
        action_cost = 0
        required_floors = set()

        # Keep track of origin floors for quick lookup from the current state
        origin_floors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                origin_floors[parts[1]] = parts[2]

        for p in self.passengers_to_serve:
            if f"(served {p})" not in state:
                # Passenger needs to be served
                if f"(boarded {p})" in state:
                    # Passenger is boarded, needs depart
                    action_cost += 1 # Needs 1 depart action
                    # Needs lift to visit destination floor
                    if p in self.passenger_destinations:
                        required_floors.add(self.passenger_destinations[p])
                    # else: destination not found? Invalid problem setup.
                elif p in origin_floors:
                    # Passenger is unboarded (at origin), needs board and depart
                    action_cost += 2 # Needs 1 board + 1 depart action
                    origin_floor = origin_floors[p]
                    # Needs lift to visit origin floor (for pickup)
                    required_floors.add(origin_floor)
                    # Needs lift to visit destination floor (for dropoff)
                    if p in self.passenger_destinations:
                         required_floors.add(self.passenger_destinations[p])
                    # else: destination not found? Invalid problem setup.
                # else: Passenger is unserved but neither boarded nor at origin.
                # This state should not be reachable in a valid plan from the initial state.
                # We ignore such passengers for heuristic calculation, assuming they don't exist
                # or are in an unrecoverable state.


        # 4. Calculate travel cost
        travel_cost = 0
        if required_floors:
            current_lift_idx = self.floor_to_index[current_lift_floor]
            # Filter out required floors that are not in our floor index map (invalid floors)
            valid_required_floors = {f for f in required_floors if f in self.floor_to_index}

            if valid_required_floors:
                required_indices = sorted([self.floor_to_index[f] for f in valid_required_floors])
                min_req_idx = required_indices[0]
                max_req_idx = required_indices[-1]

                # Travel cost is the distance to cover the range [min_req_idx, max_req_idx]
                # starting from current_lift_idx.
                # This is min(dist(current, min_req) + dist(min_req, max_req), dist(current, max_req) + dist(max_req, min_req))
                # = min(abs(current_idx - min_req_idx) + (max_req_idx - min_req_idx), abs(current_idx - max_req_idx) + (max_req_idx - min_req_idx))
                # = (max_req_idx - min_req_idx) + min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx))
                travel_cost = (max_req_idx - min_req_idx) + min(abs(current_lift_idx - min_req_idx), abs(current_lift_idx - max_req_idx))
            # else: If required_floors was not empty but none were valid, travel_cost remains 0.

        total_heuristic = action_cost + travel_cost

        return total_heuristic
