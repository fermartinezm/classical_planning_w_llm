from fnmatch import fnmatch
# Assume Heuristic base class is imported from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class
class miconicHeuristic: # Inherit from Heuristic if base class is available
# class miconicHeuristic(Heuristic): # Use this line if inheriting

    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all unserved passengers by summing
    the estimated travel cost for the lift to visit necessary floors and the
    estimated action costs (board/depart) for each unserved passenger.

    # Assumptions
    - Floors are totally ordered by the 'above' predicate, forming a linear structure.
    - The lift can carry multiple passengers (infinite capacity assumed).
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding or departing a passenger is 1.
    - Problem instances have a valid linear floor structure and a 'lift-at' predicate in the initial state.
    - Goal conditions only involve 'served' predicates for passengers.

    # Heuristic Initialization
    - Parse the 'above' predicates from static facts to determine the floor order
      and create a mapping from floor names to their integer indices and vice-versa.
    - Parse the 'origin' and 'destin' predicates from static facts to store
      the initial origin and destination for each passenger.
    - Store the goal passengers (those who need to be 'served').

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state (all goal passengers are 'served'). If yes, return 0.
    2. Find the current floor of the lift. If not found or not a recognized floor, return infinity (invalid state).
    3. Identify all passengers who are not yet 'served' according to the goal state.
    4. For each unserved passenger:
       - Get their origin and destination floors from the pre-parsed static information. If missing, return infinity (invalid problem/state).
       - Determine if they are waiting at their origin floor (`(origin p f)`) or are already boarded (`(boarded p)`).
       - If waiting at origin `f_origin`, add `f_origin` to the set of floors the lift must visit. If `f_origin` is not a recognized floor, return infinity.
       - If boarded, add their destination floor `f_destin` to the set of floors the lift must visit. If `f_destin` is not a recognized floor, return infinity.
       - Count the number of waiting passengers (each needs a 'board' action).
       - Count the number of boarded passengers (each needs a 'depart' action).
    5. Determine the set of unique floors the lift must visit (`floors_to_visit`).
    6. If `floors_to_visit` is empty:
       - If there are unserved passengers, this indicates an invalid state (unserved, but not waiting/boarded). Return infinity.
       - If there are no unserved passengers, the initial goal check should have returned 0. This case should not be reached if the initial check is correct.
    7. If `floors_to_visit` is not empty:
       - Find the minimum (`f_min`) and maximum (`f_max`) floor in `floors_to_visit` based on the floor order indices.
       - Calculate the distance between any two floors `f1`, `f2` as `abs(index(f1) - index(f2))`.
       - Estimate the minimum travel cost for the lift to go from its current floor `f_current` to visit all floors in `floors_to_visit`. This is calculated as the distance from `f_current` to the closest extreme of the required range (`f_min` or `f_max`), plus the distance to traverse the entire required range (`dist(f_min, f_max)`). Formula: `min(dist(f_current, f_min), dist(f_current, f_max)) + dist(f_min, f_max)`.
    8. The total heuristic value is the estimated minimum travel cost plus the total count of 'board' actions needed (number of waiting passengers) plus the total count of 'depart' actions needed (number of boarded passengers).
    """

    def __init__(self, task):
        # If inheriting from Heuristic, uncomment the next line:
        # super().__init__(task)

        self.goals = task.goals
        self.static = task.static

        # Build floor order and floor-to-index map
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_map = {} # floor_below -> floor_above
        below_map = {} # floor_above -> floor_below
        all_floors = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (above f_above f_below)
                    f_above, f_below = parts[1:]
                    above_map[f_below] = f_above
                    below_map[f_above] = f_below
                    all_floors.add(f_above)
                    all_floors.add(f_below)

        # Find the lowest floor: a floor that is not a value in above_map (nothing is below it)
        floors_with_something_below_them = set(above_map.keys())
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_with_something_below_them:
                lowest_floor = floor
                break

        # If a lowest floor is found, build the ordered list and index map
        if lowest_floor:
            current_floor = lowest_floor
            index = 0
            while current_floor:
                self.floor_to_index[current_floor] = index
                self.index_to_floor[index] = current_floor
                index += 1
                current_floor = above_map.get(current_floor) # Get the floor directly above this one
        # Note: If lowest_floor is None and all_floors is not empty, it implies a non-linear
        # or disconnected floor structure, which is not handled by this linear distance heuristic.
        # Assuming valid miconic problems have a linear floor structure.

        # Store passenger origins and destinations from static facts
        self.passenger_info = {} # { passenger: {'origin': floor, 'destin': floor} }
        for fact in self.static:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (origin p f)
                    p, f = parts[1:]
                    if p not in self.passenger_info:
                        self.passenger_info[p] = {}
                    self.passenger_info[p]['origin'] = f
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (destin p f)
                    p, f = parts[1:]
                    if p not in self.passenger_info:
                        self.passenger_info[p] = {}
                    self.passenger_info[p]['destin'] = f

        # Store goal passengers (those who need to be served)
        self.goal_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 2: # Ensure it's (served p)
                     p = parts[1]
                     self.goal_passengers.add(p)


    def get_floor_index(self, floor):
        """Helper to get the index of a floor."""
        # Return -1 if floor not found (e.g., invalid state fact)
        return self.floor_to_index.get(floor, -1)

    def get_distance(self, floor1, floor2):
        """Helper to calculate distance between two floors."""
        idx1 = self.get_floor_index(floor1)
        idx2 = self.get_floor_index(floor2)
        # If either floor is not in our index map (e.g., invalid floor name in state),
        # return a large value indicating this path is likely invalid/unreachable.
        if idx1 == -1 or idx2 == -1:
             return float('inf')
        return abs(idx1 - idx2)


    def __call__(self, node):
        state = node.state

        # Check if goal state is reached
        # A state is a goal state if all goal predicates are true.
        # In miconic, the goal is typically (served p) for a set of passengers.
        # We need to check if all self.goal_passengers are served in the current state.
        # If self.goal_passengers is empty, the goal is trivially met (h=0).
        if all(f"(served {p})" in state for p in self.goal_passengers):
            return 0 # Heuristic is 0 for goal states

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: # Ensure it's (lift-at f)
                    current_lift_floor = parts[1]
                    break

        # If lift location is unknown or not a recognized floor, the state is likely invalid or terminal.
        # Return infinity or a large value. Assuming valid states have lift-at at a valid floor.
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This shouldn't happen in valid miconic states reachable from initial state.
             # Returning a large value might guide search away from such states.
             return float('inf') # Or a large integer like 1000000

        # Identify unserved passengers and required floors
        unserved_passengers = set()
        waiting_passengers = set() # Passengers at origin, not boarded
        boarded_passengers = set() # Passengers boarded

        floors_to_visit = set() # Floors the lift must visit

        # Iterate through all passengers who are in the goal state (need serving)
        for p in self.goal_passengers:
            # Check if passenger is served in the current state
            if f"(served {p})" not in state:
                unserved_passengers.add(p)

                # Get origin and destination from stored info (from static facts)
                p_info = self.passenger_info.get(p)
                if p_info is None:
                    # Passenger in goal but not in static origin/destin facts? Invalid problem?
                    # Assuming valid problems where goal passengers have origin/destin in static.
                    return float('inf') # Indicate problem/state issue

                origin_floor = p_info.get('origin')
                destin_floor = p_info.get('destin')

                if origin_floor is None or destin_floor is None:
                     # Missing origin or destin for a goal passenger. Invalid problem.
                     return float('inf') # Indicate problem/state issue

                # Check if passenger is waiting at origin
                origin_fact = f"(origin {p} {origin_floor})"
                if origin_fact in state:
                    waiting_passengers.add(p)
                    # Add origin floor to required visits only if it's a valid floor
                    if origin_floor in self.floor_to_index:
                        floors_to_visit.add(origin_floor)
                    else:
                         # Origin floor not in floor map. Invalid state or problem.
                         return float('inf') # Indicate problem/state issue

                # Check if passenger is boarded
                boarded_fact = f"(boarded {p})"
                if boarded_fact in state:
                    boarded_passengers.add(p)
                    # Add destination floor to required visits only if it's a valid floor
                    if destin_floor in self.floor_to_index:
                        floors_to_visit.add(destin_floor)
                    else:
                         # Destination floor not in floor map. Invalid state or problem.
                         return float('inf') # Indicate problem/state issue

        # If no unserved passengers, the initial goal check should have returned 0.
        # If unserved passengers exist but floors_to_visit is empty, it means
        # these passengers are neither waiting at origin nor boarded. This is an invalid state.
        # Or maybe they are served but the goal check failed? No, goal check is explicit.
        # So, unserved passengers must be waiting or boarded in a valid state.
        # If unserved_passengers > 0 and floors_to_visit is empty, return inf.
        if unserved_passengers and not floors_to_visit:
             # This state implies unserved passengers are not in a state requiring lift interaction.
             # This shouldn't happen in valid reachable states unless they are already at destination but not served?
             # No, (destin p f) is a static fact. (served p) is the goal.
             # If (origin p f) is false and (boarded p) is false and (served p) is false, where is the passenger?
             # This state is likely unreachable or invalid.
             return float('inf')


        # Calculate travel cost
        travel_cost = 0
        if floors_to_visit:
            # Find min and max floor indices among floors_to_visit
            required_indices = sorted([self.get_floor_index(f) for f in floors_to_visit])
            min_idx_to_visit = required_indices[0]
            max_idx_to_visit = required_indices[-1]

            f_min_to_visit = self.index_to_floor[min_idx_to_visit]
            f_max_to_visit = self.index_to_floor[max_idx_to_visit]

            current_idx = self.get_floor_index(current_lift_floor)

            # Minimum travel to cover the range [min_idx, max_idx] starting from current_idx
            # This is min distance from current to one extreme + distance to traverse the range.
            dist_current_to_min_required = abs(current_idx - min_idx_to_visit)
            dist_current_to_max_required = abs(current_idx - max_idx_to_visit)
            dist_min_to_max_required = max_idx_to_visit - min_idx_to_visit # Distance is always positive

            travel_cost = min(dist_current_to_min_required, dist_current_to_max_required) + dist_min_to_max_required

        # Calculate action cost
        # Each waiting passenger needs 1 board action.
        # Each boarded passenger needs 1 depart action.
        action_cost = len(waiting_passengers) + len(boarded_passengers)

        # Total heuristic is travel cost + action cost
        total_cost = travel_cost + action_cost

        return total_cost
