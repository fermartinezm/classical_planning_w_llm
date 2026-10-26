from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the remaining effort by summing:
    1. Twice the number of unboarded unserved passengers plus the number of boarded unserved passengers. This accounts for the minimum board and depart actions required.
    2. An estimate of the vertical movement required by the lift to visit all necessary floors. The movement estimate is the distance from the current floor to the highest floor requiring service, plus the distance from the highest floor requiring service down to the lowest floor requiring service. This models a simple up-and-down sweep strategy.

    # Assumptions
    - The floor structure is a linear sequence defined by `above` predicates.
    - Each `board` and `depart` action costs 1.
    - Movement cost is estimated based on the range of floors requiring service and the lift's current position relative to this range.

    # Heuristic Initialization
    - Parses the `above` predicates from static facts to establish the floor order
      and create a mapping from floor names to integer indices (highest floor gets highest index).
    - Stores the goal conditions (which passengers need to be served).
    - Stores static origin and destination facts for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
       - Determine the lift's current floor.
       - Identify all passengers that are part of the goal (need to be served).
       - Identify which goal passengers are already served in the current state.
       - Identify which goal passengers are currently boarded in the lift.

    2. Identify Unserved Passengers:
       - An unserved passenger is a goal passenger who is not yet marked as served.

    3. Identify Floors Requiring Service:
       - For each unserved passenger:
         - If the passenger is not boarded, their origin floor requires a pickup service.
         - If the passenger is boarded, their destination floor requires a dropoff service.
       - Collect all unique origin floors of unboarded unserved passengers and all unique destination floors of boarded unserved passengers into a set of "service floors".

    4. Calculate Base Heuristic (Passenger Actions):
       - The base heuristic component is calculated as (2 * number of unboarded unserved passengers) + (1 * number of boarded unserved passengers). This accounts for the minimum board and depart actions required (a waiting passenger needs board+depart, a boarded needs depart).

    5. Calculate Movement Heuristic:
       - If there are no service floors (meaning all unserved passengers are either already at their destination and boarded, or there are no unserved passengers), the movement cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the service floors.
       - Find the index of the lift's current floor.
       - The movement cost is estimated as the sum of two distances:
         - The distance from the lift's current floor to the highest floor requiring service (`abs(current_idx - max_req_idx)`).
         - The distance from the highest floor requiring service down to the lowest floor requiring service (`max_req_idx - min_req_idx`).
       This models a strategy where the lift first travels to the highest required floor and then sweeps downwards, visiting all required floors.

    6. Sum Components:
       - The total heuristic value is the sum of the base heuristic (passenger actions) and the estimated movement heuristic.
       - Ensure the heuristic is 0 if and only if all goal passengers are served.
    """

    def _setup_static_info(self):
        """
        Parses static facts to determine floor order, goal passengers,
        and static origin/destin facts.
        """
        above_map = {} # Maps f_above -> f_below
        all_floors = set()
        floors_below_something = set() # Floors that appear as f_below

        for fact in self.task.static:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                above_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_below_something.add(f_below)

        # Find the highest floor (a floor that is not below any other floor)
        highest_floor = None
        for floor in all_floors:
            if floor not in floors_below_something:
                 highest_floor = floor
                 break

        self.floor_to_index = {}
        if highest_floor:
            current_floor = highest_floor
            index = len(all_floors) - 1 # Assign highest index to highest floor

            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                next_floor = above_map.get(current_floor) # Get the floor directly below
                current_floor = next_floor
                index -= 1
        elif all_floors:
             # Handle case with a single floor and no 'above' facts
             self.floor_to_index[next(iter(all_floors))] = 0


        # Store goal passengers
        self.goal_passengers = set()
        for goal in self.task.goals:
            if match(goal, "served", "*"):
                passenger = get_parts(goal)[1]
                self.goal_passengers.add(passenger)

        # Store static origin/destin facts for easy lookup
        self.static_origins = {}
        self.static_destins = {}
        for fact in self.task.static:
             if match(fact, "origin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 self.static_origins[p] = f
             elif match(fact, "destin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 self.static_destins[p] = f


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Determine the lift's current floor.
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        if lift_floor is None:
             # Should not happen in a valid miconic state
             return float('inf')

        current_idx = self.floor_to_index.get(lift_floor, -1)
        if current_idx == -1:
             # Lift is at a floor not in our mapping - invalid state?
             return float('inf')


        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        unserved_passengers = self.goal_passengers - served_passengers

        # 3. Identify Floors Requiring Service
        service_floors = set()
        waiting_unserved_count = 0
        boarded_unserved_count = 0

        for p in unserved_passengers:
            if p in boarded_passengers:
                # Boarded unserved passenger needs dropoff
                dest_floor = self.static_destins.get(p)
                if dest_floor:
                    service_floors.add(dest_floor)
                    boarded_unserved_count += 1
            else:
                # Waiting unserved passenger needs pickup and dropoff
                origin_floor = self.static_origins.get(p)
                dest_floor = self.static_destins.get(p)
                if origin_floor:
                    service_floors.add(origin_floor)
                    waiting_unserved_count += 1
                if dest_floor:
                     service_floors.add(dest_floor)


        # 4. Calculate Base Heuristic (Passenger Actions)
        # Each waiting passenger needs board (1) + depart (1) = 2 actions.
        # Each boarded passenger needs depart (1) action.
        base_h = (waiting_unserved_count * 2) + (boarded_unserved_count * 1)


        # 5. Calculate Movement Heuristic
        movement_cost = 0
        if service_floors:
            service_indices = {self.floor_to_index.get(f, -1) for f in service_floors if f in self.floor_to_index}
            if -1 in service_indices:
                 # Handle error: a required floor was not found in the floor map
                 return float('inf')

            min_req_idx = min(service_indices)
            max_req_idx = max(service_indices)

            # Movement cost: distance from current to max, plus distance from max to min
            # This models going to the top-most required floor and sweeping down.
            movement_cost = abs(current_idx - max_req_idx) + (max_req_idx - min_req_idx)


        # 6. Sum Components
        h_value = base_h + movement_cost

        # Ensure heuristic is 0 at goal
        if len(unserved_passengers) == 0:
             return 0

        return h_value
