from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError


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
    # Ensure the number of parts is at least the number of args,
    # and match up to the number of args provided.
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the number of necessary actions (board, depart, and move)
    to serve all passengers. It sums the required board and depart actions for
    unserved passengers and adds an estimate of the minimum lift movement cost
    to visit all floors where passengers need to be picked up or dropped off.

    # Assumptions
    - Floors are linearly ordered, defined by the `(above f_lower f_higher)` facts.
    - The lift can carry any number of passengers.
    - The cost of each action (board, depart, move) is 1.
    - The lift movement cost is estimated by the distance required to travel
      from the current floor to the span of necessary floors (origins of waiting
      passengers and destinations of boarded passengers) and traverse that span.

    # Heuristic Initialization
    - Parses the `(above f_lower f_higher)` facts from static information to
      build a mapping from floor names to numerical indices, representing the
      floor order.
    - Parses the `(destin ?p ?f)` facts from static information to store the
      destination floor for each passenger.
    - Identifies all relevant passengers from the goal conditions (`(served ?p)`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current State Information:**
        - Find the current floor of the lift.
        - Determine which passengers are waiting at their origin (`(origin ?p ?f)`).
        - Determine which passengers are currently boarded (`(boarded ?p)`).
        - Determine which passengers have been served (`(served ?p)`).

    2.  **Calculate Non-Move Actions:**
        - Count the number of passengers who are currently waiting at their origin. Each of these needs a `board` action.
        - Count the total number of passengers who are *not* yet served. Each of these needs a `depart` action eventually.
        - The total non-move cost is the sum of these two counts.

    3.  **Identify Required Floors for Lift Visits:**
        - Collect the origin floor for every passenger currently waiting at their origin.
        - Collect the destination floor for every passenger currently boarded in the lift (using the pre-calculated destination map).
        - The set of required floors is the union of these two sets.

    4.  **Calculate Move Actions:**
        - If there are no required floors, the move cost is 0.
        - Otherwise, find the numerical indices for all required floors using the pre-calculated floor index map.
        - Determine the minimum (`min_idx`) and maximum (`max_idx`) floor indices among the required floors.
        - Get the numerical index for the current lift floor (`current_f_idx`).
        - The estimated minimum move cost is the distance required to travel from the current floor to reach the span of required floors (`[min_idx, max_idx]`) plus the distance to traverse that span. This is calculated as `(max_idx - min_idx) + min(abs(current_f_idx - min_idx), abs(current_f_idx - max_idx))`.

    5.  **Sum Costs:**
        - The total heuristic value is the sum of the non_move_cost and the move_cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of all passengers from the task definition.
        """
        self.task = task
        static_facts = task.static

        # 1. Build floor order and indices from (above f_lower f_higher) facts
        above_map = {}
        all_floors = set()
        floors_as_second_arg = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_lower, f_higher = get_parts(fact)
                above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_as_second_arg.add(f_higher)

        # Find the lowest floor (appears as first arg in 'above' but never as second)
        lowest_floor = None
        potential_lowest = [f for f in all_floors if f not in floors_as_second_arg]
        if len(potential_lowest) == 1:
            lowest_floor = potential_lowest[0]
        elif len(potential_lowest) > 1:
             # Should not happen in standard miconic, pick one
             lowest_floor = potential_lowest[0]
        # If potential_lowest is empty, all floors are part of a cycle or there are no floors.
        # Assume linear structure from problem description.

        self.floor_indices = {}
        if lowest_floor:
            current_floor = lowest_floor
            index = 1
            while current_floor is not None:
                self.floor_indices[current_floor] = index
                current_floor = above_map.get(current_floor)
                index += 1
        # else: No floors found or non-linear structure. Heuristic might fail or return inf.

        # 2. Store passenger destinations from (destin ?p ?f) facts
        self.destinations = {}
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # Also get passengers from goal facts if not already found (e.g., if goal is just (served p))
        for goal in task.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        current_lift_floor = None
        num_waiting = 0
        num_served = 0
        pickup_floors = set()
        dropoff_floors = set()

        # 1. Identify Current State Information
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
            elif match(fact, "served", "*"):
                num_served += 1

        # Ensure current_lift_floor was found (should always be in a valid state)
        if current_lift_floor is None:
             # print("Warning: Lift location not found in state. Returning infinity.")
             return float('inf')

        # Iterate through all known passengers to find their state
        unserved_passengers = set()
        for passenger in self.all_passengers:
            if f"(served {passenger})" not in state:
                unserved_passengers.add(passenger)
                # Check if waiting at origin
                found_origin = False
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, _, origin_floor = get_parts(fact)
                        num_waiting += 1
                        pickup_floors.add(origin_floor)
                        found_origin = True
                        break # A passenger should only have one origin fact

                # If not waiting and not served, they must be boarded
                if not found_origin and f"(boarded {passenger})" in state:
                     # Passenger is boarded, needs to go to destination
                     if passenger in self.destinations: # Should always be true for relevant passengers
                         dropoff_floors.add(self.destinations[passenger])
                     # else: problem definition error? Assume valid problem.

        # 2. Calculate Non-Move Actions
        # Each waiting passenger needs a board action.
        # Each unserved passenger needs a depart action.
        # Total non-move cost = (count of waiting passengers) + (count of unserved passengers)
        non_move_cost = num_waiting + len(unserved_passengers)

        # 3. Identify all required floors (origins for waiting + destinations for boarded)
        required_floors = pickup_floors | dropoff_floors

        # 4. Calculate Move Actions
        move_cost = 0
        if required_floors:
            # Get indices for required floors
            try:
                req_indices = sorted([self.floor_indices[f] for f in required_floors])
                current_f_idx = self.floor_indices[current_lift_floor]

                min_idx = req_indices[0]
                max_idx = req_indices[-1]

                # Estimate move cost: travel to the span + traverse the span
                # min(dist from current to min_idx, dist from current to max_idx) + span
                move_cost = (max_idx - min_idx) + min(abs(current_f_idx - min_idx), abs(current_f_idx - max_idx))

            except KeyError as e:
                 # This might happen if a floor in state/goals wasn't in static 'above' facts
                 # or if current_lift_floor is missing. Should not happen in valid PDDL.
                 # Return infinity or a large value to prune this state.
                 # print(f"Warning: Floor index not found for {e}. Returning infinity.")
                 return float('inf')


        # 5. Sum Costs
        total_cost = non_move_cost + move_cost

        return total_cost
