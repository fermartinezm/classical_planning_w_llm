# Add necessary imports
from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
# If the path is different, this import needs adjustment.
# Based on the example code, this path seems standard.
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Example: '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    # Handles potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining effort by summing up the individual costs
        for each unserved passenger. The cost for a passenger depends on
        whether they are waiting at their origin or are boarded in the lift.
        The cost includes travel time (number of floors) and the board/depart
        actions (each counted as 1 action). Travel time is calculated based
        on the absolute difference in floor indices, derived from the 'above'
        predicates. This heuristic is non-admissible as it sums costs
        independently for passengers, potentially double-counting lift travel.

    Assumptions:
        - The 'above' predicates define a total linear order of floors.
        - The state representation includes dynamic facts like (lift-at),
          (origin), (boarded), (served).
        - Static facts like (destin) and (above) are available in task.static.
        - The cost of moving one floor (up or down), boarding, and departing
          is considered 1 action each for simplicity in the heuristic.
        - Every passenger relevant to the goal has a defined destination
          in the static facts.
        - The PDDL instance is valid and describes a connected set of floors
          with a clear highest and lowest floor via 'above' predicates.

    Heuristic Initialization:
        1. Parses the 'above' predicates from static facts to determine the
           linear order of floors. It builds a mapping from floor name to
           a numerical index (0-based). This is done by finding the highest
           floor (one that is not immediately below any other floor) and
           then traversing downwards using the 'above' relationships.
        2. Parses the 'destin' predicates from static facts to store the
           destination floor for each passenger in a dictionary.
        3. Collects the names of all passengers involved in the problem
           (those with a defined destination) from the keys of the destinations dictionary.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state by comparing the state
           with the task's goal set. If the goal is a subset of the state,
           the heuristic is 0.
        2. Identify the current floor of the lift by finding the fact
           '(lift-at ?floor)' in the current state. If not found (should only
           happen in goal state, which is handled), return infinity.
        3. Initialize total_cost to 0.
        4. Iterate through each passenger collected during initialization.
        5. For the current passenger:
           a. Check if the fact '(served <passenger_name>)' is present in the
              current state. If yes, this passenger is served, and their
              contribution to the heuristic is 0. Continue to the next passenger.
           b. If the passenger is not served, retrieve their destination floor
              from the pre-computed destinations map. If the destination is not
              found (invalid problem), return infinity.
           c. Check if the fact '(boarded <passenger_name>)' is present in the
              current state.
              - If yes: The passenger is currently inside the lift. The remaining
                estimated cost for this passenger is the number of floors the
                lift needs to travel from its current floor to the passenger's
                destination floor (calculated using the floor index distance),
                plus 1 action for the 'depart' operation.
                Add this calculated cost to the total_cost.
              - If no: The passenger must be waiting at their origin floor. Find
                the fact '(origin <passenger_name> ?floor)' in the current state
                to determine their origin floor. If the origin is not found
                (invalid state), return infinity. The remaining estimated cost for
                this passenger is the number of floors the lift needs to travel
                from its current floor to the passenger's origin floor (to pick
                them up), plus 1 action for the 'board' operation, plus the
                number of floors the lift needs to travel from the origin floor
                to the destination floor (to drop them off), plus 1 action for
                the 'depart' operation. Add this calculated cost to the total_cost.
        6. Return the final calculated total_cost as the heuristic value for the state.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor order and index mapping
        # floor_below[f_above] = f_below if (above f_above f_below) is true (f_above is immediately above f_below)
        floor_below = {}
        all_floors = set()
        floors_that_are_below_something = set() # Floors that appear as the second argument in (above f_i f_j)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_above, f_below = parts[1:]
                floor_below[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_that_are_below_something.add(f_below)

        # Find the highest floor (a floor in all_floors but not in floors_that_are_below_something)
        # Assumes there is at least one floor and a linear ordering
        highest_floor = None
        potential_highest_floors = all_floors - floors_that_are_below_something

        if len(potential_highest_floors) == 1:
             highest_floor = potential_highest_floors.pop()
        elif len(all_floors) == 1: # Case with only one floor
             highest_floor = list(all_floors)[0]
        # else: # Indicates invalid or non-linear floor structure or empty domain.

        self.floor_to_index = {}
        if highest_floor:
            current_floor = highest_floor
            index = 0
            # Traverse downwards using the floor_below map
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                current_floor = floor_below.get(current_floor)
                index += 1
        # If highest_floor is None or traversal didn't cover all_floors,
        # self.floor_to_index might be incomplete or empty.
        # The distance function handles floors not in the map by returning inf.


        # 2. Build destinations mapping and collect passengers
        self.destinations = {}
        self.passengers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                p, d_floor = parts[1:]
                self.destinations[p] = d_floor
                self.passengers.add(p)

    def distance(self, floor1, floor2):
        """Calculates the distance (number of floors) between two floors."""
        # Return a large value if floor not in mapping (invalid state/domain)
        if floor1 not in self.floor_to_index or floor2 not in self.floor_to_index:
             # This indicates an issue with the state or domain definition
             # Returning inf is appropriate as this path is likely invalid
             return float('inf')

        return abs(self.floor_to_index[floor1] - self.floor_to_index[floor2])

    def __call__(self, node):
        state = node.state

        # 1. Check for goal state
        # The goal is (and (served p1) (served p2) ...).
        # Check if all goal facts are in the current state.
        if self.goals <= state:
             return 0

        # 2. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break
        # If lift location is not found in a non-goal state, something is wrong.
        if current_lift_floor is None:
             return float('inf') # Should not happen in a reachable non-goal state

        # 3. Initialize total cost
        total_cost = 0

        # 4. Iterate through each passenger relevant to the goal
        for passenger in self.passengers:
            # 5a. Check if served
            if f'(served {passenger})' in state:
                continue # Passenger is served, cost is 0 for this one

            # 5b. Get destination
            # Assumes every passenger in self.passengers has a destination
            destin_f = self.destinations.get(passenger)
            if destin_f is None:
                 # Passenger exists but has no destination? Invalid domain/problem.
                 return float('inf') # Invalid problem state

            # 5c. Check if boarded or waiting
            if f'(boarded {passenger})' in state:
                # Passenger is boarded
                # Cost = travel to destination + depart
                travel_cost = self.distance(current_lift_floor, destin_f)
                if travel_cost == float('inf'): return float('inf') # Propagate infinity
                total_cost += travel_cost + 1 # +1 for depart action
            else:
                # Passenger is waiting at origin
                # Find origin floor
                origin_f = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "origin" and len(parts) == 3 and parts[1] == passenger:
                        origin_f = parts[2]
                        break

                # If not served and not boarded, they must be at origin.
                # If origin is not found, state is invalid.
                if origin_f is None:
                     return float('inf') # Invalid state

                # Cost = travel to origin + board + travel to destination + depart
                travel_to_origin_cost = self.distance(current_lift_floor, origin_f)
                if travel_to_origin_cost == float('inf'): return float('inf') # Propagate infinity
                travel_to_dest_cost = self.distance(origin_f, destin_f)
                if travel_to_dest_cost == float('inf'): return float('inf') # Propagate infinity

                total_cost += travel_to_origin_cost + 1 + travel_to_dest_cost + 1 # +1 for board, +1 for depart

        # 6. Return total cost
        return total_cost
