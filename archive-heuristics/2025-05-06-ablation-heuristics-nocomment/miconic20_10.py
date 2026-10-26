from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts represented as strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Example: '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern of predicate and arguments."""
    # Example: match('(at obj room)', 'at', '*', 'room') -> True
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining cost by summing up the individual costs
        for each unserved passenger. The cost for a passenger depends on
        whether they are waiting at their origin or are already boarded.
        Travel cost is estimated as the absolute difference in floor indices.
        This heuristic is non-admissible as it sums independent costs which
        might double-count lift travel between floors needed by multiple passengers.
        It aims to guide a greedy best-first search by prioritizing states
        where passengers are closer to being served, either by being boarded
        or by the lift being closer to their pickup/dropoff floor.

    Assumptions:
        - Floors are named 'f<number>' and can be ordered numerically.
        - The 'above' predicates in static facts define the linear order of floors.
        - The goal is to have all specified passengers served.
        - In any valid state, a passenger is either at their origin, boarded, or served.
        - In any valid non-goal state, the lift location is specified by '(lift-at ?f)'.

    Heuristic Initialization:
        - Stores the task goals (`self.goals`) and static facts (`self.static`)
          from the `task` object provided to the constructor (via `super().__init__`).
        - Parses static facts to determine each passenger's destination floor,
          storing this mapping in `self.destinations`.
        - Parses static facts (specifically 'above' predicates) to identify all
          floor names present in the problem.
        - Sorts the identified floor names numerically based on the number part
          (e.g., 'f1', 'f2', 'f10' are sorted as f1, f2, f10).
        - Creates a mapping from floor names to their numerical index in the
          sorted list (`self.floor_to_index`) and the reverse mapping
          (`self.index_to_floor`). These mappings are used to calculate floor distances.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the search node (`node.state`). The state
           is a frozenset of PDDL fact strings.
        2. Find the current floor of the lift by iterating through the state facts
           and identifying the fact matching the pattern '(lift-at ?f)'. Extract
           the floor name from this fact.
        3. If the lift location (`current_lift_floor`) is not found in the state:
           Check if the current state is a goal state by verifying if all facts
           in `self.goals` are present in the state. If it is a goal state,
           return a heuristic value of 0. If it is not a goal state and the
           lift location is missing, the state is considered invalid for this
           domain; return infinity (`float('inf')`) to prune this state.
        4. Initialize the total heuristic cost (`total_cost`) to 0. This variable
           will accumulate the estimated costs for all unserved passengers.
        5. Identify all passengers who need to be served according to the task goals.
           These are the passengers `?p` for whom `(served ?p)` is a fact in `self.goals`.
        6. Iterate through each passenger identified in step 5.
        7. For the current passenger:
           a. Check if the passenger is already served by looking for the fact
              '(served <passenger_name>)' in the current state. If this fact is
              found, the passenger is served, and the remaining cost for this
              passenger is 0; continue to the next passenger in the loop.
           b. If the passenger is not served, retrieve their destination floor
              (`f_destin`) from the `self.destinations` map (which was populated
              during initialization from static facts). If the destination is
              not found for a passenger listed in the goals, this indicates an
              invalid problem definition; return infinity.
           c. Check if the passenger is currently boarded by looking for the fact
              '(boarded <passenger_name>)' in the current state.
           d. If the passenger is boarded (`is_boarded` is True):
              i. The remaining steps for this passenger involve moving the lift
                 from its current floor (`current_lift_floor`) to their destination
                 floor (`f_destin`) and then performing the 'depart' action.
              ii. Calculate the estimated travel cost as the absolute difference
                  between the index of the current lift floor and the index of
                  the destination floor using the `self.dist()` helper method.
              iii. Add 1 for the 'depart' action itself.
              iv. The estimated cost for this passenger is `travel_cost + 1`.
           e. If the passenger is not boarded (`is_boarded` is False): By the domain
              rules, if a passenger is not served and not boarded, they must be
              waiting at their origin floor.
              i. Find the passenger's origin floor (`f_origin`) by iterating
                 through the state facts and identifying the fact matching the
                 pattern '(origin <passenger_name> ?f)'. Extract the origin floor
                 name from this fact. If the origin fact is not found for an
                 unserved, unboarded passenger, the state is invalid; return infinity.
              ii. The remaining steps are to travel from the current lift floor
                  to the origin floor, perform the 'board' action, travel from
                  the origin floor to the destination floor, and perform the
                  'depart' action.
              iii. Calculate the estimated travel cost 1 (from `current_lift_floor`
                  to `f_origin`) using `self.dist()`.
              iv. Add 1 for the 'board' action.
              v. Calculate the estimated travel cost 2 (from `f_origin` to
                  `f_destin`) using `self.dist()`.
              vi. Add 1 for the 'depart' action.
              vii. The estimated cost for this passenger is `travel_cost_1 + 1 + travel_cost_2 + 1`.
           f. Add the calculated cost for the current passenger to the `total_cost`.
        8. After iterating through all passengers who need to be served, the final
           `total_cost` represents the heuristic estimate for the current state.
           Return `total_cost`.
    """
    def __init__(self, task):
        # The base class Heuristic is assumed to store task.goals and task.static
        # in self.goals and self.static respectively.
        super().__init__(task)

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Extract floor ordering from static facts
        floor_names = set()
        # Collect all floors mentioned in 'above' facts
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                floor_names.add(f1)
                floor_names.add(f2)

        # Sort floor names numerically (assuming f1, f2, f10, etc. naming)
        # This requires a custom sort key to handle multi-digit numbers correctly
        def floor_sort_key(floor_name):
            # Extract the number part, handle potential errors
            try:
                # Assumes floor names are like 'f1', 'f10', 'f2'
                return int(floor_name[1:])
            except (ValueError, IndexError):
                # Fallback for unexpected names, though problem description implies f<number>
                # If names are not f<number>, this heuristic might not work correctly
                # regarding floor distance, but we provide a fallback sort key.
                return floor_name # Use string itself for sorting

        sorted_floor_names = sorted(list(floor_names), key=floor_sort_key)

        self.floor_to_index = {floor: i for i, floor in enumerate(sorted_floor_names)}
        self.index_to_floor = {i: floor for i, floor in enumerate(sorted_floor_names)}

    def dist(self, f1, f2):
        """Calculates the number of floors between f1 and f2."""
        # Distance is the absolute difference in indices
        idx1 = self.floor_to_index.get(f1)
        idx2 = self.floor_to_index.get(f2)

        if idx1 is None or idx2 is None:
            # This indicates a floor name from the state/goals was not found
            # in the static 'above' facts used to build the index map.
            # This shouldn't happen in valid PDDL instances for this domain.
            # Returning infinity indicates an unreachable or invalid state segment.
            return float('inf')

        return abs(idx1 - idx2)

    def __call__(self, node):
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        # Iterate through state facts to find the lift location
        for fact in state:
            if match(fact, "lift-at", "*"):
                # Fact is like '(lift-at f2)'
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is not found, the state is likely invalid or terminal
        if current_lift_floor is None:
             # Check if it's a goal state first by checking if all goals are in the state
             if self.goals <= state:
                 return 0 # Goal state
             else:
                 # Invalid non-goal state (lift-at should always be present otherwise)
                 return float('inf') # Indicate invalid state

        total_cost = 0

        # Identify passengers that need to be served
        # Goals are like '(served p1)', '(served p2)', etc.
        passengers_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        for passenger in passengers_to_serve:
            # Check if passenger is already served
            if f'(served {passenger})' in state:
                continue # Already served, cost is 0 for this passenger

            # Passenger is not served, calculate remaining cost
            f_destin = self.destinations.get(passenger)
            if f_destin is None:
                 # This passenger is in the goal but has no destination in static facts.
                 # This indicates an invalid problem definition.
                 return float('inf') # Indicate invalid problem/state

            is_boarded = f'(boarded {passenger})' in state

            if is_boarded:
                # Passenger is boarded
                # Need to travel from current lift floor to destination and depart
                cost_travel = self.dist(current_lift_floor, f_destin)
                cost_depart = 1
                cost_for_passenger = cost_travel + cost_depart
                total_cost += cost_for_passenger
            else:
                # Passenger is not boarded, must be at origin
                # Need to find origin floor from the current state
                f_origin = None
                # Iterate through state facts to find the origin location for this passenger
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        # Fact is like '(origin p1 f6)'
                        f_origin = get_parts(fact)[2]
                        break

                if f_origin is None:
                    # Passenger is not served, not boarded, and not at an origin.
                    # This state seems invalid based on domain rules.
                    # A passenger is either at origin, boarded, or served.
                    # If not served and not boarded, they must be at origin.
                    return float('inf') # Indicate invalid state

                # Need to travel from current lift floor to origin, board,
                # travel from origin to destination, and depart
                cost_travel_to_origin = self.dist(current_lift_floor, f_origin)
                cost_board = 1
                cost_travel_origin_to_destin = self.dist(f_origin, f_destin)
                cost_depart = 1
                cost_for_passenger = cost_travel_to_origin + cost_board + cost_travel_origin_to_destin + cost_depart
                total_cost += cost_for_passenger

        return total_cost
