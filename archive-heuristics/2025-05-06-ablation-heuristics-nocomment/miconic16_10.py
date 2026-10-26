from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have the same number of parts and args for matching
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the remaining cost to reach the goal (all passengers served)
    by summing the estimated costs for each unserved passenger independently.
    The cost for a passenger includes the estimated movement of the lift
    to their origin floor (if not boarded), the 'board' action, the estimated
    movement to their destination floor, and the 'depart' action. Movement
    cost between floors is estimated by the absolute difference in floor indices,
    derived from the floor ordering.

    Assumptions:
    - The floor names allow mapping to a linear order (e.g., f1, f2, ...).
    - The 'above' predicate in static facts defines a total order on floors,
      where `(above f_higher f_lower)` means `f_higher` is anywhere above `f_lower`.
    - The 'above' predicate in action preconditions `(above ?f1 ?f2)` means
      `?f1` is immediately above `?f2` for the 'up' action (moving from `?f1` to `?f2`),
      and `?f2` is immediately above `?f1` for the 'down' action (moving from `?f1` to `?f2`).
      Note: The action names 'up' and 'down' appear counter-intuitive based on standard
      PDDL conventions and the `above` predicate definition, but the heuristic
      calculates movement steps based on the derived floor order, which is consistent
      with the number of 'up'/'down' actions required regardless of their names.
    - The cost of 'board', 'depart', 'up', and 'down' actions is 1.
    - The distance between two floors is the number of immediate steps between them,
      equal to the absolute difference in their indices in the ordered list of floors.
    - The heuristic calculates the sum of costs for serving each passenger
      independently, ignoring potential optimizations from batching passengers
      in the lift. This makes it non-admissible but potentially effective for GBFS.
    - Passenger origins and destinations are available in the initial state or static facts
      via the `(origin ?p ?f)` and `(destin ?p ?f)` predicates.

    Heuristic Initialization:
    - Parses all floors from static 'above' predicates.
    - Determines the 'immediately above' relation from the transitive 'above' relation.
    - Builds an ordered list of floors from lowest to highest using the
      'immediately above' relation.
    - Creates a mapping from floor names to integer indices based on this order.
    - Stores passenger origins by parsing `(origin ?p ?f)` facts from
      the initial state and static facts.
    - Stores passenger destinations by parsing `(destin ?p ?f)` facts from
      the initial state and static facts.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for origins/destins

        # 1. Get all floors and transitive above pairs
        all_floors = set()
        above_pairs = set() # Stores (f_higher, f_lower)
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                above_pairs.add((f_higher, f_lower))
        self.all_floors = list(all_floors) # Keep a list for iterating

        # 2. Determine 'immediately above' relation
        direct_above = set() # Stores (f_immediately_higher, f_lower)
        for f_i, f_j in above_pairs:
            is_direct = True
            for f_k in self.all_floors:
                if f_k != f_i and f_k != f_j:
                    # Check if f_k is between f_i and f_j transitively
                    if (f_i, f_k) in above_pairs and (f_k, f_j) in above_pairs:
                        is_direct = False
                        break
            if is_direct:
                direct_above.add((f_i, f_j))

        # 3. Build ordered list of floors from lowest to highest
        ordered_floors = []
        if self.all_floors:
            # Find the lowest floor: a floor f such that no (f_higher, f) is in direct_above
            is_lower_in_direct_pair = {f_lower for _, f_lower in direct_above}
            lowest_floor = None
            # Find a floor that is not the lower part of any direct_above pair
            potential_lowest = all_floors - is_lower_in_direct_pair
            if len(potential_lowest) == 1:
                 lowest_floor = list(potential_lowest)[0]
            elif len(self.all_floors) == 1:
                 lowest_floor = list(self.all_floors)[0]
            else:
                 # Fallback: Find floor that is above the fewest others (transitive)
                 # This handles cases where the 'above' facts might not perfectly define a single chain
                 above_counts = {}
                 for f1 in self.all_floors:
                     above_counts[f1] = sum(1 for f2 in self.all_floors if f1 != f2 and (f1, f2) in above_pairs)
                 # The floor above the fewest others is the lowest
                 lowest_floor = min(self.all_floors, key=lambda f: above_counts.get(f, 0)) # Use 0 if floor not in above_counts (e.g. single floor)


            if lowest_floor:
                ordered_floors.append(lowest_floor)
                current = lowest_floor
                # Build map from lower floor to immediately higher floor
                immediately_above_map = {f_lower: f_higher for f_higher, f_lower in direct_above}
                while current in immediately_above_map:
                    current = immediately_above_map[current]
                    ordered_floors.append(current)
            # else: ordered_floors remains empty if no floors or lowest floor not found

        self.ordered_floors = ordered_floors
        self.floor_to_int = {floor: i for i, floor in enumerate(self.ordered_floors)}

        # 4. Store passenger origins and destinations
        self.passenger_origin = {}
        self.passenger_destin = {}
        # Origins and destinations are typically in the initial state or static facts
        for fact in initial_state.union(static_facts):
             parts = get_parts(fact)
             if parts[0] == "origin":
                 p, f = parts[1:]
                 self.passenger_origin[p] = f
             elif parts[0] == "destin":
                 p, f = parts[1:]
                 self.passenger_destin[p] = f

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current location of the lift by finding the `(lift-at ?f)` fact.
        2. Identify the set of passengers who are currently 'served' and 'boarded'
           by checking for `(served ?p)` and `(boarded ?p)` facts in the current state.
        3. Initialize the total heuristic cost to 0.
        4. Iterate through all passengers known to the problem (those with a destination
           recorded in `self.passenger_destin`):
           - If the passenger is in the set of 'served' passengers, their cost contribution is 0.
           - If the passenger is in the set of 'boarded' passengers (and not served):
             - Add 1 to the cost (for the 'depart' action).
             - Get the passenger's destination floor from `self.passenger_destin`.
             - Add the estimated movement cost for the lift to travel from its
               current floor to the passenger's destination floor. This is calculated
               as the absolute difference in floor indices using the pre-computed mapping.
           - If the passenger is neither 'served' nor 'boarded' (they must be at their origin):
             - Add 1 to the cost (for the 'board' action).
             - Add 1 to the cost (for the 'depart' action).
             - Get the passenger's origin floor from `self.passenger_origin` (pre-parsed in `__init__`).
             - Get the passenger's destination floor from `self.passenger_destin`.
             - Add the estimated movement cost for the lift to travel from its
               current floor to the passenger's origin floor.
             - Add the estimated movement cost for the lift to travel from the
               passenger's origin floor to their destination floor.
        5. The total heuristic value is the sum of these individual passenger costs.
        6. Ensure the heuristic is exactly 0 if and only if the state is a goal state
           (all passengers with destinations are served).
        """
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # Identify served and boarded passengers in the current state
        served_passengers_in_state = set()
        boarded_passengers_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_passengers_in_state.add(parts[1])
            elif parts[0] == 'boarded':
                boarded_passengers_in_state.add(parts[1])

        total_cost = 0

        # Iterate through all passengers whose destination is known (all passengers in the problem)
        for p, dest_floor in self.passenger_destin.items():
            if p in served_passengers_in_state:
                continue # Passenger is served, cost is 0

            # Passenger is not served. Check if boarded.
            if p in boarded_passengers_in_state:
                # Passenger is boarded, needs to reach destination and depart
                # Cost = Move(current_lift_floor, dest_floor) + Depart
                if current_lift_floor in self.floor_to_int and dest_floor in self.floor_to_int:
                    total_cost += abs(self.floor_to_int[current_lift_floor] - self.floor_to_int[dest_floor]) + 1
                # else: Assume valid floors are always in self.floor_to_int

            else: # Passenger is not served and not boarded => must be at origin
                # Passenger is at origin, needs board and depart
                # Cost = Move(current_lift_floor, origin_floor) + Board + Move(origin_floor, dest_floor) + Depart
                origin_floor = self.passenger_origin.get(p) # Get origin from pre-parsed static info
                if origin_floor and current_lift_floor in self.floor_to_int and origin_floor in self.floor_to_int and dest_floor in self.floor_to_int:
                     total_cost += abs(self.floor_to_int[current_lift_floor] - self.floor_to_int[origin_floor]) + 1 + \
                                   abs(self.floor_to_int[origin_floor] - self.floor_to_int[dest_floor]) + 1
                 # else: Assume valid floors/origins are always in maps

        # The check for goal state (h=0 iff goal) is implicitly handled by the loop.
        # If all passengers are served, the loop body is skipped for all passengers, total_cost remains 0.
        # If at least one passenger is not served, they fall into the 'boarded' or 'else' branch,
        # adding at least 1 (for board/depart) to total_cost, making it > 0.
        return total_cost
