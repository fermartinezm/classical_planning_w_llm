from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have enough parts to match the arguments
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal (all passengers served) by summing
    an estimated movement cost for the lift and an estimated action cost
    (boarding and departing). The movement cost is estimated based on the
    vertical distance needed to visit the range of floors where unserved
    passengers are waiting or need to be dropped off. The action cost is
    estimated by counting the number of board and depart actions still
    required for unserved passengers.

    Assumptions:
    - Floor names are structured such that they can be numerically sorted
      (e.g., 'f1', 'f2', ...). This is used as a fallback for determining
      floor order if the '(above ...)' facts don't form a simple chain or
      if the lowest floor isn't uniquely identifiable via graph traversal.
    - The '(above f_low f_high)' predicate indicates that f_high is
      immediately above f_low, defining a linear order of floors.
    - All passengers present in the problem instance have a defined destination
      in the static facts.
    - The heuristic is non-admissible and designed for greedy best-first search
      to minimize node expansions, not guarantee optimal solutions.

    Heuristic Initialization:
    The constructor processes the static facts to build necessary data structures:
    1.  Parses the '(above f_low f_high)' facts to determine the ordered list
        of floors from lowest to highest. This is primarily done by building
        a chain based on the 'above' relationships, starting from the floor
        that is not listed as being "above" any other floor (the lowest).
        As a fallback, if this graph traversal doesn't yield a clear linear
        order or lowest floor, floors are sorted numerically based on their names.
        A mapping from floor name to its index in this ordered list is created,
        along with a reverse mapping from index to name.
    2.  Parses the '(destin p f_destin)' facts to store the destination floor
        for each passenger in a dictionary.
    3.  Identifies all passengers present in the problem instance by collecting
        all passenger names found in the destination facts.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic `h(s)` for a given state `s` is computed as follows:
    1.  Identify the current floor of the lift from the state fact '(lift-at f)'.
    2.  Identify all passengers who are not yet served. This is done by checking
        for the absence of the '(served p)' fact for each passenger in the state.
    3.  If there are no unserved passengers, the state is a goal state, and the
        heuristic value is 0.
    4.  If there are unserved passengers, determine the set of "required floors"
        that the lift must visit:
        -   For each unserved passenger `p` who is currently waiting at their
            origin floor (i.e., '(origin p f_origin)' is in the state), add
            `f_origin` to the set of required floors. Count this passenger
            towards the total number of required 'board' actions.
        -   For each unserved passenger `p` who is currently boarded (i.e.,
            '(boarded p)' is in the state), add their destination floor
            `f_destin` (looked up from the pre-parsed destinations) to the set
            of required floors. Count this passenger towards the total number
            of required 'depart' actions.
    5.  If the set of required floors is empty (meaning all unserved passengers
        are boarded and the lift is already at their destination floors), the
        heuristic is simply the number of boarded unserved passengers (each
        needs one 'depart' action).
    6.  If the set of required floors is not empty, calculate the estimated
        movement cost:
        -   Find the minimum and maximum floor indices among the required floors.
        -   Get the index of the current lift floor.
        -   The movement cost is estimated as the minimum vertical travel needed
            to visit the range of floors from the minimum required floor index
            to the maximum required floor index, starting from the current floor.
            This is calculated as:
            -   If the current floor index is below the minimum required floor index:
                `(min_required_idx - current_idx) + (max_required_idx - min_required_idx)`
                (distance down to the lowest required floor, then distance up to the highest)
            -   If the current floor index is above the maximum required floor index:
                `(current_idx - max_required_idx) + (max_required_idx - min_required_idx)`
                (distance up to the highest required floor, then distance down to the lowest)
            -   If the current floor index is within the range [min_required_idx, max_required_idx]:
                `(max_required_idx - min_required_idx) + min(current_idx - min_required_idx, max_required_idx - current_idx)`
                (distance to traverse the range, plus distance from current floor to the nearest end of the range)
    7.  Calculate the estimated action cost: This is the sum of the number of
        waiting passengers (each needs a 'board' action) and the number of
        boarded unserved passengers (each needs a 'depart' action).
    8.  The total heuristic value is the sum of the estimated movement cost
        and the estimated action cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor order and create index mapping
        above_map = {}
        all_floors_in_above = set()
        floors_that_are_high = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_low, f_high = get_parts(fact)[1:]
                above_map[f_low] = f_high
                all_floors_in_above.add(f_low)
                all_floors_in_above.add(f_high)
                floors_that_are_high.add(f_high)

        lowest_floor = None
        potential_lowest = all_floors_in_above - floors_that_are_high

        if len(potential_lowest) == 1:
            lowest_floor = list(potential_lowest)[0]
        elif len(potential_lowest) > 1:
             # Multiple potential lowest floors - fallback to numerical sort
             try:
                 sorted_floors = sorted(list(all_floors_in_above), key=lambda f: int(re.search(r'\d+', f).group()))
                 lowest_floor = sorted_floors[0]
             except (AttributeError, ValueError):
                  # If names are not fN, pick the first one found in potential_lowest
                  print("Warning: Multiple potential lowest floors found and names not numerical. Picking one arbitrarily.")
                  lowest_floor = list(potential_lowest)[0]
        elif not all_floors_in_above:
             # No above facts. Try to find floors from destinations or initial state lift-at
             # We'll infer floors from destinations for simplicity here.
             # A more robust parser would get all objects of type floor.
             inferred_floors = set(self.destinations.values()) # destinations are parsed below
             if inferred_floors:
                 try:
                     sorted_floors = sorted(list(inferred_floors), key=lambda f: int(re.search(r'\d+', f).group()))
                     lowest_floor = sorted_floors[0]
                 except (AttributeError, ValueError):
                      print("Warning: No 'above' facts and destination names not numerical. Picking one arbitrarily.")
                      lowest_floor = list(inferred_floors)[0]
             # If no floors found at all, floor_order will be empty, handled later.


        self.floor_order = []
        current = lowest_floor
        while current is not None and current in above_map: # Ensure current is in map to avoid infinite loop on bad data
            self.floor_order.append(current)
            current = above_map.get(current)
        # Add the last floor if the loop terminated because current was not in above_map but was the end of the chain
        if current is not None and current not in self.floor_order:
             self.floor_order.append(current)

        # If floor_order is still empty (e.g., single floor problem with no above facts)
        if not self.floor_order and all_floors_in_above:
             self.floor_order = list(all_floors_in_above)
             try:
                 self.floor_order.sort(key=lambda f: int(re.search(r'\d+', f).group()))
             except (AttributeError, ValueError):
                  print("Warning: Could not determine floor order. Using arbitrary order.")
                  self.floor_order.sort() # Fallback to alphabetical sort

        # 2. Parse passenger destinations
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, f_destin = get_parts(fact)[1:]
                self.destinations[p] = f_destin

        # 3. Identify all passengers
        # Assuming all passengers have a destination defined in static facts.
        self.all_passengers = set(self.destinations.keys())

        # Create floor index mapping after floor_order is finalized
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_order)}
        self.index_to_floor = {i: floor for i, floor in enumerate(self.floor_order)}


    def get_floor_index(self, floor_name):
        """Returns the index of a floor in the ordered list."""
        # Return a value that indicates error if floor not found,
        # but for this domain, all floors should be in the list.
        return self.floor_to_index.get(floor_name)

    def distance(self, floor1_name, floor2_name):
        """Calculates the distance between two floors."""
        idx1 = self.get_floor_index(floor1_name)
        idx2 = self.get_floor_index(floor2_name)
        if idx1 is None or idx2 is None:
            # This indicates a floor name was not found in our parsed list
            # Should not happen in valid problems, but handle defensively
            return float('inf')
        return abs(idx1 - idx2)

    def __call__(self, node):
        state = node.state

        # 1. Get current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state
             # Indicates an unrecoverable state or parsing error
             return float('inf')

        current_idx = self.get_floor_index(current_lift_floor)
        if current_idx is None:
             # Current lift floor not found in our floor list - parsing error?
             return float('inf')


        # 2. Identify unserved passengers
        current_served = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_passengers - current_served

        # 3. If no unserved passengers, it's a goal state
        if not unserved_passengers:
            return 0

        # 4. Collect required floors and count actions
        required_floors = set()
        num_waiting = 0
        num_boarded_unserved = 0

        current_origins = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}
        current_boarded = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        for p in unserved_passengers:
            if p in current_origins:
                # Passenger is waiting at origin
                f_origin = current_origins[p]
                required_floors.add(f_origin)
                num_waiting += 1
            elif p in current_boarded:
                # Passenger is boarded, needs dropoff at destination
                f_destin = self.destinations.get(p)
                if f_destin: # Ensure destination exists
                    required_floors.add(f_destin)
                    num_boarded_unserved += 1
                # else: passenger boarded but no destination? Invalid state? Assume valid.
            # else: passenger is unserved but neither waiting nor boarded? Invalid state?

        # 5. Handle case where all unserved are boarded and at destination
        if not required_floors:
             # This implies all unserved passengers are boarded and the lift is
             # currently at their destination floor. They just need to depart.
             # The cost is the number of such passengers.
             return num_boarded_unserved

        # 6. Calculate movement cost
        required_indices = [self.get_floor_index(f) for f in required_floors if self.get_floor_index(f) is not None]
        if not required_indices:
             # Should not happen if required_floors was not empty, but defensive check
             return float('inf') # Or num_boarded_unserved if that's the only case left

        min_required_idx = min(required_indices)
        max_required_idx = max(required_indices)

        movement_cost = 0
        if current_idx < min_required_idx:
            # Need to go down to the lowest required floor, then up to the highest
            movement_cost = (min_required_idx - current_idx) + (max_required_idx - min_required_idx)
        elif current_idx > max_required_idx:
            # Need to go up to the highest required floor, then down to the lowest
            movement_cost = (current_idx - max_required_idx) + (max_required_idx - min_required_idx)
        else: # current_idx is within [min_required_idx, max_required_idx]
            # Need to traverse the range, starting by going to the nearest extreme
            movement_cost = (max_required_idx - min_required_idx) + min(current_idx - min_required_idx, max_required_idx - current_idx)

        # 7. Calculate action cost
        action_cost = num_waiting + num_boarded_unserved

        # 8. Total heuristic
        total_heuristic = movement_cost + action_cost

        return total_heuristic
