from heuristics.heuristic_base import Heuristic
from task import Task

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all passengers
    served) by summing two components:
    1.  The estimated number of board/depart actions required for unserved
        passengers.
    2.  The estimated number of lift movement actions required to visit all
        necessary floors (origins for unboarded passengers, destinations for
        unserved passengers).

    Assumptions:
    -   The PDDL domain follows the structure of the miconic domain provided.
    -   The 'above' predicates define a linear ordering of floors (a single tower).
    -   The 'destin' facts are static and available in task.facts (or task.static).
    -   The 'origin' facts for unboarded passengers are present in the state.
    -   The 'boarded' facts for boarded passengers are present in the state.
    -   The 'served' facts indicate completed passengers.
    -   The 'lift-at' fact indicates the current lift location.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes static information from
    task.facts:
    -   It identifies all passengers and floors in the problem by iterating
        through all possible ground facts.
    -   It stores the destination floor for each passenger by parsing 'destin' facts.
    -   It builds an ordered list of floors and a mapping from floor name to
        its index (level) based on the 'above' predicates. This allows quick
        calculation of floor differences. It assumes a single bottom floor
        (a floor not appearing as the upper floor in any 'above' relation)
        to start building the ordered list.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Check if the state is a goal state (all goals from task.goals are in the state). If yes, return 0.
    2.  Identify the current floor of the lift by finding the '(lift-at ?f)' fact in the state.
    3.  Initialize counters for unboarded passengers (N_unboarded) and boarded
        unserved passengers (N_boarded_unserved).
    4.  Initialize sets for floors that need to be visited: pickup_floors
        (origins of unboarded passengers) and dropoff_floors (destinations of
        unserved passengers).
    5.  Iterate through all known passengers in the problem (identified during initialization):
        a.  Check if the passenger is served by looking for '(served ?p)' in the state.
        b.  If the passenger is NOT served:
            i.  Check if the passenger is at their origin floor by iterating
                through state facts to find one starting with '(origin ?p '.
            ii. If the passenger is at their origin: Increment N_unboarded. Extract
                their origin floor from the fact and add it to pickup_floors.
                Look up their destination floor (from pre-processed static info)
                and add it to dropoff_floors.
            iii. If the passenger is NOT at their origin (and not served), they
                 must be boarded. Check for '(boarded ?p)' in the state. If true:
                 Increment N_boarded_unserved. Look up their destination floor
                 and add it to dropoff_floors.
    6.  Calculate the estimated cost for board/depart actions:
        board_depart_cost = (2 * N_unboarded) + N_boarded_unserved.
        (Each unboarded passenger needs a board action and a depart action;
         each boarded unserved passenger needs a depart action).
    7.  Calculate the estimated cost for lift movement:
        a.  Combine pickup_floors and dropoff_floors into a set of all
            required_floors to visit.
        b.  If required_floors is empty, movement_cost is 0.
        c.  Otherwise, find the minimum and maximum floor indices among the
            current lift floor and all floors in required_floors using the
            pre-calculated floor_to_index map.
        d.  The movement_cost is the difference between the maximum and minimum
            indices (representing the vertical range the lift must cover).
    8.  The total heuristic value is the sum of board_depart_cost and movement_cost.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals

        # --- Heuristic Initialization ---
        self.all_passengers = set()
        self.all_floors = set()
        self.destinations = {}
        above_map = {}  # f_below -> f_above

        # Parse all possible ground facts to find objects and static relations
        for fact_str in task.facts:
            parts = fact_str[1:-1].split()
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'origin' or predicate == 'destin':
                if len(args) == 2:
                    self.all_passengers.add(args[0])
                    self.all_floors.add(args[1])
                    if predicate == 'destin':
                        self.destinations[args[0]] = args[1]
            elif predicate == 'above':
                 if len(args) == 2:
                    self.all_floors.add(args[0])
                    self.all_floors.add(args[1])
                    above_map[args[1]] = args[0]  # args[1] is below args[0]
            elif predicate == 'lift-at':
                 if len(args) == 1:
                    self.all_floors.add(args[0])
            # 'boarded' and 'served' only involve passengers, already covered

        # Build floor list and index map based on 'above' relations
        floors_that_are_above_others = set(above_map.values())
        bottom_floors = self.all_floors - floors_that_are_above_others

        self.floor_list = []
        if len(bottom_floors) == 1:
             bottom_floor = bottom_floors.pop()
             self._build_floor_list_from_bottom(bottom_floor, above_map)
        elif self.all_floors:
             # Handle cases with potentially multiple bottom floors or complex 'above'
             # For robustness, sort floors and try to build from the first one not above another
             sorted_floors = sorted(list(self.all_floors))
             for floor in sorted_floors:
                 if floor not in floors_that_are_above_others:
                     self._build_floor_list_from_bottom(floor, above_map)
                     break # Assume we found the intended bottom and built the main tower
             # If floor_list is still empty, handle error or use sorted list as fallback
             if not self.floor_list:
                  self.floor_list = sorted_floors # Fallback: alphabetical order

        self.floor_to_index = {f: i for i, f in enumerate(self.floor_list)}


    def _build_floor_list_from_bottom(self, bottom_floor, above_map):
        """Helper to build the ordered floor list starting from the bottom."""
        self.floor_list = [bottom_floor]
        current = bottom_floor
        # Use a set to detect cycles
        visited_floors = {bottom_floor}
        while current in above_map:
            next_floor = above_map[current]
            # Prevent infinite loops in case of cycles in 'above' relations (invalid domain)
            if next_floor in visited_floors:
                 # Cycle detected, stop building this chain
                 break
            self.floor_list.append(next_floor)
            visited_floors.add(next_floor)
            current = next_floor


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 2. Identify current lift floor
        current_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                parts = fact_str[1:-1].split()
                if len(parts) == 2:
                    current_floor = parts[1]
                    break

        if current_floor is None or current_floor not in self.floor_to_index:
             # Lift location not found or is an unknown floor - indicates invalid state
             return float('inf')

        # 3. Initialize counters and sets
        N_unboarded = 0
        N_boarded_unserved = 0
        pickup_floors = set()
        dropoff_floors = set()

        # 4. Iterate through all passengers
        # Use state as a set for efficient membership testing
        state_set = set(state)

        for p in self.all_passengers:
            p_served_fact = '(served {})'.format(p)

            # a. Check if the passenger is served
            if p_served_fact not in state_set:
                # Passenger is NOT served

                # i. Check if the passenger is at their origin floor
                p_origin_fact_str = None
                f_origin = None
                # Iterate through state facts starting with '(origin p '
                for fact_str in state:
                    if fact_str.startswith(f'(origin {p} '):
                        parts = fact_str[1:-1].split()
                        if len(parts) == 3:
                            p_origin_fact_str = fact_str
                            f_origin = parts[2]
                            break

                # ii. If the passenger is at their origin
                if p_origin_fact_str:
                    N_unboarded += 1
                    if f_origin in self.all_floors: # Ensure origin floor is known
                        pickup_floors.add(f_origin)
                    # Add destination to dropoff_floors
                    if p in self.destinations and self.destinations[p] in self.all_floors:
                        dropoff_floors.add(self.destinations[p])

                # iii. If the passenger is NOT at their origin (and not served), check if boarded
                else:
                    p_boarded_fact = '(boarded {})'.format(p)
                    if p_boarded_fact in state_set:
                        N_boarded_unserved += 1
                        # Add destination to dropoff_floors
                        if p in self.destinations and self.destinations[p] in self.all_floors:
                            dropoff_floors.add(self.destinations[p])

        # 6. Calculate board/depart cost
        board_depart_cost = (2 * N_unboarded) + N_boarded_unserved

        # 7. Calculate movement cost
        all_required_floors = pickup_floors.union(dropoff_floors)

        if not all_required_floors:
            # This case should only happen if all passengers are served (handled by goal check)
            # or if there are unserved passengers but no required stops (invalid problem state)
            # If unserved passengers exist but no required floors, movement cost is 0
            # but board/depart cost will be > 0.
            movement_cost = 0
        else:
            required_floor_indices = {self.floor_to_index[f] for f in all_required_floors if f in self.floor_to_index}

            # Add current floor index to the set for min/max calculation
            all_indices_to_consider = required_floor_indices.union({self.floor_to_index[current_floor]})

            if not all_indices_to_consider: # Should not happen if all_required_floors is not empty and current_floor is valid
                 movement_cost = 0
            else:
                 min_idx = min(all_indices_to_consider)
                 max_idx = max(all_indices_to_consider)
                 movement_cost = max_idx - min_idx

        # 8. Total heuristic value
        h_value = board_depart_cost + movement_cost

        return h_value
