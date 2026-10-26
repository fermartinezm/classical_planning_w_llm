from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from .heuristic_base import Heuristic # Or wherever it is located

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all passengers served)
    by summing two components: the estimated number of actions (board/depart)
    and the estimated vertical movement cost for the lift. It is designed for
    greedy best-first search and does not need to be admissible, aiming instead
    for accuracy and efficiency.

    Assumptions:
    - The floors are arranged linearly and ordered by the 'above' predicate,
      forming a single tower.
    - The 'above' predicate defines immediate adjacency (e.g., (above f2 f1)
      means f2 is directly above f1).
    - Passenger origins and destinations are static facts (except for the
      'origin' predicate which is removed upon boarding).

    Heuristic Initialization:
    In the constructor (__init__), the heuristic pre-processes static information
    from the task:
    1. It parses the 'above' facts to determine the linear order of floors
       and creates mappings between floor names and numerical indices. This
       allows efficient calculation of floor distances.
    2. It parses the initial 'origin' and 'destin' facts for all passengers
       to store their required trips.

    Step-By-Step Thinking for Computing Heuristic:
    In the __call__ method, for a given state:
    1. Check if the state is a goal state (all passengers are served). If so,
       the heuristic value is 0.
    2. Identify the current floor of the lift.
    3. Iterate through all passengers to identify those who are not yet served.
    4. For each unserved passenger, determine their current status:
       - Waiting at their origin floor (predicate `(origin p f)` is true).
       - Boarded in the lift (predicate `(boarded p)` is true).
    5. Calculate the estimated number of actions needed:
       - Each waiting passenger requires a 'board' action and a 'depart' action (2 actions).
       - Each boarded passenger requires a 'depart' action (1 action).
       Sum these up for all unserved passengers.
    6. Identify the set of floors that require the lift to stop for an action
       (either a waiting passenger's origin floor or a boarded passenger's
       destination floor).
    7. If there are no floors requiring action (which should only happen if
       all passengers are served, already handled in step 1), the movement
       cost is 0.
    8. If there are action floors, find the minimum and maximum floor indices
       among them.
    9. Calculate the estimated movement cost: This is the minimum vertical
       distance the lift must travel to reach the range of action floors
       [min_action_idx, max_action_idx] from its current floor, plus the
       distance to traverse that entire range. The minimum distance to reach
       the range and traverse it is calculated as
       `min(abs(current_idx - min_action_idx), abs(current_idx - max_action_idx)) + (max_action_idx - min_action_idx)`.
    10. The total heuristic value is the sum of the estimated actions needed
        and the estimated movement cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Helper to parse fact strings
        def get_parts(fact):
            return fact[1:-1].split()

        # 1. Parse floor order and create index mapping
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_facts = [fact for fact in static_facts if fact.startswith('(above ')]

        # Build the 'below' relationship map {floor_below: floor_above}
        below_map = {}
        all_floors = set()
        for fact in above_facts:
            _, f_above, f_below = get_parts(fact)
            below_map[f_below] = f_above
            all_floors.add(f_above)
            all_floors.add(f_below)

        # Find the lowest floor (a floor that is not a value in below_map)
        all_floors_that_are_above_something = set(below_map.values())
        # Handle case with only one floor or no above facts
        if not all_floors:
             lowest_floor = None
        else:
            potential_lowest = all_floors - all_floors_that_are_above_something
            if not potential_lowest:
                 # This might happen if floors form a cycle or there's only one floor
                 # In a standard miconic problem, there's a unique lowest floor
                 # Let's assume a unique lowest floor exists if all_floors is not empty
                 lowest_floor = next(iter(all_floors)) # Fallback: pick any floor
            else:
                 lowest_floor = potential_lowest.pop()


        # Build the ordered list of floors and index map
        ordered_floors = []
        current_floor = lowest_floor
        index = 0
        while current_floor is not None:
            ordered_floors.append(current_floor)
            self.floor_to_index[current_floor] = index
            self.index_to_floor[index] = current_floor
            index += 1
            current_floor = below_map.get(current_floor) # Get the floor above the current one

        self.highest_floor_index = index - 1

        # 2. Parse passenger initial origins and destinations
        self.passenger_info = {} # { passenger: {'origin': floor, 'destin': floor} }
        origin_facts = [fact for fact in static_facts if fact.startswith('(origin ')]
        destin_facts = [fact for fact in static_facts if fact.startswith('(destin ')]

        # Get all passenger names from origin/destin facts
        all_passengers_set = set()
        for fact in origin_facts + destin_facts:
             all_passengers_set.add(get_parts(fact)[1])

        self.all_passengers = list(all_passengers_set)

        for p in self.all_passengers:
             self.passenger_info[p] = {'origin': None, 'destin': None}

        for fact in origin_facts:
            _, p, f = get_parts(fact)
            self.passenger_info[p]['origin'] = f

        for fact in destin_facts:
            _, p, f = get_parts(fact)
            self.passenger_info[p]['destin'] = f

    def get_floor_index(self, floor_name):
        """Helper to get the numerical index for a floor name."""
        return self.floor_to_index.get(floor_name, -1) # Return -1 or handle error if floor not found

    def floor_distance(self, floor1, floor2):
        """Helper to calculate distance between two floors."""
        idx1 = self.get_floor_index(floor1)
        idx2 = self.get_floor_index(floor2)
        if idx1 == -1 or idx2 == -1:
             # Should not happen in valid problems, but handle defensively
             return float('inf')
        return abs(idx1 - idx2)

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        # Goal is (served p) for all p in self.all_passengers
        if all(f'(served {p})' in state for p in self.all_passengers):
            return 0

        # 2. Find current lift location
        lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                lift_floor = fact[1:-1].split()[1]
                break
        # If lift location is not found, something is wrong with the state representation
        if lift_floor is None:
             # This state is likely invalid or unreachable in a standard problem
             # Return infinity or a very large number
             return float('inf')

        current_idx = self.get_floor_index(lift_floor)

        # 3. Identify unserved passengers and their status
        unserved_waiting = {} # {passenger: origin_floor}
        unserved_boarded = [] # [passenger]

        for p in self.all_passengers:
            if f'(served {p})' not in state:
                if f'(boarded {p})' in state:
                    unserved_boarded.append(p)
                else:
                    # Passenger is unserved and not boarded, must be waiting at origin
                    # Find origin floor from state if possible (in case origin changed, though not in this domain)
                    origin_floor = None
                    for fact in state:
                         if fact.startswith(f'(origin {p} '):
                             origin_floor = fact[1:-1].split()[2]
                             break
                    # If not found in state, use the initial origin from static info
                    if origin_floor is None:
                         origin_floor = self.passenger_info[p]['origin']

                    if origin_floor: # Ensure origin was found
                        unserved_waiting[p] = origin_floor
                    # else: # Passenger exists but has no origin/destin info? Invalid problem?

        # 4. Calculate actions needed
        # Each waiting passenger needs board + depart (2 actions)
        # Each boarded passenger needs depart (1 action)
        actions_needed = len(unserved_waiting) * 2 + len(unserved_boarded)

        # 5. Identify floors requiring action
        waiting_floors = set(unserved_waiting.values())
        boarded_dest_floors = {self.passenger_info[p]['destin'] for p in unserved_boarded if self.passenger_info[p]['destin'] is not None}
        action_floors = waiting_floors | boarded_dest_floors

        # If no action floors, but unserved passengers exist, this implies an issue
        # (e.g., passenger waiting at destination, but destination wasn't added to action_floors)
        # The definition of waiting_floors should include passengers waiting at destination.
        # The definition of boarded_dest_floors includes destinations for boarded passengers.
        # If action_floors is empty, actions_needed must be 0, handled by initial goal check.
        if not action_floors:
             # This case should ideally not be reached if num_unserved > 0
             # If it is reached, it suggests unserved passengers exist but aren't
             # waiting anywhere or boarded going anywhere specific.
             # Return actions_needed (which would be 0 if action_floors is empty and logic is sound)
             return actions_needed


        # 6. Calculate movement cost
        action_indices = {self.get_floor_index(f) for f in action_floors if self.get_floor_index(f) != -1}

        if not action_indices: # Should not happen if action_floors is not empty and floors are valid
             return actions_needed # No valid floors to move to

        min_action_idx = min(action_indices)
        max_action_idx = max(action_indices)

        # Movement cost: minimum distance to reach the range [min_action_idx, max_action_idx]
        # from current_idx, plus the distance to traverse the range.
        # This is min(dist_to_min_end, dist_to_max_end) + range_size
        # where range_size = max_action_idx - min_action_idx
        # dist_to_min_end = abs(current_idx - min_action_idx)
        # dist_to_max_end = abs(current_idx - max_action_idx)

        movement_cost = min(abs(current_idx - min_action_idx), abs(current_idx - max_action_idx)) + (max_action_idx - min_action_idx)

        # Total heuristic
        heuristic_value = actions_needed + movement_cost

        return heuristic_value

