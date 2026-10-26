import re
import math

def natural_sort_key(s):
    """Key for natural sorting (e.g., f1, f2, f10)."""
    # Handles strings like 'f1', 'f10', 'floor1', 'floor10'
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal state
    by summing two components:
    1. The estimated number of boarding and departing actions needed for all
       unserved passengers.
    2. The estimated number of lift movement actions needed to visit all
       relevant floors (origin floors for waiting passengers and destination
       floors for all unserved passengers), plus the current floor.

    Assumptions:
    - The floor names are ordered such that natural sorting (e.g., f1, f2, f10)
      corresponds to the physical order of floors. This is inferred from the
      structure of the 'above' predicates in the example instance files.
    - The 'above' predicates implicitly define a total order on floors.
    - The cost of each action (board, depart, up, down) is 1.
    - Valid states always contain exactly one `(lift-at ?f)` fact.
    - All passengers mentioned in the state or static facts have a destination
      defined in the static facts.

    Heuristic Initialization:
    The constructor processes the static information from the task:
    - It identifies all floor objects mentioned in the problem and creates a
      mapping from floor name to an integer index based on natural sorting.
    - It extracts the destination floor for each passenger from the 'destin'
      predicates.
    - It identifies all passengers mentioned in the problem.
    This precomputation avoids redundant parsing during heuristic evaluation
    for each state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state using `task.goal_reached(state)`.
       If it is, the heuristic value is 0.
    2. Identify the current floor of the lift from the `(lift-at ?f)` fact
       in the state. If not found or invalid, return infinity.
    3. Identify all unserved passengers. These are passengers for whom
       `(served ?p)` is not true in the state.
    4. If there are no unserved passengers, the state is a goal state (already
       checked in step 1), return 0.
    5. Calculate the base cost from boarding and departing actions:
       Initialize heuristic `h = 0`.
       Identify passengers waiting at their origin (`(origin ?p ?f)` in state)
       and passengers already boarded (`(boarded ?p)` in state) among the
       unserved passengers.
       For each unserved passenger `p`:
       - If `p` is waiting at their origin, add 2 to `h` (for board and depart).
       - If `p` is boarded, add 1 to `h` (for depart).
    6. Identify the set of floors the lift *must* visit to serve the unserved
       passengers. This set includes:
       - The origin floor for every unserved passenger waiting at their origin.
       - The destination floor for every unserved passenger (whether waiting
         or boarded).
       Let this set be `F_required`. If any required floor is not a known floor,
       return infinity.
    7. Calculate the lift movement cost:
       - Get the integer index of the current lift floor.
       - Get the integer indices for all floors in `F_required`.
       - Consider the set of all floor indices that must be visited, including
         the current floor index and all required floor indices. Let this set
         be `I_visit`.
       - If `I_visit` has more than one element, the estimated minimum number
         of lift moves required to traverse the range of floors from the lowest
         index in `I_visit` to the highest index in `I_visit` is
         `max(I_visit) - min(I_visit)`.
       - Add this lift movement cost to `h`.
    8. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.passenger_destinations = {}
        self.all_passengers = set()
        self.floor_to_index = {}
        self.index_to_floor = {}

        # Collect all objects and identify passengers and floors
        passengers_set = set()
        floors_set = set()

        # Process initial state and static facts
        for fact_str in task.static | task.initial_state:
            parts = fact_str.strip("()").split()
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'destin' and len(args) == 2:
                 p, f = args
                 self.passenger_destinations[p] = f
                 passengers_set.add(p)
                 floors_set.add(f)
            elif predicate == 'origin' and len(args) == 2:
                 p, f = args
                 passengers_set.add(p)
                 floors_set.add(f)
            elif predicate == 'boarded' and len(args) == 1:
                 p = args[0]
                 passengers_set.add(p)
            elif predicate == 'served' and len(args) == 1:
                 p = args[0]
                 passengers_set.add(p)
            elif predicate == 'lift-at' and len(args) == 1:
                 f = args[0]
                 floors_set.add(f)
            elif predicate == 'above' and len(args) == 2:
                 f1, f2 = args
                 floors_set.add(f1)
                 floors_set.add(f2)
            # Add other predicates if they involve objects

        self.all_passengers = frozenset(passengers_set)

        # Sort floor names using natural sort and create index mapping
        sorted_floors = sorted(list(floors_set), key=natural_sort_key)
        for i, floor in enumerate(sorted_floors):
            self.floor_to_index[floor] = i
            self.index_to_floor[i] = floor

        # Check if floors were found
        if not self.floor_to_index and floors_set:
             # Found floor-like objects but couldn't map them?
             # This might indicate an issue with floor naming or parsing assumption.
             print(f"Warning: Could not map floors {floors_set} to indices. Heuristic may return infinity.")


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach the goal.
        """
        # 1. Check if goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Identify current lift floor
        current_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                parts = fact_str.strip("()").split()
                if len(parts) == 2:
                    current_floor = parts[1]
                    break # Assuming only one lift-at fact

        if current_floor is None or current_floor not in self.floor_to_index:
             # Invalid state (no lift-at) or current floor not recognized
             return float('inf')

        idx_current = self.floor_to_index[current_floor]

        # 3. Identify unserved passengers and their state (waiting/boarded)
        passengers_waiting = set()
        passengers_boarded = set()
        unserved_passengers = set()

        # Required floors for movement
        required_floors = set()

        # Iterate through all known passengers
        for p in self.all_passengers:
            served_fact = f'(served {p})'
            if served_fact not in state:
                unserved_passengers.add(p)

                boarded_fact = f'(boarded {p})'
                is_boarded = boarded_fact in state

                if is_boarded:
                    passengers_boarded.add(p)
                else:
                    # Must be waiting at origin. Find the origin floor from state facts.
                    origin_floor = None
                    # Iterate through state facts to find the origin for this passenger
                    for fact_str in state:
                         if fact_str.startswith(f'(origin {p} '):
                              parts = fact_str.strip("()").split()
                              if len(parts) == 3:
                                   origin_floor = parts[2]
                                   break # Found origin fact for this passenger

                    if origin_floor:
                         passengers_waiting.add(p)
                         required_floors.add(origin_floor)
                    else:
                         # Unserved but neither boarded nor at origin? Invalid state.
                         # This could happen if origin fact is missing or malformed.
                         return float('inf') # Cannot compute heuristic for invalid state

                # Add destination floor for all unserved passengers
                dest_floor = self.passenger_destinations.get(p)
                if dest_floor:
                     required_floors.add(dest_floor)
                else:
                     # Destination not found in static facts - problem definition error
                     return float('inf') # Cannot compute heuristic


        # If no unserved passengers, return 0 (handled by goal check)
        if not unserved_passengers:
            return 0 # Should not be reached if goal check is first

        # 5. Calculate base cost (board/depart actions)
        h = len(passengers_waiting) * 2 + len(passengers_boarded)

        # 6. Calculate lift movement cost
        # Consider the set of indices including the current floor and all required floors
        indices_to_visit = {idx_current}
        valid_required_indices = set()
        for f in required_floors:
             idx = self.floor_to_index.get(f)
             if idx is not None:
                  valid_required_indices.add(idx)
             else:
                  # Required floor not found in parsed floors - problem with parsing or state
                  return float('inf') # Cannot compute heuristic

        indices_to_visit.update(valid_required_indices)

        # If indices_to_visit has more than one element, calculate the range.
        if len(indices_to_visit) > 1:
            min_visit_idx = min(indices_to_visit)
            max_visit_idx = max(indices_to_visit)
            lift_moves = max_visit_idx - min_visit_idx
            h += lift_moves
        # else: lift_moves is 0, h remains unchanged

        return h
