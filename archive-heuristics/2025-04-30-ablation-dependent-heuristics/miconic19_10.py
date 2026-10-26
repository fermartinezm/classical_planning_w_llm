import re

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the number of actions required to serve all passengers.
    The heuristic sums the estimated actions for each unserved passenger:
    1. One 'board' action if the passenger is waiting at their origin.
    2. One 'depart' action if the passenger is boarded.
    3. Estimates the lift movement cost required to visit all necessary floors
       (origins of waiting passengers and destinations of boarded passengers).
       The movement cost is estimated as the span of the required floors plus
       the minimum distance from the current lift floor to either end of the span.

    Assumptions:
    - The floor structure is a single linear tower, defined by 'above' predicates.
    - Floors are ordered such that if (above f_i f_j) is true, then f_j is higher than f_i.
    - The goal is to serve all passengers defined in the problem's destinations.
    - Valid states adhere to the domain rules (e.g., lift is at exactly one floor,
      unserved passengers are either waiting at an origin or boarded).

    Heuristic Initialization:
    - Parses the static facts to determine the floor ordering and assign indices.
      It identifies the 'immediately above' relation (f_b is immediately above f_a if (above f_a f_b) is true and there's no intermediate floor f_k).
      It finds the lowest floor (the one not immediately above any other floor).
      It traverses the 'immediately above' chain to assign a unique index to each floor.
    - Stores passenger destinations and identifies all passengers in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using the task's goal definition (`self.task.goal_reached(state)`). If yes, return 0.
    2. If the floor index map was not successfully built during initialization (e.g., due to malformed static facts or complex floor structure), return infinity as the heuristic cannot be computed reliably.
    3. Find the current floor of the lift from the state facts. If not found or the floor is not indexed, return infinity.
    4. Initialize the heuristic value (representing board/depart actions) to 0.
    5. Initialize a set of required stop floors (required_stop_floors_str).
    6. Get the set of all passengers in the problem (from pre-calculated destinations).
    7. Convert the state frozenset to a set for faster fact lookups.
    8. For each passenger `p` in the set of all passengers:
       - If `(served p)` is not in the state facts:
         - This passenger is unserved.
         - Check if `(origin p o)` is true in the state facts for some floor `o`. If yes:
           - Add 1 to the board/depart actions count (for the 'board' action).
           - Add floor `o` to the set of required stop floors. If `o` is not indexed, return infinity.
         - Else, check if `(boarded p)` is true in the state facts. If yes:
           - Add 1 to the board/depart actions count (for the 'depart' action).
           - Get the destination floor `d` for passenger `p` from the pre-calculated destinations.
           - Add floor `d` to the set of required stop floors. If `d` is not found or not indexed, return infinity.
         - (Note: A valid unserved passenger should be either waiting or boarded).
    9. Calculate the movement cost based on the current lift floor index and the set of required stop floor strings:
       - If the set of required stop floors is empty, the movement cost is 0.
       - Otherwise, get the indices for all required stop floors using the pre-calculated floor-to-index map.
       - Find the minimum and maximum indices among the required stop floors.
       - The estimated movement cost is calculated as `min(abs(current_floor_index - min_required_index), abs(current_floor_index - max_required_index)) + (max_required_index - min_required_index)`.
    10. The total heuristic value is the sum of the board/depart actions count and the estimated movement cost.
    11. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information.

        @param task: The planning task object (instance of Task class).
        """
        self.task = task
        self.destinations = {}
        self.floor_to_index = {}
        self.index_to_floor = {}
        self._all_passengers = set() # Keep track of all passengers in the problem

        # Extract destinations from static facts
        for fact in task.static:
            if fact.startswith('(destin '):
                parts = fact.strip('()').split()
                if len(parts) == 3: # Ensure correct format
                    passenger = parts[1]
                    destination = parts[2]
                    self.destinations[passenger] = destination
                    self._all_passengers.add(passenger)


        # Extract floor ordering from static facts
        all_floors = set()
        above_relations = set() # Store as tuples (f1, f2)
        for fact in task.static:
            if fact.startswith('(above '):
                parts = fact.strip('()').split()
                if len(parts) == 3: # Ensure correct format
                    f1 = parts[1]
                    f2 = parts[2]
                    above_relations.add((f1, f2))
                    all_floors.add(f1)
                    all_floors.add(f2)


        if not all_floors:
             # Cannot build floor map, heuristic will return inf for any state
             return

        # Build the 'immediately above' map
        # f_b is immediately above f_a if (above f_a f_b) is true
        # AND there is no f_k such that (above f_a f_k) and (above f_k f_b) are true.
        immediate_above_map = {}

        # Create a set of all floors for quick lookup
        all_floors_set = set(all_floors)

        for f_a in all_floors:
            for f_b in all_floors:
                if (f_a, f_b) in above_relations:
                    # Check if there's an intermediate floor f_k
                    is_intermediate = False
                    for f_k in all_floors_set:
                        if f_k != f_a and f_k != f_b and (f_a, f_k) in above_relations and (f_k, f_b) in above_relations:
                            is_intermediate = True
                            break
                    if not is_intermediate:
                        # f_b is immediately above f_a
                        # Assuming a linear tower, each floor (except top) is immediately below exactly one other floor.
                        # If we find multiple immediate_above for f_a, it's a non-linear structure.
                        if f_a in immediate_above_map:
                             # Keep the first one found for f_a.
                             pass
                        else:
                            immediate_above_map[f_a] = f_b


        # Find the lowest floor: The floor that is not the value in any immediate_above_map entry.
        floors_that_are_immediately_above = set(immediate_above_map.values())
        potential_lowest_floors = all_floors_set - floors_that_are_immediately_above

        lowest_floor = None
        if len(potential_lowest_floors) == 1:
            lowest_floor = list(potential_lowest_floors)[0]
        elif len(potential_lowest_floors) == 0 and all_floors:
             # This can happen if there's only one floor, or a cycle (invalid).
             # If only one floor, it's the lowest.
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # No clear lowest floor from immediate relations.
                 # Fallback: Try to find a floor that is the first element in 'above' facts but never the second.
                 # This is the absolute lowest in a DAG.
                 f1_set = {f1 for f1, f2 in above_relations}
                 f2_set = {f2 for f1, f2 in above_relations}
                 potential_lowest_fallback = f1_set - f2_set
                 if len(potential_lowest_fallback) == 1:
                     lowest_floor = list(potential_lowest_fallback)[0]
                 elif len(potential_lowest_fallback) == 0 and all_floors:
                     # Still no clear lowest. Could be a single floor, or a structure where the lowest is also the highest (cycle).
                     if len(all_floors) == 1:
                         lowest_floor = list(all_floors)[0]
                     else:
                         # Final fallback: Sort floors alphabetically. This is a weak assumption.
                         sorted_floors = sorted(list(all_floors))
                         lowest_floor = sorted_floors[0]


        if lowest_floor is None:
             # Cannot build floor map, heuristic will return inf for any state
             self.floor_to_index = {} # Ensure it's empty if building failed
             self.index_to_floor = {}
             return


        # Traverse the 'immediately above' chain to assign indices
        current_floor = lowest_floor
        current_index = 0
        indexed_floors = set()
        while current_floor is not None and current_floor not in indexed_floors:
            self.floor_to_index[current_floor] = current_index
            self.index_to_floor[current_index] = current_floor
            indexed_floors.add(current_floor)

            current_floor = immediate_above_map.get(current_floor)
            current_index += 1

        # Check if all floors were indexed
        if len(self.floor_to_index) != len(all_floors):
             # This happens if there are disjoint sets of floors or non-linear structures not handled by immediate_above_map.
             # The heuristic will return infinity if it encounters an unindexed floor.
             pass


    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach a goal state.
        """
        # Check if it's a goal state using the task's goal definition
        if self.task.goal_reached(state):
            return 0

        # If floor map wasn't built successfully, return infinity
        if not self.floor_to_index:
             return float('inf')

        # Find current lift floor
        current_floor_str = None
        # Convert state to set for faster lookups
        state_facts = set(state)

        for fact in state_facts:
            if fact.startswith('(lift-at '):
                parts = fact.strip('()').split()
                if len(parts) == 2:
                    current_floor_str = parts[1]
                    break


        if current_floor_str is None or current_floor_str not in self.floor_to_index:
             # Invalid state (no lift-at) or lift at an unindexed floor
             return float('inf')

        current_floor_index = self.floor_to_index[current_floor_str]

        # Identify unserved passengers and required stops
        required_stop_floors_str = set()
        board_depart_actions = 0

        for p in self._all_passengers:
            if f'(served {p})' not in state_facts:
                # Passenger is unserved
                is_boarded = f'(boarded {p})' in state_facts
                origin_floor_str = None
                # Find if passenger is waiting and where
                # Iterate through state_facts to find the origin fact for this passenger
                for fact in state_facts:
                     if fact.startswith(f'(origin {p} '):
                         parts = fact.strip('()').split()
                         if len(parts) == 3:
                             origin_floor_str = parts[2]
                             break


                if origin_floor_str is not None: # Passenger is waiting at origin_floor_str
                    board_depart_actions += 1 # Need a board action
                    if origin_floor_str in self.floor_to_index:
                        required_stop_floors_str.add(origin_floor_str)
                    else:
                         # Origin floor not indexed, indicates problem
                         return float('inf') # Cannot compute heuristic reliably
                elif is_boarded: # Passenger is boarded
                    board_depart_actions += 1 # Need a depart action
                    destin_floor_str = self.destinations.get(p)
                    if destin_floor_str is None:
                         return float('inf') # Cannot compute heuristic reliably (boarded passenger without destination)
                    elif destin_floor_str in self.floor_to_index:
                        required_stop_floors_str.add(destin_floor_str)
                    else: # destin_floor_str is not None but not in self.floor_to_index
                         return float('inf') # Cannot compute heuristic reliably (destination floor not indexed)
                # else: Passenger is unserved but neither waiting nor boarded.
                # This case implies an invalid state or a problem definition where passengers
                # can exist without being at an origin or boarded.
                # Assuming valid miconic problems, this case shouldn't occur for unserved passengers.
                # If it does, they don't contribute to board/depart actions or required stops
                # based on the current state facts, which is reasonable.

        # Calculate movement cost
        movement_cost = 0
        if required_stop_floors_str:
            # Ensure all required floors were successfully indexed
            required_stop_indices = []
            for f in required_stop_floors_str:
                 # Check if floor is indexed. This check is technically redundant
                 # if the checks when adding to the set were sufficient, but safer.
                 if f in self.floor_to_index:
                      required_stop_indices.append(self.floor_to_index[f])
                 else:
                      # This case should have been caught above when adding to the set,
                      # but double-checking for safety.
                      return float('inf') # Cannot compute heuristic reliably


            if required_stop_indices: # Should be true if required_stop_floors_str was not empty and all floors indexed
                min_required_index = min(required_stop_indices)
                max_required_index = max(required_stop_indices)

                # Estimated moves = distance to nearest required floor endpoint + span of required floors
                dist_to_min = abs(current_floor_index - min_required_index)
                dist_to_max = abs(current_floor_index - max_required_index)
                span = max_required_index - min_required_index

                movement_cost = min(dist_to_min, dist_to_max) + span
            # else: This branch should technically not be reached if required_stop_floors_str was not empty
            # and the checks inside the loop passed.


        # Total heuristic
        heuristic_value = board_depart_actions + movement_cost

        return heuristic_value
