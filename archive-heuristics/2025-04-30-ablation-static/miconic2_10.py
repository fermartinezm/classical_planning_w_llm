from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # This case indicates an unexpected fact format.
         # In a typical PDDL planner, facts are represented consistently.
         # Returning an empty list or raising an error might be appropriate,
         # but for this context, we assume valid PDDL fact string format.
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the estimated board/depart actions for each unserved
    passenger and the estimated minimum lift movement cost to visit all
    necessary floors (origins for waiting passengers, destinations for all
    unserved passengers).

    # Assumptions
    - Floors are linearly ordered, defined by `above` facts like `(above f_higher f_lower)`.
    - The cost of each action (board, depart, up, down) is 1.
    - The lift can carry multiple passengers.
    - Passengers waiting at their origin floor are represented by `(origin p f)`
      and not `(boarded p)`.

    # Heuristic Initialization
    - Parses static `above` facts to build a mapping from floor names to
      numerical indices, representing their vertical order. The lowest floor
      is assigned index 0. This mapping is crucial for calculating movement costs.
    - Parses initial state and goal facts to identify all passengers and
      store their origin and destination floors. This information is static
      for each passenger throughout the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the lift's current floor by finding the fact `(lift-at ?f)` in the state.
       Convert this floor name to its numerical index using the map built during initialization.
    2. Initialize counters for estimated actions: `n_wait` (unserved, unboarded passengers)
       and `n_boarded` (unserved, boarded passengers).
    3. Initialize sets to store the indices of floors the lift must visit:
       `f_pick_indices` (origin floors for unserved, unboarded passengers) and
       `f_drop_indices` (destination floors for all unserved passengers).
    4. Iterate through all passengers identified during initialization:
       - Check if the passenger is already `served` by looking for `(served p)` in the state. If served, skip this passenger.
       - If the passenger is unserved:
         - Get the passenger's destination floor and add its index to `f_drop_indices`.
         - Check if the passenger is `boarded` by looking for `(boarded p)` in the state.
         - If `boarded`: Increment `n_boarded`.
         - If not `boarded` (and unserved): Increment `n_wait`. Get the passenger's origin floor and add its index to `f_pick_indices`.
    5. Calculate the total estimated board/depart actions: `action_cost = (2 * n_wait) + n_boarded`.
       Each unboarded passenger needs a `board` and a `depart` action (2 actions).
       Each boarded, unserved passenger needs a `depart` action (1 action).
    6. Combine the required pick-up and drop-off floor indices: `f_visit_indices = f_pick_indices | f_drop_indices`.
       These are all the floors the lift must stop at to interact with unserved passengers.
    7. Calculate the estimated minimum movement cost:
       - If `f_visit_indices` is empty (meaning all passengers are served), the movement cost is 0.
       - Otherwise, find the minimum (`min_visit_idx`) and maximum (`max_visit_idx`) floor indices in `f_visit_indices`.
       - The estimated movement cost is the minimum number of steps to reach either `min_visit_idx` or `max_visit_idx` from the current floor index (`current_floor_idx`), plus the number of steps to traverse the entire range between `min_visit_idx` and `max_visit_idx`. This assumes the lift makes one efficient sweep (up or down) covering all necessary floors.
       - `movement_cost = min(abs(current_floor_idx - min_visit_idx), abs(current_floor_idx - max_visit_idx)) + (max_visit_idx - min_visit_idx)`.
    8. The total heuristic value is the sum of `action_cost` and `movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger info.
        """
        self.task = task
        self.goals = task.goals

        # Build floor order and index mapping from static 'above' facts
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_facts = [fact for fact in task.static if match(fact, "above", "*", "*")]

        # Build a map from a floor to the floor immediately below it
        below_map = {}
        # Keep track of all floors mentioned
        all_floors = set()
        # Keep track of floors that are mentioned as the lower floor in an 'above' fact
        lower_floors = set()

        for fact in above_facts:
            parts = get_parts(fact)
            if len(parts) == 3: # Ensure correct fact structure
                _, f_higher, f_lower = parts
                below_map[f_higher] = f_lower
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                lower_floors.add(f_lower)

        # Handle case with no floors or no above facts (e.g., single floor problem)
        if not all_floors:
             # Try to extract floors from initial state facts mentioning floors
             floors_from_state = set()
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 # Simple check for terms starting with 'f' (common floor naming)
                 for part in parts:
                     if part.startswith('f'):
                         floors_from_state.add(part)

             if floors_from_state:
                 all_floors = floors_from_state
                 # If no above facts, assume alphabetical/numerical sort defines order
                 sorted_floors = sorted(list(all_floors))
                 for i, floor in enumerate(sorted_floors):
                     self.floor_to_index[floor] = i
                     self.index_to_floor[i] = floor
                 # Floor map built, exit __init__
                 return
             else:
                 # No floors found at all, likely an invalid problem setup
                 print("Warning: No floors found in static facts or initial state.")
                 # The heuristic will likely return inf or behave unexpectedly if no floors.
                 pass # Continue with empty floor maps

        # Find the highest floor (a floor that is in all_floors but not in lower_floors)
        highest_floor = None
        for floor in all_floors:
            if floor not in lower_floors:
                highest_floor = floor
                break

        if highest_floor is None and len(all_floors) > 0:
             # This implies a cycle or disconnected floors in 'above' facts,
             # which shouldn't happen in valid linear Miconic problems.
             print("Warning: Could not find a unique highest floor from 'above' facts. Falling back to alphabetical sort.")
             sorted_floors = sorted(list(all_floors))
             for i, floor in enumerate(sorted_floors):
                 self.floor_to_index[floor] = i
                 self.index_to_floor[i] = floor
             # Floor map built, exit __init__
             return
        elif highest_floor is not None:
            # Build the ordered list of floors from highest to lowest using below_map
            ordered_floors_desc = []
            current = highest_floor
            while current is not None and current in all_floors: # Add check for validity
                ordered_floors_desc.append(current)
                current = below_map.get(current)

            # Check if all floors were included (handles disconnected components if any)
            if len(ordered_floors_desc) != len(all_floors):
                 print("Warning: 'above' facts do not form a single linear sequence covering all floors. Falling back to alphabetical sort.")
                 sorted_floors = sorted(list(all_floors))
                 for i, floor in enumerate(sorted_floors):
                     self.floor_to_index[floor] = i
                     self.index_to_floor[i] = floor
            else:
                # Assign indices (lowest floor gets index 0)
                num_floors = len(ordered_floors_desc)
                for i, floor in enumerate(reversed(ordered_floors_desc)):
                    self.floor_to_index[floor] = i
                    self.index_to_floor[i] = floor


        # Extract passenger origins and destinations
        self.passenger_origin = {}
        self.passenger_destin = {}
        self.all_passengers = set()

        # Get origins and destinations from initial state
        for fact in task.initial_state:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, p, f = parts
                    self.passenger_origin[p] = f
                    self.all_passengers.add(p)
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, p, f = parts
                    self.passenger_destin[p] = f
                    self.all_passengers.add(p)
            # Also add passengers mentioned as boarded or served in initial state
            elif match(fact, "boarded", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2:
                     _, p = parts
                     self.all_passengers.add(p)
            elif match(fact, "served", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2:
                     _, p = parts
                     self.all_passengers.add(p)


        # Also get passengers from goals, in case some are only mentioned there
        for goal in task.goals:
             if match(goal, "served", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 2:
                     _, p = parts
                     self.all_passengers.add(p)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, current_lift_floor = parts
                    break

        if current_lift_floor is None:
             # Should not happen in a valid state, but handle defensively
             # If lift location is unknown, we can't estimate movement.
             # Return infinity or a very large number to discourage this state.
             return float('inf')

        current_floor_idx = self.floor_to_index.get(current_lift_floor)
        if current_floor_idx is None:
             # Should not happen if floor map is built correctly from problem definition
             print(f"Error: Unknown floor '{current_lift_floor}' found in state.")
             return float('inf')


        n_wait = 0 # Unserved, unboarded passengers
        n_boarded = 0 # Unserved, boarded passengers
        f_pick_indices = set() # Origin floors for unserved, unboarded
        f_drop_indices = set() # Destination floors for all unserved

        # Create sets for quick lookup of passenger status
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*") and len(get_parts(fact)) == 2}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*") and len(get_parts(fact)) == 2}

        for p in self.all_passengers:
            if p in served_passengers:
                continue # This passenger is already served

            # This passenger is unserved
            dest_floor = self.passenger_destin.get(p)
            if dest_floor is None:
                 # Should not happen if passenger_destin is populated correctly
                 # print(f"Warning: Destination not found for passenger '{p}'. Skipping.")
                 continue # Skip this passenger if destination is unknown

            dest_floor_idx = self.floor_to_index.get(dest_floor)
            if dest_floor_idx is None:
                 # Should not happen if floor map is complete
                 # print(f"Warning: Unknown destination floor '{dest_floor}' for passenger '{p}'. Skipping.")
                 continue # Skip this passenger if destination floor is unknown

            f_drop_indices.add(dest_floor_idx)

            if p in boarded_passengers:
                n_boarded += 1
            else:
                # Passenger is unserved and unboarded, assumed to be at origin
                n_wait += 1
                origin_floor = self.passenger_origin.get(p)
                if origin_floor is None:
                     # Should not happen if passenger_origin is populated correctly
                     # print(f"Warning: Origin not found for passenger '{p}'. Skipping.")
                     continue # Skip this passenger if origin is unknown

                origin_floor_idx = self.floor_to_index.get(origin_floor)
                if origin_floor_idx is None:
                     # Should not happen if floor map is complete
                     # print(f"Warning: Unknown origin floor '{origin_floor}' for passenger '{p}'. Skipping.")
                     continue # Skip this passenger if origin floor is unknown

                f_pick_indices.add(origin_floor_idx)

        # Calculate board/depart action cost
        # Each unboarded needs board (1) + depart (1) = 2 actions
        # Each boarded needs depart (1) = 1 action
        action_cost = (2 * n_wait) + n_boarded

        # Calculate movement cost
        f_visit_indices = f_pick_indices | f_drop_indices

        movement_cost = 0
        if f_visit_indices: # Only calculate if there are floors to visit
            min_visit_idx = min(f_visit_indices)
            max_visit_idx = max(f_visit_indices)

            # Minimum moves to reach the range [min_visit_idx, max_visit_idx]
            # from current_floor_idx, plus the moves to traverse the range.
            # This assumes a single sweep strategy.
            dist_to_min = abs(current_floor_idx - min_visit_idx)
            dist_to_max = abs(current_floor_idx - max_visit_idx)
            range_span = max_visit_idx - min_visit_idx

            movement_cost = min(dist_to_min, dist_to_max) + range_span

        total_cost = action_cost + movement_cost

        return total_cost
