import re
from heuristics.heuristic_base import Heuristic

# Helper function to parse a PDDL fact string
def parse_fact(fact_string):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)' into ('predicate', ['arg1', 'arg2']).
    """
    # Removes leading/trailing brackets and splits by spaces
    parts = fact_string.strip('()').split()
    if not parts:
        return None, [] # Should not happen with valid facts
    return parts[0], parts[1:]

class miconicHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for the Miconic domain.
        Estimates the cost to reach the goal (all passengers served) by summing
        the minimum number of elevator moves required to visit all necessary
        floors and the total number of board/depart actions needed for unserved
        passengers.

    Assumptions:
        - The PDDL domain is Miconic as provided.
        - Floor ordering is defined by '(above f_i f_j)' facts in static info,
          forming a linear order from a lowest floor up to a highest floor.
        - Passenger destinations are provided in static info via '(destin p f)' facts.
        - Passenger origins (if unboarded) and boarded status are in the state.
        - The state representation includes facts like '(lift-at f)', '(origin p f)',
          '(boarded p)', '(served p)'.

    Heuristic Initialization:
        - Parses static facts to determine the linear order of floors and creates
          a mapping from floor name to its index (0-based).
        - Parses static facts to map each passenger to their destination floor.
        - Collects the names of all passengers involved in the problem.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the elevator from the state.
        2. Identify all passengers who have not yet been served.
        3. If no passengers are unserved, the state is a goal state, return 0.
        4. Separate unserved passengers into those who are unboarded (waiting at an origin floor)
           and those who are boarded (travelling to a destination floor).
        5. Determine the set of "required floors" that the elevator must visit. This set
           includes the origin floor for each unboarded unserved passenger and the
           destination floor for each boarded unserved passenger.
        6. If the set of required floors is empty (meaning all unserved passengers are
           currently at the elevator's location), the move cost is 0. The heuristic
           is simply the number of unserved passengers (representing the board/depart actions).
        7. If the set of required floors is not empty:
           a. Find the minimum and maximum floor indices among the required floors.
           b. Calculate the minimum number of elevator moves required to visit all
              required floors starting from the current floor. This is the distance
              between the current floor and the furthest required floor in either
              direction, plus the distance between the minimum and maximum required
              floors. This simplifies to `max(current_floor_index, max_required_index) - min(current_floor_index, min_required_index)`.
           c. The total heuristic value is the sum of this move cost and the total
              number of unserved passengers (each needing one board and one depart action).
              Heuristic = `Move cost + Number of unserved passengers`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.floor_to_index = {}
        self.passenger_destin = {}
        self.all_passengers = set()

        # Data structures to build floor order graph
        floor_above_map = {} # Maps floor -> floor immediately above it
        in_degree = {} # Maps floor -> number of floors immediately below it (i.e., number of floors for which this floor is immediately above)
        all_floors = set()

        # 1. Parse static facts to build floor order graph and passenger destinations
        for fact_string in task.static:
            pred, args = parse_fact(fact_string)
            if pred == 'above':
                f_above, f_below = args
                all_floors.add(f_above)
                all_floors.add(f_below)
                # Build graph where edge is f_below -> f_above
                floor_above_map[f_below] = f_above
                in_degree[f_above] = in_degree.get(f_above, 0) + 1
                in_degree[f_below] = in_degree.get(f_below, 0) # Ensure all floors are in in_degree
            elif pred == 'destin':
                p, f = args
                self.passenger_destin[p] = f
                self.all_passengers.add(p)

        # Collect all passengers from task.facts that involve passengers.
        # task.facts contains all possible ground atoms in the domain/problem.
        passenger_predicates = {'origin', 'destin', 'boarded', 'served'}
        for fact_string in task.facts:
             pred, args = parse_fact(fact_string)
             if pred in passenger_predicates and args:
                 # Assuming the first argument is always the passenger
                 self.all_passengers.add(args[0])


        # 2. Determine floor order and build floor_to_index map
        if not all_floors:
             # Handle case with no floors (should not happen in miconic)
             # Cannot build floor_to_index map, heuristic will likely return inf
             return

        # Find the lowest floor (in-degree 0 in the f_below -> f_above graph)
        lowest_floor = None
        for floor in all_floors:
            if in_degree.get(floor, 0) == 0:
                lowest_floor = floor
                break

        if lowest_floor is None:
             # This implies a problem with the static definition of floors (e.g., cycle or no base)
             # In a well-formed miconic problem, there should be exactly one floor with in-degree 0.
             # Cannot build floor_to_index map. Heuristic will likely return inf.
             return # Exit init early if floor structure is invalid

        # Traverse from lowest floor to build ordered list and index map
        current_floor = lowest_floor
        floor_index = 0
        while current_floor is not None:
            self.floor_to_index[current_floor] = floor_index
            floor_index += 1
            # Find the floor immediately above current_floor
            current_floor = floor_above_map.get(current_floor) # Get the floor this one is immediately below

        # Check if all floors were included in the mapping (indicates a linear chain)
        if len(self.floor_to_index) != len(all_floors):
             # This implies the 'above' relations don't form a single linear chain
             # (e.g., disconnected floors, branching). Heuristic might be inaccurate
             # or fail if a required floor is not in the map.
             pass # Continue, but be aware of potential issues


    def __call__(self, node):
        state_set = set(node.state)

        # 1. Find current lift floor
        f_current = None
        for fact_string in state_set:
            pred, args = parse_fact(fact_string)
            if pred == 'lift-at':
                f_current = args[0]
                break

        # If floor structure was invalid in init, self.floor_to_index might be empty
        # or f_current might not be in it.
        if not self.floor_to_index or f_current not in self.floor_to_index:
             # Cannot compute heuristic without a valid floor map or current location
             return float('inf')


        # 2. Find unserved passengers
        unserved_passengers = {p for p in self.all_passengers if '(served {})'.format(p) not in state_set}

        # 3. If no passengers are unserved, it's a goal state
        if not unserved_passengers:
            return 0

        # 4. Separate unboarded and boarded unserved passengers
        unboarded_passengers = set()
        boarded_passengers = set()
        for p in unserved_passengers:
            if '(boarded {})'.format(p) in state_set:
                boarded_passengers.add(p)
            else:
                unboarded_passengers.add(p)


        # 5. Determine required floors
        required_floors = set()
        # Origins of unboarded passengers
        for p in unboarded_passengers:
            # Find origin floor from state
            found_origin = False
            for fact_string in state_set:
                 pred, args = parse_fact(fact_string)
                 if pred == 'origin' and args and args[0] == p:
                      required_floors.add(args[1])
                      found_origin = True
                      break
            # If origin not found in state for an unboarded passenger, something is wrong
            # with the state representation or domain model assumptions.
            # We cannot pick up this passenger. Heuristic might be inaccurate.
            # For robustness, maybe return inf if a required origin/destin is missing?
            # Let's assume valid states for now.
            if not found_origin:
                 # print(f"Warning: Unboarded passenger {p} has no origin fact in state.")
                 pass # Heuristic might be inaccurate

        # Destinations of boarded passengers
        for p in boarded_passengers:
            # Destination is from static info
            if p in self.passenger_destin:
                required_floors.add(self.passenger_destin[p])
            else:
                 # Boarded passenger with no destination in static? Invalid problem?
                 # print(f"Warning: Boarded passenger {p} has no destination in static info.")
                 pass # Heuristic might be inaccurate


        # 6. Handle case with no required floors
        if not required_floors:
            # This happens if all unserved passengers are at the current floor,
            # either waiting to board or waiting to depart.
            return len(unserved_passengers) # Only board/depart actions needed

        # 7. Calculate move cost and total heuristic
        current_idx = self.floor_to_index[f_current]
        required_floors_indices = set()
        for f in required_floors:
             if f in self.floor_to_index:
                  required_floors_indices.add(self.floor_to_index[f])
             else:
                  # Required floor not in index map? Problem with floor parsing or state?
                  # print(f"Warning: Required floor {f} not found in floor index map.")
                  return float('inf') # Cannot compute heuristic reliably

        if not required_floors_indices: # Should be non-empty if required_floors was non-empty
             # This case should not be reached if required_floors was non-empty and all floors were mapped.
             # It might indicate a required floor was not in the floor_to_index map.
             # The check above handles this by returning inf.
             # If we reach here, it means required_floors was non-empty, but required_floors_indices became empty
             # because none of the required floors were in the floor_to_index map.
             # This is an error condition.
             return float('inf')


        min_idx = min(required_floors_indices)
        max_idx = max(required_floors_indices)

        # Minimum moves to cover the range [min_idx, max_idx] starting from current_idx
        move_cost = max(current_idx, max_idx) - min(current_idx, min_idx)

        # Total heuristic = move cost + actions at floors
        # We count the number of unserved passengers as a proxy for the number of
        # board/depart actions remaining.
        heuristic_value = move_cost + len(unserved_passengers)

        return heuristic_value
