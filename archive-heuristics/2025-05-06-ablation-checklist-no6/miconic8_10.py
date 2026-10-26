# Need to import the base class if it's in a separate file
# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available in the environment where this code runs.

from fnmatch import fnmatch

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class inheriting from Heuristic (assuming it exists)
# If Heuristic base class is not provided, this code might need a dummy base class definition
# or run in an environment where it's available.
# For the purpose of providing the code as requested, I will assume Heuristic is available.

class miconicHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions (board, depart, move) required to serve
    all unserved passengers.

    Heuristic calculation:
    - Count the number of unserved, unboarded passengers (needs 1 board action each).
    - Count the number of unserved, boarded passengers (needs 1 depart action each).
    - Estimate movement cost: Minimum moves for the lift to visit all required
      pickup and dropoff floors. This is estimated as the distance between the
      lowest and highest required floors plus the minimum distance from the
      current lift floor to either of these extreme floors.
    - Total heuristic = board_actions + depart_actions + movement_cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.

        Args:
            task: The planning task object containing initial state, goals, static facts, etc.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Needed for single floor case

        # Build floor order and mapping (floor_name -> index)
        above_facts = [fact for fact in self.static if match(fact, "above", "*", "*")]
        floor_names = set()
        above_map = {} # Maps floor_i to floor_j if (above floor_i floor_j)
        is_above_target = set() # Floors that are the second argument of an above fact

        for fact in above_facts:
            _, f_i, f_j = get_parts(fact)
            floor_names.add(f_i)
            floor_names.add(f_j)
            above_map[f_i] = f_j
            is_above_target.add(f_j)

        self.floor_to_index = {}
        if not floor_names:
            # Handle single floor case - find the floor name from lift-at in initial state
            initial_lift_fact = next(iter([fact for fact in self.initial_state if match(fact, "lift-at", "*")]), None)
            if initial_lift_fact:
                 _, single_floor = get_parts(initial_lift_fact)
                 self.floor_to_index[single_floor] = 1
        else:
            # Find the lowest floor (the one that is in floor_names but not a target of any 'above' fact)
            lowest_floor = None
            candidate_lowest = floor_names - is_above_target
            if candidate_lowest:
                 # Assuming a single connected component for floors, there should be only one lowest floor
                 lowest_floor = candidate_lowest.pop()
            elif floor_names:
                 # This case implies floor_names is not empty, but all floors are targets of 'above' facts,
                 # suggesting a cycle or disconnected components where no floor is a clear 'lowest'.
                 # This shouldn't happen in standard miconic. Fallback: pick an arbitrary floor.
                 # Find a floor that is a source but not a target (should be the lowest in a chain)
                 is_above_source = set(above_map.keys())
                 potential_lowest_source = is_above_source - is_above_target
                 if potential_lowest_source:
                      lowest_floor = potential_lowest_source.pop()
                 else:
                      # Still no clear lowest? Pick any floor name.
                      lowest_floor = next(iter(floor_names))


            # Build the ordered list of floors and the floor_name -> index mapping
            current_floor = lowest_floor
            index = 1
            # Use a set to track visited floors to prevent infinite loops in case of cycles (invalid PDDL)
            visited_floors = set()
            # Only proceed if a lowest floor was found (handles empty floor_names or weird structures)
            while current_floor is not None and current_floor not in visited_floors:
                visited_floors.add(current_floor)
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = above_map.get(current_floor) # Get the floor immediately above

            # Note: Floors not reachable from the lowest_floor via 'above' links will not be in floor_to_index.
            # The heuristic will ignore these floors for movement cost calculation, which is reasonable
            # as the lift cannot move to them using 'up'/'down' actions defined by 'above'.
            # This might underestimate if there are disconnected floor subgraphs with passengers.
            # Assuming standard miconic structure where all floors are connected linearly.


        # Extract passenger destinations from static facts
        self.passenger_destin = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destin_floor = get_parts(fact)
                self.passenger_destin[passenger] = destin_floor

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        Args:
            node: The search node containing the current state.

        Returns:
            The estimated heuristic value (integer).
        """
        state = node.state

        board_actions_needed = 0
        depart_actions_needed = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Find current lift location
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_floor = get_parts(fact)
                break

        # If lift location is not found, state is likely invalid or goal reached (lift-at might be deleted?)
        # In miconic, lift-at is always present and just changes floor.
        # If goal is reached, h=0 is handled by action counts being 0.
        # If current_floor is None for a non-goal state, something is wrong.
        # Assuming current_floor is always found in non-goal states.

        # Identify unserved passengers and their needs
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_facts = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        # Consider all passengers that are goals or currently in the state (origin/boarded)
        # This covers all passengers relevant to the current state and goal.
        all_relevant_passengers = set(self.passenger_destin.keys()) | set(origin_facts.keys()) | boarded_passengers

        for passenger in all_relevant_passengers:
            if passenger not in served_passengers:
                # Passenger is unserved
                if passenger in boarded_passengers:
                    # Unserved and boarded: needs to depart
                    depart_actions_needed += 1
                    # Add destination floor to dropoff targets
                    if passenger in self.passenger_destin: # Should always be true for valid problems
                        dropoff_floors.add(self.passenger_destin[passenger])
                    # else: Invalid state, passenger boarded but no destination? Heuristic might be inaccurate.
                elif passenger in origin_facts:
                    # Unserved and unboarded: needs to board
                    board_actions_needed += 1
                    pickup_floors.add(origin_facts[passenger])
                # else: passenger is unserved, not boarded, not at origin - implies unreachable?
                # Assuming reachable states only, this case shouldn't happen for unserved passengers.

        # Calculate movement cost
        all_target_floors = pickup_floors | dropoff_floors
        movement_cost = 0

        # Only calculate movement if there are floors to visit and more than one floor exists
        if all_target_floors and len(self.floor_to_index) > 1:
            # Filter out target floors that weren't successfully mapped to an index
            # (e.g., if the floor structure was malformed and not all floors were ordered)
            mappable_target_floors = {f for f in all_target_floors if f in self.floor_to_index}

            if mappable_target_floors:
                all_target_indices = {self.floor_to_index[f] for f in mappable_target_floors}
                min_idx = min(all_target_indices)
                max_idx = max(all_target_indices)

                # Ensure current_floor is also mappable (should be if state is valid)
                if current_floor in self.floor_to_index:
                    current_idx = self.floor_to_index[current_floor]

                    # Movement cost: distance to nearest endpoint + distance between endpoints
                    movement_cost = (max_idx - min_idx) + min(abs(current_idx - min_idx), abs(current_idx - max_idx))
                # else: current_floor not mappable? Invalid state? Movement cost remains 0.

        # Total heuristic is sum of actions and movement estimate
        total_cost = board_actions_needed + depart_actions_needed + movement_cost

        return total_cost
