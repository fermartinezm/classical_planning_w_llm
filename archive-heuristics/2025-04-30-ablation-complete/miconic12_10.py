from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums two components:
    1. An estimate of the number of board/depart actions needed for unserved passengers.
    2. An estimate of the minimum lift movement actions needed to visit all floors
       where passengers are waiting or need to be dropped off.

    # Assumptions
    - Floors are arranged linearly, and `(above f_higher f_lower)` defines the
      immediate floor adjacency and order.
    - The lift has unlimited capacity.
    - The cost of move, board, and depart actions is 1.

    # Heuristic Initialization
    - Parses the `(above f_higher f_lower)` facts from the static information
      to build a mapping from floor names to numerical levels, starting from 0
      for the lowest floor.
    - Parses the `(destin p d)` facts from the static information to map
      each passenger to their destination floor.
    - Stores a set of all passenger names.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the goal state is reached. If yes, return 0.
    2. Identify the current floor of the lift.
    3. Identify all unserved passengers by checking against the goal conditions.
    4. For each unserved passenger, determine if they are waiting at their origin
       (`(origin p o)`) or are currently boarded (`(boarded p)`).
    5. Calculate the **Action Cost Component**:
       - Count the number of passengers currently waiting at their origin (unserved and not boarded). Each needs a 'board' action.
       - Count the number of passengers currently boarded (unserved and boarded). Each needs a 'depart' action.
       - The action cost estimate is the sum of these two counts.
    6. Calculate the **Movement Cost Component**:
       - Determine the set of 'required floors' the lift must visit:
         - All origin floors of passengers waiting at their origin.
         - All destination floors of passengers currently boarded.
       - If there are no required floors, the movement cost is 0.
       - If there are required floors:
         - Get the numerical levels for all required floors and the current lift floor using the pre-calculated mapping.
         - Find the minimum and maximum floor levels among the required floors.
         - Estimate the minimum movement actions needed to travel from the current
           lift floor to visit all required floors. This is estimated as the minimum
           of two simple paths:
           a) Travel from the current floor to the lowest required floor, then sweep
              upwards to the highest required floor. Cost: `abs(current_level - min_req_level) + (max_req_level - min_req_level)`.
           b) Travel from the current floor to the highest required floor, then sweep
              downwards to the lowest required floor. Cost: `abs(current_level - max_req_level) + (max_req_level - min_req_level)`.
         - The movement cost estimate is the minimum of these two path costs.
    7. The total heuristic value is the sum of the Action Cost Component and the Movement Cost Component.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals # Goal conditions (used to check if goal is reached)
        static_facts = task.static # Facts that are not affected by actions.

        # 1. Build floor levels mapping
        self.floor_levels = {}
        above_facts = [fact for fact in static_facts if get_parts(fact)[0] == 'above']

        # Build adjacency list for the 'above' relationship (f_lower -> f_higher)
        above_graph = {}
        all_floors = set()
        for fact in above_facts:
            _, f_higher, f_lower = get_parts(fact)
            above_graph.setdefault(f_lower, []).append(f_higher)
            all_floors.add(f_higher)
            all_floors.add(f_lower)

        # Find the lowest floor(s) (those not appearing as f_higher in any above fact)
        higher_floors = {get_parts(fact)[1] for fact in above_facts}
        lowest_floors = [f for f in all_floors if f not in higher_floors]

        # Use BFS to assign levels starting from the lowest floor(s)
        # Assuming a single linear tower, BFS from any lowest floor will cover all floors.
        if lowest_floors:
            # Start BFS from the first lowest floor found.
            # If there are multiple disconnected floor sections, this will only map one.
            # Assuming standard miconic structure with a single tower.
            queue = [(lowest_floors[0], 0)]
            visited = {lowest_floors[0]}

            while queue:
                current_floor, level = queue.pop(0)
                self.floor_levels[current_floor] = level

                # Find floors immediately above the current floor
                floors_immediately_above = above_graph.get(current_floor, [])

                for next_floor in floors_immediately_above:
                    if next_floor not in visited:
                        visited.add(next_floor)
                        queue.append((next_floor, level + 1))
        # else: If lowest_floors is empty, it implies a problem structure without a clear bottom.
        # The floor_levels map will remain empty, potentially causing errors later.
        # For valid miconic problems, this case should not occur.

        # 2. Store passenger destinations
        self.destinations = {}
        for fact in static_facts:
            if get_parts(fact)[0] == 'destin':
                _, passenger, destination_floor = get_parts(fact)
                self.destinations[passenger] = destination_floor

        # 3. Store all passenger names for quick lookup
        self.all_passengers = set(self.destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # In a valid state, the lift location must be specified.

        # Get the level of the current lift floor. Default to 0 if floor not found
        # in levels map (e.g., if floor was not in 'above' facts, unlikely in miconic).
        current_level = self.floor_levels.get(current_lift_floor, 0)


        # 3. Identify unserved passengers and their state
        unserved_passengers = {p for p in self.all_passengers if f'(served {p})' not in state}

        waiting_passengers = set() # Unserved passengers at origin, not boarded
        boarded_passengers = set() # Unserved passengers boarded

        for p in unserved_passengers:
            if f'(boarded {p})' in state:
                boarded_passengers.add(p)
            else:
                 # If not served and not boarded, they must be waiting at origin
                 waiting_passengers.add(p)


        # 4. Calculate Action Cost Component
        # Each waiting passenger needs a board action.
        # Each boarded passenger needs a depart action.
        action_cost = len(waiting_passengers) + len(boarded_passengers)


        # 5. Calculate Movement Cost Component
        # Floors where passengers are waiting at origin
        waiting_floors = {get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*") and get_parts(fact)[1] in waiting_passengers}
        # Floors where boarded passengers need to go
        boarded_dest_floors = {self.destinations[p] for p in boarded_passengers}

        required_floors = waiting_floors.union(boarded_dest_floors)

        movement_cost = 0
        if required_floors:
            # Get levels for all required floors. Default to 0 if floor not found.
            required_levels = {self.floor_levels.get(f, 0) for f in required_floors}
            min_req_lvl = min(required_levels)
            max_req_lvl = max(required_levels)

            # Estimate movement cost as minimum travel to cover the range of required floors
            # starting from the current floor.
            # Path 1: Go to lowest required floor, then sweep up to highest.
            cost_up_sweep = abs(current_level - min_req_lvl) + (max_req_lvl - min_req_lvl)
            # Path 2: Go to highest required floor, then sweep down to lowest.
            cost_down_sweep = abs(current_level - max_req_lvl) + (max_req_lvl - min_req_lvl)

            movement_cost = min(cost_up_sweep, cost_down_sweep)

        # 6. Total heuristic value
        total_cost = action_cost + movement_cost

        return total_cost
