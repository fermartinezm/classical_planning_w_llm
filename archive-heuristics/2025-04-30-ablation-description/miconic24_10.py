# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle basic format
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Return empty list for invalid format
         return []
    return fact_str[1:-1].split()

# Assuming Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#         self.initial_state = task.initial_state # Assuming initial_state is available

#     def __call__(self, node):
#         raise NotImplementedError

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. An estimate of the minimum number of board/depart actions required for unserved passengers.
    2. An estimate of the minimum vertical movement required for the elevator to visit all necessary floors (origin floors for waiting passengers, destination floors for boarded passengers).

    # Assumptions
    - Each board and depart action costs 1.
    - Each move (up/down) action between adjacent floors costs 1.
    - The 'above' predicates define a total order on floors, allowing assignment of integer levels.
    - The minimum vertical movement to visit a set of floors from the current floor is lower bounded by the maximum distance from the current floor to any floor in the set.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts and initial state.
    - Identifies all passengers that need to be served (from the goal state).
    - Builds a mapping from floor names to integer levels based on the 'above' predicates. This is done by treating 'above' as defining a directed graph from lower to higher floors and performing a BFS starting from the lowest floor(s).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the goal state is reached. If yes, the heuristic is 0.
    2. Initialize `action_cost = 0` and `f_stops_levels = set()`.
    3. Find the current floor of the elevator from the state fact `(lift-at ?f)`.
    4. Iterate through all passengers that need to be served (identified during initialization).
    5. For each such passenger `p`:
       - Check if `(served p)` is true in the current state. If yes, this passenger is done, continue to the next.
       - If `(served p)` is false:
         - Check if `(boarded p)` is true in the current state.
         - If `(boarded p)` is true:
           - Add 1 to `action_cost` (for the future `depart` action).
           - Add the integer level of their destination floor (looked up from initialization) to `f_stops_levels`.
         - If `(boarded p)` is false:
           - Find their origin floor `f_origin` from the state fact `(origin p f_origin)`.
           - Add 2 to `action_cost` (for the future `board` and `depart` actions).
           - Add the integer level of their origin floor to `f_stops_levels`.
    6. Calculate the vertical movement cost:
       - If `f_stops_levels` is empty, `vertical_cost = 0`.
       - If `f_stops_levels` is not empty:
         - Get the integer level for the current elevator floor.
         - Find the minimum and maximum levels among `f_stops_levels`.
         - `vertical_cost` is the maximum of the absolute difference between the current level and the minimum stop level, and the absolute difference between the current level and the maximum stop level. This is an admissible lower bound on the vertical travel.
    7. The total heuristic value is `action_cost + vertical_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floors for each passenger.
        - Floor levels based on 'above' relationships.
        - Passengers that need to be served.
        """
        # Call parent constructor if needed, assuming it takes task
        # super().__init__(task)

        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find origin facts

        # Store destination floors for each passenger.
        self.destinations = {}
        # Collect destin facts from static and initial state
        for fact in self.static | self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                self.destinations[passenger] = destination_floor

        # Identify all passengers that need to be served (from the goal state).
        self.passengers_to_serve = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                 self.passengers_to_serve.add(parts[1])

        # Build floor levels map from 'above' facts.
        # (above f_lower f_higher) means f_higher is above f_lower
        above_relations = []
        all_floors = set()
        # Collect floors and above relations from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_lower, f_higher = parts[1], parts[2]
                above_relations.append((f_lower, f_higher))
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        # Also collect floors from initial state and goals to be comprehensive
        for fact in self.initial_state | self.goals:
             parts = get_parts(fact)
             # Check predicates that involve floors
             if parts and parts[0] in ["lift-at", "origin", "destin"]:
                  if len(parts) > 1: # Ensure there's an argument
                      # Floor is usually the last argument for these predicates
                      all_floors.add(parts[-1])


        # Build graph: edge f_lower -> f_higher if (above f_higher f_lower)
        # This means f_higher is one level above f_lower
        adj = {f: [] for f in all_floors}
        in_degree = {f: 0 for f in all_floors}
        for f_lower, f_higher in above_relations:
            # Ensure floors are in our collected set before adding edge/degree
            if f_lower in adj and f_higher in adj:
                 adj[f_lower].append(f_higher)
                 in_degree[f_higher] += 1
            # else: print(f"Warning: Floor in above relation not found in all_floors: {f_lower}, {f_higher}")


        # Find lowest floor(s) (in-degree 0)
        q = [f for f in all_floors if in_degree[f] == 0]
        self.floor_levels = {}
        level = 0

        # BFS to assign levels
        # Process level by level
        while q:
            next_q = []
            current_level_floors = []
            # Collect all floors for the current level first
            for f in q:
                 if f not in self.floor_levels: # Avoid re-processing
                      self.floor_levels[f] = level
                      current_level_floors.append(f)

            # Process neighbors of floors at the current level
            for f_current_level in current_level_floors:
                 # Get neighbors safely
                 for f_higher in adj.get(f_current_level, []):
                      in_degree[f_higher] -= 1
                      # If all predecessors are processed, add to next level queue
                      if in_degree[f_higher] == 0:
                          next_q.append(f_higher)

            q = next_q
            level += 1

        # Optional: Check if all floors were assigned a level
        # if len(self.floor_levels) != len(all_floors):
        #     print(f"Warning: Not all floors ({len(all_floors)}) could be assigned a level ({len(self.floor_levels)}).")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
             return 0

        action_cost = 0
        f_stops_levels = set()
        current_floor = None

        # Find current elevator location
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_floor = parts[1]
                break

        # If lift location is missing or doesn't have a level, cannot compute heuristic
        if current_floor is None or current_floor not in self.floor_levels:
             # print(f"Error: Lift location {current_floor} not found or has no level.")
             return float('inf')

        current_level = self.floor_levels[current_floor]

        # Identify required stops and calculate action cost component
        for passenger in self.passengers_to_serve:
            # Check if passenger is already served in the current state
            is_served = f"(served {passenger})" in state

            if not is_served:
                # Check if passenger is boarded
                is_boarded = f"(boarded {passenger})" in state

                if is_boarded:
                    # Passenger is boarded, needs to depart at destination
                    action_cost += 1 # Cost for depart action
                    dest_floor = self.destinations.get(passenger)
                    # Add destination floor level to required stops if valid
                    if dest_floor and dest_floor in self.floor_levels:
                         f_stops_levels.add(self.floor_levels[dest_floor])
                    # else: print(f"Warning: Destination floor {dest_floor} for {passenger} not found or has no level.")
                else:
                    # Passenger is waiting at origin, needs to board and depart
                    action_cost += 2 # Cost for board + depart actions
                    # Find origin floor in the current state
                    origin_floor = None
                    for fact in state:
                         parts = get_parts(fact)
                         if parts and parts[0] == "origin" and len(parts) == 3 and parts[1] == passenger:
                             origin_floor = parts[2]
                             break
                    # Add origin floor level to required stops if valid
                    if origin_floor and origin_floor in self.floor_levels:
                         f_stops_levels.add(self.floor_levels[origin_floor])
                    # else: print(f"Warning: Origin floor {origin_floor} for {passenger} not found or has no level.")


        # Calculate vertical distance component
        vertical_cost = 0
        if f_stops_levels:
            min_stop_level = min(f_stops_levels)
            max_stop_level = max(f_stops_levels)

            # Vertical cost is the distance to the furthest required floor from current
            # This is an admissible lower bound on vertical travel
            vertical_cost = max(abs(current_level - min_stop_level), abs(current_level - max_stop_level))

        # Total heuristic is the sum of action costs and vertical movement costs
        total_cost = action_cost + vertical_cost

        return total_cost
