import math
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts parts from a PDDL fact string by removing parentheses
    and splitting by space. Handles potential empty strings or malformed facts gracefully.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    Returns an empty list if the fact is not a valid parenthesized string.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the total number of actions (move, board, depart)
    required to serve all passengers specified in the goal. It calculates the cost
    for each unserved goal passenger individually and sums these costs. The cost
    for a passenger includes the estimated lift movements needed to pick them up
    (if waiting at origin) and drop them off at their destination, plus the
    board (if applicable) and depart actions.

    # Assumptions
    - The floor structure is linear and defined by the `(above f_i f_j)` predicates,
      meaning floor `f_i` is directly above floor `f_j`. This implies a strict
      vertical ordering.
    - There is a single lift operating in the building.
    - The heuristic calculates costs as if each passenger is served sequentially,
      starting from the current lift position. This might overestimate movement costs
      when multiple passengers can be served efficiently in a single trip (e.g.,
      picking up multiple people at the same floor or dropping off multiple people),
      but it provides a reasonable and efficiently computable estimate suitable for
      greedy best-first search. Admissibility is not required.
    - All passengers mentioned in the goal `(served p)` have a corresponding
      static `(destin p f)` fact defining their destination.
    - Floor names are consistent and correctly referenced in predicates across static
      facts, initial state, and goals.
    - The problem instance is well-formed (e.g., lift exists, floors are defined if needed).

    # Heuristic Initialization
    - Parses static `(above f_i f_j)` facts to determine the numerical level of each
      floor. This allows calculating the distance between floors simply as the
      absolute difference in their levels (number of 'up'/'down' actions).
    - Parses static `(destin p f)` facts to store the destination floor for
      each passenger relevant to the goal.
    - Identifies the set of all passengers that need to be served to reach the
      goal state based on the `(served p)` goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal Completion:** If all goal passengers are already in the `(served p)`
        state (i.e., the set of goal passengers is a subset of passengers served in
        the current state), the goal is reached, and the heuristic value is 0.
    2.  **Identify Lift Location:** Determine the current floor `L` where the lift is
        located using the `(lift-at f)` fact in the current state. If the lift's
        location is unknown, return infinity as the state is likely invalid.
    3.  **Identify Passenger States:** For the current state, determine the status of each
        *unserved* goal passenger by checking for `(origin p O_p)` or `(boarded p)` facts.
    4.  **Calculate Floor Distances:** Use the precomputed floor levels map. The
        distance between any two floors `f1`, `f2` is `abs(level(f1) - level(f2))`.
        Handle cases where floor levels might be unknown (e.g., single-floor problems
        or incomplete definitions) by returning a default distance (e.g., 1 if floors
        are different, 0 if same).
    5.  **Calculate Cost per Unserved Passenger:** Iterate through all passengers `p`
        that are part of the goal but not yet served in the current state:
        - If passenger `p` is **boarded** (`(boarded p)` is true):
          The estimated cost involves moving the lift from its current location `L` to the
          passenger's destination `D_p`, and the passenger departing.
          Cost = `distance(L, D_p)` + 1 (for `depart` action).
        - If passenger `p` is **waiting at origin** (`(origin p O_p)` is true):
          The estimated cost involves moving the lift from `L` to the origin `O_p`,
          the passenger boarding, the lift moving from `O_p` to the destination `D_p`,
          and the passenger departing.
          Cost = `distance(L, O_p)` + 1 (for `board`) + `distance(O_p, D_p)` + 1 (for `depart`).
        - If passenger `p` is unserved but neither boarded nor at origin: This indicates
          an unusual state. A fallback cost estimation is applied (e.g., assuming they
          need to be transported from the current lift location to their destination).
    6.  **Sum Costs:** The total heuristic value `h` is the sum of the costs calculated
        in step 5 for all unserved goal passengers.
    7.  **Return Value:** Return the calculated total cost `h`. Ensure the value is non-negative.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        - Computes floor levels based on 'above' predicates.
        - Stores passenger destinations.
        - Identifies passengers required to be served for the goal.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Precompute floor levels
        self.floor_levels = {}
        adj_above = {} # floor -> floor directly below it
        adj_below = {} # floor -> floor directly above it
        all_floors = set()

        # Extract floor information and 'above' relationships
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            # Assuming domain uses standard typing, e.g., (floor f1)
            # If not, rely on 'above', 'lift-at', 'origin', 'destin' to find floors
            if predicate == 'floor' and len(parts) == 2:
                 all_floors.add(parts[1])
            elif predicate == 'above' and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                # Store adjacency based on (above f_higher f_lower)
                adj_above[f_above] = f_below
                adj_below[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Also consider floors mentioned in init/goal if not typed explicitly
        for fact in task.initial_state.union(task.goals):
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred in ['lift-at', 'origin', 'destin', 'served']: # Predicates involving floors/passengers
                 for part in parts[1:]:
                     # Basic check if it looks like a floor name (often starts with 'f')
                     # This is heuristic; proper type checking would be better if available
                     if isinstance(part, str) and part.startswith('f'): # Simple check
                          # Need a robust way to distinguish floors from passengers if types aren't explicit
                          # Let's assume floors are those involved in 'above', 'lift-at', 'origin', 'destin'
                          # We already add floors from 'above'. Add from others too.
                          if pred == 'lift-at' and len(parts) == 2: all_floors.add(parts[1])
                          if pred == 'origin' and len(parts) == 3: all_floors.add(parts[2])
                          if pred == 'destin' and len(parts) == 3: all_floors.add(parts[2])


        if all_floors:
            # Find a bottom-most floor (has nothing below it according to 'above' facts)
            bottom_floor = None
            potential_bottom = all_floors - set(adj_above.keys())

            if len(potential_bottom) == 1:
                 bottom_floor = list(potential_bottom)[0]
            elif len(potential_bottom) > 1:
                 # If multiple floors seem to be bottom floors, it might indicate disconnected components
                 # or a non-standard setup. We need one starting point for level calculation.
                 # Pick one arbitrarily, assuming the relevant part of the structure is connected.
                 bottom_floor = list(potential_bottom)[0]
                 print(f"Warning: Multiple potential bottom floors found: {potential_bottom}. Using {bottom_floor} as base for level calculation.")
            elif not potential_bottom and len(all_floors) > 0:
                 # No floor lacks something below it (and floors exist). Could be a cycle or single floor.
                 if len(all_floors) == 1:
                      bottom_floor = list(all_floors)[0] # Single floor is its own "bottom"
                 else:
                      print("Warning: Could not find a clear bottom floor (possible cycle or definition issue). Floor levels might be inaccurate.")
                      # Heuristic will rely on default distances if levels aren't computed.


            if bottom_floor is not None:
                # Perform BFS from the bottom floor to assign levels upwards
                queue = [(bottom_floor, 0)]
                visited = {bottom_floor}
                self.floor_levels[bottom_floor] = 0
                head = 0
                while head < len(queue):
                    curr_f, curr_level = queue[head]
                    head += 1

                    # Check floor directly above the current floor using adj_below map
                    if curr_f in adj_below:
                        f_above = adj_below[curr_f]
                        if f_above not in visited:
                            visited.add(f_above)
                            self.floor_levels[f_above] = curr_level + 1
                            queue.append((f_above, curr_level + 1))

                # Optional: Check if all known floors were assigned a level
                if len(self.floor_levels) != len(all_floors):
                     unreached_floors = all_floors - set(self.floor_levels.keys())
                     print(f"Warning: Floor level calculation might be incomplete. {len(unreached_floors)} floors were not reached (e.g., {list(unreached_floors)[:3]}).")

            # If bottom_floor was None and not single floor case, self.floor_levels remains empty.

        # 2. Store passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # 3. Identify passengers who need to be served for the goal
        self.goal_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'served' and len(parts) == 2:
                self.goal_passengers.add(parts[1])

        # Verify all goal passengers have destinations defined
        for p in self.goal_passengers:
            if p not in self.destinations:
                 print(f"Error: Goal requires serving passenger {p}, but their destination is not defined in static facts.")
                 # This might require raising an error or handling gracefully depending on requirements.


    def _get_distance(self, f1, f2):
        """
        Calculates the distance between two floors based on their precomputed levels.
        Distance is the absolute difference in levels (number of up/down moves).
        Handles cases where floor levels might be unknown.
        """
        if f1 == f2:
            return 0

        level1 = self.floor_levels.get(f1)
        level2 = self.floor_levels.get(f2)

        if level1 is not None and level2 is not None:
            # Both levels are known
            return abs(level1 - level2)
        else:
            # Levels are unknown for one or both floors.
            # This might happen with incomplete 'above' definitions, single-floor problems,
            # or if level computation failed.
            # If floors are different but levels unknown, return 1 as a minimum default cost.
            print(f"Debug: Floor level unknown for calculating distance between {f1} and {f2}. Assuming distance 1.")
            return 1

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Sums the estimated costs for serving each unserved goal passenger.
        """
        state = node.state

        # Identify passengers already served in the current state
        served_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'served' and len(parts) == 2:
                  served_in_state.add(parts[1])

        # Check if all goal passengers are served
        if self.goal_passengers.issubset(served_in_state):
            return 0 # Goal reached

        # Find current lift location
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'lift-at' and len(parts) == 2:
                lift_floor = parts[1]
                break

        if lift_floor is None:
            # This should not happen in a valid Miconic state during search.
            print("Error: Lift location ('lift-at') not found in state. Cannot compute heuristic.")
            return float('inf') # Return infinity to indicate an issue or invalid state

        total_cost = 0
        passengers_at_origin = {} # Map: passenger -> origin_floor
        passengers_boarded = set() # Set: passenger

        # Parse current state for status of goal passengers who are not yet served
        unserved_goal_passengers = self.goal_passengers - served_in_state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'origin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                if passenger in unserved_goal_passengers:
                    passengers_at_origin[passenger] = floor
            elif predicate == 'boarded' and len(parts) == 2:
                passenger = parts[1]
                if passenger in unserved_goal_passengers:
                    passengers_boarded.add(passenger)

        # Calculate cost for each unserved goal passenger
        for p in unserved_goal_passengers:
            dest_floor = self.destinations.get(p)
            if dest_floor is None:
                # This should have been caught in __init__, but double-check.
                print(f"Error: Destination unknown for unserved goal passenger {p}. Skipping cost.")
                continue

            if p in passengers_boarded:
                # Passenger is boarded: needs move to destination + depart
                cost_move_to_dest = self._get_distance(lift_floor, dest_floor)
                cost_depart = 1
                total_cost += cost_move_to_dest + cost_depart
            elif p in passengers_at_origin:
                # Passenger is waiting at origin: needs move to origin + board + move to dest + depart
                origin_floor = passengers_at_origin[p]
                cost_move_to_origin = self._get_distance(lift_floor, origin_floor)
                cost_board = 1
                cost_move_origin_to_dest = self._get_distance(origin_floor, dest_floor)
                cost_depart = 1
                total_cost += cost_move_to_origin + cost_board + cost_move_origin_to_dest + cost_depart
            else:
                # Passenger 'p' is required for the goal, is not served,
                # but is neither 'boarded' nor at 'origin'. This is an unexpected state
                # under normal operation (origin -> boarded -> served).
                # Possible reasons: initial state anomaly, intermediate inconsistent state (less likely),
                # or passenger was dropped off incorrectly in a previous step (if possible).
                # Apply a fallback cost: assume they need transport from lift's current location.
                print(f"Warning: Unserved goal passenger {p} is not boarded and not at origin. Estimating cost as move from lift location {lift_floor} to destination {dest_floor} + depart.")
                cost_move_to_dest = self._get_distance(lift_floor, dest_floor)
                cost_depart = 1
                total_cost += cost_move_to_dest + cost_depart


        # The heuristic value should be non-negative.
        # Since costs (distances >= 0, action counts = 1) are added, total_cost >= 0.
        return total_cost
