from heuristics.heuristic_base import Heuristic
from task import Task


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the cost to reach the goal state (all passengers served)
        by summing the estimated minimum actions required for each unserved passenger
        independently. The estimated cost for a passenger includes movement actions
        to their origin (if waiting) and destination floors, plus the board and
        depart actions.

    Assumptions:
        - The 'above' predicates in the static facts define a total order of floors,
          forming a single vertical chain (e.g., f1 < f2 < ... < fn).
        - Floor names are consistent and can be mapped to integer indices based on
          this 'above' order.
        - Passenger status in any reachable state is either waiting (origin fact),
          boarded (boarded fact), or served (served fact).
        - Passenger destination facts ('destin') are available in the static information.

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the static facts:
        1. It parses the 'above' facts to determine the vertical order of floors.
           It identifies the highest floor (one not immediately above any other)
           and builds the floor chain downwards using the 'above' relations.
           It creates a mapping from floor names to integer indices (0-based)
           where index 0 is the lowest floor. It checks if the 'above' facts
           form a single contiguous chain; otherwise, the floor mapping is marked
           as invalid, and the heuristic will return infinity for any state.
        2. It parses the 'destin' facts to store the destination floor for each
           passenger, which is needed to identify required drop-off locations.
        3. It identifies all passengers present in the problem from the 'destin'
           and initial state facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Check if the state is the goal state (all passengers served). If yes,
           the heuristic value is 0.
        2. Identify the current floor of the elevator from the 'lift-at' fact.
           If the floor is unknown or not in the pre-processed floor map, return
           infinity (state is likely invalid or unreachable).
        3. Identify all unserved passengers by comparing the set of all passengers
           (from initialization) with the set of passengers marked as 'served' in
           the current state.
        4. If there are no unserved passengers, return 0 (goal state).
        5. Initialize the total heuristic value to 0.
        6. For each unserved passenger `p`:
           a. Get the passenger's destination floor `f_dest` from the pre-processed
              static information. If the destination is unknown or not in the
              floor map, return infinity.
           b. Check if the passenger `p` is currently waiting at their origin floor
              `f_origin` (has an `(origin p f_origin)` fact in the state).
           c. Check if the passenger `p` is currently boarded (has a `(boarded p)`
              fact in the state).
           d. If the passenger is waiting at `f_origin`:
              Estimate the cost for this passenger's journey as the sum of:
              - Movement cost from the current elevator floor to `f_origin`:
                `abs(self.floor_to_int[current_floor] - self.floor_to_int[f_origin])`
              - Cost of the 'board' action: 1
              - Movement cost from `f_origin` to `f_dest`:
                `abs(self.floor_to_int[f_origin] - self.floor_to_int[f_dest])`
              - Cost of the 'depart' action: 1
              Add this estimated cost for passenger `p` to the total heuristic value.
           e. If the passenger is boarded:
              Estimate the cost for this passenger's remaining journey as the sum of:
              - Movement cost from the current elevator floor to `f_dest`:
                `abs(self.floor_to_int[current_floor] - self.floor_to_int[f_dest])`
              - Cost of the 'depart' action: 1
              Add this estimated cost for passenger `p` to the total heuristic value.
           f. (Note: In a valid state, an unserved passenger must be either waiting
              or boarded. If neither is true, it indicates an invalid state,
              handled by returning infinity if floor lookups fail).
        7. The final heuristic value is the accumulated sum from step 6.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task

        # --- Heuristic Initialization ---
        # Parse 'above' facts to build floor order
        above_map = {} # f_below -> f_above (f_above is immediately above f_below)
        below_map = {} # f_above -> f_below
        all_floors_in_above = set()

        for fact in task.static:
            if fact.startswith('(above '):
                parts = fact.strip('()').split()
                # PDDL: (above f_above f_below) based on action definitions
                f_above = parts[1]
                f_below = parts[2]
                above_map[f_below] = f_above
                below_map[f_above] = f_below
                all_floors_in_above.add(f_above)
                all_floors_in_above.add(f_below)

        self.floor_to_int = {}
        self.int_to_floor = {}
        self.min_floor_int = 0
        self.max_floor_int = -1 # Indicate invalid range initially

        if not all_floors_in_above:
             # No 'above' facts. Try to find floors from other facts.
             all_floor_names = set()
             for fact in task.initial_state | task.static | task.goals:
                  parts = fact.strip('()').split()
                  for part in parts:
                      # Simple check: does it start with 'f' and contain digits?
                      if part.startswith('f') and any(char.isdigit() for char in part):
                          all_floor_names.add(part)

             if len(all_floor_names) == 1:
                 # Single floor case
                 single_floor = list(all_floor_names)[0]
                 self.floor_to_int[single_floor] = 0
                 self.int_to_floor[0] = single_floor
                 self.min_floor_int = 0
                 self.max_floor_int = 0
             elif all_floor_names:
                 # Multiple floors but no 'above' facts to order them
                 print("Error: Cannot determine floor order from 'above' facts, and multiple floors exist.")
                 # floor_to_int remains empty, heuristic will return inf
             else:
                 # No floors found at all
                 print("Error: No floors found in the problem definition.")
                 # floor_to_int remains empty, heuristic will return inf
        else:
            # Find the highest floor (a floor that is not the 'above' part of any 'above' fact)
            highest_floor = None
            for f in all_floors_in_above:
                 if f not in below_map:
                     highest_floor = f
                     break

            if highest_floor is None:
                 print("Error: 'above' facts do not define a clear highest floor (possible cycle or disconnected graph).")
                 # floor_to_int remains empty, heuristic will return inf
            else:
                # Build the ordered list of floors starting from the highest, going down
                floor_index = len(all_floors_in_above) - 1 # Start indexing from the top
                current_floor = highest_floor
                visited_floors = set()

                while current_floor is not None and current_floor not in visited_floors:
                    visited_floors.add(current_floor)
                    self.floor_to_int[current_floor] = floor_index
                    self.int_to_floor[floor_index] = current_floor
                    floor_index -= 1
                    current_floor = above_map.get(current_floor) # Move down to the floor below

                # Check if all floors mentioned in 'above' facts were included in the chain
                if len(self.floor_to_int) != len(all_floors_in_above):
                     print("Error: 'above' facts do not form a single contiguous floor chain.")
                     self.floor_to_int = {} # Mark as invalid
                     self.min_floor_int = 0
                     self.max_floor_int = -1 # Indicate invalid range
                else:
                     # Successfully built chain
                     self.min_floor_int = 0
                     self.max_floor_int = len(self.floor_to_int) - 1 # Correct max index


        if not self.floor_to_int:
             print("Heuristic initialization failed: Floor mapping is empty or invalid.")


        # Store passenger destinations and identify all passengers
        self.passenger_destin = {}
        self.all_passengers = set()
        for fact in task.static:
            if fact.startswith('(destin '):
                parts = fact.strip('()').split()
                p = parts[1]
                f_dest = parts[2]
                self.passenger_destin[p] = f_dest
                self.all_passengers.add(p)

        # Add passengers from initial state if not in static destin (less common but possible)
        for fact in task.initial_state:
             if fact.startswith('(origin '):
                 parts = fact.strip('()').split()
                 self.all_passengers.add(parts[1])
             elif fact.startswith('(boarded '):
                 parts = fact.strip('()').split()
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """
        Computes the miconic heuristic for the given state.
        """
        state = node.state

        # Check for goal state
        if self.task.goal_reached(state):
            return 0

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Extract current elevator floor
        current_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_floor = fact.strip('()').split()[1]
                break

        # If elevator location is unknown or invalid, state is likely unreachable
        if current_floor is None or current_floor not in self.floor_to_int:
             return float('inf')

        current_floor_int = self.floor_to_int[current_floor]

        # 2. Identify unserved passengers
        served_passengers = {p.strip('()').split()[1] for p in state if p.startswith('(served ')}
        unserved_passengers = self.all_passengers - served_passengers

        # If no unserved passengers, it's the goal state (already handled)
        if not unserved_passengers:
             return 0

        # 5. Initialize the heuristic value
        h_value = 0

        # 6. Iterate through each unserved passenger
        for p in unserved_passengers:
            # a. Get destination floor
            f_dest = self.passenger_destin.get(p)
            if f_dest is None or f_dest not in self.floor_to_int:
                 return float('inf') # Destination must be known and valid

            f_dest_int = self.floor_to_int[f_dest]

            # b/c. Check if passenger is waiting or boarded
            is_waiting = False
            is_boarded = False
            f_origin = None

            # Check state facts for origin or boarded status
            for fact in state:
                if fact.startswith(f'(origin {p} '):
                    f_origin = fact.strip('()').split()[2]
                    is_waiting = True
                    break # Found origin, passenger is waiting
                elif fact.startswith(f'(boarded {p})'):
                    is_boarded = True
                    break # Found boarded, passenger is boarded

            # d. If the passenger is waiting
            if is_waiting:
                if f_origin is None or f_origin not in self.floor_to_int:
                     return float('inf') # Origin must be known and valid

                f_origin_int = self.floor_to_int[f_origin]

                # Estimate cost: move current -> f_origin, board, move f_origin -> f_dest, depart
                dist_to_origin = abs(current_floor_int - f_origin_int)
                dist_origin_to_dest = abs(f_origin_int - f_dest_int)
                cost_p = dist_to_origin + 1 + dist_origin_to_dest + 1
                h_value += cost_p

            # e. If the passenger is boarded
            elif is_boarded:
                # Estimate cost: move current -> f_dest, depart
                dist_to_dest = abs(current_floor_int - f_dest_int)
                cost_p = dist_to_dest + 1
                h_value += cost_p

            # f. If passenger is unserved but neither waiting nor boarded - handled by returning inf if lookups fail

        # 7. Final heuristic value is the sum
        return h_value
