from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys # Used for returning a large value for invalid states

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    return fact.strip()[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board and depart actions for unserved passengers
    and adds an estimate of the minimum lift movement actions needed to visit
    all floors where pickups or dropoffs are required.

    # Assumptions
    - The 'above' facts in the PDDL problem define a linear ordering of floors.
      If the 'above' facts do not form a single linear chain covering all relevant
      floors, the heuristic falls back to an alphabetical ordering of floors,
      which might impact its accuracy.
    - Passenger destinations ('destin' facts) are static and available in the
      initial state or static facts.
    - All passengers mentioned in 'origin' or 'destin' facts need to be served.
    - All actions have a cost of 1.

    # Heuristic Initialization
    - Parses 'above' facts from static, initial, and goal states to create a
      mapping between floor names and integer indices, representing their order.
      It attempts to build a linear chain based on 'above' relations, falling
       back to alphabetical sorting if the chain is incomplete or malformed.
    - Parses 'destin' facts from static and initial states to store the
      destination floor for each passenger.
    - Identifies all passengers present in 'origin' or 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located from the state facts `(lift-at ?f)`. Get its corresponding index using the pre-calculated floor mapping. If the lift location is not found or the floor is not mapped, return a large heuristic value (indicating an issue or difficulty).

    2.  **Identify Unserved Passengers:** Iterate through all known passengers. A passenger is unserved if the fact `(served ?p)` is not present in the current state.

    3.  **Count Board and Depart Actions:**
        - For each unserved passenger `p`:
            - Check if `p` is currently waiting at their origin floor (`(origin ?p ?f)` is in the state). If yes, increment the heuristic by 1 (for the required `board` action).
            - Check if `p` is currently boarded (`(boarded ?p)` is in the state). If yes, increment the heuristic by 1 (for the required `depart` action).
            - Note: A passenger needs both a board and a depart action to be served, unless they start boarded. This counts the *remaining* board action needed (if waiting) and the *remaining* depart action needed (if unserved).

    4.  **Identify Required Floors for Movement:** Collect the set of unique floor indices that the lift *must* visit to serve the unserved passengers. These are:
        - The origin floor for every passenger who is currently waiting (`(origin ?p ?f)`).
        - The destination floor for every passenger who is currently boarded (`(boarded ?p)`).

    5.  **Calculate Movement Cost:**
        - If the set of required floor indices is empty (meaning all unserved passengers are either waiting at the lift's current floor or are boarded and need to go to the lift's current floor, or there are no unserved passengers), the movement cost is 0.
        - Otherwise, find the minimum (`min_req_idx`) and maximum (`max_req_idx`) indices among the required floors.
        - The movement cost is estimated as the minimum travel distance required for the lift, starting from its current floor index (`current_floor_idx`), to reach and traverse the range of required floors `[min_req_idx, max_req_idx]`. A reasonable estimate for this is the distance from the current floor to the closest end of the required range, plus the length of the range: `min(abs(current_floor_idx - min_req_idx), abs(current_floor_idx - max_req_idx)) + (max_req_idx - min_req_idx)`. This captures the cost of reaching the 'action zone' and then moving within it.

    6.  **Sum Costs:** Add the calculated movement cost to the heuristic value accumulated from counting board and depart actions.

    7.  **Return Total Heuristic:** The final sum is the estimated number of actions to reach a goal state. If no passengers need serving, the heuristic will correctly be 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the list of all passengers.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Build floor order mapping
        self.floor_to_index, self.index_to_floor = self._build_floor_order(
            self.static_facts, self.initial_state, self.goals
        )

        # Store passenger destinations and identify all passengers
        self.destinations = {}
        self.all_passengers = set()

        # Look for destin facts in static or initial state
        for fact in self.static_facts | self.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "destin" and len(parts) == 3:
                 passenger, floor = parts[1], parts[2]
                 self.destinations[passenger] = floor
                 self.all_passengers.add(passenger)
             elif parts[0] == "origin" and len(parts) == 3: # Also collect passengers from origin facts
                 self.all_passengers.add(parts[1])
             elif parts[0] == "served" and len(parts) == 2: # Passengers might be served in initial state
                 self.all_passengers.add(parts[1])
             elif parts[0] == "boarded" and len(parts) == 2: # Passengers might be boarded in initial state
                 self.all_passengers.add(parts[1])


        # Filter passengers to only include those with destinations, as others cannot be served
        # This assumes valid PDDL where any passenger requiring service has a destin fact.
        self.all_passengers = {p for p in self.all_passengers if p in self.destinations}


    def _build_floor_order(self, static_facts, initial_state, goals):
        """
        Builds floor_to_index and index_to_floor maps from (above f_above f_below) facts.
        Assumes 'above' facts define a linear order. Falls back to alphabetical if needed.
        """
        above_map = {} # maps floor_below -> floor_above
        all_floors = set()

        # Collect all floors mentioned in relevant facts
        for fact in static_facts | initial_state | goals:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "above" and len(parts) == 3:
                 f_above, f_below = parts[1], parts[2]
                 above_map[f_below] = f_above
                 all_floors.add(f_above)
                 all_floors.add(f_below)
            elif predicate == "lift-at" and len(parts) == 2:
                 all_floors.add(parts[1])
            elif predicate in ["origin", "destin"] and len(parts) == 3:
                 all_floors.add(parts[2]) # The floor argument

        if not all_floors:
            return {}, {} # No floors found

        # Find the lowest floor: a floor that is not a value in above_map
        # This is the floor that no other floor is directly above.
        potential_lowest = all_floors - set(above_map.values())

        lowest_floor = None
        if len(potential_lowest) == 1:
            lowest_floor = potential_lowest.pop()
        elif potential_lowest:
             # Multiple potential lowest floors or disconnected components.
             # Fallback: Sort all floors alphabetically and use that order.
             sorted_floors = sorted(list(all_floors))
             floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
             index_to_floor = {i: f for i, f in enumerate(sorted_floors)}
             # print(f"Warning: Could not determine unique lowest floor. Using alphabetical order: {sorted_floors}", file=sys.stderr) # Debugging
             return floor_to_index, index_to_floor
        else:
             # No potential lowest found (e.g., cycle or no above facts).
             # Fallback: Sort all floors alphabetically.
             sorted_floors = sorted(list(all_floors))
             if sorted_floors:
                 floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
                 index_to_floor = {i: f for i, f in enumerate(sorted_floors)}
                 # print(f"Warning: No lowest floor found from 'above' facts. Using alphabetical order: {sorted_floors}", file=sys.stderr) # Debugging
                 return floor_to_index, index_to_index
             else:
                 # Should not happen if all_floors is not empty, but defensive.
                 # print("Warning: No floors found at all.", file=sys.stderr) # Debugging
                 return {}, {}


        floor_to_index = {}
        index_to_floor = {}
        current_floor = lowest_floor
        index = 0

        # Build the chain upwards
        while current_floor is not None:
            if current_floor in floor_to_index:
                 # Cycle detected or already visited. Stop chain building.
                 # print(f"Warning: Cycle detected in floor 'above' facts at {current_floor}. Stopping chain.", file=sys.stderr) # Debugging
                 break
            floor_to_index[current_floor] = index
            index_to_floor[index] = current_floor
            index += 1
            # Find the floor directly above the current floor
            # The floor above `f_below` is `above_map[f_below]`
            current_floor = above_map.get(current_floor)

        # If not all floors were included in the chain, add remaining ones sorted alphabetically
        chained_floors = set(floor_to_index.keys())
        if len(chained_floors) != len(all_floors):
             unchained_floors = sorted(list(all_floors - chained_floors))
             start_index = len(floor_to_index)
             for i, floor in enumerate(unchained_floors):
                 floor_to_index[floor] = start_index + i
                 index_to_floor[start_index + i] = floor
             # print(f"Warning: Not all floors in 'above' chain. Added unchained floors alphabetically: {unchained_floors}", file=sys.stderr) # Debugging


        return floor_to_index, index_to_floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # If floor mapping failed during init, we cannot compute a meaningful heuristic
        if not self.floor_to_index:
             # Check if goal is reached (all passengers served)
             passengers_to_serve = [p for p in self.all_passengers if f'(served {p})' not in state]
             if not passengers_to_serve:
                 return 0 # Goal state (or no passengers to serve)
             else:
                 # Cannot compute meaningful heuristic without floor info.
                 # Return a value indicating difficulty.
                 return len(passengers_to_serve) * 1000 # Arbitrary large penalty


        # Find current lift floor index
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This state is likely invalid (lift location unknown).
             # Return a high heuristic value.
             passengers_to_serve = [p for p in self.all_passengers if f'(served {p})' not in state]
             if not passengers_to_serve:
                 return 0 # Goal state
             else:
                 return len(passengers_to_serve) * 1000 # Arbitrary large penalty


        current_floor_idx = self.floor_to_index.get(current_lift_floor)
        if current_floor_idx is None:
             # Current lift floor is not in our floor map. Problem with floor parsing or state.
             # Return a high value.
             passengers_to_serve = [p for p in self.all_passengers if f'(served {p})' not in state]
             if not passengers_to_serve:
                 return 0 # Goal state
             else:
                 return len(passengers_to_serve) * 1000 # Arbitrary large penalty


        heuristic = 0
        required_floor_indices = set()

        # Track which passengers are waiting or boarded in the current state
        waiting_passengers_in_state = set() # {passenger}
        boarded_passengers_in_state = set() # {passenger}
        served_passengers_in_state = set() # {passenger}

        # Pre-parse state facts for quick lookup
        state_predicates = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate not in state_predicates:
                state_predicates[predicate] = []
            state_predicates[predicate].append(parts)

        # Populate waiting, boarded, served sets
        for parts in state_predicates.get("origin", []):
             if len(parts) == 3:
                 waiting_passengers_in_state.add(parts[1])
        for parts in state_predicates.get("boarded", []):
             if len(parts) == 2:
                 boarded_passengers_in_state.add(parts[1])
        for parts in state_predicates.get("served", []):
             if len(parts) == 2:
                 served_passengers_in_state.add(parts[1])


        # Iterate through all known passengers to find unserved ones and calculate costs
        for passenger in self.all_passengers:
            if passenger in served_passengers_in_state:
                continue # This passenger is served

            destin_f = self.destinations.get(passenger)

            if destin_f is None:
                 # Should not happen if self.all_passengers is filtered correctly,
                 # but defensive check. Cannot serve passenger without destination.
                 # Add a large penalty? Or assume valid input. Assuming valid input.
                 continue # Skip passenger if destination is unknown


            destin_idx = self.floor_to_index.get(destin_f)
            if destin_idx is None:
                 # Destination floor not in our floor map. Problem with floor parsing or PDDL.
                 # Return a high value.
                 # Count unserved passengers to scale penalty
                 unserved_count = len(self.all_passengers) - len(served_passengers_in_state)
                 return unserved_count * 1000 # Arbitrary large penalty


            # Check passenger state (waiting or boarded)
            is_waiting = passenger in waiting_passengers_in_state
            is_boarded = passenger in boarded_passengers_in_state

            # A passenger should be either waiting or boarded if unserved and not at origin/dest
            # Assuming valid states adhere to this.

            if is_waiting:
                # Passenger is waiting at origin
                # Find origin floor from state facts
                origin_f = None
                for parts in state_predicates.get("origin", []):
                    if len(parts) == 3 and parts[1] == passenger:
                        origin_f = parts[2]
                        break

                if origin_f is None:
                    # Passenger is marked as waiting but origin fact is missing? Invalid state.
                    # Return high penalty.
                    unserved_count = len(self.all_passengers) - len(served_passengers_in_state)
                    return unserved_count * 1000 # Arbitrary large penalty


                origin_idx = self.floor_to_index.get(origin_f)

                if origin_idx is None:
                     # Origin floor not in our floor map. Problem with floor parsing or PDDL.
                     # Return a high value.
                     unserved_count = len(self.all_passengers) - len(served_passengers_in_state)
                     return unserved_count * 1000 # Arbitrary large penalty

                heuristic += 1 # Cost for 'board' action
                required_floor_indices.add(origin_idx)
                required_floor_indices.add(destin_idx) # Need to visit destination eventually

            elif is_boarded:
                # Passenger is boarded
                heuristic += 1 # Cost for 'depart' action
                required_floor_indices.add(destin_idx)

            # If passenger is unserved but neither waiting nor boarded, it's an invalid state
            # or they are at their origin/destin but not boarded/served yet?
            # The PDDL implies unserved passengers are either waiting or boarded.
            # If they are at their origin floor and lift is there, they can board.
            # If they are at their destin floor and lift is there and they are boarded, they can depart.
            # The heuristic counts actions needed *from the current state*.
            # If a waiting passenger is at their origin and lift is there, board is applicable.
            # If a boarded passenger is at their destin and lift is there, depart is applicable.
            # The heuristic counts the *need* for board/depart regardless of lift position.
            # The movement cost covers getting the lift there.

        # Calculate movement cost
        movement_cost = 0
        if required_floor_indices:
            min_req_idx = min(required_floor_indices)
            max_req_idx = max(required_floor_indices)

            # Movement cost estimate: distance to closest end of range + range length
            dist_to_min = abs(current_floor_idx - min_req_idx)
            dist_to_max = abs(current_floor_idx - max_req_idx)

            movement_cost = min(dist_to_min, dist_to_max) + (max_req_idx - min_req_idx)

        heuristic += movement_cost

        # If no passengers need serving, heuristic will be 0.
        # This is handled by the loop over self.all_passengers and the check for served.

        return heuristic

