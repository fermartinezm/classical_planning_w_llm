import re
import math

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing up the estimated cost
    for each unserved passenger independently. The estimated cost for a
    passenger includes the movement cost for the lift to reach their
    origin (if waiting) and destination, plus the cost of the board and
    depart actions.

    Assumptions:
    - Floor names are in the format 'f' followed by a number (e.g., 'f1', 'f10').
    - The 'above' predicates define a total order on floors consistent with
      the numerical order of the floor names.
    - The state representation uses strings for facts, e.g., '(predicate arg1 arg2)'.
    - The Task object provides initial_state, goals, and static facts.
    - Every passenger mentioned in the goals has a 'destin' fact in the static information.
    - Every unserved, non-boarded passenger mentioned in the goals has an 'origin' fact in the state.

    Heuristic Initialization:
    The heuristic is initialized with the Task object.
    It extracts and stores:
    1. A mapping from floor names (e.g., 'f1') to integer indices (e.g., 1),
       based on the numerical order of floor names found in the initial state
       and static facts.
    2. A mapping from passenger names to their destination floor names,
       extracted from the 'destin' static facts.
    3. A set of all passenger names, extracted from the goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current floor of the lift from the state facts. Convert
       the floor name to its integer index using the precomputed mapping.
    2. Initialize the total heuristic value to 0.
    3. Identify which passengers are currently 'served', 'boarded', and at their 'origin'
       based on the current state facts. Store origin floors for waiting passengers.
    4. Iterate through the set of all passengers in the problem (extracted from goals).
    5. For each passenger `p`:
       a. If `p` is in the set of 'served' passengers, continue (cost is 0).
       b. If `p` is in the set of 'boarded' passengers:
          i. Get `p`'s destination floor name from the stored mapping.
          ii. Convert the destination floor name to its integer index.
          iii. The estimated cost for this passenger is the absolute difference
               between the current lift floor index and the destination floor index
               (estimated movement cost) plus 1 (for the 'depart' action). Add
               this cost to the total heuristic.
       c. If `p` is not served and not boarded, they must be waiting at their origin.
          i. Find `p`'s origin floor name from the 'origin' facts identified in step 3.
          ii. Convert the origin floor name to its integer index.
          iii. Get `p`'s destination floor name from the stored mapping.
          iv. Convert the destination floor name to its integer index.
          v. The estimated cost for this passenger is the absolute difference
             between the current lift floor index and the origin floor index
             (estimated movement to pickup) plus 1 (for the 'board' action)
             plus the absolute difference between the origin floor index and
             the destination floor index (estimated movement to dropoff) plus
             1 (for the 'depart' action). Add this cost to the total heuristic.
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        self.static_facts = task.static
        self.goal_facts = task.goals
        self.initial_state_facts = task.initial_state
        # Combine initial state and static facts to find all relevant objects (floors, passengers)
        all_facts = self.static_facts | self.initial_state_facts

        self.floor_to_idx = self._build_floor_mapping(all_facts)
        self.passenger_dest = self._extract_passenger_destinations(self.static_facts)
        self.all_passengers = self._extract_all_passengers(self.goal_facts)

    def _build_floor_mapping(self, facts):
        """
        Builds a mapping from floor names (e.g., 'f1') to integer indices (e.g., 1).
        Assumes floor names are 'f' followed by a number and sorts them numerically.
        Extracts floor names from all provided facts.
        """
        floor_names = set()
        # Extract all floor names mentioned in facts
        for fact in facts:
            parts = fact.strip("()").split()
            if len(parts) > 1:
                for part in parts[1:]:
                    # Simple check if it looks like a floor name 'f...'
                    if part.startswith('f') and len(part) > 1:
                         # Check if the rest is digits
                         if part[1:].isdigit():
                            floor_names.add(part)

        # Sort floor names numerically based on the number part
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))

        # Create mapping
        floor_to_idx = {floor_name: i + 1 for i, floor_name in enumerate(sorted_floor_names)}
        return floor_to_idx

    def _extract_passenger_destinations(self, static_facts):
        """
        Extracts passenger destination floors from static facts.
        """
        passenger_dest = {}
        for fact in static_facts:
            parts = fact.strip("()").split()
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                passenger_dest[passenger] = destination_floor
        return passenger_dest

    def _extract_all_passengers(self, goal_facts):
        """
        Extracts all passenger names from the goal facts (assuming goals are all (served p)).
        """
        all_passengers = set()
        for fact in goal_facts:
            parts = fact.strip("()").split()
            if parts[0] == 'served' and len(parts) == 2:
                all_passengers.add(parts[1])
        return all_passengers


    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.

        @param state: A frozenset of strings representing the current state facts.
        @return: An integer heuristic value. Returns float('inf') for states
                 that are likely unreachable or inconsistent based on expected
                 miconic problem structure (e.g., lift location missing,
                 passenger destination missing).
        """
        h = 0

        # Find current lift floor
        current_lift_floor_name = None
        for fact in state:
            parts = fact.strip("()").split()
            if parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_floor_name = parts[1]
                break

        if current_lift_floor_name is None:
             # Should not happen in a valid miconic state
             return float('inf')

        current_lift_floor_idx = self.floor_to_idx.get(current_lift_floor_name)
        if current_lift_floor_idx is None:
             # Should not happen if _build_floor_mapping is correct
             return float('inf')

        # Identify passenger states in the current state
        served_passengers = set()
        boarded_passengers = set()
        waiting_passengers_info = {} # {passenger_name: origin_floor_name}

        for fact in state:
            parts = fact.strip("()").split()
            if len(parts) >= 2:
                pred = parts[0]
                obj1 = parts[1]

                if pred == 'served' and len(parts) == 2:
                     served_passengers.add(obj1)
                elif pred == 'boarded' and len(parts) == 2:
                     boarded_passengers.add(obj1)
                elif pred == 'origin' and len(parts) == 3:
                    p, f = obj1, parts[2]
                    waiting_passengers_info[p] = f

        # Calculate heuristic based on the state of all passengers
        for passenger in self.all_passengers:
            if passenger in served_passengers:
                continue # Passenger is served, cost is 0

            # Passenger is not served
            dest_floor_name = self.passenger_dest.get(passenger)
            if dest_floor_name is None:
                 # Passenger has no destination? Problem definition issue.
                 # Treat as unsolvable from this state.
                 return float('inf')

            dest_floor_idx = self.floor_to_idx.get(dest_floor_name)
            if dest_floor_idx is None:
                 # Destination floor not in mapping? Problem definition issue.
                 # Treat as unsolvable.
                 return float('inf')

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to depart at destination
                # Cost = moves to destination + 1 (depart action)
                h += abs(current_lift_floor_idx - dest_floor_idx) + 1
            elif passenger in waiting_passengers_info:
                # Passenger is waiting at origin, needs board and depart
                origin_floor_name = waiting_passengers_info[passenger]
                origin_floor_idx = self.floor_to_idx.get(origin_floor_name)

                if origin_floor_idx is None:
                     # Origin floor not in mapping? Problem definition issue.
                     # Treat as unsolvable.
                     return float('inf')

                # Cost = moves to origin + 1 (board) + moves from origin to destination + 1 (depart)
                h += abs(current_lift_floor_idx - origin_floor_idx) + 1 + abs(origin_floor_idx - dest_floor_idx) + 1
            else:
                # Passenger is not served, not boarded, and not at origin.
                # This state is inconsistent with the domain model if the initial state was valid.
                # A passenger must be at origin, boarded, or served.
                # Treat as unsolvable from this state.
                return float('inf')


        return h
