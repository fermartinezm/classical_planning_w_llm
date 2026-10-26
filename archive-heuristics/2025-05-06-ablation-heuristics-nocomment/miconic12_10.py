# Assume heuristics.heuristic_base.Heuristic is available
from heuristics.heuristic_base import Heuristic

from collections import defaultdict, deque
from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def topological_sort_floors(floor_objects, above_facts):
    """
    Performs topological sort on floors based on (above f_high f_low) facts.
    Returns a list of floors from lowest to highest.
    """
    # Graph where edge f_low -> f_high means f_high is above f_low
    graph = {f: [] for f in floor_objects}
    in_degree = {f: 0 for f in floor_objects}

    for fact in above_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'above':
            f_high = parts[1]
            f_low = parts[2]
            # Only add edge if both floors are in the list of objects
            if f_low in graph and f_high in graph:
                 graph[f_low].append(f_high)
                 in_degree[f_high] += 1
            # else: print(f"Warning: 'above' fact references unknown floor: {fact}") # Optional warning


    queue = deque([f for f in floor_objects if in_degree[f] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # If the result doesn't contain all floor objects, it means the 'above' facts
    # don't form a single connected component including all floors.
    # For miconic, this implies an invalid floor structure.
    # In a valid miconic problem, all floors should be in a single chain.
    # If this happens, the heuristic might be unreliable.
    # We could potentially fall back to alphabetical sort or raise an error.
    # Assuming valid miconic structure for now.
    if len(result) != len(floor_objects):
         # Fallback or error handling could go here.
         # For this problem, we assume valid input.
         pass


    # The topological sort gives floors from lowest in-degree (lowest) to highest in-degree (highest)
    return result

def extract_floor_objects(facts):
    """Extracts all unique floor object names from a collection of facts."""
    floors = set()
    for fact in facts:
        parts = get_parts(fact)
        if len(parts) > 1:
            # Floors appear in (origin p f), (destin p f), (above f1 f2), (lift-at f)
            if parts[0] in ['origin', 'destin', 'lift-at'] and len(parts) >= 3: # Use >=3 just in case
                 floors.add(parts[2])
            elif parts[0] == 'above' and len(parts) == 3:
                 floors.add(parts[1])
                 floors.add(parts[2])
    return list(floors) # Return as list

def extract_passenger_objects(facts):
    """Extracts all unique passenger object names from a collection of facts."""
    passengers = set()
    for fact in facts:
        parts = get_parts(fact)
        if len(parts) > 1:
            # Passengers appear in (origin p f), (destin p f), (boarded p), (served p)
            if parts[0] in ['origin', 'destin'] and len(parts) >= 2: # origin/destin have 3 parts, boarded/served have 2
                 passengers.add(parts[1])
            elif parts[0] in ['boarded', 'served'] and len(parts) >= 2:
                 passengers.add(parts[1])
    return list(passengers) # Return as list


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining cost to reach the goal state (all passengers served)
        by summing the required non-move actions (board/depart) and an estimate
        of the required lift movement actions. The move cost is estimated as the
        sum of the vertical distances each passenger needs to travel while boarded,
        plus the initial distance for the lift to reach the first pickup floor.

    Assumptions:
        - Floors are ordered linearly by the `(above f_high f_low)` facts, forming a single chain.
        - All floor objects mentioned in facts are included in the list of floor objects.
        - Passenger and floor object names follow a consistent format allowing parsing.
        - The goal is always to serve all passengers defined in the problem.
        - The lift's location is always specified in the state.

    Heuristic Initialization:
        1. Extracts all floor objects and passenger objects from the initial state and static facts.
        2. Determines the floor order by performing a topological sort based on the `(above f_high f_low)` static facts.
        3. Creates a mapping from floor names to integer indices based on the sorted order (lowest floor gets index 0).
        4. Stores the destination floor index for each passenger by parsing the `(destin ?person ?floor)` static facts.
        5. Stores the set of all passenger names.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state and find the lift's current floor. Convert the current floor name to its integer index using the precomputed mapping. If the lift location is not found or the floor is unknown, return infinity (indicating an invalid state).
        2. Identify all passengers who are not yet served by checking for the absence of `(served ?person)` facts in the current state. If all passengers are served, the state is a goal state, and the heuristic value is 0.
        3. Initialize `non_move_cost` to 0, `pickup_indices` (set of floor indices where pickups are needed) to empty, and `boarded_travel_distance` (sum of vertical distances passengers need to travel while boarded) to 0.
        4. Iterate through the unserved passengers:
            a. For each unserved passenger `p`, add 1 to `non_move_cost` (for the future `depart` action).
            b. Check if the passenger `p` is currently at their origin floor (`(origin p o)` in state) or is currently boarded (`(boarded p)` in state).
            c. If `p` is at their origin `o`:
                i. Add 1 to `non_move_cost` (for the future `board` action).
                ii. Get the index `o_idx` for floor `o`. If the origin floor is known, add `o_idx` to the `pickup_indices` set.
                iii. Get the destination index `d_idx` for passenger `p`. If both origin and destination floors are known, add the absolute difference `abs(d_idx - o_idx)` to `boarded_travel_distance`. This estimates the vertical distance this passenger will travel inside the lift after boarding.
            d. If `p` is boarded:
                i. Get the destination index `d_idx` for passenger `p`. If the destination floor is known, add the absolute difference `abs(d_idx - current_idx)` to `boarded_travel_distance`. This estimates the remaining vertical distance this passenger needs to travel inside the lift from the current floor.
            e. If a passenger's origin or destination floor is not found in the floor map, this indicates an invalid problem setup, and the heuristic might be inaccurate or return infinity.
        5. Calculate `move_cost`:
            a. Initialize `move_cost` to `boarded_travel_distance`. This accounts for the vertical travel needed *while* passengers are inside the lift.
            b. If the `pickup_indices` set is not empty (meaning there are passengers waiting at their origins):
                i. Calculate the minimum distance from the lift's `current_idx` to any floor index in `pickup_indices`.
                ii. Add this minimum distance to `move_cost`. This accounts for the initial travel needed to reach the first passenger pickup location.
        6. The total heuristic value is `non_move_cost + move_cost`.
    """
    def __init__(self, task):
        # Extract all floor and passenger objects from initial state and static facts
        all_facts = set(task.initial_state) | set(task.static)
        floor_objects = extract_floor_objects(all_facts)
        self.all_passengers = set(extract_passenger_objects(all_facts))

        # Extract 'above' facts to determine floor order
        above_facts = [fact for fact in task.static if get_parts(fact)[0] == 'above']

        # Determine floor order using topological sort
        # The sort order is lowest to highest floor
        sorted_floors = topological_sort_floors(floor_objects, above_facts)

        # Create floor name to index mapping
        self.floor_to_idx = {floor: i for i, floor in enumerate(sorted_floors)}

        # Store passenger destinations (as floor indices)
        self.destinations = {}
        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'destin':
                p, d = parts[1], parts[2]
                if d in self.floor_to_idx: # Ensure destination floor is known
                    self.destinations[p] = self.floor_to_idx[d]
                # else: print(f"Warning: Destination floor {d} for passenger {p} not found in floor map.") # Optional warning


    def match(self, fact, *args):
        """Checks if a fact matches a pattern using fnmatch."""
        parts = get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def __call__(self, node):
        state = node.state

        # Check if goal is reached (all passengers served)
        unserved_passengers = {p for p in self.all_passengers if f"(served {p})" not in state}
        if not unserved_passengers:
            return 0 # Goal state

        # Find current lift floor
        current_floor = None
        for fact in state:
            if self.match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if current_floor is None or current_floor not in self.floor_to_idx:
             # This state is likely invalid or unreachable in a standard miconic problem
             # Or it indicates a problem in state representation/parsing
             # For robustness, return infinity or a very high value?
             # Assuming valid states where lift location is always known and valid.
             return float('inf') # Should not happen in valid states

        current_idx = self.floor_to_idx[current_floor]

        non_move_cost = 0
        pickup_indices = set()
        boarded_travel_distance = 0

        # Iterate through unserved passengers to calculate costs
        for p in unserved_passengers:
            # All unserved passengers need a 'depart' action eventually
            non_move_cost += 1

            is_at_origin = False
            is_boarded = False
            origin_floor = None

            # Check current status of the passenger
            for fact in state:
                if self.match(fact, "origin", p, "*"):
                    is_at_origin = True
                    origin_floor = get_parts(fact)[2]
                    break # Found origin, no need to check further for this passenger status
                if self.match(fact, "boarded", p):
                    is_boarded = True
                    break # Found boarded, no need to check further for this passenger status

            # Ensure passenger destination is known (should be from static facts)
            if p not in self.destinations:
                 # This indicates an invalid problem instance
                 # print(f"Error: Destination for passenger {p} not found.") # Optional error
                 return float('inf') # Should not happen in valid problems

            d_idx = self.destinations[p]

            if is_at_origin:
                # Passenger needs to be boarded
                non_move_cost += 1
                if origin_floor in self.floor_to_idx:
                    o_idx = self.floor_to_idx[origin_floor]
                    pickup_indices.add(o_idx)
                    # Add vertical distance from origin to destination
                    boarded_travel_distance += abs(d_idx - o_idx)
                # else: print(f"Warning: Origin floor {origin_floor} for passenger {p} not found in floor map.") # Optional warning
                # If origin floor is unknown, we can't calculate pickup_indices or travel distance for this passenger.
                # This might make the heuristic underestimate. Assuming valid origin floors.

            elif is_boarded:
                # Passenger is already boarded, just needs to depart
                # Add vertical distance from current lift floor to destination
                boarded_travel_distance += abs(d_idx - current_idx)

            # If passenger is neither at origin nor boarded, they must be served.
            # This case is excluded by iterating over unserved_passengers.


        # Calculate move cost
        move_cost = boarded_travel_distance

        # Add cost to reach the first pickup floor if any pickups are needed
        if pickup_indices:
            # Find the minimum distance from the current floor to any pickup floor
            cost_to_first_pickup = min(abs(current_idx - o_idx) for o_idx in pickup_indices)
            move_cost += cost_to_first_pickup

        # Total heuristic is the sum of non-move actions and estimated move actions
        total_heuristic = non_move_cost + move_cost

        return total_heuristic
