from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the remaining cost by summing:
    1. The number of passengers waiting at their origin (need boarding).
    2. The number of passengers currently boarded (need departing).
    3. An estimate of the vertical travel cost for the lift.

    The travel cost is estimated as the minimum number of floor moves
    required to go from the current lift floor to the lowest required
    floor, sweep up to the highest required floor, or vice-versa.
    Required floors are origin floors of waiting passengers and
    destination floors of boarded passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Destination floor for each passenger.
        - Mapping from floor name to numerical index based on 'above' predicates.
        - List of all passengers.
        """
        super().__init__(task)

        self.goals = task.goals
        self.static_facts = task.static

        # Extract passenger destinations from goals
        self.passengers = set()
        self.destinations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "served":
                passenger = args[0]
                self.passengers.add(passenger)
                # Find the destination for this passenger in static facts
                for fact in self.static_facts:
                    if match(fact, "destin", passenger, "*"):
                        self.destinations[passenger] = get_parts(fact)[2]
                        break

        # Build floor mapping based on 'above' predicates (Topological Sort)
        self.floor_to_num = {}
        self.num_to_floor = {}
        floor_names = set()
        above_relations = [] # Store (higher_floor, lower_floor)

        # Collect all floor names and 'above' relations
        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                floor_names.add(f_higher)
                floor_names.add(f_lower)
                above_relations.append((f_higher, f_lower))

        # Add floors from initial state that might not be in 'above' (e.g., only one floor)
        for fact in task.initial_state:
             if match(fact, "lift-at", "*"):
                 floor_names.add(get_parts(fact)[1])
             if match(fact, "origin", "*", "*"):
                 floor_names.add(get_parts(fact)[2])
             # Destinations are covered by goals/static

        # Handle case with 0 or 1 floor
        if not floor_names:
             # No floors, heuristic will always be 0 if no goals
             # If there are goals but no floors, problem is likely unsolvable or ill-defined
             # Returning 0 seems appropriate if no floors mean no movement is possible/needed
             return

        if len(floor_names) == 1:
             floor_name = list(floor_names)[0]
             self.floor_to_num[floor_name] = 1
             self.num_to_floor[1] = floor_name
             return

        # Build graph where edge is lower -> higher
        above_graph = defaultdict(list) # lower_floor -> [higher_floor, ...]
        # Count how many floors are *below* each floor
        below_count = {f: 0 for f in floor_names}

        for f_higher, f_lower in above_relations:
             above_graph[f_lower].append(f_higher)
             below_count[f_higher] += 1 # f_higher has f_lower below it

        # Find the lowest floor(s) (those with 0 floors below them)
        queue = deque([f for f in floor_names if below_count[f] == 0])
        current_num = 1

        # Perform topological sort to assign numbers
        while queue:
            # If multiple floors have below_count == 0, they are parallel lowest floors
            # This shouldn't happen in standard miconic linear floor structure.
            # Assuming a single lowest floor for correct numbering.
            # If multiple, sorting them alphabetically or arbitrarily might be needed,
            # but the problem implies a strict linear order.
            f_lower = queue.popleft()

            # Check if floor already numbered (could happen with weird 'above' facts)
            if f_lower in self.floor_to_num:
                 continue # Skip if already processed

            self.floor_to_num[f_lower] = current_num
            self.num_to_floor[current_num] = f_lower
            current_num += 1

            # For each floor directly above f_lower
            for f_higher in above_graph[f_lower]:
                below_count[f_higher] -= 1 # One less floor below f_higher
                if below_count[f_higher] == 0:
                    queue.append(f_higher)

        # Check if all floors were numbered
        if len(self.floor_to_num) != len(floor_names):
             # This indicates a problem with the 'above' facts (e.g., cycles, disconnected)
             # For a heuristic, we can potentially fall back or signal an issue.
             # Let's raise an error or return inf if floor mapping is incomplete.
             # An incomplete mapping means we can't calculate distances reliably.
             if len(self.floor_to_num) < len(floor_names):
                 # Some floors couldn't be ordered, likely due to invalid 'above' facts
                 # print("Error: Could not establish a linear order for all floors.")
                 # Clear mapping to force inf return in __call__
                 self.floor_to_num = {}
                 self.num_to_floor = {}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current lift floor
        lift_at_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if lift_at_fact is None:
             # Should not happen in a valid miconic state
             return float('inf')

        f_lift_name = get_parts(lift_at_fact)[1]
        f_lift_num = self.floor_to_num.get(f_lift_name)
        if f_lift_num is None:
             # Floor name not in our mapping - indicates an issue with floor parsing
             # or a floor in state not mentioned in init/static, or floor mapping failed
             return float('inf')

        cost = 0
        required_floors_nums = set()
        passengers_needing_pickup = 0
        passengers_needing_dropoff = 0

        # Count passengers needing board/depart and collect required floors
        for passenger in self.passengers:
            is_served = f"(served {passenger})" in state
            if is_served:
                continue # This passenger is done

            # Check if passenger is waiting at origin
            origin_fact = next((fact for fact in state if match(fact, "origin", passenger, "*")), None)
            if origin_fact:
                passengers_needing_pickup += 1
                f_origin_name = get_parts(origin_fact)[2]
                f_origin_num = self.floor_to_num.get(f_origin_name)
                if f_origin_num is not None:
                    required_floors_nums.add(f_origin_num)
                else:
                    # Unknown floor name or floor mapping failed
                    return float('inf')

            # Check if passenger is boarded
            is_boarded = f"(boarded {passenger})" in state
            if is_boarded:
                passengers_needing_dropoff += 1
                f_destin_name = self.destinations.get(passenger)
                if f_destin_name:
                    f_destin_num = self.floor_to_num.get(f_destin_name)
                    if f_destin_num is not None:
                        required_floors_nums.add(f_destin_num)
                    else:
                        # Unknown floor name or floor mapping failed
                        return float('inf')
                else:
                    # Destination not found for boarded passenger - indicates init/static issue
                    return float('inf')

        # Add cost for board and depart actions
        cost += passengers_needing_pickup
        cost += passengers_needing_dropoff

        # If no floors need visiting, travel cost is 0
        if not required_floors_nums:
            return cost # Should be 0 if no passengers need service

        # Calculate travel cost
        min_f_num = min(required_floors_nums)
        max_f_num = max(required_floors_nums)

        # Estimate travel moves: distance to closest end of range + size of range
        # This estimates the moves for a sweep that covers the required range, starting from the closest end.
        travel_cost = min(abs(f_lift_num - min_f_num), abs(f_lift_num - max_f_num)) + (max_f_num - min_f_num)

        cost += travel_cost

        return cost
