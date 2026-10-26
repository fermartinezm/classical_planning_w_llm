# The Heuristic base class is assumed to be available in the environment.
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args if args are not wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assume Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass

class miconicHeuristic: # Inherit from Heuristic if base class is provided
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all unserved passengers.
    It calculates the cost for each unserved passenger independently and sums these costs.
    For a waiting passenger, the cost includes moving the lift to their origin floor, boarding,
    moving the lift to their destination floor, and departing.
    For a boarded passenger, the cost includes moving the lift to their destination floor and departing.
    Movement cost between floors is the absolute difference in floor levels.

    # Assumptions
    - The domain uses 'above' predicates to define a linear order of floors.
    - Action costs are uniform (cost 1 per action).
    - The heuristic overestimates movement costs by calculating them independently for each passenger,
      but aims to provide a good estimate for greedy search.
    - The PDDL instance has a valid floor structure defined by 'above' facts, forming a single chain.
    - All floors mentioned in 'above' facts or passenger origins/destinations are defined as 'floor' objects.

    # Heuristic Initialization
    - Parses 'above' facts from static information to build a mapping from floor names to integer levels.
    - Parses 'destin' facts from static information to map passengers to their destination floors.
    - Collects all passenger names from static 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current floor of the lift from the state and get its level using the pre-calculated floor level map. If the lift floor is not mapped, return infinity.
    3. Identify all passengers and their current status (waiting at origin, boarded, or served) from the state.
    4. For each passenger that is not yet 'served' (identified from the initial problem's 'destin' facts):
       - Get the passenger's destination floor and its level using the pre-calculated map. If the destination floor is not mapped, return infinity.
       - If the passenger is 'waiting' at their origin floor 'o' (found in the current state):
         - Get the origin floor's level. If the origin floor is not mapped, return infinity.
         - Calculate the cost for this passenger: `abs(current_level - origin_level)` (move to origin) + 1 (board) + `abs(origin_level - dest_level)` (move to destin) + 1 (depart).
         - Add this cost to the total.
       - If the passenger is 'boarded' (found in the current state):
         - Calculate the cost for this passenger: `abs(current_level - dest_level)` (move to destin) + 1 (depart).
         - Add this cost to the total.
    5. The total heuristic value is the sum of the contributions of all unserved passengers.
    6. If all passengers are served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        # self.goals = task.goals # Not strictly needed for this heuristic calculation, but good practice.
        static_facts = task.static
        task_objects = task.objects # Access objects from task

        # Build floor level mapping
        self.floor_to_level = self._build_floor_levels(static_facts, task_objects)

        # Build passenger destination mapping and collect all passenger names
        self.passenger_to_destin = {}
        self.all_passenger_names = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, passenger, floor = parts
                    self.passenger_to_destin[passenger] = floor
                    self.all_passenger_names.add(passenger)

    def _build_floor_levels(self, static_facts, task_objects):
        """
        Build a mapping from floor names to integer levels based on 'above' facts.
        Assumes 'above f_higher f_lower' means f_higher is one level above f_lower.
        Assumes floors form a single linear chain.
        """
        higher_to_lower = {}
        all_floors = set()

        # Collect all floors defined in the problem objects
        for obj_type_str in task_objects:
             parts = obj_type_str.split(' - ')
             if len(parts) == 2 and parts[1] == 'floor':
                 floor_names = parts[0].split()
                 all_floors.update(floor_names)

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, f_higher, f_lower = parts
                    higher_to_lower[f_higher] = f_lower
                    # Ensure floors from 'above' facts are also included, although objects should cover this
                    all_floors.add(f_higher)
                    all_floors.add(f_lower)

        floor_to_level = {}
        if not all_floors:
             return floor_to_level # No floors defined

        # Find the lowest floor: a floor that is a value in higher_to_lower but not a key
        potential_lowest_floors = set(higher_to_lower.values()) - set(higher_to_lower.keys())

        lowest_floor = None
        if len(potential_lowest_floors) == 1:
            lowest_floor = list(potential_lowest_floors)[0]
        elif len(all_floors) == 1 and not higher_to_lower:
             # Case: single floor, no above facts
             lowest_floor = list(all_floors)[0]
        # Note: If len(potential_lowest_floors) > 1 or 0 (and >1 floor exists),
        # the floor structure is not a single linear chain as assumed.
        # The heuristic will be inaccurate or fail (return inf) if it encounters an unmapped floor.

        if lowest_floor:
            current_floor = lowest_floor
            level = 1
            floor_to_level[current_floor] = level
            # Build levels upwards
            reverse_higher_to_lower = {v: k for k, v in higher_to_lower.items()} # lower_to_higher
            while current_floor in reverse_higher_to_lower:
                next_floor = reverse_higher_to_lower[current_floor]
                level += 1
                floor_to_level[next_floor] = level
                current_floor = next_floor

        # Final check: ensure all floors found were assigned a level
        # If not all floors are mapped, it means the 'above' facts don't form a single chain
        # covering all floors defined in objects. This heuristic relies on a full mapping.
        # Missing floors will cause lookups to fail later, resulting in inf heuristic.
        # We don't need to explicitly handle it here, the __call__ method checks for mapped floors.

        return floor_to_level

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    current_lift_floor = parts[1]
                    break

        if current_lift_floor is None or current_lift_floor not in self.floor_to_level:
             # Lift location not found in state or floor not mapped (invalid state/problem)
             return float('inf')

        current_level = self.floor_to_level[current_lift_floor]

        # Identify passenger statuses and origins
        served_passengers = set()
        boarded_passengers = set()
        waiting_passengers_info = {} # {p: origin_floor}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "served" and len(parts) == 2:
                served_passengers.add(parts[1])
            elif predicate == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == "origin" and len(parts) == 3:
                p, o = parts[1:]
                waiting_passengers_info[p] = o

        # Identify unserved passengers based on the complete list from static facts
        unserved_passengers = self.all_passenger_names - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        total_cost = 0

        for p in unserved_passengers:
            dest_floor = self.passenger_to_destin.get(p)
            if dest_floor is None or dest_floor not in self.floor_to_level:
                 # Destination floor not found or not mapped (problem instance issue)
                 return float('inf')

            dest_level = self.floor_to_level[dest_floor]

            if p in waiting_passengers_info:
                # Passenger is waiting at origin
                origin_floor = waiting_passengers_info[p]
                if origin_floor not in self.floor_to_level:
                     # Origin floor not mapped (problem instance issue)
                     return float('inf')

                origin_level = self.floor_to_level[origin_floor]

                # Cost for waiting passenger: move to origin + board + move to destin + depart
                # Movement cost: abs(current_level - origin_level) + abs(origin_level - dest_level)
                # Actions cost: 1 (board) + 1 (depart)
                total_cost += abs(current_level - origin_level) + 1 + abs(origin_level - dest_level) + 1

            elif p in boarded_passengers:
                # Passenger is boarded
                # Cost for boarded passenger: move to destin + depart
                # Movement cost: abs(current_level - dest_level)
                # Actions cost: 1 (depart)
                total_cost += abs(current_level - dest_level) + 1
            # else: passenger is unserved but neither waiting nor boarded?
            # This should not happen in a valid state. A passenger is either served, waiting, or boarded.
            # If a passenger is in all_passenger_names but not in served, waiting_passengers_info, or boarded_passengers,
            # it implies an inconsistency in the state representation. We assume valid states.


        return total_cost
