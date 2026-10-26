# Assuming Heuristic base class is available from a library like 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, use a dummy base class:
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.objects = task.objects

    def __call__(self, node):
        raise NotImplementedError

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

from fnmatch import fnmatch

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args if args are not wildcards
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers, assuming each unserved passenger is transported independently.
    For a passenger waiting at their origin floor, the estimated cost includes
    travel to the origin, boarding, travel to the destination, and departing.
    For a passenger already boarded, the estimated cost includes travel to the
    destination and departing.

    # Assumptions
    - Each move action (up/down) costs 1.
    - Each board action costs 1.
    - Each depart action costs 1.
    - The cost for each unserved passenger is calculated independently and summed.
      This ignores potential optimizations from transporting multiple passengers
      simultaneously but provides a reasonable estimate for greedy search.
    - The floor structure is a simple linear sequence defined by `(above f_high f_low)`
      predicates, where `f_high` is immediately above `f_low`.
    - All passengers and floors are listed in the objects section.
    - Every passenger that is part of the goal has a destination defined in the static facts.
    - Every valid state includes exactly one `(lift-at ?f)` fact.
    - Unserved passengers are either at their origin or boarded.

    # Heuristic Initialization
    - Extract all passengers and floors from the task objects.
    - Determine the floor order by parsing `(above f_high f_low)` facts and
      create a mapping from floor name to its integer index (level). This involves
      finding the bottom floor and traversing upwards using the `above` relations.
    - Store the destination floor for each passenger from `(destin ?p ?f)` facts.
    - Store the set of passengers that need to be served according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Process the current state facts to quickly find the lift's location,
        which passengers are boarded, and which are at their origin.
    2.  Identify the current floor of the lift.
    3.  Get the integer index for the current lift floor using the precomputed map.
    4.  Initialize the total estimated cost to 0.
    5.  Iterate through all passengers that are part of the goal (i.e., need to be served).
    6.  For each such passenger `p`:
        a.  Check if the passenger is already served in the current state (`(served ?p)`). If yes, skip this passenger.
        b.  If not served, check if the passenger is currently boarded (`(boarded ?p)`).
        c.  Find the passenger's destination floor `f_destin` using the precomputed destination map. Get its index.
        d.  If boarded:
            i.  The estimated cost for this passenger is the travel distance from the current lift floor to `f_destin` plus the cost of the `depart` action.
                Cost = `abs(current_lift_floor_index - floor_index[f_destin]) + 1`.
        e.  If not boarded:
            i.  Find the passenger's current origin floor `f_origin` from the state (`(origin ?p ?f_origin)`). Get its index.
            ii. The estimated cost for this passenger is the travel distance from the current lift floor to `f_origin`, plus the cost of the `board` action, plus the travel distance from `f_origin` to `f_destin`, plus the cost of the `depart` action.
                Cost = `abs(current_lift_floor_index - floor_index[f_origin]) + 1 + abs(floor_index[f_destin] - floor_index[f_origin]) + 1`.
    7.  Add the calculated cost for the unserved passenger to the total estimated cost.
    8.  Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of all passengers.
        """
        self.goals = task.goals # Goal conditions (e.g., (served p1), (served p2))
        self.static = task.static # Static facts (e.g., (above f1 f2), (destin p1 f10))
        self.objects = task.objects # All objects (passengers, floors)

        # Extract all passengers and floors based on type
        self.all_passengers = {obj for obj, obj_type in self.objects.items() if obj_type == 'passenger'}
        self.all_floors = {obj for obj, obj_type in self.objects.items() if obj_type == 'floor'}

        # Determine floor order and create floor name -> index mapping
        # Build map: lower_floor -> higher_floor
        floor_above_map = {}
        # Keep track of floors that are the 'lower' part of an 'above' fact
        is_lower_floor_in_above = set()
        # Keep track of floors that are the 'higher' part of an 'above' fact
        is_higher_floor_in_above = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                f_high, f_low = get_parts(fact)[1:]
                floor_above_map[f_low] = f_high
                is_lower_floor_in_above.add(f_low)
                is_higher_floor_in_above.add(f_high)

        self.floor_index = {}

        # Find the bottom floor: a floor that is a lower floor in some 'above' fact
        # but is not a higher floor in any 'above' fact.
        bottom_floor = None
        # Check floors that are lower floors in 'above' facts
        for floor in is_lower_floor_in_above:
             if floor not in is_higher_floor_in_above:
                 bottom_floor = floor
                 break

        # Handle case with only one floor or no 'above' facts defining a chain
        if not floor_above_map:
             if len(self.all_floors) > 0:
                 # If there's only one floor, its index is 0.
                 if len(self.all_floors) == 1:
                     self.floor_index = {list(self.all_floors)[0]: 0}
                 # If multiple floors but no 'above' facts, cannot determine order.
                 # This indicates an invalid problem setup for this domain.
                 # floor_index remains empty, heuristic will return inf.
        # If a bottom floor was found, build the index map by traversing upwards
        elif bottom_floor is not None:
            current_floor = bottom_floor
            index = 0
            while current_floor is not None:
                self.floor_index[current_floor] = index
                index += 1
                # Get the floor directly above the current one using the map
                current_floor = floor_above_map.get(current_floor)
        # else: multiple floors, no bottom found - problem setup issue. floor_index remains empty.


        # Store destination floor for each passenger
        self.passenger_destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                passenger, floor = get_parts(fact)[1:]
                self.passenger_destinations[passenger] = floor

        # Store the set of passengers that need to be served according to the goal
        self.goal_served_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 self.goal_served_passengers.add(get_parts(goal)[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Check if goal is reached
        # A state is a goal state if all passengers in the goal are served.
        is_goal_state = all(f'(served {p})' in state for p in self.goal_served_passengers)
        if is_goal_state:
            return 0

        # If floor indexing failed during initialization (invalid problem setup)
        if not self.floor_index:
             # Cannot compute travel costs. Return infinity.
             return float('inf')

        # --- Process state facts for quick lookups ---
        lift_at_floor = None
        boarded_passengers_in_state = set()
        origin_facts_in_state = {} # passenger -> origin_floor
        served_passengers_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "lift-at":
                # Assuming only one lift-at fact exists
                lift_at_floor = parts[1]
            elif predicate == "boarded":
                boarded_passengers_in_state.add(parts[1])
            elif predicate == "origin":
                passenger, floor = parts[1:]
                origin_facts_in_state[passenger] = floor
            elif predicate == "served":
                 served_passengers_in_state.add(parts[1])
        # --- End state fact processing ---


        # This should always be true in a valid state for this domain
        if lift_at_floor is None or lift_at_floor not in self.floor_index:
             # Cannot determine lift location or its index. Invalid state or setup.
             return float('inf') # Should not be reached in valid problems

        current_lift_floor_index = self.floor_index[lift_at_floor]

        total_estimated_cost = 0

        # Iterate through all passengers that are part of the goal
        for passenger in self.goal_served_passengers:
            # Check if passenger is already served
            if passenger in served_passengers_in_state:
                continue # This passenger is done

            # Passenger is unserved. Check if boarded or waiting at origin.
            is_boarded = passenger in boarded_passengers_in_state

            destination_floor = self.passenger_destinations.get(passenger)
            if destination_floor is None or destination_floor not in self.floor_index:
                 # Passenger destination is missing or floor is not indexed. Invalid setup.
                 return float('inf') # Should not be reached in valid problems

            destination_floor_index = self.floor_index[destination_floor]

            if is_boarded:
                # Passenger is boarded, needs to depart at destination
                # Cost = travel to destination + depart action
                travel_cost = abs(current_lift_floor_index - destination_floor_index)
                depart_cost = 1
                total_estimated_cost += travel_cost + depart_cost

            else:
                # Passenger is not boarded. Find their origin from the pre-processed facts.
                origin_floor = origin_facts_in_state.get(passenger)

                if origin_floor is None or origin_floor not in self.floor_index:
                     # Passenger is unserved, not boarded, and not at an origin floor.
                     # This state shouldn't be reachable in a valid problem execution
                     # if they started at an origin and weren't served or boarded.
                     # It implies an invalid state or domain issue.
                     return float('inf') # Should not be reached in valid problems

                origin_floor_index = self.floor_index[origin_floor]

                # Cost = travel to origin + board action + travel from origin to destination + depart action
                travel_to_origin_cost = abs(current_lift_floor_index - origin_floor_index)
                board_cost = 1
                travel_from_origin_to_dest_cost = abs(destination_floor_index - origin_floor_index)
                depart_cost = 1
                total_estimated_cost += travel_to_origin_cost + board_cost + travel_from_origin_to_dest_cost + depart_cost

        return total_estimated_cost
