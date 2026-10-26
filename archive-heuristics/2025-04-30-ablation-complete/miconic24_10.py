from fnmatch import fnmatch
# Assuming Heuristic base class is available from the planning framework
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to get all
    passengers to their destination floors. It sums the estimated cost
    for each passenger who is not yet served. The cost for a passenger
    depends on whether they are waiting at their origin or already boarded.
    It includes lift movement, boarding, and departing actions.

    # Assumptions
    - The domain follows the standard Miconic rules (passengers wait at origin,
      board, are transported, and depart at destination).
    - The 'above' predicate defines a linear order of floors.
    - The lift has unlimited capacity.
    - The heuristic calculates costs for passengers independently, ignoring
      potential synergies (e.g., picking up/dropping off multiple passengers
      at the same floor). This makes it non-admissible but potentially
      effective for greedy search.

    # Heuristic Initialization
    - Extract the goal conditions to identify all passengers that need to be served.
    - Extract static facts to determine the destination floor for each passenger
      and to build a mapping from floor names to integer indices based on the
      'above' predicate.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. For each passenger who is not yet served (based on the goal state):
       a. Check if the passenger is currently boarded in the lift.
       b. If the passenger is boarded:
          - Find their destination floor using the pre-calculated map.
          - Get the integer index for the current lift floor and the destination floor.
          - Calculate the estimated cost: Absolute difference in floor indices
            between the current lift floor and the destination floor (for movement)
            plus 1 (for the depart action).
       c. If the passenger is not boarded (they must be waiting at their origin,
          assuming valid states):
          - Find their origin floor from the state and destination floor from the map.
          - Get the integer index for the current lift floor, the origin floor,
            and the destination floor.
          - Calculate the estimated cost: Absolute difference in floor indices
            between the current lift floor and the origin floor (for movement to pick up)
            plus 1 (for the board action)
            plus absolute difference in floor indices between the origin floor
            and the destination floor (for movement to drop off)
            plus 1 (for the depart action).
    3. Sum the estimated costs for all passengers who are not yet served.
    4. If all passengers are served (goal state), the heuristic is 0.
    5. If the floor structure cannot be determined or a required floor (lift, origin, destination)
       is missing from the indexed floors, return infinity to indicate an unreachable
       or invalid state for this heuristic.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions,
        passenger destinations, and floor ordering.
        """
        # The set of facts that must hold in goal states. We assume goals are (served ?p).
        self.goals = task.goals
        # Static facts are facts that do not change during planning.
        static_facts = task.static

        # Store goal passengers (those who need to be served)
        self.goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Store passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination = get_parts(fact)
                self.passenger_destinations[passenger] = destination

        # Build floor index mapping from static 'above' facts
        self.floor_indices = self._build_floor_indices(static_facts)

    def _build_floor_indices(self, static_facts):
        """
        Build a mapping from floor names to integer indices based on 'above' facts.
        Assumes (above f_lower f_higher) means f_lower is immediately below f_higher.
        Returns an empty dict if floor order cannot be determined for multiple floors.
        """
        above_relations = {} # Maps lower floor to higher floor: {f_lower: f_higher}
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_lower, f_higher = get_parts(fact)
                above_relations[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        floor_indices = {}

        if not all_floors:
            # No floors defined. Should not happen in a valid problem.
            return {}

        if len(all_floors) == 1:
            # Single floor problem. Index is 0.
            floor_indices[list(all_floors)[0]] = 0
            return floor_indices

        # Find the lowest floor: a floor that is never the 'higher' floor in an 'above' relation.
        higher_floors_in_above = set(above_relations.values())
        lowest_floor_candidates = all_floors - higher_floors_in_above

        if len(lowest_floor_candidates) != 1:
            # Problematic: Multiple lowest floors (disconnected towers?) or no lowest floor (cycle?).
            # Cannot build a linear index.
            return {} # Indicate failure to build index

        lowest_floor = list(lowest_floor_candidates)[0]
        current_floor = lowest_floor
        current_index = 0

        # Traverse up the floors from the lowest
        while current_floor is not None and current_floor in all_floors:
            floor_indices[current_floor] = current_index
            current_index += 1
            current_floor = above_relations.get(current_floor) # Get the floor immediately above

        # Check if all floors were successfully indexed.
        if len(floor_indices) != len(all_floors):
             return {} # Indicate failure if not all floors in all_floors were indexed

        return floor_indices


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all passengers.
        """
        state = node.state  # Current world state (frozenset of strings)

        # Check if there are multiple floors present in the state facts themselves.
        # This helps identify if we are in a multi-floor problem context.
        floors_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ["lift-at", "origin", "destin"]:
                 if len(parts) > 1: # Ensure there's an argument for floor
                     floors_in_state.add(parts[-1]) # The last argument is the floor

        # If we have multiple floors in the state but failed to build floor indices,
        # the heuristic cannot be computed meaningfully. Return infinity.
        if len(floors_in_state) > 1 and not self.floor_indices:
             return float('inf')

        # Find the current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, heuristic is infinite (shouldn't happen in valid states)
        if lift_floor is None:
             return float('inf')

        # Get the index of the current lift floor.
        # If floor_indices was built successfully, lookup the index.
        # If floor_indices is empty (implies single floor or unorderable multi-floor),
        # and we haven't returned inf yet (meaning it's likely a single floor handled by _build_floor_indices),
        # the lookup should succeed and return 0. If floor_indices is not empty, but lift_floor isn't in it,
        # it's inconsistent -> inf.
        current_lift_floor_index = self.floor_indices.get(lift_floor)
        if current_lift_floor_index is None:
             # This should only happen if floor_indices was built successfully (not empty)
             # but the lift_floor from the state is not in the indexed floors.
             # This indicates an inconsistency between state and static facts.
             return float('inf')


        total_cost = 0

        # Track which passengers are currently boarded or at their origin
        boarded_passengers = set()
        passengers_at_origin = {} # {passenger: origin_floor}
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "boarded":
                if len(parts) > 1: boarded_passengers.add(parts[1])
            elif parts and parts[0] == "origin":
                if len(parts) > 2: passengers_at_origin[parts[1]] = parts[2]
            elif parts and parts[0] == "served":
                if len(parts) > 1: served_passengers.add(parts[1])

        # Consider passengers who are not yet served
        passengers_to_serve = self.goal_passengers - served_passengers

        if not passengers_to_serve:
            # All goal passengers are served
            return 0

        for passenger in passengers_to_serve:
            destination_floor = self.passenger_destinations.get(passenger)
            if destination_floor is None:
                 # Passenger needs serving but has no destination defined in static facts.
                 # This indicates an inconsistent problem definition. Treat as unsolveable.
                 return float('inf')

            # Get the index of the destination floor.
            dest_floor_index = self.floor_indices.get(destination_floor)
            if dest_floor_index is None:
                 # Destination floor was present in static facts but not found in indexed floors.
                 # Indicates inconsistency or floor_indices is empty for multi-floor problem.
                 # The multi-floor empty index case is caught at the start.
                 # So if we are here and dest_floor_index is None, floor_indices is not empty
                 # but the floor wasn't found.
                 return float('inf')


            if passenger in boarded_passengers:
                # Passenger is in the lift, needs to go to destination and depart
                # Cost = Move to destination + Depart
                move_cost = abs(current_lift_floor_index - dest_floor_index)
                depart_cost = 1
                total_cost += move_cost + depart_cost

            elif passenger in passengers_at_origin:
                # Passenger is waiting at origin, needs pickup, transport, and dropoff
                origin_floor = passengers_at_origin[passenger]

                # Get the index of the origin floor.
                origin_floor_index = self.floor_indices.get(origin_floor)
                if origin_floor_index is None:
                     # Origin floor was present in state but not found in indexed floors.
                     # Indicates inconsistency or floor_indices is empty for multi-floor problem.
                     # The multi-floor empty index case is caught at the start.
                     # So if we are here and origin_floor_index is None, floor_indices is not empty
                     # but the floor wasn't found.
                     return float('inf')

                # Cost = Move to origin + Board + Move to destination + Depart
                move_to_origin_cost = abs(current_lift_floor_index - origin_floor_index)
                board_cost = 1
                move_to_dest_cost = abs(origin_floor_index - dest_floor_index)
                depart_cost = 1
                total_cost += move_to_origin_cost + board_cost + move_to_dest_cost + depart_cost

            # else: Passenger is not served, not boarded, and not at origin.
            # This state should ideally not be reachable in a valid problem/plan.
            # We ignore such passengers for the heuristic calculation.

        return total_cost
