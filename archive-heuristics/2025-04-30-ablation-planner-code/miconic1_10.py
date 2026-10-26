import re
from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if needed

# Define a dummy Heuristic base class if not provided
# This is just for standalone testing/linting; the actual environment provides it.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string.
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It sums the estimated cost for each unserved passenger,
    considering the actions (board, depart) and the lift movement needed
    to pick them up and drop them off individually.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates the cost for each unserved passenger independently
      and sums them up. This overestimates the cost as the lift can batch
      passengers, but provides a non-zero estimate for non-goal states.
    - Floor names follow the pattern 'fN' where N is an integer, and the
      'above' predicates imply a linear ordering corresponding to these integers.
      The heuristic derives a numerical index for each floor based on this assumption.

    # Heuristic Initialization
    - Parses static facts to identify all floor names and sort them numerically
      to create a mapping from floor names to numerical indices.
    - Parses static facts to map each passenger to their destination floor.
    - Identifies the set of all passengers who need to be served based on the
      goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost to 0.
    2.  Identify the current floor of the lift by finding the fact `(lift-at ?f)` in the state.
    3.  Iterate through each passenger that needs to be served (identified during initialization from goal conditions).
    4.  For the current passenger `p`:
        -   Check if `(served p)` is true in the current state. If yes, this passenger is served, and no cost is added for them.
        -   If `(served p)` is false:
            -   Find the passenger's origin floor `o` by looking for `(origin p o)` in the state.
            -   Find the passenger's destination floor `d` using the pre-calculated destination map.
            -   Check if `(boarded p)` is true in the state.
            -   If `(origin p o)` is true (passenger is waiting):
                -   Estimate movement cost from current lift floor to origin floor `o`: `abs(index(current_floor) - index(o))`.
                -   Add cost for boarding: +1.
                -   Estimate movement cost from origin floor `o` to destination floor `d`: `abs(index(o) - index(d))`.
                -   Add cost for departing: +1.
                -   Add these costs to the total heuristic.
            -   If `(boarded p)` is true (passenger is in the lift):
                -   Estimate movement cost from current lift floor to destination floor `d`: `abs(index(current_floor) - index(d))`.
                -   Add cost for departing: +1.
                -   Add these costs to the total heuristic.
            # Note: A passenger cannot be both waiting and boarded in a valid state.
            # If a passenger is neither waiting nor boarded, and not served,
            # this implies an invalid state or a different domain structure.
            # Based on the domain, unserved passengers are either at origin, boarded, or served.
            # We assume valid states here.
    5.  Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of passengers to be served.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Determine floor ordering by parsing floor names and sorting
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_high, f_low = get_parts(fact)[1:]
                all_floors.add(f_high)
                all_floors.add(f_low)

        # Sort floors based on the numerical suffix (e.g., f1, f2, f10)
        # Use regex to extract the number
        def get_floor_number(floor_name):
            match = re.match(r'f(\d+)', floor_name)
            if match:
                return int(match.group(1))
            # This case should ideally not happen with standard miconic benchmarks
            # print(f"Warning: Unexpected floor name format: {floor_name}")
            return float('inf') # Assign a large value to sort it last

        sorted_floors = sorted(list(all_floors), key=get_floor_number)

        # Create floor_name -> index map
        self.floor_indices = {floor: i for i, floor in enumerate(sorted_floors)}

        # 2. Map passenger to destination floor
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, d_floor = get_parts(fact)[1:]
                self.passenger_destinations[p] = d_floor

        # 3. Identify all passengers who need to be served (from goal)
        self.passengers_to_serve = set()
        for goal in self.goals:
            # Goal facts are typically simple predicates like (served p)
            if match(goal, "served", "*"):
                p = get_parts(goal)[1]
                self.passengers_to_serve.add(p)
            # Handle potential AND goals like (and (served p1) (served p2))
            # The task.goals structure usually flattens this, but being explicit is safer.
            # Assuming task.goals is a collection of facts. The example supports this.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid state reachable from initial state
             # print("Error: Lift location not found in state.")
             return float('inf') # Indicate an unreachable or invalid state

        # Ensure current_lift_floor is in floor_indices (should be if state is valid)
        if current_lift_floor not in self.floor_indices:
             # print(f"Error: Current lift floor {current_lift_floor} not found in floor indices.")
             return float('inf') # Indicate an invalid state

        current_lift_index = self.floor_indices[current_lift_floor]

        # Iterate through passengers who need to be served
        for passenger in self.passengers_to_serve:
            # Check if passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, no cost

            # Passenger is not served. Find their state (waiting or boarded)
            is_boarded = f"(boarded {passenger})" in state
            origin_floor = None
            # Find origin floor if waiting
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break

            destination_floor = self.passenger_destinations.get(passenger) # Use .get for safety
            if destination_floor is None:
                 # Should not happen if passenger is in goals but no destination in static facts
                 # print(f"Error: Destination not found for passenger {passenger}")
                 continue # Cannot calculate cost for this passenger

            # Ensure destination_floor is in floor_indices (should be if static facts are valid)
            if destination_floor not in self.floor_indices:
                 # print(f"Error: Destination floor {destination_floor} not found in floor indices.")
                 continue # Cannot calculate cost for this passenger

            destination_index = self.floor_indices[destination_floor]

            if origin_floor: # Passenger is waiting at origin_floor
                # Ensure origin_floor is in floor_indices
                if origin_floor not in self.floor_indices:
                     # print(f"Error: Origin floor {origin_floor} not found in floor indices.")
                     continue # Cannot calculate cost for this passenger

                origin_index = self.floor_indices[origin_floor]
                # Cost to pick up: move to origin + board
                cost_to_pickup = abs(current_lift_index - origin_index) + 1
                # Cost to drop off: move from origin to destination + depart
                cost_to_dropoff = abs(origin_index - destination_index) + 1
                total_cost += cost_to_pickup + cost_to_dropoff

            elif is_boarded: # Passenger is boarded
                # Cost to drop off: move to destination + depart
                cost_to_dropoff = abs(current_lift_index - destination_index) + 1
                total_cost += cost_to_dropoff

            # Else: Passenger is unserved, not waiting, not boarded. This state is likely invalid
            # based on the domain definition (passengers are either at origin, boarded, or served).
            # We assume valid states here and add 0 cost for such passengers, as they are not in
            # a state that requires immediate action based on the known predicates. This might
            # underestimate if the state is valid but implies a complex sequence not captured.
            # However, for typical miconic, this case implies an error in state representation
            # or domain understanding if the passenger is truly unserved.

        return total_cost
