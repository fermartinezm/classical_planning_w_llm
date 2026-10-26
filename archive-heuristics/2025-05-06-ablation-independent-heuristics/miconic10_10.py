import logging
from heuristics.heuristic_base import Heuristic
from task import Operator, Task

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    two components:
    1.  An estimate of the number of board/depart actions required for
        unserved passengers.
    2.  An estimate of the lift movement actions required to visit the
        relevant floors (origins of waiting passengers, destinations of
        boarded passengers).

    Assumptions:
    -   The problem instance is valid and solvable.
    -   The 'above' predicates define a linear ordering of floors.
    -   Unserved passengers are either waiting at their origin floor or
        are boarded.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It extracts:
    -   The destination floor for each passenger using the `(destin ?p ?f)`
        facts. This is stored in `self.passenger_to_destin`.
    -   The linear ordering of floors using the `(above ?f1 ?f2)` facts.
        `(above f1 f2)` means f1 is physically above f2. The floors are
        indexed from 0 (lowest) to N-1 (highest). This mapping is stored
        in `self.floor_to_index`.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Identify the current floor of the lift.
    2.  Identify which passengers are currently served, boarded, or waiting
        at their origin floor by examining the state facts.
    3.  Initialize the heuristic value `h` to 0.
    4.  Initialize a set `required_floors` to store floors the lift must visit.
    5.  Iterate through all passengers defined in the problem.
        For each passenger `p`:
        -   If `p` is not marked as `served` in the current state:
            -   If `p` is marked as `boarded`:
                -   Add 1 to `h` (for the future `depart` action).
                -   Add `p`'s destination floor (looked up from static info)
                    to `required_floors`.
            -   If `p` is waiting at their `origin` floor (found in state):
                -   Add 2 to `h` (for the future `board` and `depart` actions).
                -   Add `p`'s origin floor to `required_floors`.
            -   (Passengers not served, not boarded, and not at origin are
                 assumed not to exist in valid reachable states).
    6.  Calculate the estimated lift travel cost:
        -   If `required_floors` is empty, travel cost is 0.
        -   Otherwise, get the floor indices for the current lift floor and
            all floors in `required_floors` using `self.floor_to_index`.
        -   Find the minimum (`min_req_idx`) and maximum (`max_req_idx`)
            indices among the `required_floors`.
        -   The estimated travel cost is the minimum of the distance from the
            current floor index to `min_req_idx` and the distance from the
            current floor index to `max_req_idx`, plus the span of the
            required floors (`max_req_idx - min_req_idx`). This estimates
            the cost to reach one end of the required range and then traverse
            the range.
    7.  Add the estimated travel cost to `h`.
    8.  Return `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.passenger_to_destin = {}
        self.floor_to_index = {}

        all_floors = set()
        floors_below_count = {}

        # Parse static facts to get passenger destinations and floor ordering
        for fact_str in task.static:
            parts = fact_str.strip('()').split()
            predicate = parts[0]

            if predicate == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.passenger_to_destin[passenger] = floor
                all_floors.add(floor)
            elif predicate == 'above':
                floor_above = parts[1]
                floor_below = parts[2]
                all_floors.add(floor_above)
                all_floors.add(floor_below)
                # Count how many floors are directly below each floor
                floors_below_count[floor_above] = floors_below_count.get(floor_above, 0) + 1

        # Ensure all floors found are in the count dictionary (those with 0 floors below)
        for floor in all_floors:
             floors_below_count.setdefault(floor, 0)

        # Sort floors based on the count of floors below them (ascending)
        # This gives the order from lowest to highest floor
        sorted_floors = sorted(all_floors, key=lambda f: floors_below_count[f])

        # Assign index based on the sorted order
        for i, floor in enumerate(sorted_floors):
            self.floor_to_index[floor] = i

        # logging.info(f"Initialized miconicHeuristic: floor_to_index={self.floor_to_index}, passenger_to_destin={self.passenger_to_destin}")


    def __call__(self, node):
        state = node.state
        h = 0
        required_floors = set()
        current_floor = None

        # Extract dynamic state information
        served_passengers = set()
        boarded_passengers = set()
        passengers_at_origin = {} # {passenger: floor}

        for fact_str in state:
            parts = fact_str.strip('()').split()
            predicate = parts[0]

            if predicate == 'served':
                served_passengers.add(parts[1])
            elif predicate == 'boarded':
                boarded_passengers.add(parts[1])
            elif predicate == 'origin':
                passengers_at_origin[parts[1]] = parts[2]
            elif predicate == 'lift-at':
                current_floor = parts[1]

        # Calculate action cost and identify required floors based on unserved passengers
        all_passengers = set(self.passenger_to_destin.keys()) # Get all passengers from static info

        for p in all_passengers:
            if p not in served_passengers:
                if p in boarded_passengers:
                    # Passenger is boarded but not served
                    h += 1 # Needs depart action
                    dest_floor = self.passenger_to_destin.get(p)
                    if dest_floor:
                        required_floors.add(dest_floor)
                    # else: Should not happen in valid problems - boarded passenger without destination?
                elif p in passengers_at_origin:
                    # Passenger is at origin but not served (and not boarded)
                    h += 2 # Needs board + depart actions
                    origin_floor = passengers_at_origin[p]
                    required_floors.add(origin_floor)
                # else: Passenger is unserved, not boarded, and not at origin.
                # This state should ideally not be reachable in a valid problem,
                # or implies the passenger is lost. We ignore them for heuristic calculation
                # assuming valid states only have passengers at origin, boarded, or served.


        # Calculate estimated lift travel cost
        travel_cost = 0
        if required_floors:
            # Ensure current floor is included in the set of floors to consider for span
            all_relevant_floors = required_floors | {current_floor}

            # Get indices for all relevant floors
            relevant_indices = sorted([self.floor_to_index[f] for f in all_relevant_floors if f in self.floor_to_index])

            if relevant_indices:
                min_idx = relevant_indices[0]
                max_idx = relevant_indices[-1]
                curr_idx = self.floor_to_index.get(current_floor, min_idx) # Default to min_idx if current_floor not found (shouldn't happen)

                # Estimate travel as distance to closest end of the range + the span
                dist_to_min = abs(curr_idx - min_idx)
                dist_to_max = abs(curr_idx - max_idx)

                travel_cost = min(dist_to_min, dist_to_max) + (max_idx - min_idx)

        h += travel_cost

        return h

