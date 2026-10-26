from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of necessary board and depart actions for unserved passengers
    and adds an estimate of the minimum lift movement needed to visit all floors
    where pickups or dropoffs are required.

    # Assumptions
    - The floor order is defined by the `(above f_higher f_lower)` static facts,
      where `f_higher` is immediately above `f_lower`.
    - All passengers needing service are listed in `(destin ?p ?d)` static facts.
    - The goal is to have all such passengers `(served ?p)`.

    # Heuristic Initialization
    - Build a mapping from floor names to integer floor numbers based on the `above` facts.
    - Store the destination floor for each passenger from the `destin` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor.
    2. Identify all unserved passengers (those not `served`).
    3. Categorize unserved passengers into those waiting at an origin floor (`origin ?p ?o`)
       and those already boarded (`boarded ?p`).
    4. Count the number of waiting passengers (`NumWaiting`) and the total number
       of unserved passengers (`NumUnserved`).
    5. The number of required `board` actions is at least `NumWaiting`.
    6. The number of required `depart` actions is at least `NumUnserved`.
    7. Identify the set of floors the lift *must* visit: the origin floors of waiting
       passengers and the destination floors of boarded passengers. Let this set be `StopFloors`.
    8. If `NumUnserved` is 0, the goal is reached, and the heuristic is 0.
    9. If `NumUnserved` > 0, calculate the estimated movement cost:
       - Find the minimum (`min_s`) and maximum (`max_s`) floor numbers among `StopFloors`.
       - Get the current floor number of the lift (`f_current_num`).
       - The minimum movement cost is estimated as the distance from the current floor
         to the closer end of the required floor range `[min_s, max_s]`, plus the size
         of the range itself: `min(abs(f_current_num - min_s), abs(f_current_num - max_s)) + (max_s - min_s)`.
    10. The total heuristic value is the sum of the estimated board actions,
        estimated depart actions, and the estimated movement cost:
        `NumWaiting + NumUnserved + Movement cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build floor map: floor name -> integer number
        # Find the 'above' relationships: f_lower -> f_higher
        # This map stores which floor is immediately above another
        lower_to_higher_map = {}
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                lower_to_higher_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the lowest floor (the one that is a key in lower_to_higher_map but not a value,
        # or the only floor if there's only one)
        lowest_floor = None
        if len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
        elif len(all_floors) > 1:
            f_lowers = set(lower_to_higher_map.keys())
            f_highers = set(lower_to_higher_map.values())
            lowest_floor_candidates = f_lowers - f_highers
            if len(lowest_floor_candidates) == 1:
                 lowest_floor = list(lowest_floor_candidates)[0]
            # Fallback: if the lowest floor isn't explicitly a key with nothing below it
            # (e.g., a single floor problem, or unusual above facts),
            # find the floor that is not a value in the map (nothing is below it).
            elif len(lowest_floor_candidates) == 0:
                 potential_lowest = all_floors - f_highers
                 if len(potential_lowest) == 1:
                      lowest_floor = list(potential_lowest)[0]

        # If lowest_floor is still None and there are floors, it indicates an issue
        # with the static facts defining the floor order.
        if lowest_floor is None and len(all_floors) > 0:
             # As a fallback, if the lowest floor couldn't be determined from 'above' chain,
             # we could potentially sort floor names alphabetically or numerically if they
             # follow a pattern (like f1, f2, ...). However, relying on a specific naming
             # convention is fragile. The current logic should work for valid chains.
             # If it fails, the problem instance might be ill-defined regarding floors.
             # For now, we proceed assuming lowest_floor was found or all_floors is empty.
             pass


        self.floor_map = {}
        current = lowest_floor
        num = 1
        # Iterate upwards from the lowest floor
        while current is not None:
             self.floor_map[current] = num
             num += 1
             current = lower_to_higher_map.get(current) # Get floor immediately above

        # Store passenger destinations from static facts
        self.goal_locations = {}
        for static_fact in static_facts:
            parts = get_parts(static_fact)
            if parts[0] == "destin":
                 passenger, destination = parts[1], parts[2]
                 self.goal_locations[passenger] = destination


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        lift_at = None
        served_passengers = set()
        waiting_passengers = set()
        boarded_passengers = set()
        origin_map = {} # passenger -> origin_floor

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "lift-at":
                lift_at = parts[1]
            elif predicate == "served":
                served_passengers.add(parts[1])
            elif predicate == "origin":
                passenger, origin_floor = parts[1], parts[2]
                waiting_passengers.add(passenger)
                origin_map[passenger] = origin_floor
            elif predicate == "boarded":
                boarded_passengers.add(parts[1])

        # All passengers that need to be served are those in self.goal_locations
        all_passengers = set(self.goal_locations.keys())
        unserved_passengers = all_passengers - served_passengers

        num_waiting = len(waiting_passengers)
        # num_boarded = len(boarded_passengers) # Not directly used in final sum, but needed for stop_floors

        # Heuristic is 0 if all passengers are served
        if len(unserved_passengers) == 0:
            return 0

        # Floors the lift must visit for pickups or dropoffs
        stop_floors = set()
        for p in waiting_passengers:
            # Need to pick up at origin
            stop_floors.add(origin_map[p])
            # Need to drop off at destination
            stop_floors.add(self.goal_locations[p])
        for p in boarded_passengers:
            # Need to drop off at destination
            stop_floors.add(self.goal_locations[p])

        # Convert stop floors to numbers
        stop_floor_nums = {self.floor_map[f] for f in stop_floors}

        min_s = min(stop_floor_nums)
        max_s = max(stop_floor_nums)
        f_current_num = self.floor_map[lift_at]

        # Estimated movement cost
        # Distance to the closer end of the required range + the size of the range
        movement_cost = min(abs(f_current_num - min_s), abs(f_current_num - max_s)) + (max_s - min_s)

        # Total heuristic: board actions + depart actions + movement actions
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        total_cost = num_waiting + len(unserved_passengers) + movement_cost

        return total_cost
