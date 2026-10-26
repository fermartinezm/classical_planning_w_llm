from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the required board actions, the required depart actions, and an
    estimate of the necessary floor movements. The movement estimate assumes
    the lift travels from its current floor to the lowest required floor,
    and then sweeps upwards to the highest required floor, visiting all
    required floors in between.

    # Assumptions
    - Floor names are structured as 'f' followed by a number (e.g., f1, f2, f10),
      and these numbers represent the floor order (f1 < f2 < f10).
    - The goal is always to have a set of passengers 'served'.

    # Heuristic Initialization
    - Extracts the origin and destination floors for all passengers from static facts.
    - Determines the ordered list of all floors in the problem instance by
      parsing floor names and sorting them numerically. Creates mappings
      between floor names and their numerical indices.
    - Stores the set of passengers that need to be served (from goal conditions).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift.
    2. Determine which passengers are unserved (not yet in the 'served' state).
    3. Separate unserved passengers into those waiting at their origin floor
       and those already boarded in the lift.
    4. Count the number of 'board' actions needed: This is the number of
       unserved passengers waiting at their origin floor.
    5. Count the number of 'depart' actions needed: This is the total number
       of unserved passengers.
    6. Identify the set of floors the lift *must* visit to serve the remaining
       passengers. This includes the origin floors of waiting passengers and
       the destination floors of all unserved passengers (both waiting and boarded).
    7. If there are no floors to visit (all passengers served), the movement
       cost is 0.
    8. If there are floors to visit, find the minimum and maximum floor indices
       among these required floors.
    9. Calculate the estimated movement cost: This is the distance from the
       current lift floor to the lowest required floor, plus the distance
       between the lowest and highest required floors. This models a strategy
       where the lift first goes to the lowest required floor and then sweeps
       upwards to the highest required floor.
       Movement Cost = abs(current_floor_index - min_required_floor_index) +
                       (max_required_floor_index - min_required_floor_index).
    10. The total heuristic value is the sum of the board count, the depart count,
        and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger origins/destinations
        and floor ordering.
        """
        self.goals = task.goals  # Goal conditions

        # Extract passenger origins and destinations from static facts
        self.passenger_origins = {}
        self.passenger_destins = {}
        all_floors_set = set()

        # Collect all facts from initial state, goals, and static to find all floors
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)

        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == "origin":
                p, f = parts[1], parts[2]
                self.passenger_origins[p] = f
                all_floors_set.add(f)
            elif parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.passenger_destins[p] = f
                all_floors_set.add(f)
            elif parts[0] == "lift-at":
                 f = parts[1]
                 all_floors_set.add(f)
            elif parts[0] == "above":
                 f1, f2 = parts[1], parts[2]
                 all_floors_set.add(f1)
                 all_floors_set.add(f2)


        # Determine floor order by numerical suffix
        # Assumes floor names are like 'f1', 'f2', 'f10'
        def floor_sort_key(floor_name):
            # Extract the number part after 'f' and convert to int
            return int(floor_name[1:])

        self.floor_list = sorted(list(all_floors_set), key=floor_sort_key)
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_list)}
        self.index_to_floor = {i: floor for i, floor in enumerate(self.floor_list)}

        # Identify passengers that need to be served
        self.goal_passengers = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.goal_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is not found, the state is likely invalid or terminal
        # For a heuristic, returning infinity or a very large number is appropriate
        # if the state is not the goal but seems problematic.
        # Assuming valid states in typical planning problems, this check might be redundant.
        # However, if current_lift_floor is None, we cannot compute movement cost.
        # If unserved_passengers is empty, we return 0 anyway.
        # If unserved_passengers is not empty but lift location is unknown, something is wrong.
        # Let's proceed assuming current_lift_floor is found if unserved_passengers is not empty.


        # 2. Determine unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.goal_passengers - served_passengers

        # If all goal passengers are served, it's a goal state
        if not unserved_passengers:
            return 0

        # 3. Separate unserved passengers
        waiting_unserved = set()
        boarded_unserved = set()
        for p in unserved_passengers:
            if any(match(fact, "origin", p, "*") for fact in state):
                waiting_unserved.add(p)
            elif any(match(fact, "boarded", p) for fact in state):
                boarded_unserved.add(p)
            # Note: A passenger should be either waiting or boarded if not served
            # and not at their origin (which implies they were picked up).
            # The domain definition ensures this partition.

        # 4. Count board actions needed
        board_actions_needed = len(waiting_unserved)

        # 5. Count depart actions needed
        depart_actions_needed = len(unserved_passengers)

        # 6. Identify floors the lift must visit
        floors_to_visit = set()
        for p in waiting_unserved:
            # Need to visit origin to pick up
            floors_to_visit.add(self.passenger_origins[p])
        for p in unserved_passengers:
             # Need to visit destination to drop off (whether waiting or boarded)
             floors_to_visit.add(self.passenger_destins[p])

        # 7. Calculate movement cost
        movement_cost = 0
        if floors_to_visit:
            floor_indices_to_visit = {self.floor_to_index[f] for f in floors_to_visit}
            min_idx = min(floor_indices_to_visit)
            max_idx = max(floor_indices_to_visit)
            curr_idx = self.floor_to_index[current_lift_floor]

            # 9. Estimated movement cost: go to lowest required, sweep up
            # This estimates the distance from the current floor to the lowest
            # required floor, plus the distance to traverse from the lowest
            # required floor up to the highest required floor.
            movement_cost = abs(curr_idx - min_idx) + (max_idx - min_idx)

            # An alternative strategy (go to highest, sweep down) could be:
            # movement_cost_alt = abs(curr_idx - max_idx) + (max_idx - min_idx)
            # Using only the up-sweep strategy for simplicity and potentially
            # better performance characteristics for GBFS on this domain.

        # 10. Total heuristic value
        total_cost = board_actions_needed + depart_actions_needed + movement_cost

        return total_cost
