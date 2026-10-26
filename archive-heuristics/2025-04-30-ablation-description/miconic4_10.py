from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the minimum number of actions (lift movements, board, depart)
    required to serve all passengers specified in the goal. It calculates the cost
    for each unserved passenger independently, summing up the estimated costs.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic assumes passengers are served individually, ignoring potential
      optimizations like picking up/dropping off multiple passengers on a single trip.
    - Floors are ordered linearly based on the `above` predicate, forming a single tower.

    # Heuristic Initialization
    - Parses the static facts (`above` predicates) to determine the floor ordering
      and assign a numerical level to each floor. The lowest floor is assigned level 1.
    - Extracts the destination floor for each passenger from the static facts,
      focusing only on passengers that are part of the goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of the estimated costs for each
    passenger that has not yet been served.

    For each passenger `p` that needs to be served (i.e., `(served p)` is in the goal state):
    1. Check if the passenger is already served in the current state.
       - If `(served p)` is true: The passenger is already served. Cost for this passenger is 0.
    2. If the passenger is not served, check their current state:
       - If `(boarded p)` is true in the current state: The passenger is inside the lift.
         - Find the lift's current floor (`(lift-at ?f_current)`).
         - Find the passenger's destination floor (`self.destinations[p]`).
         - The estimated cost for this passenger is the number of floor movements from the current lift floor to the destination floor plus the cost of departing.
         - Cost = `abs(floor_level[destination] - floor_level[current_lift_floor]) + 1` (for depart action).
       - If `(origin p f_origin)` is true in the current state: The passenger is waiting at their origin floor.
         - Find the lift's current floor (`(lift-at ?f_current)`).
         - Find the passenger's origin floor (`f_origin`).
         - Find the passenger's destination floor (`self.destinations[p]`).
         - The estimated cost for this passenger involves two phases:
           a) Getting the lift to the origin floor and boarding the passenger:
              Cost = `abs(floor_level[origin] - floor_level[current_lift_floor]) + 1` (for board action).
           b) Moving the passenger from the origin floor to the destination floor and departing:
              Cost = `abs(floor_level[destination] - floor_level[origin]) + 1` (for depart action).
         - Total estimated cost for this passenger = (Cost from phase a) + (Cost from phase b).
       - If the passenger is neither served, boarded, nor at an origin, this indicates an unexpected state. The heuristic returns infinity.

    3. Sum the estimated costs for all unserved passengers. This sum is the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract goal passengers
        self.goal_passengers = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "served":
                self.goal_passengers.add(args[0])

        # Find destinations for all passengers that are in the goal
        self.destinations = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "destin":
                 passenger, floor = args
                 if passenger in self.goal_passengers:
                     self.destinations[passenger] = floor

        # Build floor level mapping using BFS on the 'above' relations
        # (above f_higher f_lower) means f_higher is one level above f_lower
        # Graph edge: f_lower -> f_higher
        adj_up = defaultdict(list) # f_lower -> [f_higher, ...]
        in_degree_up = defaultdict(int)
        all_floors_set = set()

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "above":
                f_higher, f_lower = args
                adj_up[f_lower].append(f_higher)
                in_degree_up[f_higher] += 1
                all_floors_set.add(f_higher)
                all_floors_set.add(f_lower)

        # Find the lowest floor(s) (in-degree 0 in the f_lower -> f_higher graph)
        # Assuming a single linear tower structure, there should be exactly one floor with in-degree 0.
        lowest_floors = [f for f in all_floors_set if in_degree_up[f] == 0]

        self.floor_level = {}
        q = deque()
        level = 1

        # Start BFS from all identified lowest floors
        for f in lowest_floors:
            if f not in self.floor_level: # Avoid processing same floor if multiple lowest found (unlikely in Miconic)
                self.floor_level[f] = level
                q.append(f)

        # If no 'above' facts but floors exist (e.g., single floor problem)
        if not all_floors_set:
             # Attempt to find floors from initial state or goals if no 'above' facts
             # This is a fallback and might not be exhaustive without full object list
             potential_floors = set()
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if len(parts) > 1 and parts[0] in ['lift-at', 'origin', 'destin']:
                     for part in parts[1:]:
                         # Simple check: assume parts starting with 'f' are floors
                         if part.startswith('f'):
                             potential_floors.add(part)
             for goal in task.goals:
                  parts = get_parts(goal)
                  if len(parts) > 1 and parts[0] in ['lift-at', 'origin', 'destin']: # Goal might contain these for some domains
                      for part in parts[1:]:
                          if part.startswith('f'):
                              potential_floors.add(part)

             if potential_floors:
                 # If we found floors but no 'above' facts, assume they are all on the same level or order doesn't matter
                 # For this heuristic, we need levels. This scenario is problematic.
                 # Let's assign level 1 to all found floors as a fallback, though this breaks distance calculation.
                 # A more robust approach would be to fail or return infinity.
                 # For simplicity in this fallback, assign level 1.
                 for f in potential_floors:
                     self.floor_level[f] = 1
                 # print("Warning: No 'above' facts found. Assuming all floors are at level 1.")
             # else: no floors found at all, floor_level remains empty


        # Perform BFS to assign levels
        while q:
            current_floor = q.popleft()
            current_level = self.floor_level[current_floor]

            # Find floors immediately above current_floor
            # These are neighbors of current_floor in the adj_up graph
            for next_floor in adj_up.get(current_floor, []):
                 if next_floor not in self.floor_level:
                     self.floor_level[next_floor] = current_level + 1
                     q.append(next_floor)

        # Optional: Check if all floors were assigned a level
        # If len(self.floor_level) != len(all_floors_set), it implies a disconnected
        # or cyclic 'above' graph, which is invalid for a simple tower.
        # We proceed but the heuristic might be inaccurate if required levels are missing.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "lift-at":
                current_lift_floor = args[0]
                break

        if current_lift_floor is None:
             # Should not happen in a valid Miconic state
             # print("Error: Lift location not found in state.")
             return float('inf') # Cannot estimate cost without lift location

        current_lift_level = self.floor_level.get(current_lift_floor)
        if current_lift_level is None:
             # Should not happen if floor_level was built correctly and state is valid
             # print(f"Error: Level for current lift floor {current_lift_floor} not found.")
             return float('inf') # Cannot estimate cost


        # Track which passengers are served, boarded, or waiting at origin
        served_passengers = set()
        boarded_passengers = set()
        origin_locations = {} # passenger -> floor

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "served":
                served_passengers.add(args[0])
            elif predicate == "boarded":
                boarded_passengers.add(args[0])
            elif predicate == "origin":
                passenger, floor = args
                origin_locations[passenger] = floor

        # Calculate cost for each goal passenger not yet served
        for passenger in self.goal_passengers:
            if passenger in served_passengers:
                continue # Already served, cost is 0 for this passenger

            destin_floor = self.destinations.get(passenger)
            if destin_floor is None:
                 # Destination not found (should be in self.destinations from init)
                 # print(f"Error: Destination not found for goal passenger {passenger}")
                 return float('inf') # Cannot estimate cost

            destin_level = self.floor_level.get(destin_floor)
            if destin_level is None:
                 # print(f"Error: Destination floor {destin_floor} level unknown for passenger {passenger}")
                 return float('inf') # Cannot estimate cost


            if passenger in boarded_passengers:
                # Passenger is in the lift
                # Needs to travel to destination and depart
                cost_move_to_destin = abs(destin_level - current_lift_level)
                cost_depart = 1
                total_cost += cost_move_to_destin + cost_depart

            elif passenger in origin_locations:
                # Passenger is waiting at origin
                origin_floor = origin_locations[passenger]
                origin_level = self.floor_level.get(origin_floor)
                if origin_level is None:
                     # print(f"Error: Origin floor {origin_floor} level unknown for passenger {passenger}")
                     return float('inf') # Cannot estimate cost

                # Needs lift to come to origin, board, travel to destin, depart
                cost_move_to_origin = abs(origin_level - current_lift_level)
                cost_board = 1
                cost_move_to_destin = abs(destin_level - origin_level)
                cost_depart = 1

                total_cost += cost_move_to_origin + cost_board + cost_move_to_destin + cost_depart

            else:
                 # Passenger is neither served, boarded, nor at an origin.
                 # This indicates an unexpected state for a goal passenger.
                 # print(f"Error: Goal passenger {passenger} not served, boarded, or at origin.")
                 return float('inf') # Cannot estimate cost from invalid state

        return total_cost
