from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact string gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the required boarding actions, the required departing
    actions, and an estimate of the lift movement cost.

    # Assumptions
    - The domain uses standard miconic predicates: origin, destin, above,
      boarded, served, lift-at.
    - The 'above' predicates define a linear ordering of floors.
    - All passengers needing service are listed in the task goals as (served ?p).
    - Passenger origins and destinations are static or defined in the initial state.

    # Heuristic Initialization
    - Parses 'above' predicates from static facts to build a mapping from
      floor names to numerical levels. This allows calculating vertical distance.
    - Parses 'destin' predicates from static facts or initial state to map
      each passenger to their destination floor.
    - Stores the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift. Get its numerical level.
    2. Initialize counters for required 'board' and 'depart' actions to zero.
    3. Initialize sets for floors needing a pickup ('PickupFloors') and floors
       needing a dropoff ('DropoffFloors').
    4. Iterate through each passenger that needs to be served (from task goals):
       - If the passenger is already '(served ?p)' in the current state, ignore them.
       - If the passenger is not served:
         - Increment the count for 'depart' actions needed (each unserved passenger needs one depart).
         - Get the passenger's destination floor from the pre-calculated map and add it to 'DropoffFloors'.
         - Check if the passenger is currently '(boarded ?p)'.
           - If yes, they are in the lift; no 'board' action is needed for them in the future.
           - If no, check if the passenger is at their origin '(origin ?p ?orig)'.
             - If yes, they are waiting at their origin; increment the count for 'board' actions needed and add their origin floor to 'PickupFloors'.
             - If no, the passenger is not served, not boarded, and not at their origin (this state should ideally not occur in a solvable problem unless the passenger was moved by a non-standard action, which is not the case in this domain). We assume valid states where unserved, unboarded passengers are at their origin.
    5. Combine 'PickupFloors' and 'DropoffFloors' into 'ServiceFloors'. These are all floors the lift must visit to make progress.
    6. Calculate the estimated lift movement cost:
       - If 'ServiceFloors' is empty (all relevant passengers are served or already at the lift's current location and ready to depart), the movement cost is 0.
       - If 'ServiceFloors' is not empty:
         - Find the minimum and maximum floor levels among 'ServiceFloors'.
         - The estimated movement cost is the minimum vertical distance required to reach either the lowest or highest service floor from the current lift floor, plus the vertical distance spanning all service floors. This estimates the cost of moving to one end of the required range and then traversing the range.
    7. The total heuristic value is the sum of 'board_actions_needed', 'depart_actions_needed', and 'movement_cost'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions, used to identify all passengers.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Build floor level map from 'above' predicates.
        above_relations = {} # map floor_below -> floor_above
        all_floors = set()
        floors_above = set() # Floors that appear as the first argument in 'above'

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_above, f_below = get_parts(fact)
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_above.add(f_above)
                above_relations[f_below] = f_above # f_below is below f_above

        # Also add floors from initial state that might not be in 'above' facts
        for fact in initial_state:
             if match(fact, "lift-at", "*"):
                 all_floors.add(get_parts(fact)[1])
             elif match(fact, "origin", "*", "*"):
                  all_floors.add(get_parts(fact)[2])
             elif match(fact, "destin", "*", "*"):
                  all_floors.add(get_parts(fact)[2])

        self.floor_levels = {}
        if not all_floors:
             raise ValueError("Could not identify any floors in the problem.")
        elif len(all_floors) == 1:
             self.floor_levels = {list(all_floors)[0]: 1}
        elif not above_relations:
             # Multiple floors but no 'above' relations to define order.
             # Fallback: Assign levels based on alphabetical sort.
             print("Warning: Multiple floors found but no 'above' facts to define order. Assuming alphabetical order for levels.")
             sorted_floors = sorted(list(all_floors))
             self.floor_levels = {f: i+1 for i, f in enumerate(sorted_floors)}
        else:
            # Standard case with 'above' relations. Find the lowest floor.
            # The lowest floor is one that is in all_floors but is never the first argument of an 'above' predicate.
            lowest_floor_candidates = all_floors - floors_above
            if len(lowest_floor_candidates) != 1:
                 print("Warning: Could not uniquely determine lowest floor from 'above' facts. Assuming alphabetical order for levels.")
                 sorted_floors = sorted(list(all_floors))
                 self.floor_levels = {f: i+1 for i, f in enumerate(sorted_floors)}
            else:
                lowest_floor = lowest_floor_candidates.pop()
                current_floor = lowest_floor
                level = 1
                # Build map upwards using the above_relations (mapping floor_below -> floor_above)
                while current_floor in above_relations:
                    self.floor_levels[current_floor] = level
                    current_floor = above_relations[current_floor]
                    level += 1
                # Add the highest floor which is not a key in above_relations
                self.floor_levels[current_floor] = level

                # Double check if all floors were assigned a level.
                if len(self.floor_levels) != len(all_floors):
                     print("Warning: Not all floors are connected by 'above' facts. Assuming alphabetical order for levels.")
                     sorted_floors = sorted(list(all_floors))
                     self.floor_levels = {f: i+1 for i, f in enumerate(sorted_floors)}


        # 2. Build passenger destination map.
        self.passenger_destinations = {}
        # Get all passengers from goals
        all_goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Also collect destinations from initial state and static facts for any passenger
        # (even if not in goal, though typically goals list all relevant passengers)
        for fact in static_facts.union(initial_state):
             if match(fact, "destin", "*", "*"):
                  _, p, dest_floor = get_parts(fact)
                  self.passenger_destinations[p] = dest_floor

        # Ensure all goal passengers have a destination recorded
        for p in all_goal_passengers:
             if p not in self.passenger_destinations:
                  print(f"Warning: Destination not found for goal passenger {p}. This may lead to inaccurate heuristic values.")
                  # Cannot estimate cost for this passenger's destination movement.
                  # The heuristic will still count the depart action, but not movement to destination.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find current lift floor and level.
        current_lift_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if current_lift_fact is None:
             # This should not happen in a valid miconic state
             # If the lift location is unknown, we can't compute movement cost.
             # Return a large value.
             print("Warning: lift-at predicate not found in state.")
             # Fallback: Count unserved passengers * base cost
             unserved_count = sum(1 for goal in self.goals if get_parts(goal)[1] not in {get_parts(fact)[1] for fact in state if match(fact, "served", "*")})
             return unserved_count * 3 # Arbitrary large cost

        current_floor = get_parts(current_lift_fact)[1]
        current_level = self.floor_levels.get(current_floor)
        if current_level is None:
             # This floor was not in the initial 'above' facts or initial state floors.
             # This indicates an issue with floor parsing or problem definition.
             # Return a large value.
             print(f"Warning: Level for floor {current_floor} not found in floor_levels map.")
             # Fallback: Count unserved passengers * base cost
             unserved_count = sum(1 for goal in self.goals if get_parts(goal)[1] not in {get_parts(fact)[1] for fact in state if match(fact, "served", "*")})
             return unserved_count * 3 # Arbitrary large cost


        # 2. & 3. Initialize counters and sets.
        board_actions_needed = 0
        depart_actions_needed = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Get sets/maps from current state for quick lookup
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_locations_in_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}


        # 4. Iterate through each passenger that needs to be served (from task goals).
        all_goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        for p in all_goal_passengers:
            if p in served_passengers_in_state:
                continue # Passenger is already served

            # Passenger is not served
            depart_actions_needed += 1 # Needs a depart action eventually

            dest_floor = self.passenger_destinations.get(p)
            if dest_floor:
                 dropoff_floors.add(dest_floor)
            else:
                 # Destination not found for a goal passenger. Problematic instance.
                 # Cannot estimate movement to destination. Heuristic will be less accurate.
                 # The depart action is already counted. No movement added for this dest.
                 pass # Warning printed in __init__


            if p not in boarded_passengers_in_state:
                # Passenger is not served and not boarded, must be at origin (in a valid state)
                orig_floor = origin_locations_in_state.get(p)
                if orig_floor:
                    board_actions_needed += 1 # Needs a board action
                    pickup_floors.add(orig_floor)
                else:
                    # Passenger not served, not boarded, and not at origin in this state.
                    # This indicates a potentially invalid state or problem definition.
                    # We counted the board action, but cannot add origin to pickup floors.
                    print(f"Warning: Passenger {p} is not served, not boarded, and not at origin in state. Assuming they need boarding but origin is unknown.")


        # 5. Combine service floors.
        service_floors = pickup_floors.union(dropoff_floors)

        # 6. Calculate movement cost.
        movement_cost = 0
        if service_floors:
            try:
                # Ensure all service floors have a level mapping
                if not all(f in self.floor_levels for f in service_floors):
                     # This should have been caught during initialization or indicates a dynamic floor?
                     # Fallback to a simple action count if floor levels are missing.
                     print("Warning: Service floor found without a level mapping. Using fallback movement cost.")
                     movement_cost = len(service_floors) * 2 # Arbitrary cost per floor
                else:
                    min_service_level = min(self.floor_levels[f] for f in service_floors)
                    max_service_level = max(self.floor_levels[f] for f in service_floors)

                    # Estimate movement to cover the range [min_service_level, max_service_level]
                    # starting from current_level.
                    # Cost = distance to nearest end + span of the range
                    dist_to_min = abs(current_level - min_service_level)
                    dist_to_max = abs(current_level - max_service_level)
                    span = max_service_level - min_service_level

                    movement_cost = min(dist_to_min, dist_to_max) + span

            except Exception as e:
                 # Catch any other unexpected errors during movement calculation
                 print(f"Warning: Error during movement cost calculation: {e}. Using fallback movement cost.")
                 movement_cost = len(service_floors) * 2 # Arbitrary cost per floor


        # 7. Total heuristic.
        total_cost = board_actions_needed + depart_actions_needed + movement_cost

        # The heuristic is 0 iff board_actions_needed=0, depart_actions_needed=0, and movement_cost=0.
        # depart_actions_needed=0 implies all goal passengers are served.
        # board_actions_needed=0 implies no passengers are waiting at origin.
        # movement_cost=0 implies service_floors is empty.
        # service_floors empty implies pickup_floors empty and dropoff_floors empty.
        # pickup_floors empty implies no passengers waiting at origin.
        # dropoff_floors empty implies no non-served passengers have destinations added.
        # Since destinations for *all* non-served passengers are added to dropoff_floors,
        # dropoff_floors empty implies there are NO non-served passengers.
        # Thus, total_cost=0 iff all goal passengers are served (goal state).

        return total_cost
