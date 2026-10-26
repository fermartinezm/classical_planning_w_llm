from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class if heuristic_base is not provided
# In a real scenario, this would be imported from the planning framework
class Heuristic:
    def __init__(self, task):
        pass

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing the number of
    unserved passengers (each needing at least a board and a depart action)
    and the minimum travel distance required for the lift to reach any floor
    where a passenger needs to be picked up or dropped off.

    # Assumptions
    - The floors are arranged linearly, and the 'above' predicates define this order.
    - The goal is to serve all specified passengers.
    - Each unserved passenger requires at least two actions: board and depart.

    # Heuristic Initialization
    - Parses 'above' facts from static information to build a mapping between
      floor names and their corresponding integer indices, assuming a linear
      floor structure starting from a single lowest floor.
    - Extracts the list of all passengers who need to be served from the goal
      conditions.
    - Extracts the destination floor for each passenger from static information.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Passengers: Determine which passengers from the goal
       list have not yet been marked as 'served' in the current state.
    2. Check for Goal State: If there are no unserved passengers, the state is
       a goal state, and the heuristic value is 0.
    3. Find Lift Location: Determine the current floor of the lift.
    4. Identify Relevant Floors: Collect the set of floors that the lift
       *must* visit to make progress towards serving the unserved passengers.
       This includes:
       - The origin floor for any unserved passenger who is currently waiting
         at their origin ('origin' predicate).
       - The destination floor for any unserved passenger who is currently
         boarded in the lift ('boarded' predicate).
    5. Calculate Minimum Distance to Relevant Floors: If there are relevant
       floors, calculate the minimum absolute difference between the lift's
       current floor index and the index of any relevant floor. This estimates
       the immediate travel cost to get to a useful location. If there are no
       relevant floors (e.g., all unserved passengers are already boarded and
       at their destination), this distance is 0.
    6. Calculate Total Heuristic: The heuristic value is the sum of:
       - Twice the number of unserved passengers (representing the board and
         depart actions needed per passenger).
       - The minimum distance calculated in the previous step (representing
         the immediate travel cost).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about floors
        and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # --- Build Floor Mapping ---
        above_facts_parts = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]

        # Map: floor_lower -> floor_higher
        above_map = {f_lower: f_higher for _, f_lower, f_higher in above_facts_parts}

        # Find all floors mentioned in above facts
        all_floors_in_above = set(above_map.keys()) | set(above_map.values())

        # Find the lowest floor: a floor that is a key but not a value in above_map
        # This assumes a single linear chain starting from one lowest floor
        potential_lowest = set(above_map.keys()) - set(above_map.values())

        lowest_floor = None
        if len(potential_lowest) == 1:
            lowest_floor = list(potential_lowest)[0]
        elif not above_facts_parts and all_floors_in_above:
             # Case with a single floor and no above facts? Or multiple floors without order?
             # If there's only one floor mentioned anywhere, that's the lowest.
             if len(all_floors_in_above) == 1:
                 lowest_floor = list(all_floors_in_above)[0]
             else:
                 # If multiple floors but no above facts, alphabetical might be the only guess.
                 # Or get all floors from all facts.
                 all_floor_names_from_facts = set()
                 for fact in task.initial_state | task.static | task.goals:
                      parts = get_parts(fact)
                      if len(parts) > 1 and parts[0] in ["origin", "destin", "lift-at"]:
                          all_floor_names_from_facts.add(parts[2] if parts[0] in ["origin", "destin"] else parts[1])
                      elif len(parts) > 2 and parts[0] == "above":
                          all_floor_names_from_facts.add(parts[1])
                          all_floor_names_from_facts.add(parts[2])
                 if all_floor_names_from_facts:
                     lowest_floor = sorted(list(all_floor_names_from_facts))[0] # Arbitrary fallback
                 else:
                     # No floors found at all?
                     lowest_floor = None

        # Build the ordered list and maps
        self.floor_to_index = {}
        self.index_to_floor = {}
        if lowest_floor:
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                self.index_to_floor[index] = current_floor
                index += 1
                current_floor = above_map.get(current_floor)
        else:
             # Fallback if floor structure is unexpected or empty
             print("Warning: Could not determine linear floor order. Heuristic may be inaccurate or fail.")
             # Attempt to create a map based on any floor names found, using alphabetical order as index
             all_floor_names_from_facts = set()
             for fact in task.initial_state | task.static | task.goals:
                  parts = get_parts(fact)
                  if len(parts) > 1 and parts[0] in ["origin", "destin", "lift-at"]:
                      all_floor_names_from_facts.add(parts[2] if parts[0] in ["origin", "destin"] else parts[1])
                  elif len(parts) > 2 and parts[0] == "above":
                      all_floor_names_from_facts.add(parts[1])
                      all_floor_names_from_facts.add(parts[2])
             sorted_floors = sorted(list(all_floor_names_from_facts))
             self.floor_to_index = {floor: i for i, floor in enumerate(sorted_floors)}
             self.index_to_floor = {i: floor for i, floor in enumerate(sorted_floors)}


        # --- Extract Passenger Info ---
        # Get all passengers from goal facts (those who need to be served)
        self.all_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Get passenger destinations from static facts
        self.passenger_destinations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "destin", "*", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if f"(served {p})" not in state}

        # If all passengers are served, the goal is reached.
        if not unserved_passengers:
            return 0

        # Find the lift's current floor
        lift_floor_str = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                 lift_floor_str = parts[1]
                 break

        # If lift location is unknown (shouldn't happen in valid states, but for robustness)
        if lift_floor_str is None or lift_floor_str not in self.floor_to_index:
             # Cannot compute distance, return a simple count
             print(f"Warning: Lift location '{lift_floor_str}' not found in floor map. Using fallback heuristic.")
             return 2 * len(unserved_passengers) # Fallback: just count unserved passengers * 2

        lift_floor_index = self.floor_to_index[lift_floor_str]

        # Identify relevant floors that the lift needs to visit
        relevant_floors_str = set()
        for p in unserved_passengers:
            # Check if passenger is waiting at origin
            origin_fact_parts = next((get_parts(fact) for fact in state if match(fact, "origin", p, "*")), None)
            if origin_fact_parts:
                relevant_floors_str.add(origin_fact_parts[2]) # Add origin floor

            # Check if passenger is boarded
            boarded_fact = f"(boarded {p})"
            if boarded_fact in state:
                # Add destination floor for boarded passenger
                # Ensure passenger has a known destination (should be in static facts)
                if p in self.passenger_destinations:
                    relevant_floors_str.add(self.passenger_destinations[p])
                else:
                    print(f"Warning: Boarded passenger {p} has no known destination.")


        # Calculate the minimum distance from the lift's current floor to any relevant floor
        min_dist_to_relevant = 0
        if relevant_floors_str:
            # Filter out any relevant floors that weren't successfully mapped to an index
            mappable_relevant_indices = {self.floor_to_index[f] for f in relevant_floors_str if f in self.floor_to_index}
            if mappable_relevant_indices:
                 min_dist_to_relevant = min(abs(lift_floor_index - idx) for idx in mappable_relevant_indices)
            # If relevant_floors_str is not empty but none are mappable, min_dist remains 0, which is a reasonable fallback


        # Heuristic calculation:
        # 2 * number of unserved passengers (estimate for board + depart per passenger)
        # + minimum distance to a floor where work is needed
        heuristic_value = 2 * len(unserved_passengers) + min_dist_to_relevant

        return heuristic_value

