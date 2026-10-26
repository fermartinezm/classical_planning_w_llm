from fnmatch import fnmatch
# Assuming Heuristic base class is imported or available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions (as seen in Logistics example)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't zip shorter parts list with longer args list
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Use a simple class definition if Heuristic base is not provided in the final output context
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts
    that are required to be tightened in the goal state. It considers the cost
    of acquiring a usable spanner if Bob doesn't have one, the cost of traveling
    to the vicinity of the nuts, and the cost of tightening each nut.

    # Assumptions
    - Bob is the only agent.
    - Bob needs a usable spanner to tighten a nut.
    - Bob must be at the same location as a nut to tighten it.
    - All relevant locations (where Bob, spanners, or nuts are) are part of a connected graph defined by 'link' predicates in solvable problems.
    - The cost of travel between nuts after reaching the first one is not explicitly modeled beyond reaching the closest one.
    - Each action (move, pickup, drop, tighten) has a cost of 1.

    # Heuristic Initialization
    - Extracts the set of nuts that must be tightened in the goal state.
    - Builds a graph of locations based on 'link' predicates.
    - Computes all-pairs shortest paths (distances) between connected locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of nuts that are currently 'loose' in the state and are required to be 'tightened' in the goal.
    2. If this set is empty, the goal is effectively reached for all nuts, so the heuristic is 0.
    3. Find Bob's current location in the state.
    4. Find the current location for each of the loose nuts identified in step 1.
    5. Find the current location for each usable spanner in the state.
    6. Determine if Bob is currently carrying a usable spanner.
    7. Calculate the 'spanner setup cost' and the 'effective start location' for Bob's subsequent travel:
       - If Bob is carrying a usable spanner, the cost is 0, and the effective start location is Bob's current location.
       - If Bob is not carrying a usable spanner:
         - If Bob is carrying *something* else, add 1 for the 'drop' action.
         - Find the usable spanner location closest to Bob's current location among reachable spanner locations.
         - Add the distance to this closest reachable spanner location plus 1 for the 'pickup' action to the cost.
         - The effective start location for travel is the location where Bob picks up the spanner.
       - If no usable spanners are found or reachable, the problem is likely unsolvable, return infinity.
    8. Calculate the 'travel to nuts cost':
       - Find the target nut location that is closest to the effective start location determined in step 7 among reachable nut locations.
       - The travel cost is the distance from the effective start location to this closest reachable nut location. (This simplifies travel between multiple nuts).
       - If no target nuts are found or reachable, return infinity.
    9. Calculate the 'tighten cost': This is simply the number of loose nuts that need tightening, as each requires one 'tighten' action.
    10. The total heuristic value is the sum of the spanner setup cost, the travel to nuts cost, and the tighten cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and precomputing
        location distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph from static 'link' facts
        all_linked_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                all_linked_locations.add(parts[1])
                all_linked_locations.add(parts[2])

        self.location_graph = {loc: set() for loc in all_linked_locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1)

        # Compute all-pairs shortest paths (distances) using BFS
        self.distances = {}
        for start_loc in all_linked_locations:
            self.distances[start_loc] = {}
            queue = [(start_loc, 0)]
            visited = {start_loc}
            while queue:
                current_loc, dist = queue.pop(0)
                self.distances[start_loc][current_loc] = dist
                # Check if current_loc is in graph (handles potential isolated nodes if any were added)
                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """
        Get the precomputed shortest distance between two locations.
        Returns float('inf') if locations are not in the graph or disconnected.
        """
        # Ensure both locations are in the graph before looking up distance
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Loose Nuts that need tightening
        loose_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        loose_nuts_to_tighten = loose_nuts_in_state.intersection(self.goal_nuts)

        # 2. If this set is empty, the goal is effectively reached for all nuts, so the heuristic is 0.
        if not loose_nuts_to_tighten:
            return 0

        # 3. Find Bob's current location
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        # Assume bob is always located somewhere in a solvable problem
        if bob_location is None:
             # Should not happen in a valid state for this domain, but handle defensively
             return float('inf') # Indicates an impossible state

        # 4. Find locations of loose nuts to tighten
        nut_locations = {}
        for nut in loose_nuts_to_tighten:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations[nut] = get_parts(fact)[2]
                    break
            # Assume all nuts are located somewhere in a solvable problem
            if nut not in nut_locations:
                 # Nut exists but has no location? Unsolvable.
                 return float('inf')

        # 5. Find locations of usable spanners
        usable_spanner_locations = set()
        for fact in state:
            # Find objects that are 'at' a location
            if match(fact, "at", "*", "*"):
                obj_at_loc, loc = get_parts(fact)[1], get_parts(fact)[2]
                # Check if the object at this location is usable
                if f"(usable {obj_at_loc})" in state:
                     # Assume anything that is (usable X) is a spanner type for this domain
                     usable_spanner_locations.add(loc)

        # 6. Determine if Bob is currently carrying a usable spanner.
        carried_item = None
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                carried_item = get_parts(fact)[2]
                break

        is_carrying_usable_spanner = (carried_item is not None) and (f"(usable {carried_item})" in state)

        # 7. Calculate the 'spanner setup cost' and the 'effective start location' for Bob's subsequent travel:
        spanner_setup_cost = 0
        loc_after_spanner_setup = bob_location

        if not is_carrying_usable_spanner:
            # Bob needs to get a usable spanner
            if carried_item is not None:
                # Carrying something, but not a usable spanner - must drop it
                spanner_setup_cost += 1 # drop action

            # Find closest usable spanner location
            min_dist = float('inf')
            closest_spanner_loc = None
            # Need to consider distance from bob_location to usable_spanner_locations
            reachable_usable_spanner_locations = [
                loc for loc in usable_spanner_locations if self.get_distance(bob_location, loc) != float('inf')
            ]

            if not reachable_usable_spanner_locations:
                 # No usable spanners found or reachable - problem likely unsolvable
                 return float('inf')

            for spanner_loc in reachable_usable_spanner_locations:
                dist = self.get_distance(bob_location, spanner_loc)
                if dist < min_dist:
                    min_dist = dist
                    closest_spanner_loc = spanner_loc

            spanner_setup_cost += min_dist # moves to spanner
            spanner_setup_cost += 1 # pickup action
            loc_after_spanner_setup = closest_spanner_loc

        # 8. Calculate travel cost to nuts
        # Need to visit all locations in nut_locations.values()
        # Simple estimate: distance from loc_after_spanner_setup to the closest nut location
        min_dist_to_first_nut = float('inf')
        target_nut_locations_set = set(nut_locations.values()) # Unique locations of target nuts

        # Need to consider distance from loc_after_spanner_setup to target_nut_locations_set
        reachable_nut_locations = [
            loc for loc in target_nut_locations_set if self.get_distance(loc_after_spanner_setup, loc) != float('inf')
        ]

        if not reachable_nut_locations:
             # No target nuts found or reachable from spanner setup location - problem likely unsolvable
             return float('inf')

        for nut_loc in reachable_nut_locations:
             dist = self.get_distance(loc_after_spanner_setup, nut_loc)
             min_dist_to_first_nut = min(min_dist_to_first_nut, dist)

        travel_to_nuts_cost = min_dist_to_first_nut

        # 9. Calculate tighten cost
        tighten_cost = len(loose_nuts_to_tighten)

        # 10. Total heuristic
        total_cost = spanner_setup_cost + travel_to_nuts_cost + tighten_cost

        return total_cost
