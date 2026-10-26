from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in the execution environment
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the fact has at least as many parts as args for zipping
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the number of nuts to tighten,
    the number of spanners that need to be picked up, and the travel cost
    to reach the first location where a necessary action (tightening a nut
    or picking up a spanner) can occur.

    # Assumptions
    - Each loose nut in the goal needs to be tightened.
    - Tightening a nut consumes one usable spanner.
    - The man can only carry one spanner at a time (implied by pickup/carrying predicates).
    - The location graph is static and traversable.
    - All necessary usable spanners exist somewhere in the initial state for solvable problems.
    - There is exactly one man object in the problem.
    - Links between locations are bidirectional.

    # Heuristic Initialization
    - Identifies the man object name.
    - Extracts all location objects from the static facts (links) and initial state (object locations).
    - Builds the location graph based on `link` predicates.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the set of nuts that are goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Target Nuts: Find all nuts that are currently `loose` and are required to be `tightened` in the goal state. Let N_loose_target be the count. If N_loose_target is 0, the heuristic is 0.

    2. Count Required Pickups: To tighten N_loose_target nuts, the man needs N_loose_target usable spanners in total. If the man is currently carrying a usable spanner, he needs N_loose_target - 1 additional spanners. Otherwise, he needs N_loose_target spanners. This is the number of `pickup_spanner` actions needed. Let this be N_pickups_needed.

    3. Base Action Cost: The minimum number of actions includes N_loose_target `tighten_nut` actions and N_pickups_needed `pickup_spanner` actions. Initialize heuristic h = N_loose_target + N_pickups_needed.

    4. Identify Current State Information: Determine the man's current location (ManLoc), the locations of all loose target nuts (NutLocs), the locations of all currently usable spanners (UsableSpannerLocs), and whether the man is currently carrying a usable spanner (ManCarryingUsable).

    5. Calculate Travel Cost: The man needs to travel to perform the tighten and pickup actions. A lower bound on the travel cost is the distance from the man's current location to the *closest* location where a necessary action can first occur. Necessary actions are tightening a nut (requires being at a NutLoc) or picking up a spanner (requires being at a UsableSpannerLoc, only if pickups are needed).
        - Identify candidate locations: All NutLocs. If N_pickups_needed > 0 and there are usable spanners available at locations, also include UsableSpannerLocs.
        - If there are candidate locations, find the minimum distance from ManLoc to any candidate location using the precomputed shortest paths. Add this minimum distance to h.

    6. Return Total Heuristic: The final value of h is the estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and building
        the location graph for distance calculations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Find man name (assuming there is exactly one man object)
        self.man_name = None
        # Look for the object involved in a 'carrying' predicate first
        for fact in initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break # Found the man via carrying predicate

        if not self.man_name:
             # Fallback: Find the single object that is 'at' a location and is not a spanner or nut
             potential_men = set()
             spanners_and_nuts = set()
             for fact in initial_state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     potential_men.add(obj)
                 # Identify spanners and nuts from initial state predicates
                 if match(fact, "usable", "*"):
                     spanners_and_nuts.add(get_parts(fact)[1])
                 if match(fact, "loose", "*"):
                     spanners_and_nuts.add(get_parts(fact)[1])
                 # (tightened facts shouldn't be in initial state for goal nuts, but check anyway)
                 if match(fact, "tightened", "*"):
                      spanners_and_nuts.add(get_parts(fact)[1])

             # Remove known spanners and nuts from potential men
             potential_men -= spanners_and_nuts

             if len(potential_men) == 1:
                 self.man_name = potential_men.pop()
             elif len(potential_men) > 1:
                  # Multiple candidates, pick the first one found in an 'at' fact among candidates
                  for fact in initial_state:
                       if match(fact, "at", "*", "*"):
                           obj, loc = get_parts(fact)[1:]
                           if obj in potential_men: # Pick one from the candidates
                                self.man_name = obj
                                break
             # If self.man_name is still None, the heuristic might fail later.
             # This indicates an unusual problem instance or domain definition.


        # 1. Identify all locations
        self.locations = set()
        # Locations from links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
        # Locations from initial object positions (man, spanners, nuts)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)[1:]
                 self.locations.add(loc)

        self.locations = sorted(list(self.locations)) # Consistent order
        self.location_to_idx = {loc: i for i, loc in enumerate(self.locations)}

        # 2. Build location graph
        self.graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Assuming links are bidirectional based on examples

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {loc: float('inf') for loc in self.locations}
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                for v in self.graph.get(u, []):
                    if self.dist[start_node][v] == float('inf'):
                        self.dist[start_node][v] = self.dist[start_node][u] + 1
                        queue.append(v)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance."""
        return self.dist.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Target Nuts (loose nuts that are goals)
        loose_target_nuts = {
            nut for nut in self.goal_nuts
            if f"(loose {nut})" in state
        }
        N_loose_target = len(loose_target_nuts)

        if N_loose_target == 0:
            return 0 # Goal reached for all nuts

        # Ensure man_name was identified in __init__
        if self.man_name is None:
             # Cannot compute heuristic without knowing the man object
             # This indicates an issue during initialization or an unusual problem.
             # Return infinity as the state is likely unsolvable from here.
             return float('inf')


        # 2. Find Man's Current Location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        # If man_loc is None, the state is invalid. Return infinity if work is needed.
        if man_loc is None:
             return float('inf') # Should not happen in valid solvable states


        # 3. Find Locations of Usable Spanners and Check if Man is Carrying One
        usable_spanner_locs = set()
        man_carrying_usable = False

        usable_spanners_names = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        for spanner_name in usable_spanners_names:
             # Check if this usable spanner is at a location
             for loc_fact in state:
                 if match(loc_fact, "at", spanner_name, "*"):
                     usable_spanner_locs.add(get_parts(loc_fact)[2])
                     break # Found location, move to next usable spanner name

        # Check if man is carrying any usable spanner
        for fact in state:
             if match(fact, "carrying", self.man_name, "*"):
                  spanner_carried = get_parts(fact)[2]
                  if spanner_carried in usable_spanners_names:
                       man_carrying_usable = True
                       break # Found a usable spanner being carried


        # 4. Calculate Required Pickups
        N_pickups_needed = max(0, N_loose_target - (1 if man_carrying_usable else 0))

        # 5. Base Action Cost (tighten + pickup actions)
        h = N_loose_target + N_pickups_needed

        # 6. Calculate Travel Cost
        required_locations = set()

        # Add locations of loose target nuts
        nut_locations = set()
        for nut in loose_target_nuts:
             # Find the location of this nut in the current state
             nut_loc = None
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       nut_loc = get_parts(fact)[2]
                       break
             if nut_loc:
                  nut_locations.add(nut_loc)
             # If nut_loc is None, the nut is not 'at' any location, which is weird.
             # Assume valid states where nuts are always at a location.

        required_locations.update(nut_locations)

        # Add locations of usable spanners if pickups are needed and spanners exist at locations
        if N_pickups_needed > 0 and len(usable_spanner_locs) > 0:
             required_locations.update(usable_spanner_locs)

        # Calculate cost to reach the closest required location from ManLoc
        cost_to_start_work = float('inf')
        if man_loc in self.locations and len(required_locations) > 0: # Ensure man_loc is a known location
             # Filter required_locations to only include known locations
             known_required_locations = [loc for loc in required_locations if loc in self.locations]
             if known_required_locations:
                 cost_to_start_work = min(self.get_distance(man_loc, loc) for loc in known_required_locations)

        # Add travel cost to heuristic
        if cost_to_start_work != float('inf'):
             h += cost_to_start_work
        else:
             # If work is needed (N_loose_target > 0) but no required location is reachable,
             # the state is likely unsolvable. Return infinity.
             if N_loose_target > 0:
                  return float('inf')
             # If N_loose_target is 0, h is already 0, which is correct.

        return h
