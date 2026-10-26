# The base class Heuristic is assumed to be available in the environment.
# It typically defines the structure with __init__(self, task) and __call__(self, node).
# from heuristics.heuristic_base import Heuristic

import math
from collections import deque
from fnmatch import fnmatch

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is defined elsewhere and imported
# For standalone testing, you might need a mock Heuristic class:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass
#
# Assuming task object has .goals, .static, .init attributes
# Assuming node object has .state attribute (a frozenset of fact strings)


class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the estimated costs for tightening actions, spanner pickup actions,
    and the man's necessary travel. Travel cost is estimated based on reaching
    the location(s) of loose nuts and usable spanners.

    # Assumptions:
    - The goal is to tighten a specific set of nuts.
    - Spanners are consumed after one use (`tighten_nut` makes a usable spanner not usable).
    - The heuristic assumes, for travel cost estimation, that all loose nuts
      that need tightening are located at a single location (or it picks one nut's location as the target group location).
      This simplifies travel calculation but might be inaccurate if nuts are spread out.
    - Links between locations are bidirectional.
    - There is only one man object.

    # Heuristic Initialization
    - Extracts all locations from the initial state and static facts (`link`).
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the goal conditions (specifically, the set of nuts that need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all loose nuts that need to be tightened (i.e., are loose in the state and are in the set of goal tightened nuts).
    2. If there are no loose nuts to tighten, the heuristic is 0 (goal state or sub-goal met).
    3. Identify the man's current location.
    4. Identify usable spanners the man is carrying.
    5. Identify usable spanners on the ground and their locations.
    6. Identify the locations of all loose nuts. Determine the "nut group" location (assuming all loose nuts are at the same place).
    7. Check if there are enough usable spanners (carried + on ground) to tighten all loose nuts. If not, the problem is likely unsolvable from this state with available spanners, return infinity.
    8. Initialize heuristic `h` with the number of loose nuts (cost of `tighten_nut` actions, 1 per nut).
    9. Calculate the number of spanners the man needs to pick up from the ground (`spanners_to_pickup = max(0, NumLooseNuts - NumUsableCarried)`).
    10. Add the cost of `pickup_spanner` actions (`spanners_to_pickup`, 1 per pickup) to `h`.
    11. Estimate travel cost:
        - If the man is already carrying usable spanners: Calculate travel from his current location to the nut group location.
        - If the man is not carrying usable spanners: Calculate travel to go to the closest usable spanner location on the ground, pick it up, and then travel to the nut group location.
        - If the man needs to pick up more spanners after the first one (i.e., `spanners_to_pickup > 1` if he started with none, or `spanners_to_pickup > 0` if he started with some): Estimate the cost of round trips from the nut group location to the closest available usable spanner location on the ground and back, for each additional spanner needed.
    12. Add the estimated total travel cost to `h`.
    13. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions.
        - All locations and computing shortest paths between them.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_facts = task.init # Initial state facts

        # Extract all locations from static links and initial 'at' predicates
        locations = set()
        adj = {} # Adjacency list for graph

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                adj.setdefault(loc1, []).append(loc2)
                adj.setdefault(loc2, []).append(loc1) # Links are bidirectional

        for fact in initial_facts:
             parts = get_parts(fact)
             if parts[0] == "at":
                 # The second argument of 'at' is always a location
                 locations.add(parts[2])

        self.locations = list(locations)
        self.dist = {l: {l2: math.inf for l2 in self.locations} for l in self.locations}

        # Compute all-pairs shortest paths using BFS from each location
        for start_loc in self.locations:
            self.dist[start_loc][start_loc] = 0
            queue = deque([start_loc])
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if self.dist[start_loc][v] == math.inf:
                        self.dist[start_loc][v] = self.dist[start_loc][u] + 1
                        queue.append(v)

        # Store goal locations for nuts
        self.goal_tightened_nuts = {
            args[0] for goal in self.goals
            if get_parts(goal)[0] == "tightened"
            for args in [get_parts(goal)[1:]]
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify dynamic facts
        man_loc = None
        carried_spanners = set()
        usable_spanners_on_ground = set()
        spanner_locations = {} # Map spanner to location if on ground
        nut_locations = {} # Map nut to location
        loose_nuts_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Assuming only one man and it's the object not a spanner or nut
                # A more robust way would be to get man object from task.objects
                if match(fact, "at", "*", "*") and not obj.startswith("spanner") and not obj.startswith("nut"):
                     man_loc = loc
                elif match(fact, "at", "*", "*") and obj.startswith("spanner"):
                     spanner_locations[obj] = loc
                elif match(fact, "at", "*", "*") and obj.startswith("nut"):
                     nut_locations[obj] = loc
            elif parts[0] == "carrying":
                 man, spanner = parts[1], parts[2]
                 carried_spanners.add(spanner)
            elif parts[0] == "usable":
                 spanner = parts[1]
                 # Check if usable spanner is on the ground (not carried)
                 if spanner in spanner_locations:
                     usable_spanners_on_ground.add(spanner)
            elif parts[0] == "loose":
                 nut = parts[1]
                 loose_nuts_in_state.add(nut)

        # Identify loose nuts that are goals
        loose_nuts_to_tighten = {
            nut for nut in loose_nuts_in_state
            if nut in self.goal_tightened_nuts
        }

        num_loose_nuts = len(loose_nuts_to_tighten)

        # If no loose nuts need tightening, we are at a goal state (or sub-goal)
        if num_loose_nuts == 0:
            return 0

        num_usable_carried = len(carried_spanners)
        num_usable_on_ground = len(usable_spanners_on_ground)

        # Check if enough usable spanners exist in total
        if num_usable_carried + num_usable_on_ground < num_loose_nuts:
            return math.inf # Problem likely unsolvable from here

        h = 0

        # 1. Cost for tighten_nut actions (1 per nut)
        h += num_loose_nuts

        # 2. Cost for pickup_spanner actions (1 per spanner picked up)
        spanners_to_pickup = max(0, num_loose_nuts - num_usable_carried)
        h += spanners_to_pickup

        # 3. Estimate travel cost

        travel_cost = 0
        current_loc = man_loc

        # Find the location of the group of loose nuts (assuming they are grouped)
        # If nuts are spread out, this heuristic is less accurate.
        nut_group_loc = None
        if num_loose_nuts > 0:
             # Pick the location of the first loose nut found
             first_loose_nut = next(iter(loose_nuts_to_tighten))
             nut_group_loc = nut_locations.get(first_loose_nut)
             if nut_group_loc is None:
                 # This shouldn't happen if state is consistent, but handle defensively
                 return math.inf # Loose nut has no location?

        # Locations of usable spanners currently on the ground
        usable_spanner_locations_on_ground = {
            spanner_locations[s] for s in usable_spanners_on_ground
        }

        if num_usable_carried > 0:
            # Man has the first spanner, travel to the nut group
            if nut_group_loc in self.dist[current_loc]:
                 travel_cost += self.dist[current_loc][nut_group_loc]
            else:
                 # Cannot reach nut group from current location
                 return math.inf

            # Need more spanners? Estimate round trips from nut group
            if spanners_to_pickup > 0:
                min_round_trip = math.inf
                for loc_s in usable_spanner_locations_on_ground:
                    if loc_s in self.dist[nut_group_loc]:
                         min_round_trip = min(min_round_trip, self.dist[nut_group_loc][loc_s] + self.dist[loc_s][nut_group_loc])

                if min_round_trip != math.inf:
                    travel_cost += spanners_to_pickup * min_round_trip
                elif spanners_to_pickup > 0:
                     # Needed spanners on ground are unreachable from nut group
                     return math.inf

        else: # num_usable_carried == 0
            # Man needs the first spanner. Go get closest, bring to nut group.
            min_first_trip = math.inf

            if not usable_spanner_locations_on_ground:
                 # Should be caught by total usable check, but double check
                 return math.inf

            for loc_s in usable_spanner_locations_on_ground:
                 if loc_s in self.dist[current_loc] and nut_group_loc in self.dist[loc_s]:
                      trip_cost = self.dist[current_loc][loc_s] + self.dist[loc_s][nut_group_loc]
                      min_first_trip = min(min_first_trip, trip_cost)

            if min_first_trip != math.inf:
                travel_cost += min_first_trip
            elif num_loose_nuts > 0:
                 # Cannot reach any usable spanner on ground and bring to nut group
                 return math.inf

            # Need more spanners after the first? Estimate round trips from nut group
            if spanners_to_pickup - 1 > 0:
                min_round_trip = math.inf
                for loc_s in usable_spanner_locations_on_ground:
                     if loc_s in self.dist[nut_group_loc]:
                          min_round_trip = min(min_round_trip, self.dist[nut_group_loc][loc_s] + self.dist[loc_s][nut_group_loc])

                if min_round_trip != math.inf:
                    travel_cost += (spanners_to_pickup - 1) * min_round_trip
                elif spanners_to_pickup - 1 > 0:
                     # Needed spanners on ground are unreachable from nut group
                     return math.inf

        h += travel_cost

        return h
