from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It considers the costs of tightening, picking up spanners,
    and traveling to necessary locations.

    # Heuristic Initialization
    - Precomputes distances between all locations using BFS on the link graph.
    - Identifies goal nuts and their static locations.
    - Identifies the man object and all spanner objects based on initial state predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify which goal nuts are still loose.
    3. If no goal nuts are loose, the heuristic is 0.
    4. Count usable spanners the man is currently carrying.
    5. Count usable spanners available at locations.
    6. Determine how many additional spanners the man needs to pick up
       (number of loose goal nuts minus usable spanners carried, minimum 0).
    7. If there aren't enough usable spanners in the world (carried + at locations)
       to tighten all loose goal nuts, the state is unsolvable (return infinity).
    8. The base heuristic cost is the number of loose goal nuts (for tighten actions)
       plus the number of spanners that need to be picked up (for pickup actions).
    9. Identify the set of locations the man must visit:
       - The location of each loose goal nut.
       - The locations of the required number of usable spanners that need to be picked up.
         (Select spanners at locations closest to the man's current location).
    10. Estimate travel cost:
        - Distance from the man's current location to the closest required visit location.
        - Plus, a simple estimate for visiting the remaining required locations
          (e.g., number of additional unique locations - 1).
    11. Sum the base cost and the estimated travel cost.
    12. If any required location is unreachable from the man's current location,
        the state is unsolvable (return infinity).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and initial state information to precompute distances and identify key objects/locations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Find all locations from links and initial/goal 'at' facts
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)
        # Goal locations are not typically specified for spanner domain goals like (tightened nut)
        # but include this loop just in case 'at' facts appear in goals in other variations.
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 locations.add(loc)

        self.locations = list(locations)

        # Build adjacency list for location graph
        self.adj = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adj[l1].add(l2)
                self.adj[l2].add(l1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Identify goal nuts (those that need to be tightened)
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)

        # Find static locations of objects (nuts, spanners, man) from initial state
        self.object_initial_locations = {}
        self.all_spanners = set()
        self.man = None

        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)(fact)
                 self.object_initial_locations[obj] = loc

        # Identify man and spanners based on predicates in initial state
        initial_spanner_candidates = set()
        initial_man_candidate = None

        for fact in initial_state:
             if match(fact, "carrying", "*", "*"):
                 _, m, s = get_parts(fact)
                 initial_man_candidate = m # Assume the one carrying is the man
                 initial_spanner_candidates.add(s)
             elif match(fact, "usable", "*"):
                 _, s = get_parts(fact)
                 initial_spanner_candidates.add(s)

        self.all_spanners = initial_spanner_candidates
        self.man = initial_man_candidate

        # If man wasn't found by carrying, find the one object in initial 'at' facts
        # that is not a spanner or goal nut. This is a fallback assumption.
        if self.man is None:
             for fact in initial_state:
                 if match(fact, "at", "*", "*"):
                     _, obj, loc = get_parts(fact)
                     if obj not in self.all_spanners and obj not in self.goal_nuts:
                          self.man = obj
                          break

        # Ensure all goal nuts have an initial location recorded (they should be in initial state 'at' facts)
        # If not, get_location will return None later, leading to infinity heuristic.


    def _bfs(self, start_node):
        """Compute distances from start_node to all reachable nodes."""
        distances = {loc: math.inf for loc in self.locations}
        if start_node not in distances:
             # Start node is not in the list of known locations (e.g., malformed problem)
             return {loc: math.inf for loc in self.locations} # All locations are unreachable from here

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            if current_loc not in self.adj: continue # Should not happen if locations are from links/initial/goal at

            for neighbor in self.adj.get(current_loc, []):
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
        return distances

    def get_location(self, state, obj):
        """Find the current location of an object in the state."""
        # Check if object is at a location
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
        # Check if object is carried by the man (only spanners can be carried)
        if obj in self.all_spanners and self.man is not None:
             for fact in state:
                 if match(fact, "carrying", self.man, obj):
                     # If carried, its location is the man's location
                     return self.get_location(state, self.man) # Recursive call to find man's location
        return None # Object location not found

    def __call__(self, node):
        state = node.state

        # 1. Find man's current location
        man_loc = self.get_location(state, self.man)
        # Check if man_loc is valid and reachable from itself (basic graph check)
        if man_loc is None or man_loc not in self.dist or self.dist[man_loc] is None:
             # Man must always be at a known, reachable location in a solvable state
             return math.inf # Should not happen in valid states

        # 2. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f"(tightened {n})" not in state}
        num_loose_goals = len(loose_goal_nuts)

        # 3. If all goals are met, heuristic is 0
        if num_loose_goals == 0:
            return 0

        # 4. Find usable spanners currently carried by the man
        carried_usable_spanners = {s for s in self.all_spanners
                                   if f"(carrying {self.man} {s})" in state and f"(usable {s})" in state}
        num_carried_usable = len(carried_usable_spanners)

        # 5. Find usable spanners available at locations (not carried)
        available_usable_at_loc = {s for s in self.all_spanners
                                   if f"(at {s} {self.get_location(state, s)})" in state and f"(usable {s})" in state}

        # 6. Calculate needed spanners to pick up
        num_spanners_to_get = max(0, num_loose_goals - num_carried_usable)

        # 7. Check solvability based on available usable spanners
        if num_spanners_to_get > len(available_usable_at_loc):
             # Not enough usable spanners exist at locations to tighten all remaining nuts
             return math.inf # Unsolvable

        # 8. Base heuristic: tighten actions + pickup actions
        h = num_loose_goals # Cost for tighten_nut actions
        h += num_spanners_to_get # Cost for pickup_spanner actions

        # 9. Identify required locations to visit
        required_visits = set()

        # Add locations of loose goal nuts
        for nut in loose_goal_nuts:
             nut_loc = self.get_location(state, nut)
             # Check if nut_loc is valid and reachable from man_loc
             if nut_loc is None or nut_loc not in self.dist or self.dist[man_loc][nut_loc] == math.inf:
                  return math.inf # Unsolvable
             required_visits.add(nut_loc)

        # Add locations of spanners to pick up
        spanner_pickup_locs = set()
        if num_spanners_to_get > 0:
            # Find the num_spanners_to_get usable spanners at locations closest to the man
            usable_at_loc_with_dist = []
            for spanner in available_usable_at_loc:
                 spanner_loc = self.get_location(state, spanner)
                 # Only consider spanners at known, reachable locations
                 if spanner_loc is not None and spanner_loc in self.dist[man_loc] and self.dist[man_loc][spanner_loc] != math.inf:
                      usable_at_loc_with_dist.append((self.dist[man_loc][spanner_loc], spanner_loc))
                 # else: spanner is usable but its location is unknown or unreachable - ignore for pickup

            # Sort by distance and pick unique locations
            usable_at_loc_with_dist.sort()
            picked_count = 0
            for dist, loc in usable_at_loc_with_dist:
                 if loc not in spanner_pickup_locs:
                      spanner_pickup_locs.add(loc)
                      picked_count += 1
                      if picked_count == num_spanners_to_get:
                           break

            # If we couldn't find enough reachable spanner locations to pick up
            if picked_count < num_spanners_to_get:
                 return math.inf # Unsolvable

            required_visits.update(spanner_pickup_locs)

        # 10. Calculate travel cost
        if not required_visits: # Should not happen if num_loose_goals > 0
             travel_cost = 0
        else:
             # Find the closest required location from man's current location
             min_dist_to_target = math.inf
             # No need to find closest_target_loc object, just the distance
             for target_loc in required_visits:
                  # Reachability check already done when adding to required_visits or spanner_pickup_locs
                  dist = self.dist[man_loc][target_loc]
                  if dist < min_dist_to_target:
                       min_dist_to_target = dist

             travel_cost = min_dist_to_target

             # Add cost for visiting remaining required locations
             # Simple estimate: 1 step per additional unique location
             travel_cost += max(0, len(required_visits) - 1)

        h += travel_cost

        return h
