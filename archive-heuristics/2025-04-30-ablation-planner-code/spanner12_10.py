# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

from collections import deque
from fnmatch import fnmatch
import math # For math.inf

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph: # Handle nodes with no outgoing links
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the number of required 'tighten_nut' actions, the number of required
    'pickup_spanner' actions (if the man doesn't carry enough usable spanners),
    and an estimate of the travel cost to reach a relevant location (either a
    loose nut or a usable spanner).

    # Assumptions:
    - The man can carry multiple spanners.
    - Each 'tighten_nut' action consumes the usability of one spanner.
    - Nut locations are static (defined in the initial state).
    - Link predicates define bidirectional connections between locations.
    - There is only one man object, identifiable from 'carrying' facts in the initial state or as the sole non-nut/non-spanner locatable.
    - Locations are identifiable as arguments in 'at' and 'link' facts.

    # Heuristic Initialization
    - Identifies the man, nuts, spanners, and locations from the initial state and static facts.
    - Stores the static locations of nuts.
    - Builds a graph of locations based on 'link' facts (assuming bidirectionality).
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are currently 'loose'. If none, the heuristic is 0 (goal state).
    3. Count the number of usable spanners the man is currently carrying.
    4. Count the total number of usable spanners available in the environment (not carried) and their locations.
    5. Check if the total number of usable spanners (carried + environment) is less than the number of loose nuts. If so, the problem is unsolvable from this state, return infinity.
    6. Calculate the number of additional usable spanners the man needs to pick up from the environment: `needed_pickups = max(0, num_loose_nuts - num_carried_usable_spanners)`.
    7. The base heuristic cost is the sum of the required 'tighten_nut' actions (equal to the number of loose nuts) and the required 'pickup_spanner' actions (equal to `needed_pickups`).
    8. Estimate the travel cost: Find the set of 'relevant' locations. This set includes the locations of all loose nuts (which are static). If `needed_pickups > 0`, it also includes the locations of all usable spanners currently in the environment.
    9. The travel cost is estimated as the shortest distance from the man's current location to the closest location in the set of relevant locations.
    10. The total heuristic value is the sum of the base cost and the estimated travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and computing distances."""
        # Store goal facts for checking goal state
        self.goals = task.goals

        static_facts = task.static
        initial_state_facts = task.initial_state

        self.man = None
        self.nut_locations = {} # {nut_obj: location}
        self.all_spanners = set()
        self.all_nuts = set()
        locations = set()
        self.location_graph = {}

        # Pass 1: Identify objects and locations from initial state facts
        locatables_in_initial_at = set()
        for fact in initial_state_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                locatables_in_initial_at.add(obj)
                locations.add(loc)
            elif predicate == 'carrying':
                 man_obj, spanner_obj = parts[1], parts[2]
                 self.man = man_obj # Assuming one man and he is initially carrying something
                 self.all_spanners.add(spanner_obj)
            elif predicate == 'usable':
                 spanner_obj = parts[1]
                 self.all_spanners.add(spanner_obj)
            elif predicate in ['loose', 'tightened']:
                 nut_obj = parts[1]
                 self.all_nuts.add(nut_obj)

        # Fallback to identify man if not found via 'carrying'
        if self.man is None:
             # Assume the only locatable in the initial state that is not a nut or spanner is the man
             potential_man_objects = locatables_in_initial_at - self.all_nuts - self.all_spanners
             if len(potential_man_objects) == 1:
                  self.man = list(potential_man_objects)[0]
             # else: self.man remains None, handled in __call__


        # Pass 2: Get static nut locations from initial state facts
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1] in self.all_nuts:
                  self.nut_locations[parts[1]] = parts[2]

        # Pass 3: Build location graph from static facts (links)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                # Assuming links are bidirectional
                self.location_graph.setdefault(l2, []).append(l1)

        # Ensure all identified locations are in the graph keys, even if they have no links
        for loc in locations:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.location_graph:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # BFS result gives distances from start_loc to reachable nodes.
        # If loc2 is not in distances[loc1], it's unreachable.
        return self.distances.get(loc1, {}).get(loc2, math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Check if goal is reached
        if self.goals.issubset(state):
             return 0

        # 1. Find man's current location
        man_location = None
        # self.man should have been identified in __init__
        if self.man is None:
             # If man wasn't identified in __init__ (e.g., no carrying fact),
             # try to identify him now from the current state.
             locatables = set()
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at':
                       locatables.add(parts[1])
             # Assume the only locatable that is not a nut or spanner is the man
             potential_man_objects = locatables - self.all_nuts - self.all_spanners
             if len(potential_man_objects) == 1:
                  self.man = list(potential_man_objects)[0]
             else:
                  # Cannot identify the man uniquely or at all
                  return math.inf # Problematic state

        # Now find the man's location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man:
                man_location = parts[2]
                break

        if man_location is None:
             # Man should always be at a location in a valid state
             return math.inf


        # 2. Identify loose nuts
        loose_nuts = {parts[1] for fact in state if get_parts(fact)[0] == 'loose'}
        num_loose_nuts = len(loose_nuts)

        # 3. Identify usable spanners carried
        carried_spanners = {parts[2] for fact in state if get_parts(fact)[0] == 'carrying' and parts[1] == self.man}
        usable_spanners_in_state = {parts[1] for fact in state if get_parts(fact)[0] == 'usable'}
        carried_usable_spanners = carried_spanners.intersection(usable_spanners_in_state)
        num_carried_usable_spanners = len(carried_usable_spanners)

        # 4. Identify usable spanners in environment and their locations
        env_usable_spanners = usable_spanners_in_state - carried_spanners
        env_usable_spanner_locations = set()
        num_env_usable_spanners = 0
        for spanner in env_usable_spanners:
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and parts[1] == spanner:
                       env_usable_spanner_locations.add(parts[2])
                       num_env_usable_spanners += 1 # Count spanners, not locations
                       break # Found location for this spanner

        # 5. Check solvability based on spanners
        if num_loose_nuts > num_carried_usable_spanners + num_env_usable_spanners:
             return math.inf # Not enough usable spanners in total

        # 6. Calculate needed pickups
        needed_pickups = max(0, num_loose_nuts - num_carried_usable_spanners)

        # 7. Base cost (tighten + pickup actions)
        # Each loose nut needs a tighten action (cost 1)
        # Each needed spanner needs a pickup action (cost 1)
        base_cost = num_loose_nuts + needed_pickups

        # 8. Estimate travel cost
        loose_nut_locations = {self.nut_locations[nut] for nut in loose_nuts}

        relevant_locations = set(loose_nut_locations)
        if needed_pickups > 0:
             relevant_locations.update(env_usable_spanner_locations)

        # Find minimum distance from man's current location to any relevant location
        min_dist_to_relevant_location = math.inf
        # relevant_locations should be non-empty if num_loose_nuts > 0 and solvable.
        # If it's empty here, something is wrong or the problem is unsolvable.
        if not relevant_locations:
             return math.inf


        for loc in relevant_locations:
             dist = self.get_distance(man_location, loc)
             min_dist_to_relevant_location = min(min_dist_to_relevant_location, dist)

        # If min_dist_to_relevant_location is inf, it means no relevant location is reachable.
        # Since num_loose_nuts > 0 (otherwise we would have returned 0), this state is unsolvable.
        if min_dist_to_relevant_location == math.inf:
             return math.inf

        # 9. Total heuristic
        total_heuristic = base_cost + min_dist_to_relevant_location

        return total_heuristic
