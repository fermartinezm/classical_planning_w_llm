from fnmatch import fnmatch
from collections import deque # For BFS
# Assuming heuristic_base.py is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions (outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_nuts_in_goal(goals):
    """Extract the names of nuts that must be tightened in the goal state."""
    goal_nuts = set()
    for goal in goals:
        parts = get_parts(goal)
        if parts and parts[0] == 'tightened':
            goal_nuts.add(parts[1])
    return goal_nuts

def get_nut_location(state, nut_name):
    """Find the current location of a specific nut in the state."""
    for fact in state:
        if match(fact, 'at', nut_name, '*'):
            return get_parts(fact)[2]
    return None # Should not happen if nut exists and is 'at' a location

def usable_spanners_on_ground_with_locs(state):
    """Find usable spanners currently on the ground and their locations."""
    spanners = []
    for fact in state:
        parts = get_parts(fact)
        if parts and parts[0] == 'at' and parts[1].startswith('spanner'):
            spanner_name = parts[1]
            location = parts[2]
            if f'(usable {spanner_name})' in state:
                spanners.append((spanner_name, location))
    return spanners


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all required nuts.
    It considers the cost of tightening actions, picking up necessary spanners,
    and the travel cost for the man to reach the locations of nuts and spanners.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Tightening a nut requires the man to be at the nut's location, carrying a usable spanner.
    - Tightening consumes one usable spanner, making it unusable.
    - The man can carry multiple spanners.
    - Spanners and nuts do not move unless carried by the man (spanners) or are the target of tighten (nuts).
    - Links between locations are bidirectional.
    - All locations involved in the problem (mentioned in links, initial state, or goals) are considered.

    # Heuristic Initialization
    - Identify the set of nuts that must be tightened in the goal state.
    - Identify the name of the man object.
    - Build the location graph based on `link` facts from static information and locations from initial/goal states.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of nuts that are currently `loose` but are required to be `tightened` in the goal state (`GoalLooseNuts`).
    2. If `GoalLooseNuts` is empty, the goal is reached for these nuts, return 0.
    3. Calculate the base cost for tightening: Add `|GoalLooseNuts|` to the heuristic (each tighten action costs 1).
    4. Identify the man's current location.
    5. Count the number of `usable` spanners the man is currently `carrying`.
    6. Calculate how many additional usable spanners the man needs to pick up from the ground: `pickups_needed = max(0, |GoalLooseNuts| - num_carrying_usable)`. Limit this by the actual number of usable spanners available on the ground. Add `pickups_needed` to the heuristic (each pickup action costs 1).
    7. Identify the locations of all nuts in `GoalLooseNuts` (`NutLocations`).
    8. Identify the locations of all `usable` spanners currently on the ground (`GroundUsableSpannerLocations`).
    9. Select `pickups_needed` locations from `GroundUsableSpannerLocations` that are closest to the man's current location.
    10. Create a set of required locations (`RequiredLocationsSet`) by combining `NutLocations` and the selected spanner pickup locations. Remove the man's current location if it's in the set (no travel needed to start).
    11. Calculate the travel cost: Sum the shortest path distances from the man's current location to each distinct location in `RequiredLocationsSet`. Add this sum to the heuristic.
    12. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and precomputing distances."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Needed to identify man

        # Identify the set of nuts that must be tightened in the goal state
        self.goal_nuts = get_nuts_in_goal(self.goals)

        # Identify the man object name
        self.man_name = None
        # Look for an object being carried (only man can carry in this domain)
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'carrying':
                 self.man_name = parts[1]
                 break
        # Fallback: Look for a locatable object in initial state that isn't a spanner or nut
        if self.man_name is None:
             for fact in self.initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'at':
                     obj_name = parts[1]
                     # Assuming objects starting with 'spanner' or 'nut' are those types
                     if not obj_name.startswith('spanner') and not obj_name.startswith('nut'):
                          self.man_name = obj_name
                          break
        assert self.man_name is not None, "Could not identify the man object from initial state."


        # Build the location graph and precompute distances
        self.location_graph = {}
        all_locations = set()

        # Add locations from link facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Assuming links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Add any locations mentioned in initial state or goals that might not be linked
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at':
                 all_locations.add(parts[2])
        for goal in self.goals:
             parts = get_parts(goal)
             # Goal facts can be '(tightened nut)' or '(at obj loc)' etc.
             if parts and parts[0] == 'at':
                 all_locations.add(parts[2])

        # Ensure all locations in the graph keys are in all_locations set
        for loc in self.location_graph.keys():
             all_locations.add(loc)

        # Handle case with no locations (e.g., empty domain/problem)
        if not all_locations:
             self.distances = {}
        else:
            self.distances = {}
            for start_loc in all_locations:
                self.distances[start_loc] = self._bfs(start_loc, all_locations)

    def _bfs(self, start_node, all_nodes):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in all_nodes}
        if start_node in distances: # Ensure start_node is one of the known locations
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # If current_node is not in graph (isolated location), it has no neighbors
                if current_node not in self.location_graph:
                     continue

                for neighbor in self.location_graph[current_node]:
                    if neighbor in distances and distances[neighbor] == float('inf'): # Ensure neighbor is a known location
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts that are goals
        goal_loose_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}

        # 2. If GoalLooseNuts is empty, return 0.
        if not goal_loose_nuts:
            return 0

        # 3. Calculate base cost for tightening
        tighten_cost = len(goal_loose_nuts)
        total_h = tighten_cost

        # 4. Identify the man's current location.
        man_location = None
        for fact in state:
             if match(fact, 'at', self.man_name, '*'):
                  man_location = get_parts(fact)[2]
                  break
        if man_location is None:
             # Man must be somewhere if state is valid, but handle defensively
             # If man location is unknown, cannot proceed.
             return float('inf')


        # 5. Count usable spanners carried by the man
        num_carrying_usable = sum(1 for fact in state if match(fact, 'carrying', self.man_name, '*') and f'(usable {get_parts(fact)[2]})' in state)

        # 6. Calculate pickups needed and add cost
        usable_spanners_on_ground = usable_spanners_on_ground_with_locs(state)
        num_ground_usable = len(usable_spanners_on_ground)

        pickups_needed = max(0, len(goal_loose_nuts) - num_carrying_usable)
        pickups_needed = min(pickups_needed, num_ground_usable) # Cannot pick up more than exist

        pickup_cost = pickups_needed
        total_h += pickup_cost

        # 7. Identify locations of loose goal nuts
        nut_locations = set()
        for nut in goal_loose_nuts:
             loc = get_nut_location(state, nut)
             if loc: # Ensure location was found
                 nut_locations.add(loc)
             else:
                 # A goal nut is loose but has no location? Problematic state.
                 return float('inf')


        # 8 & 9. Identify and select spanner pickup locations
        # Sort usable ground spanners by distance from man
        # Use get() with default inf for robustness if man_location or spanner_loc is not in distances map
        spanners_on_ground_list_sorted = sorted(usable_spanners_on_ground,
                                                key=lambda item: self.distances.get(man_location, {}).get(item[1], float('inf')))

        spanner_pickup_locations_to_visit_set = {loc for s, loc in spanners_on_ground_list_sorted[:pickups_needed]}

        # 10. Create set of required locations for travel
        required_locations_set = set(nut_locations)
        required_locations_set.update(spanner_pickup_locations_to_visit_set)
        required_locations_set.discard(man_location) # No travel needed if already at a required location

        # 11. Calculate travel cost
        travel_cost = 0
        # Sum distances from man's current location to each distinct required location
        for loc in required_locations_set:
             # Use get() with default inf for robustness
             dist = self.distances.get(man_location, {}).get(loc, float('inf'))
             if dist == float('inf'):
                  # A required location is unreachable from man's current location
                  return float('inf')
             travel_cost += dist

        total_h += travel_cost

        # 12. Return the total heuristic value.
        return total_h
