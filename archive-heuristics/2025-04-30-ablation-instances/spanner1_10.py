from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node in distances:
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is still valid (should be if from queue)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the cost components for each loose goal nut: the tighten action itself,
    the travel cost for the man to reach the nut's location, and the cost to acquire
    a usable spanner if needed.

    # Assumptions
    - Nut locations are static and can be found in the initial state.
    - Links between locations are bidirectional.
    - The man object can be identified (e.g., by looking for the object involved in 'carrying' in the initial state, or as the unique locatable not identified as a nut or spanner).
    - Problem instances are solvable, implying enough usable spanners exist in the world to tighten all goal nuts.
    - All locations mentioned in 'link' or 'at' facts in the initial/static state are valid locations in the graph.

    # Heuristic Initialization
    - Build the location graph from 'link' facts.
    - Collect all unique locations mentioned in 'link' and 'at' facts in the initial/static state.
    - Compute all-pairs shortest path distances between all identified locations using BFS.
    - Identify the static location for each nut that appears in the goals or initial state.
    - Identify the name of the man object.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of nuts that are currently 'loose' and are required to be 'tightened' in the goal state. If this set is empty, the heuristic value is 0 (goal reached for nuts).
    2. Initialize the heuristic value `h` to 0.
    3. Add the number of loose goal nuts to `h`. This accounts for the 'tighten_nut' action needed for each.
    4. Find the man's current location in the state. If not found, return a large number (unreachable).
    5. Identify the distinct locations of the loose goal nuts. For each such location, add the shortest path distance from the man's current location to this nut location to `h`. This estimates the travel cost to reach the nuts. If any nut location is unreachable from the man's current location, return a large number.
    6. Count the number of usable spanners the man is currently carrying.
    7. Calculate the number of *additional* usable spanners the man needs to pick up to tighten all remaining loose goal nuts (this is the total number of loose goal nuts minus the number of usable spanners already carried, minimum 0).
    8. If additional spanners are needed:
        a. Add the number of additional spanners needed to `h`. This accounts for the 'pickup_spanner' action for each.
        b. Find all locations where usable spanners are currently available on the ground in the state.
        c. If no usable spanners are on the ground but needed, return a large number (unreachable).
        d. Calculate the shortest path distance from the man's current location to each of these spanner locations.
        e. Sort these distances and add the sum of the smallest 'number of additional spanners needed' distances to `h`. This estimates the travel cost to acquire the necessary spanners. If any spanner location is unreachable, return a large number.
    9. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        super().__init__(task)

        self.location_graph = {}
        self.locations = set()
        self.nut_locations = {} # Map nut object name to its static location

        # 1. Collect all locations from links and initial/static 'at' facts
        for fact in task.static:
            if match(fact, "link", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)
            elif match(fact, "at", "?obj", "?l"):
                 # Add location from static 'at' facts
                 self.locations.add(get_parts(fact)[2])

        for fact in task.initial_state:
             if match(fact, "at", "?obj", "?l"):
                 # Add location from initial state 'at' facts
                 self.locations.add(get_parts(fact)[2])

        # Ensure all collected locations are nodes in the graph dictionary
        for loc in self.locations:
            self.location_graph.setdefault(loc, set())

        # 2. Compute all-pairs shortest paths
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = bfs(self.location_graph, start_node)

        # 3. Identify static nut locations
        all_potential_nuts = set()
        for goal in task.goals:
            if match(goal, "tightened", "?n"):
                all_potential_nuts.add(get_parts(goal)[1])
        for fact in task.initial_state:
             if match(fact, "loose", "?n") or match(fact, "tightened", "?n"):
                 all_potential_nuts.add(get_parts(fact)[1])

        # Find the location of these potential nuts in the initial state (assumed static)
        for nut in all_potential_nuts:
             found_loc = False
             for fact in task.initial_state:
                 if match(fact, "at", nut, "?l"):
                     self.nut_locations[nut] = get_parts(fact)[2]
                     found_loc = True
                     break
             # If not found, it's a problematic instance, but we assume solvable.

        # 4. Identify the man object name
        self.man_name = None
        # Look for the object involved in 'carrying' in the initial state
        for fact in task.initial_state:
            if match(fact, "carrying", "?m", "?s"):
                self.man_name = get_parts(fact)[1]
                break
        # If not carrying anything initially, look for a locatable object that isn't a known nut or spanner
        # This is a fallback based on domain types (man, nut, spanner are locatable)
        if self.man_name is None:
             all_nuts = set(self.nut_locations.keys())
             # Find potential spanners from initial state (usable or carried)
             potential_spanners = set()
             for fact in task.initial_state:
                 if match(fact, "usable", "?s"):
                     potential_spanners.add(get_parts(fact)[1])
                 elif match(fact, "carrying", "?m", "?s"):
                     potential_spanners.add(get_parts(fact)[2])
             all_spanners = potential_spanners

             for fact in task.initial_state:
                 if match(fact, "at", "?obj", "?l"):
                     obj_name = get_parts(fact)[1]
                     # Assume the man is the locatable object not identified as a nut or spanner
                     if obj_name not in all_nuts and obj_name not in all_spanners:
                         self.man_name = obj_name
                         break # Assuming only one man
        # If man_name is still None, the instance is likely malformed.
        # We assume a man object exists and is identifiable.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts that are goal conditions
        goal_loose_nuts = {
            get_parts(fact)[1] for fact in state
            if match(fact, "loose", "?n") and f"(tightened {get_parts(fact)[1]})" in self.goals
        }

        if not goal_loose_nuts:
            return 0 # Goal reached for all nuts

        h = 0

        # 2. Cost for tighten actions
        h += len(goal_loose_nuts)

        # 3. Find man's current location
        man_loc = None
        if self.man_name: # Ensure man_name was identified in __init__
            for fact in state:
                if match(fact, "at", self.man_name, "?l"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None:
             # Man's location not found in state? Problematic state or man_name not identified.
             # Return large number.
             return 1000000

        # 4. Cost for travel to nuts
        nut_locations_to_visit = {self.nut_locations[n] for n in goal_loose_nuts}
        for loc in nut_locations_to_visit:
            # Check if distance is known (location might be unreachable or not in graph)
            if man_loc not in self.dist or loc not in self.dist[man_loc] or self.dist[man_loc][loc] == float('inf'):
                 # Indicates unreachable location. Return large number.
                 return 1000000
            h += self.dist[man_loc][loc]

        # 5. Cost for acquiring spanners
        usable_spanners_carried = {
            get_parts(fact)[1] for fact in state
            if match(fact, "carrying", self.man_name, "?s") and f"(usable {get_parts(fact)[1]})" in state
        }
        num_usable_spanners_carried = len(usable_spanners_carried)

        num_spanners_needed = len(goal_loose_nuts)
        num_spanners_to_pick_up = max(0, num_spanners_needed - num_usable_spanners_carried)

        if num_spanners_to_pick_up > 0:
            # Add cost for pickup actions
            h += num_spanners_to_pick_up

            # Find usable spanners on the ground and their locations
            usable_spanners_on_ground = {
                get_parts(fact)[1] for fact in state
                if match(fact, "usable", "?s") and match(fact, "at", get_parts(fact)[1], "?l")
            }
            spanner_locs_on_ground = {
                get_parts(fact)[2] for fact in state
                if match(fact, "at", "?s", "?l") and get_parts(fact)[1] in usable_spanners_on_ground
            }

            if not spanner_locs_on_ground:
                 # Spanners are needed but none are available on the ground.
                 # If not enough are carried, this is unsolvable.
                 # Return large number.
                 if num_usable_spanners_carried < num_spanners_needed:
                      return 1000000

            # Calculate distances from ManLoc to all SpannerLocsOnGround
            distances_to_spanners = []
            for loc in spanner_locs_on_ground:
                 if man_loc not in self.dist or loc not in self.dist[man_loc] or self.dist[man_loc][loc] == float('inf'):
                      # Should not happen if locations were collected correctly, but safety check
                      return 1000000 # Indicate likely unreachable
                 distances_to_spanners.append(self.dist[man_loc][loc])

            distances_to_spanners.sort()

            # Add the sum of the smallest NumSpannersToPickUp distances
            num_locations_to_consider = min(num_spanners_to_pick_up, len(distances_to_spanners))
            h += sum(distances_to_spanners[:num_locations_to_consider])

        return h
