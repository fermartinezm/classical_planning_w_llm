from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import math

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all required nuts. It sums the
    number of tighten actions, the number of spanner pickup actions needed, and
    an estimate of the travel cost. The travel cost is estimated using the
    Minimum Spanning Tree (MST) on the set of locations that must be visited
    (man's current location, locations of loose goal nuts, and locations of
    usable spanners that need to be picked up).

    # Assumptions
    - Each tighten action consumes one usable spanner.
    - The man can carry multiple spanners.
    - Spanners do not become usable again.
    - The location graph defined by `link` facts and initial `at` locations is static.
    - The heuristic can identify the man object.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task goals.
    - Builds the location graph from `link` facts and initial `at` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify all goal nuts that are currently loose in the state. If there are none, the goal is achieved for these nuts, and the heuristic is 0.
    2.  Identify the man object. This is done by looking for an object involved in a `carrying` predicate or an `at` predicate that is not a known nut or usable spanner.
    3.  Determine the man's current location from the state.
    4.  Count the number of usable spanners the man is currently carrying.
    5.  Count the number of usable spanners available on the ground and record their locations.
    6.  Check if the total number of available usable spanners (carried + on ground) is less than the number of loose goal nuts. If so, the problem is unsolvable with the current resources, and the heuristic returns infinity.
    7.  The base heuristic cost is the sum of:
        -   The number of loose goal nuts (each requires one `tighten_nut` action).
        -   The number of spanners that need to be picked up from the ground. This is calculated as `max(0, num_loose_goal_nuts - num_carried_usable_spanners)`. Each pickup requires one `pickup_spanner` action.
    8.  Identify the set of locations the man must visit. This set includes:
        -   The locations of all loose goal nuts.
        -   The locations of the `num_spanners_to_pickup` closest usable spanners found on the ground (to acquire the necessary spanners).
    9.  Compute the Minimum Spanning Tree (MST) on the set of required visit locations, including the man's current location. The edge weights for the MST are the precomputed shortest path distances between locations.
    10. Add the weight of the MST to the heuristic cost. This provides a lower bound estimate of the minimum travel cost required to connect all necessary points in the plan.
    11. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

        # Build location graph and collect all locations
        self.all_locations = set()
        self.location_graph = defaultdict(list)

        # Add locations from link facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add locations from initial state 'at' facts
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                  loc = get_parts(fact)[2]
                  self.all_locations.add(loc)

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Compute shortest path distances from start_node to all reachable nodes."""
        distances = {loc: math.inf for loc in self.all_locations}
        if start_node in distances: # Ensure start_node is in the collected locations
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                # Ensure u is a valid key in the graph
                if u in self.location_graph:
                    for v in self.location_graph[u]:
                        if distances[v] == math.inf:
                            distances[v] = distances[u] + 1
                            queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {
            n for n in self.goal_nuts if f"(loose {n})" in state
        }
        num_loose_goal_nuts = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if num_loose_goal_nuts == 0:
            return 0

        # Find the man object name
        man_name = None
        # Try finding the object involved in 'carrying'
        for fact in state:
             if match(fact, "carrying", "*", "*"):
                  man_name = get_parts(fact)[1]
                  break

        # If not found via 'carrying', try finding an object 'at' a location
        # that is not a known nut or usable spanner.
        if man_name is None:
             nuts_in_state = {get_parts(f)[1] for f in state if match(f, "at", "*", "*") and get_parts(f)[1] in self.goal_nuts}
             usable_spanners_in_state = {get_parts(f)[1] for f in state if match(f, "at", "*", "*") and f"(usable {get_parts(f)[1]})" in state}
             for fact in state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     if obj not in nuts_in_state and obj not in usable_spanners_in_state:
                          man_name = obj
                          break # Assuming the first found is the man

        if man_name is None:
             # Could not identify the man object. Problem state is likely invalid.
             return math.inf

        # 2. Determine man's location
        man_loc = None
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man object identified but has no location? Invalid state.
             return math.inf

        # 3. Count carried usable spanners
        carried_usable_spanners = {
            s for s in self._get_carried_spanners(state, man_name) if f"(usable {s})" in state
        }
        num_carried_usable = len(carried_usable_spanners)

        # 4. Count usable spanners on ground and their locations
        usable_spanners_on_ground = {} # {spanner_name: location}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Check if it's a usable spanner
                if f"(usable {obj})" in state:
                     # Assume anything 'usable' is a spanner for this domain
                     usable_spanners_on_ground[obj] = loc

        num_usable_on_ground = len(usable_spanners_on_ground)

        # 5. Check solvability
        if num_carried_usable + num_usable_on_ground < num_loose_goal_nuts:
            return math.inf # Problem is unsolvable with available spanners

        # 6. Base cost: actions
        h = num_loose_goal_nuts # tighten actions
        spanners_to_pickup = max(0, num_loose_goal_nuts - num_carried_usable)
        h += spanners_to_pickup # pickup actions

        # 7. Identify required visit locations
        nut_locations = {
            loc for n in loose_goal_nuts for fact in state if match(fact, "at", n, loc)
        }

        spanner_pickup_locations_needed = set()
        if spanners_to_pickup > 0:
            # Get locations of usable spanners on the ground
            spanner_ground_locs = list(set(usable_spanners_on_ground.values())) # Unique locations

            # Ensure man_loc has distances computed
            if man_loc not in self.dist:
                 # This indicates an issue with parsing or state representation
                 return math.inf # Indicate unsolvable

            # Sort unique spanner locations by distance from man_loc
            # Filter out locations that are not in the distance map from man_loc (unreachable)
            reachable_spanner_locs = [
                loc for loc in spanner_ground_locs
                if loc in self.dist[man_loc] and self.dist[man_loc][loc] != math.inf
            ]
            sorted_spanner_locs = sorted(reachable_spanner_locs,
                                         key=lambda loc: self.dist[man_loc][loc])

            # Add the locations of the `spanners_to_pickup` closest usable spanners
            # Ensure we don't ask for more locations than available unique reachable locations
            num_locs_to_add = min(spanners_to_pickup, len(sorted_spanner_locs))
            spanner_pickup_locations_needed.update(sorted_spanner_locs[:num_locs_to_add])

        required_visit_locations = set(nut_locations)
        required_visit_locations.update(spanner_pickup_locations_needed)

        # 8. Compute MST on {man_loc} U required_visit_locations
        nodes_for_mst = list(required_visit_locations)
        if man_loc not in nodes_for_mst: # Add man_loc if not already a target location
             nodes_for_mst.append(man_loc)

        mst_weight = 0
        if len(nodes_for_mst) > 1:
            # Prim's algorithm
            # Start MST from man_loc if it's in the list, otherwise from the first node
            start_node_mst = man_loc if man_loc in nodes_for_mst else nodes_for_mst[0]
            visited = {start_node_mst}
            unvisited = set(nodes_for_mst) - visited

            while unvisited:
                min_edge_weight = math.inf
                next_node = None

                for v in visited:
                    # Ensure v is a valid key in self.dist
                    if v not in self.dist:
                         return math.inf # Should not happen if all_locations is correct

                    for u in unvisited:
                        # Ensure u is reachable from v
                        if u in self.dist[v] and self.dist[v][u] != math.inf:
                             weight = self.dist[v][u]
                             if weight < min_edge_weight:
                                 min_edge_weight = weight
                                 next_node = u

                if next_node is not None:
                    mst_weight += min_edge_weight
                    visited.add(next_node)
                    unvisited.remove(next_node)
                else:
                    # Graph is disconnected, required locations are unreachable from visited set
                    return math.inf # Indicate unsolvable

        # 9. Add MST weight to heuristic
        h += mst_weight

        return h

    def _get_carried_spanners(self, state, man_name):
        """Helper to find spanners the given man is carrying."""
        carried = set()
        if man_name:
            for fact in state:
                if match(fact, "carrying", man_name, "*"):
                    carried.add(get_parts(fact)[2])
        return carried
