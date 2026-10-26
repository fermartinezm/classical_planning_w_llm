from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_paths(start_node, graph):
    """
    Computes shortest path distances from a start_node in a graph.
    Graph is represented as an adjacency dictionary {node: [neighbor1, ...]}
    Returns a dictionary {node: distance}
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is in the graph keys
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Handle nodes with no links
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    For each loose goal nut, it estimates the cost of a sequence:
    (walk to usable spanner) + (pickup spanner) + (walk to nut) + (tighten nut).
    If the man is already carrying a usable spanner, the first two steps are skipped.
    The total heuristic is the sum of these estimated costs for all loose goal nuts.
    It uses precomputed shortest path distances between locations.

    # Assumptions
    - Each loose goal nut requires one tighten action and one usable spanner.
    - A spanner is consumed (becomes unusable) after one tighten action.
    - The man can only carry one spanner at a time. (This is partially ignored in the simple sum, but the cost calculation for a single nut trip respects it).
    - The cost of getting a spanner and getting to the nut location can be estimated independently for each nut and summed. (This is the main non-admissible simplification).
    - All locations are connected (or required locations are connected). Unreachable locations result in a large heuristic value.
    - The object of type 'man' can be identified heuristically (e.g., by looking for 'carrying' facts or eliminating spanners/nuts based on initial state facts).

    # Heuristic Initialization
    - Identify all locations from the static `link` facts, initial state, and goals.
    - Build an adjacency list graph representation from `link` facts.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the set of nuts that are part of the goal.
    - Heuristically identify spanner and nut objects from initial state facts to help identify the man object.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's name and current location in the state.
    2. Identify all usable spanners and their current locations (on the ground or carried).
    3. Identify all loose nuts and their current locations.
    4. Filter the loose nuts to find those that are goal conditions (need tightening).
    5. Check if the number of loose goal nuts exceeds the total number of usable spanners available (carried or on the ground). If so, return a large value indicating likely unsolvability.
    6. If there are no loose goal nuts, return 0 (goal state).
    7. Initialize the total heuristic cost to 0.
    8. For each loose nut that is a goal:
        a. Add 1 to the cost for the `tighten_nut` action.
        b. Determine the cost to get the man to the nut's location while carrying a usable spanner:
            i. If the man is currently carrying a usable spanner: The cost is the shortest distance from his current location to the nut's location.
            ii. If the man is not carrying a usable spanner: Find the usable spanner `S` on the ground whose location `L_S` minimizes the total travel distance `distance(man_location, L_S) + distance(L_S, nut_location)`. The cost is this minimum distance plus 1 for the `pickup_spanner` action. If no usable spanners are available on the ground (and man isn't carrying one), return a large value (should be caught by step 5, but safety check).
        c. Add this travel/pickup cost to the nut's cost.
        d. Add the nut's total cost to the overall heuristic sum.
    9. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and identifying goal nuts."""
        self.goals = task.goals
        self.initial_state = task.initial_state # Store initial state to identify object types heuristically
        static_facts = task.static

        # 1. Identify all locations and build graph
        locations = set()
        graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                locations.add(loc1)
                locations.add(loc2)
                graph.setdefault(loc1, []).append(loc2)
                graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Add locations from initial state and goals to ensure all are included
        for fact in self.initial_state | self.goals:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)[1:]
                 locations.add(loc)

        # Ensure all locations found are keys in the graph, even if they have no links
        for loc in locations:
             graph.setdefault(loc, [])

        self.locations = list(locations) # Store list of all locations

        # 2. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs_shortest_paths(start_loc, graph)

        # 3. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 4. Heuristically identify spanner and nut objects from initial state facts
        self.spanner_objects = set()
        self.nut_objects = set()
        for fact in self.initial_state:
            if match(fact, "usable", "*"):
                self.spanner_objects.add(get_parts(fact)[1])
            elif match(fact, "loose", "*") or match(fact, "tightened", "*"):
                 self.nut_objects.add(get_parts(fact)[1])


    def get_distance(self, loc1, loc2):
        """Helper to get distance, returning a large value if unreachable."""
        # Check if locations exist in our precomputed distances
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This indicates a location wasn't found during init, which is an issue
             # Or loc2 is truly unreachable from loc1
             return 1000000 # Large value for unreachable

        dist = self.distances[loc1][loc2]
        if dist == float('inf'):
             return 1000000 # Large value for unreachable
        return dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man and location
        man_name = None
        man_location = None

        # Try finding man from 'carrying' fact in current state
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man_name = get_parts(fact)[1]
                break

        # If not found, try finding man from 'carrying' fact in initial state
        if man_name is None:
             for fact in self.initial_state:
                 if match(fact, "carrying", "*", "*"):
                     man_name = get_parts(fact)[1]
                     break

        # If still not found, heuristically find the object in initial state 'at' fact that is not a spanner or nut
        if man_name is None:
             for fact in self.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     if obj not in self.spanner_objects and obj not in self.nut_objects:
                         man_name = obj
                         break # Found the man name

        # Now find the man's current location in the state
        if man_name:
             for fact in state:
                 if match(fact, "at", man_name, "*"):
                     man_location = get_parts(fact)[2]
                     break

        if man_name is None or man_location is None:
             # Should not happen in valid states, but as a safeguard
             # print(f"Warning: Could not identify man ({man_name}) or location ({man_location}) in state: {state}")
             return 1000000 # Cannot find the man

        # 2. Identify usable spanners and locations in the current state
        usable_spanners_on_ground = {} # {spanner_name: location}
        carried_spanner_usable = False

        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if this usable spanner is carried by the man
                is_carried = False
                for carrying_fact in state:
                    if match(carrying_fact, "carrying", man_name, spanner):
                        carried_spanner_usable = True
                        break # Found the usable carried spanner
                if not is_carried:
                    # Find location if on the ground
                    for at_fact in state:
                        if match(at_fact, "at", spanner, "*"):
                            usable_spanners_on_ground[spanner] = get_parts(at_fact)[2]
                            break # Found location of usable spanner on ground

        # 3. Identify loose goal nuts and locations in the current state
        loose_goal_nuts = {} # {nut_name: location}
        for nut in self.goal_nuts:
            # Check if the nut is currently loose
            if f"(loose {nut})" in state:
                 # Find the nut's location
                 for fact in state:
                     if match(fact, "at", nut, "*"):
                         loose_goal_nuts[nut] = get_parts(fact)[2]
                         break # Found nut location

        # 4. Solvability check (basic)
        # Need at least as many usable spanners as loose goal nuts
        total_usable_spanners_available = len(usable_spanners_on_ground) + (1 if carried_spanner_usable else 0)
        if len(loose_goal_nuts) > total_usable_spanners_available:
             return 1000000 # Likely unsolvable

        # If no loose goal nuts, goal is reached
        if len(loose_goal_nuts) == 0:
            return 0

        # 5. Compute heuristic sum
        h = 0

        for nut, nut_location in loose_goal_nuts.items():
            # Cost for this nut = 1 (tighten)
            nut_cost = 1

            # Cost to get man to nut_location with a usable spanner
            if carried_spanner_usable:
                # Man already has a usable spanner, just need to walk to the nut
                dist_to_nut = self.get_distance(man_location, nut_location)
                if dist_to_nut >= 1000000: return 1000000 # Unreachable nut
                nut_cost += dist_to_nut
            else:
                # Man needs to get a spanner first
                min_trip_cost = float('inf')
                found_spanner_option = False
                for spanner, spanner_location in usable_spanners_on_ground.items():
                    # Cost = walk man to spanner + pickup + walk man from spanner_location to nut
                    dist_man_to_spanner = self.get_distance(man_location, spanner_location)
                    dist_spanner_to_nut = self.get_distance(spanner_location, nut_location)

                    # Check for unreachable locations
                    if dist_man_to_spanner >= 1000000 or dist_spanner_to_nut >= 1000000:
                         continue # This spanner path is impossible

                    trip_cost = dist_man_to_spanner + 1 + dist_spanner_to_nut
                    min_trip_cost = min(min_trip_cost, trip_cost)
                    found_spanner_option = True

                if not found_spanner_option:
                    # No usable spanners on the ground reachable, and man isn't carrying one.
                    # This nut cannot be tightened. Should be caught by the initial check, but safety.
                    return 1000000

                nut_cost += min_trip_cost

            h += nut_cost

        return h
