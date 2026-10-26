from fnmatch import fnmatch
import collections
import math
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Depending on expected input robustness, could log a warning or raise error
         # For robustness, return empty list for malformed facts
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

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbors]}.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Unreachable nodes are not included.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current = queue.popleft()
        # Ensure current node exists in graph keys, handle potential missing nodes gracefully
        # BFS should only explore nodes present in the graph
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the required number of actions to tighten all
    goal nuts. It uses a greedy approach, simulating the process of the man
    sequentially addressing each loose goal nut. In each step, it calculates
    the minimum cost to get the man (with a usable spanner) to one of the
    remaining loose goal nuts and perform the tighten action. It greedily
    chooses the cheapest such step, updates the man's location and available
    spanners, and repeats until all goal nuts are tightened.

    # Assumptions
    - There is exactly one man object, assumed to be the object whose 'at' predicate
      is found first when iterating through state facts and whose name is 'bob' based on examples.
    - Nut objects are identifiable (e.g., by naming convention or type, heuristic assumes they are the objects in 'tightened' goals).
    - Spanner objects are identifiable (e.g., by naming convention or type, heuristic assumes they are the objects in 'usable' facts).
    - Locations are connected by 'link' predicates, forming an undirected graph.
    - Each usable spanner can tighten exactly one nut.
    - The goal is always to tighten a specific set of nuts.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task definition.
    - Builds the location graph (adjacency list) from the static 'link' facts.
    - Computes All-Pairs Shortest Paths (APSP) on the location graph using BFS from each node.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to find:
       - The man's current location.
       - The current location of each nut.
       - The current location of each spanner that is on the ground.
       - The spanner the man is currently carrying (if any).
       - The set of all usable spanners.
    2. Identify the set of loose nuts that are part of the goal.
    3. Identify the set of usable spanners that are currently available (either on the ground and usable, or carried and usable).
    4. Check if the number of loose goal nuts exceeds the number of available usable spanners. If so, the problem is likely unsolvable with the current resources, return a large heuristic value (e.g., 1,000,000).
    5. Initialize the total heuristic cost to 0.
    6. Initialize the man's simulated current location to his actual current location.
    7. Create lists of remaining loose goal nuts and remaining available usable spanners on the ground.
    8. Track whether the man is currently simulated as carrying a usable spanner.
    9. While there are still loose goal nuts remaining:
        a. Determine the minimum cost to tighten one of the remaining nuts in this step. This involves considering two possibilities:
           i.  Using a spanner the man is currently simulated as carrying (if any): Calculate the cost as the distance from the simulated current location to the nut's location plus 1 (for the tighten action). Find the minimum cost over all remaining nuts.
           ii. Picking up an available spanner from the ground and using it on a nut: Calculate the cost as the distance from the simulated current location to the spanner's location, plus 1 (for the pickup action), plus the distance from the spanner's location to the nut's location, plus 1 (for the tighten action). Find the minimum cost over all pairs of remaining available spanners on the ground and remaining nuts.
        b. Select the overall minimum cost found in step 9a.
        c. Add this minimum cost to the total heuristic value.
        d. Update the man's simulated current location to the location of the nut that was just tightened.
        e. Remove the tightened nut from the list of remaining nuts.
        f. If a spanner was picked up from the ground in this step, remove it from the list of remaining available spanners on the ground.
        g. Set the flag indicating whether the man is carrying a usable spanner to False, as the spanner used in this step is now consumed.
    10. Return the total accumulated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts and building the location graph.
        """
        super().__init__(task) # Initialize base class (sets self.goals and self.goal_nuts)

        # Build the location graph from 'link' facts
        self.location_graph = {}
        locations_set = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                locations_set.add(loc1)
                locations_set.add(loc2)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                # Links are bidirectional
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

        self.all_locations = list(locations_set) # Get a list of all unique locations

        # Compute All-Pairs Shortest Paths (APSP) using BFS from each node
        self.distances = {} # {start_loc: {end_loc: distance}}
        for start_node in self.all_locations:
            self.distances[start_node] = bfs(self.location_graph, start_node)

    def get_dist(self, l1, l2):
        """Helper to get the precomputed distance between two locations."""
        # Return infinity if either location is not in the graph or if l2 is unreachable from l1.
        if l1 not in self.distances or l2 not in self.distances[l1]:
            return math.inf
        return self.distances[l1][l2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse State ---
        man_loc = None
        nut_locs = {} # nut_obj -> location_str
        spanner_locs_on_ground = {} # spanner_obj -> location_str (only for spanners on ground)
        carried_spanner = None # spanner_obj if carried
        usable_spanners_set = set() # set of usable spanner objects

        # Assuming 'bob' is the man based on examples
        man_name = 'bob' # Hardcode based on examples

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj == man_name:
                    man_loc = loc
                elif obj.startswith('nut'): # Heuristic relies on naming convention
                    nut_locs[obj] = loc
                elif obj.startswith('spanner'): # Heuristic relies on naming convention
                    spanner_locs_on_ground[obj] = loc
            elif pred == 'carrying':
                m, s = parts[1], parts[2]
                if m == man_name:
                    carried_spanner = s
            elif pred == 'usable':
                s = parts[1]
                usable_spanners_set.add(s)
            # We don't need 'tightened' or 'loose' facts here, as we filter goal nuts later

        # --- Heuristic Calculation ---

        # Identify loose goal nuts in the current state
        loose_goal_nuts = [n for n in self.goal_nuts if '(tightened ' + n + ')' not in state]

        # Identify usable spanners currently available (on ground or carried)
        available_spanners_on_ground = [s for s in usable_spanners_set if s in spanner_locs_on_ground]
        man_carrying_usable = (carried_spanner is not None) and (carried_spanner in usable_spanners_set)

        # Check if solvable (enough usable spanners)
        required_spanners = len(loose_goal_nuts)
        total_usable_spanners = len(available_spanners_on_ground) + (1 if man_carrying_usable else 0)

        if required_spanners > total_usable_spanners:
            # Problem is likely unsolvable with current usable spanners
            return 1000000 # Large heuristic value

        if required_spanners == 0:
            return 0 # Goal reached

        # Handle case where man_loc might not be in the precomputed distances (e.g., initial state not in graph)
        # This shouldn't happen in valid PDDL problems where initial locations are part of the domain.
        # If it somehow happens, distances from man_loc would be inf, leading to high heuristic.
        if man_loc not in self.distances:
             return 1000000 # Man is in an unlinked/unknown location

        # Greedy simulation
        h = 0
        current_sim_loc = man_loc
        remaining_nuts = list(loose_goal_nuts)
        remaining_available_spanners_on_ground = list(available_spanners_on_ground)
        sim_carrying_usable = man_carrying_usable

        while remaining_nuts:
            best_step_cost = math.inf
            best_nut_for_step = None
            spanner_used_in_step = None # The spanner object used in this step
            pickup_needed_in_step = False # Was a pickup action part of this step?

            # Option 1: Use carried spanner (if any) on a remaining nut
            if sim_carrying_usable:
                for nut in remaining_nuts:
                    nut_loc = nut_locs[nut]
                    dist_to_nut = self.get_dist(current_sim_loc, nut_loc)
                    if dist_to_nut == math.inf:
                         # Nut location is unreachable from current simulated location
                         continue

                    cost = dist_to_nut + 1 # walk + tighten
                    if cost < best_step_cost:
                        best_step_cost = cost
                        best_nut_for_step = nut
                        # The spanner used is the one currently carried
                        # We don't need the spanner object itself here, just that one is used.
                        # spanner_used_in_step = carried_spanner # Not needed for logic
                        pickup_needed_in_step = False

            # Option 2: Pick up an available spanner on the ground and use it on a remaining nut
            if remaining_available_spanners_on_ground:
                for spanner in remaining_available_spanners_on_ground:
                    spanner_loc = spanner_locs_on_ground[spanner]
                    dist_to_spanner = self.get_dist(current_sim_loc, spanner_loc)
                    if dist_to_spanner == math.inf: continue # Spanner unreachable

                    for nut in remaining_nuts:
                        nut_loc = nut_locs[nut]
                        dist_spanner_to_nut = self.get_dist(spanner_loc, nut_loc)
                        if dist_spanner_to_nut == math.inf: continue # Nut unreachable from spanner loc

                        # Cost = walk to spanner + pickup + walk to nut + tighten
                        cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1
                        if cost < best_step_cost:
                            best_step_cost = cost
                            best_nut_for_step = nut
                            spanner_used_in_step = spanner # Store which spanner was chosen
                            pickup_needed_in_step = True

            # Apply the best step
            if best_nut_for_step is None:
                 # This implies no remaining nut is reachable with any available spanner
                 # (Should be caught by the initial check if graph is connected)
                 # Return large value just in case.
                 return 1000000

            h += best_step_cost
            current_sim_loc = nut_locs[best_nut_for_step] # Man ends up at the nut location
            remaining_nuts.remove(best_nut_for_step)

            if pickup_needed_in_step:
                remaining_available_spanners_on_ground.remove(spanner_used_in_step)

            # After tightening, the spanner is used and no longer carried.
            sim_carrying_usable = False

        return h
