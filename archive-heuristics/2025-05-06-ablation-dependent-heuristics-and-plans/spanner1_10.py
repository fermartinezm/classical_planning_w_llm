from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from start_node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, ...]}
        start_node: The starting node for BFS.

    Returns:
        A dictionary mapping each reachable node to its distance from start_node.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        distance = distances[current_node]

        if current_node in graph: # Handle nodes with no outgoing links
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It models the process as a sequence of trips, where each trip involves
    acquiring a usable spanner (if not already carried) and traveling to a
    loose goal nut's location to tighten it. The heuristic greedily selects
    the cheapest available spanner and loose goal nut pair at each step,
    considering the man's current location.

    # Assumptions
    - Each goal nut requires one tighten action.
    - Each tighten action consumes one usable spanner.
    - The man can carry multiple spanners (based on example state representation).
    - Nuts are fixed at their initial locations.
    - The graph of locations connected by 'link' predicates is connected,
      or at least all relevant locations (man start, spanner locations, nut locations)
      are reachable from each other.
    - Action costs are uniform (cost 1).

    # Heuristic Initialization
    - Extracts all locations and 'link' predicates from static facts to build
      a graph of locations.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies goal nuts and their fixed locations by looking at goal conditions
      and initial state 'at' predicates for objects that are not locations,
      man, or spanners.
    - Identifies spanners initially carried by the man.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the total cost iteratively, simulating the process
    of tightening one nut at a time until all goal nuts are tightened.

    1.  **Identify State Information:** Get the man's current location, the set
        of loose nuts that are goal conditions, and the set of usable spanners
        with their current locations (or if carried).
    2.  **Check Solvability:** If the total number of usable spanners in the
        current state is less than the number of loose goal nuts, the problem
        is likely unsolvable in this domain (as spanners become unusable).
        Return a large value.
    3.  **Initialize:** Set the current man's location to the actual location
        from the state. Initialize total heuristic cost to 0. Create mutable
        lists of remaining loose goal nuts and remaining usable spanners
        (including their current state: location or carried).
    4.  **Iterative Tightening:** While there are still remaining loose goal nuts:
        a.  Find the minimum cost to tighten *any* of the remaining loose goal
            nuts using *any* of the remaining usable spanners, starting from
            the man's current location.
        b.  The cost for a specific (nut, spanner) pair depends on the spanner's
            current state:
            -   If the spanner is currently carried by the man: Cost =
                `distance(man_location, nut_location) + 1 (tighten)`.
            -   If the spanner is at a location: Cost =
                `distance(man_location, spanner_location) + 1 (pickup) + distance(spanner_location, nut_location) + 1 (tighten)`.
            -   If any required location is unreachable, the cost for this pair is infinite.
        c.  Select the (nut, spanner) pair that yields the minimum cost for this step.
        d.  If no reachable nut/spanner combination is found (min cost is infinite),
            return a large value indicating unsolvability.
        e.  Add this minimum cost to the total heuristic cost.
        f.  Remove the selected nut from the list of remaining loose nuts.
        g.  Remove the selected spanner from the list of remaining usable spanners.
        h.  Update the man's current location to the location of the nut that was just tightened.
    5.  **Return Total Cost:** The accumulated total cost is the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and goal info."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Precompute distances between locations
        self.locations = set()
        graph = {}

        # Extract locations and build graph from 'link' facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                graph.setdefault(loc1, []).append(loc2)
                graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(graph, start_loc)

        # 2. Identify goal nuts and their fixed locations
        self.goal_nuts = {}
        # Nuts are locatable, but seem fixed. Find their initial locations.
        initial_nut_locations = {}
        initial_spanner_names = set()
        man_name = None

        # First pass to identify object types based on initial state predicates
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj in self.locations: continue # Skip locations themselves
                 # Heuristic to distinguish man, spanners, nuts based on typical predicates
                 # This is domain-dependent and assumes the structure seen in examples
                 is_spanner = any(match(f, "usable", obj) for f in initial_state) or \
                              any(match(f, "carrying", "*", obj) for f in initial_state) # Check if initially usable or carried
                 is_man = any(match(f, "at", obj, "*") for f in initial_state) and \
                          any(match(f, "carrying", obj, "*") for f in initial_state) # Check if initially at a loc and carrying something

                 if is_spanner:
                     initial_spanner_names.add(obj)
                 elif is_man:
                     man_name = obj
                 else: # Assume it's a nut
                     initial_nut_locations[obj] = loc

        # Store goal nut locations
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                if nut_name in initial_nut_locations:
                     self.goal_nuts[nut_name] = initial_nut_locations[nut_name]
                # else: Goal nut location not found in initial state 'at' facts. Problem might be malformed.

        # 3. Identify spanners initially carried by the man
        self.initial_carried_spanners = set()
        if man_name:
             for fact in initial_state:
                 if match(fact, "carrying", man_name, "*"):
                     _, _, spanner_name = get_parts(fact)
                     self.initial_carried_spanners.add(spanner_name)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify State Information
        man_current_loc = None
        spanner_locations = {} # Current location of spanners on the ground
        carried_spanners = set() # Spanners currently carried by the man
        usable_spanners_in_state = set() # Spanners that are currently usable
        loose_nuts_in_state = set() # Nuts that are currently loose

        man_name = None # Need to identify man name from state if not already known (should be in init)
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)
                 # Simple heuristic to find the man in the current state
                 # Assumes the man is the only 'locatable' object that isn't a spanner or nut
                 # (based on domain types and typical usage)
                 if obj not in self.locations and obj not in self.goal_nuts and obj not in self.initial_carried_spanners:
                      man_name = obj
                      man_current_loc = loc
                      break # Found the man and his location

        if man_name is None:
             # Fallback: Try to find man based on initial name if available
             # This might happen if the state representation is partial
             if hasattr(self, 'man_name_from_init') and self.man_name_from_init:
                  man_name = self.man_name_from_init
                  for fact in state:
                       if match(fact, "at", man_name, "*"):
                            _, _, man_current_loc = get_parts(fact)
                            break
             # If still no man_name or location, something is wrong, return high cost
             if man_name is None or man_current_loc is None:
                  return 1000000


        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)
                 # Identify spanners at locations in the current state
                 # Assume any 'locatable' object at a location that isn't the man or a nut is a spanner
                 if obj != man_name and obj not in self.goal_nuts and obj not in self.locations:
                      spanner_locations[obj] = loc

            elif match(fact, "carrying", man_name, "*"):
                 _, _, spanner_name = get_parts(fact)
                 carried_spanners.add(spanner_name)

            elif match(fact, "usable", "*"):
                 _, spanner_name = get_parts(fact)
                 usable_spanners_in_state.add(spanner_name)

            elif match(fact, "loose", "*"):
                 _, nut_name = get_parts(fact)
                 loose_nuts_in_state.add(nut_name)


        # Filter goal nuts that are still loose
        loose_goal_nuts = [nut for nut in self.goal_nuts if nut in loose_nuts_in_state]

        # Filter usable spanners and get their current state info
        usable_spanner_info = [] # List of (spanner_name, current_location_or_carried_flag, is_carried_in_state)
        for s in usable_spanners_in_state:
            if s in carried_spanners:
                 usable_spanner_info.append((s, man_current_loc, True)) # Location is man's location if carried
            elif s in spanner_locations:
                 usable_spanner_info.append((s, spanner_locations[s], False))
            # else: usable spanner exists but is not at a location and not carried? (Shouldn't happen in STRIPS)


        # 2. Check Solvability
        if len(usable_spanner_info) < len(loose_goal_nuts):
            return 1000000 # Return a large number indicating likely unsolvable

        # 3. Initialize
        current_M_loc = man_current_loc
        total_cost = 0
        remaining_nuts = list(loose_goal_nuts)
        remaining_spanners = list(usable_spanner_info) # Make a mutable copy

        # 4. Iterative Tightening (Sequential Greedy)
        while remaining_nuts:
            min_cost_step = float('inf')
            best_nut = None
            best_spanner_info = None

            # Find the best (nut, spanner) pair for the next step
            for nut in remaining_nuts:
                l_n = self.goal_nuts[nut] # Nut location is fixed

                # Check if nut location is reachable from current man location
                if current_M_loc not in self.distances or l_n not in self.distances[current_M_loc]:
                     # If current location or nut location is not in our distance map,
                     # it means it's unreachable from the precomputed graph.
                     # This indicates an unsolvable state or a graph issue.
                     dist_man_to_nut = float('inf')
                else:
                     dist_man_to_nut = self.distances[current_M_loc][l_n]


                for spanner_info in remaining_spanners:
                    s, spanner_loc_or_carried, is_carried_in_state = spanner_info

                    if is_carried_in_state:
                        # Cost = travel to nut + tighten
                        cost = dist_man_to_nut + 1
                    else:
                        # Cost = travel to spanner + pickup + travel to nut + tighten
                        l_s = spanner_loc_or_carried
                        if current_M_loc not in self.distances or l_s not in self.distances[current_M_loc] or l_s not in self.distances or l_n not in self.distances[l_s]:
                             # If current location, spanner location, or nut location is unreachable
                             cost = float('inf')
                        else:
                             dist_man_to_spanner = self.distances[current_M_loc][l_s]
                             dist_spanner_to_nut = self.distances[l_s][l_n]
                             cost = dist_man_to_spanner + 1 + dist_spanner_to_nut + 1

                    # Update minimum cost for this step
                    if cost < min_cost_step:
                        min_cost_step = cost
                        best_nut = nut
                        best_spanner_info = spanner_info

            # If no reachable nut/spanner combination found (should only happen if unsolvable)
            if min_cost_step == float('inf'):
                 return 1000000 # Indicate unsolvable

            # Add the cost of the best step
            total_cost += min_cost_step

            # Update remaining resources and man's location
            remaining_nuts.remove(best_nut)
            remaining_spanners.remove(best_spanner_info)
            current_M_loc = self.goal_nuts[best_nut] # Man is at the nut location after tightening

        # 5. Return Total Cost
        return total_cost
