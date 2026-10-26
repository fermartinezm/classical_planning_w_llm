from fnmatch import fnmatch
# Assuming Heuristic base class is available, uncomment the line below
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a fact matches a given pattern (not strictly needed for this heuristic but good practice)
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
#     - `args`: The expected pattern (wildcards `*` allowed).
#     - Returns `True` if the fact matches the pattern, else `False`.
#     """
#     parts = get_parts(fact)
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class. Inherit from Heuristic if available.
# class spannerHeuristic(Heuristic):
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all goal nuts by simulating a greedy approach:
    repeatedly travel to the nearest untightened goal nut, acquire a spanner
    if needed (by traveling to the nearest usable spanner and picking it up),
    and then tightening the nut.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing shortest path distances
        between all locations.
        """
        self.goals = task.goals

        # Build the graph of locations from link facts
        self.locations = set()
        self.links = {} # adjacency list {loc: [neighbor1, neighbor2]}

        # Populate locations and links from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links.setdefault(l1, []).append(l2)
                self.links.setdefault(l2, []).append(l1) # Links are bidirectional

        # Ensure all locations mentioned in initial state/goals are included, even if isolated
        # This is important if the man or a nut starts in an isolated location
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 loc = parts[2] # The second argument of 'at' is the location
                 self.locations.add(loc)


        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS to find shortest distances from start_node to all reachable locations."""
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = [start_node]
        q_index = 0 # Use index for queue to avoid pop(0) cost

        while q_index < len(queue):
            curr = queue[q_index]
            q_index += 1

            # Get neighbors, handle locations with no outgoing links
            # Ensure curr is in self.links before accessing it
            neighbors = self.links.get(curr, [])

            for neighbor in neighbors:
                # If we found a shorter path (only happens the first time in unweighted graph)
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state

        # 1. Identify loose goal nuts
        goal_nuts = set()
        # Assuming goals are always (tightened nut)
        for goal in self.goals:
            g_parts = get_parts(goal)
            if g_parts[0] == 'tightened' and len(g_parts) == 2:
                nut = g_parts[1]
                # Check if this nut is currently loose
                if f"(loose {nut})" in state:
                     goal_nuts.add(nut)

        if not goal_nuts:
            return 0 # Goal reached

        # 2. Extract current state information
        man_loc = None
        carried_spanners = [] # Usable spanners carried by man
        usable_spanners_on_ground = {} # {location: [spanner1, ...]}
        nut_locations = {} # {nut: location}

        # Find the man (assuming there's only one and its name starts with 'bob')
        man_name = None
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('bob'):
                 man_name = parts[1]
                 man_loc = parts[2]
                 break # Found the man

        if man_name is None:
             # Man not found in state? Invalid state.
             return float('inf')


        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj.startswith('spanner'):
                    if f"(usable {obj})" in state:
                        usable_spanners_on_ground.setdefault(loc, []).append(obj)
                elif obj in goal_nuts: # It's a loose goal nut
                     nut_locations[obj] = loc
            elif parts[0] == 'carrying' and len(parts) == 3:
                carrier, spanner = parts[1], parts[2]
                if carrier == man_name and f"(usable {spanner})" in state:
                     carried_spanners.append(spanner)

        # Check if all goal nuts have a location in the state
        if len(nut_locations) != len(goal_nuts):
             # Some goal nuts are not located in the current state? Invalid.
             return float('inf')


        # 3. Greedy simulation to estimate cost
        sim_carried_spanners_count = len(carried_spanners)
        # Use counts for ground spanners for simplicity in simulation
        sim_available_spanners_on_ground = {loc: len(spanners) for loc, spanners in usable_spanners_on_ground.items()}
        sim_remaining_nuts = set(goal_nuts)
        sim_current_man_loc = man_loc

        h = 0

        # Greedy simulation loop
        while sim_remaining_nuts:
            # Find the closest nut to the current man location
            closest_nut = None
            min_dist_to_nut = float('inf')

            # Ensure current_man_loc is in the distance map (should be if added in __init__)
            if sim_current_man_loc not in self.dist:
                 # Man is in an isolated location not in the graph
                 # Can only reach nuts in the same isolated location
                 found_nut_in_isolated_loc = False
                 for nut in sim_remaining_nuts:
                     if nut_locations[nut] == sim_current_man_loc:
                         min_dist_to_nut = 0
                         closest_nut = nut
                         found_nut_in_isolated_loc = True
                         break # Found a nut at the current isolated location
                 if not found_nut_in_isolated_loc:
                      # Man is isolated and no goal nut is at his location
                      return float('inf') # Cannot reach any remaining nut
            else:
                # Man is in the connected graph or an isolated node that is a key in self.dist
                for nut in sim_remaining_nuts:
                    nut_loc = nut_locations[nut]
                    # Ensure nut_loc is in the distance map from sim_current_man_loc
                    if nut_loc not in self.dist[sim_current_man_loc]:
                         # This nut location was not in self.locations during BFS
                         # This shouldn't happen if __init__ is correct, but as a safeguard:
                         return float('inf') # Cannot reach this nut location

                    dist = self.dist[sim_current_man_loc][nut_loc]

                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        closest_nut = nut

                if closest_nut is None:
                     # Should not happen if remaining_nuts is not empty and reachable nuts exist
                     return float('inf') # Cannot reach any remaining nut


            # Add travel cost to the closest nut
            h += min_dist_to_nut
            sim_current_man_loc = nut_locations[closest_nut] # Man is now at the nut location

            # Check if a spanner is needed for this tightening
            # Each tighten action consumes one usable spanner.
            # If man has 0 usable spanners, he must acquire one from the ground.
            if sim_carried_spanners_count == 0:
                 # Man needs to acquire a spanner from the ground
                 closest_spanner_loc = None
                 min_dist_to_spanner = float('inf')

                 # Find the closest location with a usable spanner on the ground from the *current* man location (which is the nut location)
                 # Ensure current_man_loc is in the distance map
                 if sim_current_man_loc not in self.dist:
                      # Man is in an isolated location. Can only pick up spanner if it's here.
                      if sim_current_man_loc in sim_available_spanners_on_ground and sim_available_spanners_on_ground[sim_current_man_loc] > 0:
                           min_dist_to_spanner = 0
                           closest_spanner_loc = sim_current_man_loc
                      else:
                           # Man is isolated at nut location, needs spanner, but no spanner here or reachable
                           return float('inf')
                 else:
                    # Man is in the connected graph or an isolated node that is a key in self.dist
                    for s_loc, count in sim_available_spanners_on_ground.items():
                        if count > 0:
                            # Ensure s_loc is in the distance map from sim_current_man_loc
                            if s_loc not in self.dist[sim_current_man_loc]:
                                 # This spanner location was not in self.locations during BFS
                                 return float('inf') # Cannot reach this spanner location

                            dist = self.dist[sim_current_man_loc][s_loc]

                            if dist < min_dist_to_spanner:
                                min_dist_to_spanner = dist
                                closest_spanner_loc = s_loc

                 if closest_spanner_loc is None:
                     # No usable spanners left anywhere (carried or on ground)
                     # Problem likely unsolvable
                     return float('inf') # Cannot get a spanner

                 # Add travel cost to the spanner location
                 h += min_dist_to_spanner
                 sim_current_man_loc = closest_spanner_loc # Man is now at the spanner location

                 # Add pickup cost
                 h += 1
                 sim_carried_spanners_count += 1 # Man now carries one usable spanner
                 sim_available_spanners_on_ground[closest_spanner_loc] -= 1

                 # After picking up the spanner, the man is at closest_spanner_loc.
                 # He needs to be back at the nut location (nut_locations[closest_nut])
                 # to perform the tighten action.
                 # Add travel cost back to the nut location.
                 # Ensure nut_locations[closest_nut] is in the distance map from sim_current_man_loc
                 target_nut_loc = nut_locations[closest_nut]
                 if target_nut_loc not in self.dist[sim_current_man_loc]:
                      # This nut location was not in self.locations during BFS
                      return float('inf') # Cannot reach the nut location from spanner location

                 dist_back_to_nut = self.dist[sim_current_man_loc][target_nut_loc]

                 if dist_back_to_nut == float('inf'):
                      # Should not happen if graph is connected and locations exist
                      return float('inf')
                 h += dist_back_to_nut
                 sim_current_man_loc = nut_locations[closest_nut] # Man is back at the nut location


            # Tighten the nut
            h += 1 # tighten_nut action
            sim_carried_spanners_count -= 1 # Spanner is used up
            sim_remaining_nuts.remove(closest_nut)

        return h
