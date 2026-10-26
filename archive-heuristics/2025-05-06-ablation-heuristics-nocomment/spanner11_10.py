from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_location_graph(static_facts):
    """Builds an adjacency list representation of the location graph."""
    graph = {}
    locations = set()
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            locations.add(loc1)
            locations.add(loc2)
            graph.setdefault(loc1, []).append(loc2)
            graph.setdefault(loc2, []).append(loc1)
    # Ensure all locations are keys in the graph, even if they have no links
    for loc in locations:
        graph.setdefault(loc, [])
    return graph

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of locations using BFS."""
    dist_matrix = {}
    locations = list(graph.keys())

    for start_node in locations:
        dist_matrix[start_node] = {}
        q = deque([(start_node, 0)])
        visited = {start_node}
        
        while q:
            curr_loc, d = q.popleft()
            dist_matrix[start_node][curr_loc] = d

            for neighbor in graph.get(curr_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, d + 1))
        
        # Mark unreachable locations with infinity
        for loc in locations:
             if loc not in dist_matrix[start_node]:
                 dist_matrix[start_node][loc] = float('inf')

    return dist_matrix

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Estimates the cost to reach the goal state (all goal nuts tightened)
    by considering the actions required for each loose goal nut sequentially.
    It greedily assigns available usable spanners to nuts and calculates
    the minimum travel and action costs for each tightening operation.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object containing domain information.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract object types
        self.object_types = {}
        # task.objects is a list of strings like 'bob - man'
        for obj_str in task.objects:
            parts = obj_str.split(' - ')
            if len(parts) == 2:
                self.object_types[parts[0]] = parts[1]

        # Build location graph and compute shortest paths
        self.location_graph = build_location_graph(self.static_facts)
        self.dist_matrix = compute_all_pairs_shortest_paths(self.location_graph)

        # Identify goal nut names
        self.goal_nut_names = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimate of the remaining cost to reach the goal.
            Returns float('inf') if the state is estimated to be unsolvable.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---
        # 1. Identify current state information:
        #    - Man's location
        #    - Spanner the man is carrying (if any)
        #    - Location of all locatable objects (spanners, nuts, man)
        #    - Which spanners are usable
        #    - Which nuts are loose

        current_locations = {}
        man_name = None
        carried_spanner = None
        usable_spanner_names_in_state = set()
        loose_nut_names_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                current_locations[obj_name] = loc_name
                if self.object_types.get(obj_name) == 'man':
                    man_name = obj_name
            elif parts[0] == 'carrying' and len(parts) == 3:
                 # Assuming only man can carry and carries one thing
                 carrier, carried = parts[1], parts[2]
                 if self.object_types.get(carrier) == 'man':
                     carried_spanner = carried
            elif parts[0] == 'usable' and len(parts) == 2:
                spanner_name = parts[1]
                usable_spanner_names_in_state.add(spanner_name)
            elif parts[0] == 'loose' and len(parts) == 2:
                nut_name = parts[1]
                loose_nut_names_in_state.add(nut_name)

        if man_name is None:
             # Man must exist
             return float('inf') # Should not happen in valid problems

        man_location = current_locations.get(man_name)
        if man_location is None:
             # Man must be at a location
             return float('inf') # Should not happen in valid states

        # 2. Identify loose goal nuts in the current state.
        loose_goal_nuts_names = list(self.goal_nut_names.intersection(loose_nut_names_in_state))

        # If all goal nuts are already tightened, the heuristic is 0.
        if not loose_goal_nuts_names:
            return 0

        # 3. Check solvability based on spanner count.
        # We need at least one usable spanner for each loose goal nut.
        # The total count includes the one being carried if it's usable.
        total_usable_spanners = len(usable_spanner_names_in_state)
        if len(loose_goal_nuts_names) > total_usable_spanners:
             # Not enough usable spanners in the state to tighten all required nuts
             return float('inf')

        # 4. Initialize heuristic value and tracking variables.
        h = 0
        nuts_to_process_names = list(loose_goal_nuts_names) # Make a mutable list
        # Spanners available for *pickup* in subsequent steps.
        # Initially, this is all usable spanners in the state.
        available_spanner_names_for_pickup = set(usable_spanner_names_in_state)
        current_loc = man_location
        carrying_usable = (carried_spanner is not None and carried_spanner in usable_spanner_names_in_state)

        # 5. If man is carrying a usable spanner, use it for the first nut.
        #    Greedily pick the closest loose goal nut.
        if carrying_usable:
            closest_nut_name = None
            min_dist = float('inf')

            for nut_name in nuts_to_process_names:
                nut_loc = current_locations.get(nut_name)
                if nut_loc is None: continue # Nut must be at a location

                dist_to_nut = self.dist_matrix.get(current_loc, {}).get(nut_loc, float('inf'))
                if dist_to_nut == float('inf'):
                    # Cannot reach this nut
                    return float('inf')

                if dist_to_nut < min_dist:
                    min_dist = dist_to_nut
                    closest_nut_name = nut_name

            if closest_nut_name:
                # Cost to use carried spanner on the closest nut: walk + tighten
                h += min_dist # Walk to the first nut
                h += 1        # Tighten the nut
                current_loc = current_locations.get(closest_nut_name) # Man is now at the nut location
                nuts_to_process_names.remove(closest_nut_name)
                # The carried spanner is now used and becomes unusable.
                # It is no longer available for pickup for subsequent nuts.
                available_spanner_names_for_pickup.discard(carried_spanner)
                carrying_usable = False # Man is no longer carrying a usable spanner

        # 6. For the remaining nuts, the man must pick up a spanner first.
        #    Greedily select the pair of (remaining nut, available spanner)
        #    that minimizes the cost of the next step.
        while nuts_to_process_names:
            best_nut_name = None
            best_spanner_name = None
            min_cost_step = float('inf')

            for nut_name in nuts_to_process_names:
                nut_loc = current_locations.get(nut_name)
                if nut_loc is None: continue # Nut must be at a location

                for spanner_name in available_spanner_names_for_pickup:
                    spanner_loc = current_locations.get(spanner_name)
                    if spanner_loc is None: continue # Spanner must be at a location to be picked up

                    # Cost for this step: walk to spanner + pickup + walk to nut + tighten
                    dist_to_spanner = self.dist_matrix.get(current_loc, {}).get(spanner_loc, float('inf'))
                    dist_spanner_to_nut = self.dist_matrix.get(spanner_loc, {}).get(nut_loc, float('inf'))

                    if dist_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                        # Cannot reach spanner or nut from current/spanner location
                        continue # Try next spanner/nut pair

                    cost_this_step = dist_to_spanner + 1 + dist_spanner_to_nut + 1

                    if cost_this_step < min_cost_step:
                        min_cost_step = cost_this_step
                        best_nut_name = nut_name
                        best_spanner_name = spanner_name

            if best_nut_name is None:
                # Cannot find a way to tighten any remaining nut (e.g., no spanners left,
                # or remaining spanners/nuts are unreachable).
                return float('inf')

            # Add the cost of the best next step
            h += min_cost_step
            # Update man's location to the location of the nut just tightened
            current_loc = current_locations.get(best_nut_name)
            # The spanner used is no longer available for pickup
            available_spanner_names_for_pickup.remove(best_spanner_name)
            # The nut is now tightened
            nuts_to_process_names.remove(best_nut_name)

        # 7. Return the total estimated cost.
        return h

