import math
from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation for shortest path on a graph
def bfs_shortest_path(graph, start_node):
    """
    Finds the shortest path distances from start_node to all reachable nodes.
    Graph is an adjacency list (dict of node -> set of neighbors).
    Returns a dict mapping node to distance, or float('inf') if unreachable.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might not be in graph if it has no can_traverse edges
         # but exists as a waypoint. BFS from here is impossible.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Ensure the node exists in the graph keys
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

# Precompute all-pairs shortest paths
def precompute_all_pairs_shortest_paths(graph, nodes):
    """
    Computes shortest path distances between all pairs of nodes in a graph.
    Returns a dict mapping (start_node, end_node) to distance.
    """
    all_paths = {}
    for start_node in nodes:
        distances = bfs_shortest_path(graph, start_node)
        for end_node in nodes:
            all_paths[(start_node, end_node)] = distances.get(end_node, float('inf'))
    return all_paths


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It sums up the estimated costs for each unachieved
    goal fact independently. The cost for each goal is estimated based on
    whether data/images need to be acquired and the navigation required
    to visit necessary locations (sample/observation/calibration points)
    and finally a communication point near the lander. Navigation costs
    are estimated using precomputed shortest path distances on the
    rover-specific traversal graphs.

    # Assumptions
    - The heuristic assumes that any equipped rover can perform the necessary
      tasks (sampling, imaging, communication). It calculates the minimum cost
      among all capable rovers for each goal.
    - Resource constraints like store capacity are partially considered for
      sampling (adding a drop action if the store is full).
    - Camera calibration state is considered.
    - The heuristic ignores potential conflicts or synergies between goals
      requiring the same rover or resource simultaneously.
    - Unreachable goals (e.g., sample location gone, no path to communication)
      are assigned a large heuristic value.

    # Heuristic Initialization
    The initialization phase extracts static information from the task:
    - Lander location.
    - Rover capabilities (soil, rock, imaging).
    - Store ownership for each rover.
    - Waypoint visibility graph.
    - Objective visibility from waypoints.
    - Camera information (on-board rover, calibration target, supported modes).
    - Rover-specific traversal graphs based on `can_traverse`.
    - Precomputes all-pairs shortest path distances for each rover on its
      traversal graph using BFS.
    - Identifies communication waypoints visible from the lander.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic `h` is computed as follows:
    1. Initialize `h = 0`.
    2. Identify all goal facts that are not present in the current state.
    3. For each unachieved goal fact (e.g., `(communicated_soil_data ?w)`,
       `(communicated_rock_data ?w)`, or `(communicated_image_data ?o ?m)`):
       a. Calculate the minimum estimated cost to achieve this specific goal.
          This involves iterating through all rovers capable of achieving
          this type of goal (e.g., equipped for soil analysis for soil data).
       b. For each capable rover, estimate the cost:
          i.  Find the rover's current location.
          ii. Determine if the required data/image is already available on
              this specific rover.
          iii. If data/image is available on this rover:
              - Cost = Navigation from current location to a communication
                waypoint + 1 (communicate action). Minimize navigation over
                all reachable communication waypoints.
          iv. If data/image is NOT available on this rover:
              - For soil/rock: Cost = Navigation from current location to
                sample location + 1 (sample action) + (1 if store is full) +
                Navigation from sample location to a communication waypoint +
                1 (communicate action). Minimize navigation over reachable
                sample and communication waypoints. Check if sample exists.
              - For image: Cost = (Navigation from current location to
                calibration waypoint + 1 (calibrate action) if not calibrated) +
                Navigation from (calibration or current) location to observation
                waypoint + 1 (take_image action) +
                Navigation from observation waypoint to a communication waypoint +
                1 (communicate action). Minimize navigation over reachable
                calibration, observation, and communication waypoints.
          v. Use precomputed shortest paths for navigation costs. If any
             required waypoint sequence is unreachable for this rover, this
             rover cannot achieve the goal via this path.
       c. The minimum cost for this goal is the minimum estimated cost among
          all capable rovers. If no capable rover can achieve the goal,
          assign a large cost (representing unachievability).
       d. Add the minimum estimated cost for this goal to `h`.
    4. Return `h`. If the total sum includes any large unachievability cost,
       return a large value (e.g., 1000000) to indicate likely unsolvability
       from this state.
    """

    LARGE_COST = 1000000 # Represents infinity for practical purposes

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static facts
        self.lander_location = {} # lander -> waypoint
        self.rover_capabilities = defaultdict(set) # rover -> set of {'soil', 'rock', 'imaging'}
        self.rover_stores = {} # rover -> store
        self.waypoint_visibility = defaultdict(set) # waypoint -> set of visible waypoints
        self.objective_visibility = defaultdict(set) # objective -> set of waypoints visible from
        self.calibration_targets = {} # camera -> objective
        self.camera_on_rover = {} # camera -> rover
        self.camera_modes = defaultdict(set) # camera -> set of modes
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set of reachable waypoints
        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_cameras = set()
        self.all_stores = set()
        self.all_landers = set()

        # Parse static facts to populate sets of objects and relationships
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at_lander':
                self.lander_location[parts[1]] = parts[2]
                self.all_landers.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif predicate == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
                self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
                self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
                self.all_rovers.add(parts[1])
            elif predicate == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # rover -> store
                self.all_rovers.add(parts[2])
                self.all_stores.add(parts[1])
            elif predicate == 'can_traverse':
                self.rover_traversal_graph[parts[1]][parts[2]].add(parts[3])
                self.all_rovers.add(parts[1])
                self.all_waypoints.add(parts[2])
                self.all_waypoints.add(parts[3])
            elif predicate == 'visible':
                self.waypoint_visibility[parts[1]].add(parts[2])
                self.waypoint_visibility[parts[2]].add(parts[1]) # Visibility is symmetric
                self.all_waypoints.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif predicate == 'visible_from':
                self.objective_visibility[parts[1]].add(parts[2])
                self.all_objectives.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif predicate == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2]
                self.all_cameras.add(parts[1])
                self.all_objectives.add(parts[2])
            elif predicate == 'on_board':
                self.camera_on_rover[parts[1]] = parts[2]
                self.all_cameras.add(parts[1])
                self.all_rovers.add(parts[2])
            elif predicate == 'supports':
                self.camera_modes[parts[1]].add(parts[2])
                self.all_cameras.add(parts[1])
                self.all_modes.add(parts[2])
            # Ensure all objects mentioned in static facts are added to their sets
            elif predicate in ['rover', 'waypoint', 'store', 'camera', 'mode', 'lander', 'objective']:
                 if len(parts) > 1: # Avoid predicates without objects like (:requirements)
                     if predicate == 'rover': self.all_rovers.add(parts[1])
                     elif predicate == 'waypoint': self.all_waypoints.add(parts[1])
                     elif predicate == 'store': self.all_stores.add(parts[1])
                     elif predicate == 'camera': self.all_cameras.add(parts[1])
                     elif predicate == 'mode': self.all_modes.add(parts[1])
                     elif predicate == 'lander': self.all_landers.add(parts[1])
                     elif predicate == 'objective': self.all_objectives.add(parts[1])


        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {} # rover -> (start_wp, end_wp) -> distance
        for rover in self.all_rovers:
             # Ensure all waypoints are in the graph even if no can_traverse from/to them
            graph_with_all_wps = {wp: set() for wp in self.all_waypoints}
            if rover in self.rover_traversal_graph:
                 for w_from, neighbors in self.rover_traversal_graph[rover].items():
                     graph_with_all_wps[w_from].update(neighbors)

            self.rover_shortest_paths[rover] = precompute_all_pairs_shortest_paths(
                graph_with_all_wps, self.all_waypoints
            )

        # Identify communication waypoints (visible from any lander location)
        self.communication_waypoint_sets = {} # lander -> set of visible waypoints
        self.all_communication_waypoints = set()
        for lander, lander_wp in self.lander_location.items():
             if lander_wp in self.waypoint_visibility:
                 self.communication_waypoint_sets[lander] = self.waypoint_visibility[lander_wp]
                 self.all_communication_waypoints.update(self.waypoint_visibility[lander_wp])
             else:
                 self.communication_waypoint_sets[lander] = set()


    def get_rover_location(self, state, rover):
        """Find the current waypoint of a given rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "*"):
                return get_parts(fact)[2]
        return None # Rover location not found (shouldn't happen in valid states)

    def get_dist(self, rover, wp1, wp2):
        """Get precomputed shortest distance for a rover between two waypoints."""
        if rover not in self.rover_shortest_paths:
            return float('inf')
        return self.rover_shortest_paths[rover].get((wp1, wp2), float('inf'))


    def calculate_goal_cost(self, state, goal_fact):
        """Calculate the minimum cost for a single unachieved goal fact."""
        parts = get_parts(goal_fact)
        predicate = parts[0]

        if predicate == 'communicated_soil_data':
            w = parts[1]
            # Check if already achieved
            if goal_fact in state:
                return 0

            min_goal_cost = float('inf')

            # Find any rover equipped for soil analysis
            equipped_rovers = [r for r in self.all_rovers if 'soil' in self.rover_capabilities.get(r, set())]
            if not equipped_rovers:
                return self.LARGE_COST # No rover can do this

            # Iterate through equipped rovers to find minimum cost
            for rover in equipped_rovers:
                rover_wp = self.get_rover_location(state, rover)
                if rover_wp is None: continue # Should not happen

                data_available_on_this_rover = match(f"(have_soil_analysis {rover} {w})", "have_soil_analysis", rover, w) in state

                if data_available_on_this_rover:
                    # Need to go from rover_wp to a reachable comm_wp
                    min_nav_cost = float('inf')
                    for comm_wp in self.all_communication_waypoints:
                         dist_rc = self.get_dist(rover, rover_wp, comm_wp)
                         if dist_rc != float('inf'):
                             min_nav_cost = min(min_nav_cost, dist_rc)

                    if min_nav_cost == float('inf'): continue # This rover cannot reach comm

                    cost = min_nav_cost + 1 # communicate

                else: # Need to sample
                    if not match(f"(at_soil_sample {w})", "at_soil_sample", w) in state:
                         continue # Sample is gone, this path is blocked for this rover

                    store = self.rover_stores.get(rover)
                    drop_cost = 0
                    if store and match(f"(full {store})", "full", store) in state:
                        drop_cost = 1 # Need to drop before sampling

                    # Need to go from rover_wp to w, then from w to a reachable comm_wp
                    min_nav_nav_cost = float('inf')
                    dist_rw = self.get_dist(rover, rover_wp, w)
                    if dist_rw != float('inf'):
                        for comm_wp in self.all_communication_waypoints:
                            dist_wc = self.get_dist(rover, w, comm_wp)
                            if dist_wc != float('inf'):
                                min_nav_nav_cost = min(min_nav_nav_cost, dist_rw + dist_wc)

                    if min_nav_nav_cost == float('inf'): continue # This rover cannot complete the path

                    cost = min_nav_nav_nav_cost + 1 + drop_cost + 1 # sample + drop (if needed) + communicate

                min_goal_cost = min(min_goal_cost, cost)

            return min_goal_cost if min_goal_cost != float('inf') else self.LARGE_COST

        elif predicate == 'communicated_rock_data':
            w = parts[1]
            # Check if already achieved
            if goal_fact in state:
                return 0

            min_goal_cost = float('inf')

            # Find any rover equipped for rock analysis
            equipped_rovers = [r for r in self.all_rovers if 'rock' in self.rover_capabilities.get(r, set())]
            if not equipped_rovers:
                return self.LARGE_COST # No rover can do this

            # Iterate through equipped rovers to find minimum cost
            for rover in equipped_rovers:
                rover_wp = self.get_rover_location(state, rover)
                if rover_wp is None: continue # Should not happen

                data_available_on_this_rover = match(f"(have_rock_analysis {rover} {w})", "have_rock_analysis", rover, w) in state

                if data_available_on_this_rover:
                    # Need to go from rover_wp to a reachable comm_wp
                    min_nav_cost = float('inf')
                    for comm_wp in self.all_communication_waypoints:
                         dist_rc = self.get_dist(rover, rover_wp, comm_wp)
                         if dist_rc != float('inf'):
                             min_nav_cost = min(min_nav_cost, dist_rc)

                    if min_nav_cost == float('inf'): continue # This rover cannot reach comm

                    cost = min_nav_cost + 1 # communicate

                else: # Need to sample
                    if not match(f"(at_rock_sample {w})", "at_rock_sample", w) in state:
                         continue # Sample is gone, this path is blocked for this rover

                    store = self.rover_stores.get(rover)
                    drop_cost = 0
                    if store and match(f"(full {store})", "full", store) in state:
                        drop_cost = 1 # Need to drop before sampling

                    # Need to go from rover_wp to w, then from w to a reachable comm_wp
                    min_nav_nav_cost = float('inf')
                    dist_rw = self.get_dist(rover, rover_wp, w)
                    if dist_rw != float('inf'):
                        for comm_wp in self.all_communication_waypoints:
                            dist_wc = self.get_dist(rover, w, comm_wp)
                            if dist_wc != float('inf'):
                                min_nav_nav_cost = min(min_nav_nav_cost, dist_rw + dist_wc)

                    if min_nav_nav_cost == float('inf'): continue # This rover cannot complete the path

                    cost = min_nav_nav_nav_cost + 1 + drop_cost + 1 # sample + drop (if needed) + communicate

                min_goal_cost = min(min_goal_cost, cost)

            return min_goal_cost if min_goal_cost != float('inf') else self.LARGE_COST


        elif predicate == 'communicated_image_data':
            o, m = parts[1], parts[2]
            # Check if already achieved
            if goal_fact in state:
                return 0

            min_goal_cost = float('inf')

            # Find any rover equipped for imaging that has a camera supporting the mode
            capable_rovers = []
            for r in self.all_rovers:
                if 'imaging' in self.rover_capabilities.get(r, set()):
                    for cam in self.all_cameras:
                        if self.camera_on_rover.get(cam) == r and m in self.camera_modes.get(cam, set()):
                            capable_rovers.append((r, cam))
                            break # Found a capable camera for this rover

            if not capable_rovers:
                return self.LARGE_COST # No rover can do this

            # Find observation waypoints for the objective
            obs_wps = self.objective_visibility.get(o, set())
            if not obs_wps:
                 return self.LARGE_COST # Cannot observe objective

            # Iterate through capable rovers to find minimum cost
            for rover, cam in capable_rovers:
                rover_wp = self.get_rover_location(state, rover)
                if rover_wp is None: continue # Should not happen

                image_available_on_this_rover = match(f"(have_image {rover} {o} {m})", "have_image", rover, o, m) in state

                if image_available_on_this_rover:
                    # Need to go from rover_wp to a reachable comm_wp
                    min_nav_cost = float('inf')
                    for comm_wp in self.all_communication_waypoints:
                         dist_rc = self.get_dist(rover, rover_wp, comm_wp)
                         if dist_rc != float('inf'):
                             min_nav_cost = min(min_nav_cost, dist_rc)

                    if min_nav_cost == float('inf'): continue # This rover cannot reach comm

                    cost = min_nav_cost + 1 # communicate

                else: # Need to take image
                    calibrated_available_on_this_rover = match(f"(calibrated {cam} {rover})", "calibrated", cam, rover) in state

                    if calibrated_available_on_this_rover:
                        # Need to go from rover_wp to an obs_wp, then from obs_wp to a reachable comm_wp
                        min_nav_nav_cost = float('inf')
                        for obs_wp in obs_wps:
                            dist_ro = self.get_dist(rover, rover_wp, obs_wp)
                            if dist_ro != float('inf'):
                                for comm_wp in self.all_communication_waypoints:
                                    dist_oc = self.get_dist(rover, obs_wp, comm_wp)
                                    if dist_oc != float('inf'):
                                        min_nav_nav_cost = min(min_nav_nav_cost, dist_ro + dist_oc)

                        if min_nav_nav_cost == float('inf'): continue # This rover cannot complete the path

                        cost = min_nav_nav_nav_cost + 1 + 1 # take_image + communicate

                    else: # Need to calibrate and take image
                        cal_target = self.calibration_targets.get(cam)
                        if cal_target is None: continue # Camera has no calibration target

                        cal_wps = self.objective_visibility.get(cal_target, set())
                        if not cal_wps: continue # No waypoint to calibrate from

                        # Need to go from rover_wp to a cal_wp, then from cal_wp to an obs_wp, then from obs_wp to a reachable comm_wp
                        min_nav_nav_nav_cost = float('inf')
                        for cal_wp in cal_wps:
                            dist_rc = self.get_dist(rover, rover_wp, cal_wp)
                            if dist_rc != float('inf'):
                                for obs_wp in obs_wps:
                                    dist_co = self.get_dist(rover, cal_wp, obs_wp)
                                    if dist_co != float('inf'):
                                        for comm_wp in self.all_communication_waypoints:
                                            dist_oc = self.get_dist(rover, obs_wp, comm_wp)
                                            if dist_oc != float('inf'):
                                                min_nav_nav_nav_cost = min(min_nav_nav_nav_cost, dist_rc + dist_co + dist_oc)

                        if min_nav_nav_nav_nav_cost == float('inf'): continue # This rover cannot complete the path

                        cost = min_nav_nav_nav_nav_cost + 1 + 1 + 1 # calibrate + take_image + communicate

                min_goal_cost = min(min_goal_cost, cost)

            return min_goal_cost if min_goal_cost != float('inf') else self.LARGE_COST

        else:
            # Unknown goal type, ignore or return large cost
            # This case should ideally not be reached if goals are only communication facts
            return 0


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        total_heuristic = 0

        # Calculate cost for each unachieved goal
        for goal in self.goals:
            if goal not in state:
                goal_cost = self.calculate_goal_cost(state, goal)
                # If any goal is unreachable, the total heuristic should indicate this
                if goal_cost >= self.LARGE_COST:
                    return self.LARGE_COST # Problem likely unsolvable from here
                total_heuristic += goal_cost

        return total_heuristic
