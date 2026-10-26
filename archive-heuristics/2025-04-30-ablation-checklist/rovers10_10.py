# Add necessary imports
from fnmatch import fnmatch
from collections import deque, defaultdict
# Assuming heuristics.heuristic_base exists and defines a Heuristic base class
# from heuristics.heuristic_base import Heuristic

# Define dummy Heuristic base class if not provided in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not implemented.")


def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed facts, or raise error
        # print(f"Warning: Malformed fact string: {fact}")
        return []
    # Split by space, but handle potential empty strings from multiple spaces
    parts = fact[1:-1].split()
    return [part for part in parts if part] # Filter out empty strings


def match(fact, *args):
    """Check if a PDDL fact string matches a given pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False # Cannot match if number of parts differs
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Compute shortest distances from start_node to all other nodes in graph."""
    # Collect all nodes in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # If start_node is not in the graph nodes, it's isolated from the graph structure
    if start_node not in all_nodes:
         # Return distances assuming only the start node is reachable with cost 0
         distances = {node: float('inf') for node in all_nodes}
         distances[start_node] = 0 # Can reach itself with cost 0
         return distances


    distances = {node: float('inf') for node in all_nodes}
    distances[start_node] = 0
    queue = deque([start_node])

    # Use visited set for efficiency, especially in dense graphs or graphs with cycles
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Check if current_node is a valid key in the graph (it should be if in all_nodes and not isolated)
        if current_node not in graph:
             continue # Should not happen based on all_nodes construction

        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated minimum actions
    required for each unachieved goal fact independently.

    The heuristic considers the steps needed for each goal type:
    - Soil/Rock Data: Navigate to sample, sample, navigate to lander, communicate.
    - Image Data: Navigate to calibration target, calibrate, navigate to objective,
      take image, navigate to lander, communicate.

    Navigation costs are precomputed using BFS on the rover-specific traversal graphs.
    The heuristic is non-admissible as it ignores negative interactions (resource
    contention) and positive interactions (shared travel paths, achieving multiple
    subgoals with one action).
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing navigation costs and static info."""
        # Store goal facts
        self.goals = task.goals

        # Parse static facts to build necessary lookup structures
        static_facts = set(task.static)

        self.lander_locations = {get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")}

        # Build the full visibility graph
        self.visible_graph = defaultdict(list)
        all_waypoints_set = set() # Collect all waypoints mentioned anywhere

        # Collect waypoints from static predicates
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 2 and parts[1] and parts[2]:
                    wp1, wp2 = parts[1:3]
                    self.visible_graph[wp1].append(wp2)
                    all_waypoints_set.add(wp1)
                    all_waypoints_set.add(wp2)
            elif match(fact, "at_lander", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2 and parts[2]: all_waypoints_set.add(parts[2])
            elif match(fact, "at_soil_sample", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 1 and parts[1]: all_waypoints_set.add(parts[1])
            elif match(fact, "at_rock_sample", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 1 and parts[1]: all_waypoints_set.add(parts[1])
            elif match(fact, "visible_from", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2 and parts[2]: all_waypoints_set.add(parts[2])

        # Collect all rovers from static facts
        self.rovers = set()
        for fact in static_facts:
             if match(fact, "equipped_for_*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 1 and parts[1]: self.rovers.add(parts[1])
             elif match(fact, "on_board", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2 and parts[2]: self.rovers.add(parts[2])
             elif match(fact, "store_of", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2 and parts[2]: self.rovers.add(parts[2])

        # Add rovers and their initial locations from initial state
        initial_rover_locations = {}
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2 and parts[1] and parts[2]:
                     rover, wp = parts[1:3]
                     self.rovers.add(rover)
                     all_waypoints_set.add(wp)
                     initial_rover_locations[rover] = wp # Store initial locations for BFS starts

        # Ensure all waypoints from traversal facts are included
        for fact in static_facts:
             if match(fact, "can_traverse", "*", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 3 and parts[2] and parts[3]:
                     _, rover, wp1, wp2 = parts
                     all_waypoints_set.add(wp1)
                     all_waypoints_set.add(wp2)


        self.waypoints = list(all_waypoints_set)

        # Compute navigation costs for each rover
        self.rover_nav_costs = defaultdict(dict)
        for rover in self.rovers:
             # Build the actual graph used for navigation: intersection of visible and can_traverse
             nav_graph = defaultdict(list)
             rover_can_traverse_edges = set() # Store (wp1, wp2) tuples this rover can traverse
             for fact in static_facts:
                 if match(fact, "can_traverse", rover, "*", "*"):
                     parts = get_parts(fact)
                     if len(parts) > 3 and parts[2] and parts[3]:
                         _, r, wp1, wp2 = parts
                         rover_can_traverse_edges.add((wp1, wp2))

             # Populate nav_graph based on visible and can_traverse
             rover_graph_nodes = set()
             for wp1 in self.waypoints:
                 if wp1 in self.visible_graph:
                     for wp2 in self.visible_graph[wp1]:
                         if (wp1, wp2) in rover_can_traverse_edges:
                              nav_graph[wp1].append(wp2)
                              rover_graph_nodes.add(wp1)
                              rover_graph_nodes.add(wp2)

             # Add initial location if it's not already in the graph nodes
             initial_loc = initial_rover_locations.get(rover)
             if initial_loc:
                 rover_graph_nodes.add(initial_loc)

             # Run BFS from every waypoint that is part of this rover's graph
             for start_wp in rover_graph_nodes:
                 self.rover_nav_costs[rover][start_wp] = bfs(nav_graph, start_wp)


        # Identify communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for lander_loc in self.lander_locations:
            # Find all waypoints X such that (visible X lander_loc) is true
            for wp1 in self.visible_graph:
                if lander_loc in self.visible_graph[wp1]:
                    self.comm_waypoints.add(wp1)

        # Store equipment, cameras, modes, calibration targets, visible_from locations
        self.soil_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")}
        self.rock_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")}
        self.imaging_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")}
        self.rover_stores = {get_parts(fact)[2]: get_parts(fact)[1] for fact in static_facts if match(fact, "store_of", "*", "*")} # rover -> store name (assuming one store per rover based on examples)

        self.rover_cameras = defaultdict(list) # rover -> [camera]
        for fact in static_facts:
             if match(fact, "on_board", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     camera, rover = parts[1:3]
                     self.rover_cameras[rover].append(camera)

        self.camera_modes = defaultdict(list) # camera -> [mode]
        for fact in static_facts:
             if match(fact, "supports", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     camera, mode = parts[1:3]
                     self.camera_modes[camera].append(mode)

        self.camera_cal_targets = {} # camera -> target
        for fact in static_facts:
             if match(fact, "calibration_target", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     camera, target = parts[1:3]
                     self.camera_cal_targets[camera] = target

        self.objective_visible_wps = defaultdict(list) # objective -> [waypoint]
        for fact in static_facts:
             if match(fact, "visible_from", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     objective, waypoint = parts[1:3]
                     self.objective_visible_wps[objective].append(waypoint)

        # Store initial sample locations (they disappear when sampled)
        # We need to know if a sample *was* available initially at a waypoint
        self.initial_soil_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_soil_sample", "*")}
        self.initial_rock_samples = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_rock_sample", "*")}


    def get_nav_cost(self, rover, start_wp, end_wp):
        """Get precomputed navigation cost, handle unreachable."""
        # Check if rover exists and start/end waypoints are in its navigation graph results
        if rover not in self.rover_nav_costs:
             return float('inf')
        if start_wp not in self.rover_nav_costs[rover]:
             # Rover cannot start from this waypoint in its graph
             return float('inf')
        # The BFS result already contains float('inf') for unreachable end_wps
        return self.rover_nav_costs[rover][start_wp].get(end_wp, float('inf'))


    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        h = 0

        # Get current state facts for quick lookup
        state_facts = set(state)

        # Get current rover locations
        rover_locations = {}
        for fact in state_facts:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 2:
                    _, rover, wp = parts
                    rover_locations[rover] = wp

        # Get current store status (assuming one store per rover)
        rover_store_empty = {} # rover -> True/False
        for rover, store in self.rover_stores.items():
             rover_store_empty[rover] = f"(empty {store})" in state_facts

        # Get current sample/image status (which rover has which sample/image)
        rover_have_soil = {} # (rover, waypoint) -> True
        for fact in state_facts:
             if match(fact, "have_soil_analysis", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     _, r, w = parts
                     rover_have_soil[(r, w)] = True

        rover_have_rock = {} # (rover, waypoint) -> True
        for fact in state_facts:
             if match(fact, "have_rock_analysis", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     _, r, w = parts
                     rover_have_rock[(r, w)] = True

        rover_have_image = {} # (rover, objective, mode) -> True
        for fact in state_facts:
             if match(fact, "have_image", "*", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 3:
                     _, r, o, m = parts
                     rover_have_image[(r, o, m)] = True

        # Get current calibration status
        calibrated_cameras = {} # (camera, rover) -> True
        for fact in state_facts:
             if match(fact, "calibrated", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 2:
                     _, c, r = parts
                     calibrated_cameras[(c, r)] = True


        # Process each unachieved goal
        for goal in self.goals:
            if goal in state_facts:
                continue # Goal already achieved

            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                if not args: continue # Malformed goal
                waypoint_w = args[0]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal with minimum cost
                for rover in self.soil_rovers:
                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue # Rover not found in state?

                    current_rover_cost = float('inf')

                    # Check if this rover already has the sample analysis from W
                    if (rover, waypoint_w) in rover_have_soil:
                        # Need to communicate the sample
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.get_nav_cost(rover, current_rover_loc, comm_wp)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost != float('inf'):
                             current_rover_cost = min_comm_nav_cost + 1 # +1 for communicate action

                    else:
                        # Need to sample and communicate
                        # Assumption: Sample is available if it was in initial state and this specific goal is not yet communicated.
                        if waypoint_w not in self.initial_soil_samples:
                             continue # Sample was never here initially, cannot be sampled

                        # Check if rover has an empty store. If not, assume 1 drop action needed.
                        store_needed_cost = 0
                        if rover in self.rover_stores and not rover_store_empty.get(rover, True):
                             store_needed_cost = 1 # Cost to drop

                        # Cost to get to sample location W
                        nav_to_sample_cost = self.get_nav_cost(rover, current_rover_loc, waypoint_w)
                        if nav_to_sample_cost == float('inf'): continue

                        # Cost to sample
                        sample_cost = 1

                        # Cost to get from sample location W to a communication waypoint X
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.get_nav_cost(rover, waypoint_w, comm_wp)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost == float('inf'): continue

                        # Cost to communicate
                        communicate_cost = 1

                        current_rover_cost = store_needed_cost + nav_to_sample_cost + sample_cost + min_comm_nav_cost + communicate_cost

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost

            elif predicate == "communicated_rock_data":
                if not args: continue # Malformed goal
                waypoint_w = args[0]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal with minimum cost
                for rover in self.rock_rovers:
                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue

                    current_rover_cost = float('inf')

                    # Check if this rover already has the sample analysis from W
                    if (rover, waypoint_w) in rover_have_rock:
                        # Need to communicate the sample
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.get_nav_cost(rover, current_rover_loc, comm_wp)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost != float('inf'):
                             current_rover_cost = min_comm_nav_cost + 1 # +1 for communicate action

                    else:
                        # Need to sample and communicate
                        # Assumption: Sample is available if it was in initial state and this specific goal is not yet communicated.
                        if waypoint_w not in self.initial_rock_samples:
                             continue # Sample was never here initially

                        # Check if rover has an empty store. If not, assume 1 drop action needed.
                        store_needed_cost = 0
                        if rover in self.rover_stores and not rover_store_empty.get(rover, True):
                             store_needed_cost = 1 # Cost to drop

                        # Cost to get to sample location W
                        nav_to_sample_cost = self.get_nav_cost(rover, current_rover_loc, waypoint_w)
                        if nav_to_sample_cost == float('inf'): continue

                        # Cost to sample
                        sample_cost = 1

                        # Cost to get from sample location W to a communication waypoint X
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.get_nav_cost(rover, waypoint_w, comm_wp)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost == float('inf'): continue

                        # Cost to communicate
                        communicate_cost = 1

                        current_rover_cost = store_needed_cost + nav_to_sample_cost + sample_cost + min_comm_nav_cost + communicate_cost

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost


            elif predicate == "communicated_image_data":
                if len(args) < 2: continue # Malformed goal
                objective_o, mode_m = args
                min_goal_cost = float('inf')

                # Find a rover/camera combination that can achieve this goal with minimum cost
                for rover in self.imaging_rovers:
                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue

                    # Check if this rover already has the image
                    if (rover, objective_o, mode_m) in rover_have_image:
                        # Need to communicate the image
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.comm_waypoints:
                            nav_cost = self.get_nav_cost(rover, current_rover_loc, comm_wp)
                            min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)

                        if min_comm_nav_cost != float('inf'):
                             current_rover_cost = min_comm_nav_cost + 1 # +1 for communicate action
                             min_goal_cost = min(min_goal_cost, current_rover_cost)

                    else:
                        # Need to take image and communicate
                        suitable_cameras = [c for c in self.rover_cameras.get(rover, []) if mode_m in self.camera_modes.get(c, [])]

                        for camera_c in suitable_cameras:
                            cal_target_t = self.camera_cal_targets.get(camera_c)
                            if cal_target_t is None: continue

                            cal_wps = self.objective_visible_wps.get(cal_target_t, [])
                            image_wps = self.objective_visible_wps.get(objective_o, [])
                            comm_wps = list(self.comm_waypoints)

                            if not cal_wps or not image_wps or not comm_wps: continue

                            # Find min cost R_loc -> W + W -> P + P -> X over all W, P, X
                            min_path_cost_for_rover_camera = float('inf')

                            # Precompute min Nav(p, x) for all p in ImgWPs to any x in CommWPs
                            min_nav_p_to_any_x_for_p = {} # p -> min_nav_cost_from_p_to_any_comm_wp
                            for p in image_wps:
                                min_nav_p_to_any_x_for_p[p] = float('inf')
                                for x in comm_wps:
                                     nav_p_x = self.get_nav_cost(rover, p, x)
                                     min_nav_p_to_any_x_for_p[p] = min(min_nav_p_to_any_x_for_p[p], nav_p_x)


                            # Now find the minimum total path cost R_loc -> W -> P -> X
                            for w in cal_wps:
                                nav_r_w = self.get_nav_cost(rover, current_rover_loc, w)
                                if nav_r_w == float('inf'): continue

                                for p in image_wps:
                                    nav_w_p = self.get_nav_cost(rover, w, p)
                                    if nav_w_p == float('inf'): continue

                                    min_nav_p_to_any_x = min_nav_p_to_any_x_for_p.get(p, float('inf'))
                                    if min_nav_p_to_any_x == float('inf'): continue

                                    current_path_cost = nav_r_w + nav_w_p + min_nav_p_to_any_x
                                    min_path_cost_for_rover_camera = min(min_path_cost_for_rover_camera, current_path_cost)


                            if min_path_cost_for_rover_camera != float('inf'):
                                # Total cost for this rover/camera = min_path_cost + 3 actions (calibrate, take, communicate)
                                current_rover_camera_cost = min_path_cost_for_rover_camera + 3
                                min_goal_cost = min(min_goal_cost, current_rover_camera_cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost

        return h
