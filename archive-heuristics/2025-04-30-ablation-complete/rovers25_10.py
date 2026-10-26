from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic
from typing import Set, Dict, List, Tuple, Any, Optional

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact: str) -> List[str]:
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string gracefully
    if not fact or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact: str, *args: str) -> bool:
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern,
    # unless wildcards make the length flexible (which fnmatch handles per part).
    # A simple check for basic pattern structure:
    if len(parts) != len(args):
         return False
    # Check if each part matches the corresponding argument pattern using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Graph building and BFS for shortest paths
def bfs(graph: Dict[str, Set[str]], start_node: str, all_nodes: Set[str]) -> Dict[str, float]:
    """
    Computes shortest path distances from start_node to all other nodes using BFS.

    Args:
        graph: Adjacency list representation of the directed graph {node: {neighbor1, neighbor2, ...}}.
        start_node: The starting node for the BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary {node: distance} containing shortest distances.
        Distance is float('inf') if a node is unreachable.
    """
    distances: Dict[str, float] = {node: float('inf') for node in all_nodes}

    # Ensure the start node is valid and in the set of nodes we care about
    if start_node not in all_nodes:
        return distances # Cannot start from an unknown node

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # If current_node is not in the graph keys, it has no outgoing edges
        if current_node not in graph:
            continue

        for neighbor in graph[current_node]:
            # Ensure neighbor is a node we care about
            if neighbor in all_nodes and distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each uncommunicated goal independently. It considers the steps needed
    (sampling/imaging, calibrating, communicating) and estimates movement costs
    using precomputed shortest paths on the navigation graph for each rover.
    It finds the minimum cost across suitable rovers and waypoints for each step.

    # Assumptions
    - The heuristic assumes that each uncommunicated goal can be achieved independently
      of others (ignores resource conflicts like multiple rovers needing the same
      waypoint or store).
    - It assumes a rover that samples/images data is the one that communicates it,
      or at least that the data/image is associated with a specific rover via
      `(have_soil_analysis r w)`, `(have_rock_analysis r w)`, or `(have_image r o m)`.
      If the state fact exists but no specific rover is listed (e.g., due to simplified
      state representation or domain features not fully captured), it falls back
      to assuming any capable rover can perform the communication from its current location.
    - It assumes store capacity and state (empty/full) can be ignored for sampling
      actions (i.e., a rover can always sample if equipped and at the location).
    - It assumes camera calibration is a prerequisite for taking an image and adds
      a cost for it if the image goal is not yet achieved. It doesn't track
      calibration state explicitly per camera/rover in the heuristic calculation
      beyond the initial state, effectively assuming recalibration is needed before
      each image action for an uncollected image goal.
    - Movement cost between waypoints is estimated by the shortest path distance
      in the rover's navigation graph (each step is 1 action).
    - Action costs for sample, drop, calibrate, take_image, communicate are assumed to be 1.

    # Heuristic Initialization
    - Extracts static facts from the task, including:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Camera information (on board, supported modes, calibration target).
        - Waypoint visibility for communication, imaging, and calibration.
        - Rover navigation graph (can_traverse).
        - Collects all declared objects of relevant types (rover, waypoint, etc.).
    - Precomputes all-pairs shortest paths for each rover on its navigation graph
      using BFS, considering all known waypoints as potential nodes.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal condition
    that is not yet satisfied in the current state.

    For each goal `g` in the task's goals:
    1.  If `g` is already true in the current state, its cost contribution is 0.
    2.  If `g` is `(communicated_soil_data w)` and not in state:
        a.  Check if `(have_soil_analysis r w)` is true for any rover `r`.
        b.  If data is collected (`have_soil_analysis` is true for some `r`):
            - Find all rovers `r_has_data` that have the data (based on state facts).
            - Find the minimum cost across these rovers:
                - Get the rover's current location `loc_r`.
                - Find the minimum cost to move `r_has_data` from `loc_r` to any waypoint `comm_wp` visible from the lander, plus 1 for the `communicate_soil_data` action.
            - If no specific rover is found with the data, fall back to finding the minimum communication cost across *any* soil-equipped rover from its current location.
            - Add this minimum cost to the total heuristic.
        c.  If data is not collected (`have_soil_analysis` is false for all rovers):
            - Find the minimum cost across all soil-equipped rovers `r_soil`:
                - Cost to move `r_soil` from its current location `loc_r` to `w`, plus 1 for the `sample_soil` action.
                - Plus the minimum cost to move `r_soil` from `w` (where it is after sampling) to any waypoint `comm_wp` visible from the lander, plus 1 for the `communicate_soil_data` action.
            - Add the minimum total cost over all suitable `r_soil` to the heuristic.
    3.  If `g` is `(communicated_rock_data w)` and not in state:
        - Follow the same logic as for soil data, substituting rock-specific predicates and equipment.
    4.  If `g` is `(communicated_image_data o m)` and not in state:
        a.  Check if `(have_image r o m)` is true for any rover `r`.
        b.  If image is collected (`have_image` is true for some `r`):
            - Find all rovers `r_has_image` that have the image (based on state facts).
            - Find the minimum cost across these rovers:
                - Get the rover's current location `loc_r`.
                - Find the minimum cost to move `r_has_image` from `loc_r` to any waypoint `comm_wp` visible from the lander, plus 1 for the `communicate_image_data` action.
            - If no specific rover is found with the image, fall back to finding the minimum communication cost across *any* imaging-equipped rover with a suitable camera/mode from its current location.
            - Add this minimum cost to the heuristic.
        c.  If image is not collected (`have_image` is false for all rovers):
            - Find the minimum cost across all imaging-equipped rovers `r_img` that have a camera `i` supporting mode `m`:
                - Find calibration target `t` for camera `i`.
                - Find calibration waypoints `cal_wps` visible from `t`.
                - Find imaging waypoints `img_wps` visible from `o`.
                - Find communication waypoints `comm_wps` visible from the lander.
                - Find the minimum total cost for this rover/camera combination:
                    - Cost to move from `loc_r` to any `cal_wp`, plus 1 (calibrate).
                    - Plus cost to move from that `cal_wp` to any `img_wp`, plus 1 (take_image).
                    - Plus cost to move from that `img_wp` to any `comm_wp`, plus 1 (communicate).
                - The minimum total cost for this rover/camera is the minimum over all combinations of `cal_wp`, `img_wp`, and `comm_wp`.
            - Add the minimum total cost over all suitable `r_img`/`i` combinations to the heuristic.

    The total heuristic value is the sum of the minimum costs calculated for each uncommunicated goal. If any necessary path or resource is unavailable (e.g., no equipped rover, no path, no visible waypoint), the cost for that goal component becomes infinity, propagating to the total heuristic.
    """

    def __init__(self, task: Any):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state

        # Data structures to store static information
        self.lander_location: Optional[str] = None
        self.rovers: Set[str] = set()
        self.waypoints: Set[str] = set() # Collect all waypoints declared or mentioned
        self.objectives: Set[str] = set()
        self.modes: Set[str] = set()
        self.cameras: Set[str] = set()
        self.stores: Set[str] = set()
        self.landers: Set[str] = set() # Added landers type

        self.rover_equipment: Dict[str, Set[str]] = defaultdict(set) # {rover: {soil, rock, imaging}}
        self.rover_stores: Dict[str, str] = {} # {rover: store}
        self.rover_cameras: Dict[str, Set[str]] = defaultdict(set) # {rover: {camera}}
        self.camera_modes: Dict[str, Set[str]] = defaultdict(set) # {camera: {mode}}
        self.camera_calibration_target: Dict[str, str] = {} # {camera: objective}

        self.visible_waypoints: Dict[str, Set[str]] = defaultdict(set) # {waypoint: {visible_waypoint}}
        self.objective_visible_from: Dict[str, Set[str]] = defaultdict(set) # {objective: {waypoint}}
        self.calibration_target_visible_from: Dict[str, Set[str]] = defaultdict(set) # {objective: {waypoint}}

        # Store can_traverse facts per rover for graph building
        rover_can_traverse_facts: Dict[str, List[str]] = defaultdict(list)

        # Collect all objects by type first from static facts (usually includes object declarations)
        for fact in static_facts:
             parts = get_parts(fact)
             if len(parts) == 2: # Might be a type declaration like '(rover rover1)'
                  obj_type, obj_name = parts
                  if obj_type == 'rover': self.rovers.add(obj_name)
                  elif obj_type == 'waypoint': self.waypoints.add(obj_name)
                  elif obj_type == 'objective': self.objectives.add(obj_name)
                  elif obj_type == 'mode': self.modes.add(obj_name)
                  elif obj_type == 'camera': self.cameras.add(obj_name)
                  elif obj_type == 'store': self.stores.add(obj_name)
                  elif obj_type == 'lander': self.landers.add(obj_name)


        # Now parse static facts again for predicates and relationships
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            pred = parts[0]

            if pred == 'at_lander':
                if len(parts) > 2:
                    self.lander_location = parts[2]
                    if parts[1] not in self.landers: self.landers.add(parts[1])
                    if parts[2] not in self.waypoints: self.waypoints.add(parts[2])
            elif pred == 'equipped_for_soil_analysis':
                if len(parts) > 1:
                    self.rover_equipment[parts[1]].add('soil')
                    if parts[1] not in self.rovers: self.rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                if len(parts) > 1:
                    self.rover_equipment[parts[1]].add('rock')
                    if parts[1] not in self.rovers: self.rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                if len(parts) > 1:
                    self.rover_equipment[parts[1]].add('imaging')
                    if parts[1] not in self.rovers: self.rovers.add(parts[1])
            elif pred == 'store_of':
                if len(parts) > 2:
                    self.rover_stores[parts[2]] = parts[1] # rover -> store
                    if parts[1] not in self.stores: self.stores.add(parts[1])
                    if parts[2] not in self.rovers: self.rovers.add(parts[2])
            elif pred == 'on_board':
                if len(parts) > 2:
                    self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
                    if parts[1] not in self.cameras: self.cameras.add(parts[1])
                    if parts[2] not in self.rovers: self.rovers.add(parts[2])
            elif pred == 'supports':
                if len(parts) > 2:
                    self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
                    if parts[1] not in self.cameras: self.cameras.add(parts[1])
                    if parts[2] not in self.modes: self.modes.add(parts[2])
            elif pred == 'calibration_target':
                if len(parts) > 2:
                    self.camera_calibration_target[parts[1]] = parts[2] # camera -> objective (target)
                    if parts[1] not in self.cameras: self.cameras.add(parts[1])
                    if parts[2] not in self.objectives: self.objectives.add(parts[2])
            elif pred == 'visible':
                if len(parts) > 2:
                    self.visible_waypoints[parts[1]].add(parts[2])
                    if parts[1] not in self.waypoints: self.waypoints.add(parts[1])
                    if parts[2] not in self.waypoints: self.waypoints.add(parts[2])
            elif pred == 'can_traverse':
                 if len(parts) > 3:
                    rover_name = parts[1]
                    rover_can_traverse_facts[rover_name].append(fact)
                    if parts[1] not in self.rovers: self.rovers.add(parts[1])
                    if parts[2] not in self.waypoints: self.waypoints.add(parts[2])
                    if parts[3] not in self.waypoints: self.waypoints.add(parts[3])
            # visible_from is handled after calibration_targets are fully known

        # Collect calibration targets again after parsing all static facts
        calibration_targets_set = set(self.camera_calibration_target.values())

        # Parse visible_from using the collected calibration targets
        for fact in static_facts:
             if match(fact, 'visible_from', '*', '*'):
                  parts = get_parts(fact)
                  if len(parts) > 2:
                      obj_or_target = parts[1]
                      waypoint = parts[2]
                      if waypoint not in self.waypoints: self.waypoints.add(waypoint)
                      # Check if obj_or_target is a known objective or calibration target
                      if obj_or_target in calibration_targets_set:
                           self.calibration_target_visible_from[obj_or_target].add(waypoint)
                      elif obj_or_target in self.objectives:
                           self.objective_visible_from[obj_or_target].add(waypoint)
                      # If obj_or_target is not in self.objectives but is mentioned here,
                      # it might be an objective not explicitly declared but used.
                      # Add it to objectives for robustness.
                      elif obj_or_target.startswith('objective'): # Simple check based on naming convention
                           self.objectives.add(obj_or_target)
                           self.objective_visible_from[obj_or_target].add(waypoint)


        # Compute shortest paths for each rover
        for rover in self.rovers:
            # Build the specific directed graph for this rover using can_traverse
            rover_graph: Dict[str, Set[str]] = defaultdict(set)
            for fact in rover_can_traverse_facts[rover]:
                 parts = get_parts(fact)
                 if len(parts) > 3:
                      rover_graph[parts[2]].add(parts[3])

            # Compute BFS from each waypoint to all other waypoints
            for start_wp in self.waypoints:
                 self.rover_shortest_paths[rover][start_wp] = bfs(rover_graph, start_wp, self.waypoints)


    def get_shortest_path(self, rover: str, start_wp: str, end_wp: str) -> float:
        """Retrieves precomputed shortest path distance."""
        # Check if rover, start, and end waypoints are valid and path was computed
        if rover not in self.rover_shortest_paths or \
           start_wp not in self.rover_shortest_paths[rover] or \
           end_wp not in self.rover_shortest_paths[rover][start_wp]:
             # This indicates an issue in graph building or BFS, or an invalid waypoint
             # For robustness, return infinity
             return float('inf')

        return self.rover_shortest_paths[rover][start_wp][end_wp]


    def __call__(self, node: Any) -> float:
        """Compute an estimate of the minimal number of required actions."""
        state: frozenset[str] = node.state
        h: float = 0

        # 1. Parse current state to get relevant facts
        rover_locations: Dict[str, str] = {} # {rover: waypoint}
        soil_data: Set[str] = set() # {waypoint} # Stores waypoints where soil data is collected (global fact)
        rock_data: Set[str] = set() # {waypoint} # Stores waypoints where rock data is collected (global fact)
        images: Set[Tuple[str, str]] = set() # {(objective, mode)} # Stores (objective, mode) for collected images (global fact)
        communicated_soil: Set[str] = set() # {waypoint}
        communicated_rock: Set[str] = set() # {waypoint}
        communicated_images: Set[Tuple[str, str]] = set() # {(objective, mode)}

        # Store which rover has which data/image (based on state facts)
        rover_soil_data: Dict[str, Set[str]] = defaultdict(set) # {rover: {waypoint}}
        rover_rock_data: Dict[str, Set[str]] = defaultdict(set) # {rover: {waypoint}}
        rover_images: Dict[str, Set[Tuple[str, str]]] = defaultdict(set) # {rover: {(objective, mode)}}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) > 2 and parts[1] in self.rovers:
                rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis' and len(parts) > 2:
                rover_soil_data[parts[1]].add(parts[2]) # rover -> waypoint
                soil_data.add(parts[2]) # just the waypoint
            elif pred == 'have_rock_analysis' and len(parts) > 2:
                rover_rock_data[parts[1]].add(parts[2]) # rover -> waypoint
                rock_data.add(parts[2]) # just the waypoint
            elif pred == 'have_image' and len(parts) > 3:
                rover_images[parts[1]].add((parts[2], parts[3])) # rover -> (objective, mode)
                images.add((parts[2], parts[3])) # just the (objective, mode)
            elif pred == 'communicated_soil_data' and len(parts) > 1:
                communicated_soil.add(parts[1]) # Add waypoint
            elif pred == 'communicated_rock_data' and len(parts) > 1:
                communicated_rock.add(parts[1]) # Add waypoint
            elif pred == 'communicated_image_data' and len(parts) > 2:
                communicated_images.add((parts[1], parts[2])) # Add (objective, mode)


        # Find communication waypoints visible *from* the lander location
        lander_comm_wps: Set[str] = set()
        if self.lander_location:
             # The communicate action requires (visible ?x ?y) where ?x is rover loc and ?y is lander loc.
             # So we need waypoints ?x such that (visible ?x lander_location) is true.
             for wp in self.waypoints:
                  if wp in self.visible_waypoints and self.lander_location in self.visible_waypoints[wp]:
                       lander_comm_wps.add(wp)


        # 2. Calculate cost for uncommunicated soil goals
        uncomm_soil_goals = {get_parts(g)[1] for g in self.goals if match(g, 'communicated_soil_data', '*')} - communicated_soil
        for w in uncomm_soil_goals:
            if w not in soil_data: # Need to sample and communicate
                min_sample_comm_cost = float('inf')
                # Find suitable rovers
                suitable_rovers = [r for r in self.rovers if 'soil' in self.rover_equipment.get(r, set())]
                if not suitable_rovers:
                    h += float('inf') # Cannot sample if no equipped rover
                    continue

                for r_soil in suitable_rovers:
                    loc_r = rover_locations.get(r_soil)
                    if loc_r is None: continue # Rover location unknown? Should not happen in valid state.

                    cost_to_sample = self.get_shortest_path(r_soil, loc_r, w)
                    if cost_to_sample == float('inf'): continue # Cannot reach sample location

                    cost_to_sample += 1 # sample_soil action

                    # After sampling, rover is at 'w'. Need to communicate.
                    min_comm_cost_after_sample = float('inf')
                    if not lander_comm_wps: # No waypoints visible from lander
                         min_comm_cost_after_sample = float('inf')
                    else:
                        for comm_wp in lander_comm_wps:
                            cost = self.get_shortest_path(r_soil, w, comm_wp)
                            if cost != float('inf'):
                                min_comm_cost_after_sample = min(min_comm_cost_after_sample, cost + 1) # communicate action

                    if min_comm_cost_after_sample != float('inf'):
                         total_cost_this_rover = cost_to_sample + min_comm_cost_after_sample
                         min_sample_comm_cost = min(min_sample_comm_cost, total_cost_this_rover)

                h += min_sample_comm_cost

            else: # Already sampled, need to communicate
                min_comm_cost = float('inf')
                # Find all rovers that have the data. Check rover_soil_data mapping.
                rovers_with_data = [r for r, waypoints in rover_soil_data.items() if w in waypoints]

                if not rovers_with_data:
                     # Fallback: Assume any soil-equipped rover can communicate it from its current location
                     suitable_rovers = [r for r in self.rovers if 'soil' in self.rover_equipment.get(r, set())]
                     if not suitable_rovers:
                          h += float('inf')
                          continue

                     min_cost_across_rovers = float('inf')
                     for r_comm in suitable_rovers:
                          loc_r = rover_locations.get(r_comm)
                          if loc_r is None: continue

                          if not lander_comm_wps:
                               min_cost_this_rover = float('inf')
                          else:
                               min_cost_this_rover = float('inf')
                               for comm_wp in lander_comm_wps:
                                    cost = self.get_shortest_path(r_comm, loc_r, comm_wp)
                                    if cost != float('inf'):
                                         min_cost_this_rover = min(min_cost_this_rover, cost + 1)
                          min_cost_across_rovers = min(min_cost_across_rovers, min_cost_this_rover)
                     h += min_cost_across_rovers

                else: # Found specific rovers with the data
                    min_cost_across_rovers = float('inf')
                    for rover_with_data in rovers_with_data:
                        loc_r = rover_locations.get(rover_with_data)
                        if loc_r is None: continue

                        min_comm_cost_this_rover = float('inf')
                        if not lander_comm_wps:
                             min_comm_cost_this_rover = float('inf')
                        else:
                            for comm_wp in lander_comm_wps:
                                cost = self.get_shortest_path(rover_with_data, loc_r, comm_wp)
                                if cost != float('inf'):
                                    min_comm_cost_this_rover = min(min_comm_cost_this_rover, cost + 1)
                        min_cost_across_rovers = min(min_cost_across_rovers, min_comm_cost_this_rover)
                    h += min_cost_across_rovers


        # 3. Calculate cost for uncommunicated rock goals
        uncomm_rock_goals = {get_parts(g)[1] for g in self.goals if match(g, 'communicated_rock_data', '*')} - communicated_rock
        for w in uncomm_rock_goals:
            if w not in rock_data: # Need to sample and communicate
                min_sample_comm_cost = float('inf')
                suitable_rovers = [r for r in self.rovers if 'rock' in self.rover_equipment.get(r, set())]
                if not suitable_rovers:
                    h += float('inf') # Cannot sample if no equipped rover
                    continue

                for r_rock in suitable_rovers:
                    loc_r = rover_locations.get(r_rock)
                    if loc_r is None: continue

                    cost_to_sample = self.get_shortest_path(r_rock, loc_r, w)
                    if cost_to_sample == float('inf'): continue

                    cost_to_sample += 1 # sample_rock action

                    min_comm_cost_after_sample = float('inf')
                    if not lander_comm_wps:
                         min_comm_cost_after_sample = float('inf')
                    else:
                        for comm_wp in lander_comm_wps:
                            cost = self.get_shortest_path(r_rock, w, comm_wp)
                            if cost != float('inf'):
                                min_comm_cost_after_sample = min(min_comm_cost_after_sample, cost + 1) # communicate action

                    if min_comm_cost_after_sample != float('inf'):
                         total_cost_this_rover = cost_to_sample + min_comm_cost_after_sample
                         min_sample_comm_cost = min(min_sample_comm_cost, total_cost_this_rover)

                h += min_sample_comm_cost

            else: # Already sampled, need to communicate
                min_comm_cost = float('inf')
                # Find all rovers that have the data. Check rover_rock_data mapping.
                rovers_with_data = [r for r, waypoints in rover_rock_data.items() if w in waypoints]

                if not rovers_with_data:
                     # Fallback: Assume any rock-equipped rover can communicate it
                     suitable_rovers = [r for r in self.rovers if 'rock' in self.rover_equipment.get(r, set())]
                     if not suitable_rovers:
                          h += float('inf')
                          continue

                     min_cost_across_rovers = float('inf')
                     for r_comm in suitable_rovers:
                          loc_r = rover_locations.get(r_comm)
                          if loc_r is None: continue

                          if not lander_comm_wps:
                               min_cost_this_rover = float('inf')
                          else:
                               min_cost_this_rover = float('inf')
                               for comm_wp in lander_comm_wps:
                                    cost = self.get_shortest_path(r_comm, loc_r, comm_wp)
                                    if cost != float('inf'):
                                         min_cost_this_rover = min(min_cost_this_rover, cost + 1)
                          min_cost_across_rovers = min(min_cost_across_rovers, min_cost_this_rover)
                     h += min_cost_across_rovers
                else: # Found specific rovers
                    min_cost_across_rovers = float('inf')
                    for rover_with_data in rovers_with_data:
                        loc_r = rover_locations.get(rover_with_data)
                        if loc_r is None: continue

                        min_comm_cost_this_rover = float('inf')
                        if not lander_comm_wps:
                             min_comm_cost_this_rover = float('inf')
                        else:
                            for comm_wp in lander_comm_wps:
                                cost = self.get_shortest_path(rover_with_data, loc_r, comm_wp)
                                if cost != float('inf'):
                                    min_comm_cost_this_rover = min(min_comm_cost_this_rover, cost + 1)
                        min_cost_across_rovers = min(min_cost_across_rovers, min_comm_cost_this_rover)
                    h += min_cost_across_rovers


        # 4. Calculate cost for uncommunicated image goals
        uncomm_image_goals = {(get_parts(g)[1], get_parts(g)[2]) for g in self.goals if match(g, 'communicated_image_data', '*', '*')} - communicated_images
        for (o, m) in uncomm_image_goals:
            if (o, m) not in images: # Need to image and communicate
                min_image_comm_cost = float('inf')
                # Find suitable rovers/cameras
                suitable_rover_cameras: List[Tuple[str, str]] = [] # List of (rover, camera) tuples
                for r in self.rovers:
                    if 'imaging' in self.rover_equipment.get(r, set()):
                        for cam in self.rover_cameras.get(r, set()):
                            if m in self.camera_modes.get(cam, set()):
                                suitable_rover_cameras.append((r, cam))

                if not suitable_rover_cameras:
                    h += float('inf') # Cannot image if no equipped rover/camera/mode
                    continue

                for r_img, i in suitable_rover_cameras:
                    loc_r = rover_locations.get(r_img)
                    if loc_r is None: continue

                    # Find calibration target for camera i
                    t = self.camera_calibration_target.get(i)
                    if t is None: continue # Camera has no calibration target

                    # Find calibration waypoints visible from target t
                    cal_wps = self.calibration_target_visible_from.get(t, set())
                    if not cal_wps: continue # No waypoints to calibrate from

                    # Find imaging waypoints visible from objective o
                    img_wps = self.objective_visible_from.get(o, set())
                    if not img_wps: continue # No waypoints to image from

                    # Find communication waypoints visible from lander
                    if not lander_comm_wps: continue # No waypoints to communicate from

                    # Calculate min cost for this rover/camera combination
                    min_cost_this_rover_camera = float('inf')
                    for cal_wp in cal_wps:
                        cost_to_cal = self.get_shortest_path(r_img, loc_r, cal_wp)
                        if cost_to_cal == float('inf'): continue

                        for img_wp in img_wps:
                            cost_cal_to_img = self.get_shortest_path(r_img, cal_wp, img_wp)
                            if cost_cal_to_img == float('inf'): continue

                            for comm_wp in lander_comm_wps:
                                cost_img_to_comm = self.get_shortest_path(r_img, img_wp, comm_wp)
                                if cost_img_to_comm == float('inf'): continue

                                # Total cost: move(loc_r -> cal_wp) + calibrate + move(cal_wp -> img_wp) + take_image + move(img_wp -> comm_wp) + communicate
                                total_path_actions = cost_to_cal + cost_cal_to_img + cost_img_to_comm
                                total_action_cost = 1 + 1 + 1 # calibrate, take_image, communicate
                                min_cost_this_rover_camera = min(min_cost_this_rover_camera, total_path_actions + total_action_cost)

                    min_image_comm_cost = min(min_image_comm_cost, min_cost_this_rover_camera)

                h += min_image_comm_cost

            else: # Already imaged, need to communicate
                min_comm_cost = float('inf')
                # Find all rovers that have the image. Check rover_images mapping.
                rovers_with_image = [r for r, images_set in rover_images.items() if (o, m) in images_set]

                if not rovers_with_image:
                     # Fallback: Assume any imaging-equipped rover with suitable camera/mode can communicate it
                     suitable_rovers = [r for r in self.rovers if 'imaging' in self.rover_equipment.get(r, set())]
                     suitable_rover_cameras: List[Tuple[str, str]] = []
                     for r in suitable_rovers:
                          for cam in self.rover_cameras.get(r, set()):
                               if m in self.camera_modes.get(cam, set()):
                                    suitable_rover_cameras.append((r, cam))

                     if not suitable_rover_cameras:
                          h += float('inf')
                          continue

                     min_cost_across_rovers = float('inf')
                     for r_comm, _ in suitable_rover_cameras: # We only need the rover name
                          loc_r = rover_locations.get(r_comm)
                          if loc_r is None: continue

                          if not lander_comm_wps:
                               min_cost_this_rover = float('inf')
                          else:
                               min_cost_this_rover = float('inf')
                               for comm_wp in lander_comm_wps:
                                    cost = self.get_shortest_path(r_comm, loc_r, comm_wp)
                                    if cost != float('inf'):
                                         min_cost_this_rover = min(min_cost_this_rover, cost + 1)
                          min_cost_across_rovers = min(min_cost_across_rovers, min_cost_this_rover)
                     h += min_cost_across_rovers

                else: # Found specific rovers
                    min_cost_across_rovers = float('inf')
                    for rover_with_image in rovers_with_image:
                        loc_r = rover_locations.get(rover_with_image)
                        if loc_r is None: continue

                        min_comm_cost_this_rover = float('inf')
                        if not lander_comm_wps:
                             min_comm_cost_this_rover = float('inf')
                        else:
                            for comm_wp in lander_comm_wps:
                                cost = self.get_shortest_path(rover_with_image, loc_r, comm_wp)
                                if cost != float('inf'):
                                    min_comm_cost_this_rover = min(min_comm_cost_this_rover, cost + 1)
                        min_cost_across_rovers = min(min_cost_across_rovers, min_comm_cost_this_rover)
                    h += min_cost_across_rovers


        # Ensure heuristic is 0 only at goal state
        # The logic sums costs for *unmet* goals. If h is 0, it means all goals were met.
        # The goal_reached check is the definitive source of truth.
        # If not goal, h should be > 0 unless the problem is truly unsolvable from here (h=inf).
        # Returning h directly is correct.
        return h
