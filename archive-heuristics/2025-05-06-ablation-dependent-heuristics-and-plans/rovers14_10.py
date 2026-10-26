from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: '(at rover1 waypoint1)' -> ['at', 'rover1', 'waypoint1']
    # Example: '(communicated_image_data objective1 high_res)' -> ['communicated_image_data', 'objective1', 'high_res']
    
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Not a valid PDDL fact string format expected
        return []
        
    # Remove outer parentheses
    content = fact[1:-1].strip()
    if not content:
        return [] # Empty fact like '()'

    # Split by spaces
    return content.split()


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

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest paths from a start node
    in a graph represented as an adjacency dictionary.
    Returns a dictionary {end_node: distance}.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    ungratified goal conditions. It sums the minimum estimated cost for each
    individual ungratified goal, assuming the best available rover and path
    can be used for each goal independently. The cost includes navigation
    (estimated by shortest path distance on the traverse graph), sampling,
    imaging, calibration, dropping, and communication actions.

    # Assumptions
    - The cost of each action (navigate, sample, image, calibrate, drop, communicate) is 1.
    - Navigation cost between waypoints is the shortest path distance on the
      `can_traverse` graph for the specific rover.
    - For sampling goals, if the analysis is not held, it needs to be sampled.
      If the sample is not currently at the waypoint but was initially, it is
      assumed to require re-sampling (implying it was sampled and dropped).
    - For imaging goals, if the image is not held, it needs to be taken.
      Taking an image requires calibration if the camera is not currently calibrated.
    - The heuristic assumes the best-equipped and best-positioned rover/camera
      combination is available for each ungratified goal independently.
    - It ignores negative interactions between goals (e.g., one rover sampling
      consumes the sample for another, taking an image uncalibrates the camera,
      filling a store prevents further sampling). This makes the heuristic
      non-admissible but potentially more informative than a simple goal count.
    - Assumes all necessary static conditions (like equipped status, camera
      on_board/supports/calibration_target, objective/target visibility) are met
      for at least one rover/camera combination if the goal is achievable.
      If no such combination exists or required waypoints are unreachable,
      the heuristic returns infinity for that goal component.
    - A simplified cost of 1 is added for drop actions if any sampling is needed
      and any equipped rover for that sample type has a full store. This avoids
      complex reasoning about which specific rover samples or drops.

    # Heuristic Initialization
    - Parses all objects by type from the task's facts, initial state, and goals
      (using a heuristic approach based on predicate arguments).
    - Extracts static facts such as rover capabilities, store ownership,
      lander location, waypoint visibility (`visible`), and rover traverse
      capabilities (`can_traverse`).
    - Extracts initial state facts such as soil/rock sample locations.
    - Builds navigation graphs for each rover based on `can_traverse` facts.
    - Computes all-pairs shortest paths for each rover's navigation graph
      using BFS.
    - Identifies communication waypoints (visible from the lander location).
    - Pre-calculates potential image tasks: for each objective/mode goal,
      finds all possible (rover, camera, image_waypoint, calibration_waypoint)
      combinations that could fulfill the image requirement based on static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Extract current state information: rover locations, held samples/images,
       calibrated cameras, full stores.
    3. Identify all goal conditions that are not yet satisfied in the current state.
    4. Add a simplified cost for drop actions: If any soil sampling is needed
       (ungratified soil goal and analysis not held) AND any equipped soil rover
       has a full store, add 1 to `h`. Do the same for rock sampling.
    5. For each ungratified goal (soil, rock, image):
       - If the required data (analysis or image) is already held by a rover:
         - Add 1 (communicate) + minimum navigation cost for that rover
           from its current location to any communication waypoint.
       - If the required data is NOT held:
         - For soil/rock: Need to sample and communicate. Find the equipped rover
           that minimizes the total navigation cost (current location -> sample
           waypoint -> communication waypoint). Add this minimum navigation cost
           + 1 (sample) + 1 (communicate). If no equipped rover can reach the
           necessary points, the goal is unreachable (h=inf).
         - For image: Need to take image and communicate. Find the (rover, camera,
           image waypoint, calibration waypoint) combination that minimizes the
           total cost. The cost for a combo depends on whether calibration is needed
           from the current state. Add the minimum cost over all valid combos.
           If no combo can reach the necessary points, the goal is unreachable (h=inf).
    6. Sum up the costs for all ungratified goals. If any goal is unreachable,
       the total heuristic is infinity.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Parse objects by type (using fallback heuristic method)
        # This is a heuristic approach as PDDL does not strictly require type facts
        # in a specific format in the initial state or static facts.
        all_objects = set()
        for fact in static_facts | initial_state | self.goals:
             parts = get_parts(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects

        self.objects = {
            'rover': set(), 'waypoint': set(), 'store': set(),
            'camera': set(), 'mode': set(), 'lander': set(), 'objective': set()
        }
        # Heuristically assign types based on common predicates
        for obj in all_objects:
            # Check common predicates to guess type
            if any(match(f, '*', obj, '*') or match(f, '*', '*', obj, '*') for f in static_facts | initial_state | self.goals if f.startswith('(at ')):
                 if obj.startswith('rover'): self.objects['rover'].add(obj)
                 elif obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
                 elif obj.startswith('lander'): self.objects['lander'].add(obj)
            if any(match(f, '*', obj) for f in static_facts | initial_state | self.goals if f.startswith('(equipped_for_')):
                 if obj.startswith('rover'): self.objects['rover'].add(obj)
            if any(match(f, '*', obj, '*') for f in static_facts | initial_state | self.goals if f.startswith('(store_of ')):
                 if obj.startswith('store'): self.objects['store'].add(obj)
                 if obj.startswith('rover'): self.objects['rover'].add(obj)
            if any(match(f, '*', obj, '*') for f in static_facts | initial_state | self.goals if f.startswith('(on_board ')):
                 if obj.startswith('camera'): self.objects['camera'].add(obj)
                 if obj.startswith('rover'): self.objects['rover'].add(obj)
            if any(match(f, '*', obj) for f in static_facts | initial_state | self.goals if f.startswith('(supports ')):
                 if obj.startswith('camera'): self.objects['camera'].add(obj)
                 # Modes are often just names like high_res, low_res, colour
                 if obj in ['high_res', 'low_res', 'colour']: self.objects['mode'].add(obj)
            if any(match(f, '*', obj, '*') for f in static_facts | initial_state | self.goals if f.startswith('(calibration_target ')):
                 if obj.startswith('camera'): self.objects['camera'].add(obj)
                 if obj.startswith('objective'): self.objects['objective'].add(obj)
            if any(match(f, '*', obj, '*') for f in static_facts | initial_state | self.goals if f.startswith('(visible_from ')):
                 if obj.startswith('objective'): self.objects['objective'].add(obj)
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
            if any(match(f, '*', obj) for f in static_facts | initial_state | self.goals if f.startswith('(at_soil_sample ')):
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
            if any(match(f, '*', obj) for f in static_facts | initial_state | self.goals if f.startswith('(at_rock_sample ')):
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
            if any(match(f, '*', obj) for f in self.goals if f.startswith('(communicated_soil_data ')):
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
            if any(match(f, '*', obj) for f in self.goals if f.startswith('(communicated_rock_data ')):
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
            if any(match(f, '*', obj, '*') for f in self.goals if f.startswith('(communicated_image_data ')):
                 if obj.startswith('objective'): self.objects['objective'].add(obj)
                 if obj in ['high_res', 'low_res', 'colour']: self.objects['mode'].add(obj) # Special case for modes
            if any(match(f, '*', obj, '*', '*') for f in static_facts if f.startswith('(can_traverse ')):
                 if obj.startswith('rover'): self.objects['rover'].add(obj)
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)
            if any(match(f, '*', obj, '*') for f in static_facts if f.startswith('(visible ')):
                 if obj.startswith('waypoint'): self.objects['waypoint'].add(obj)


        # 2. Extract static facts into useful structures
        self.rover_capabilities = {r: set() for r in self.objects.get('rover', [])}
        for fact in static_facts:
             if match(fact, 'equipped_for_soil_analysis', r):
                  self.rover_capabilities[r].add(fact)
             elif match(fact, 'equipped_for_rock_analysis', r):
                  self.rover_capabilities[r].add(fact)
             elif match(fact, 'equipped_for_imaging', r):
                  self.rover_capabilities[r].add(fact)

        self.rover_stores = {r: s for f in static_facts if match(f, 'store_of', s, r) for s, r in [get_parts(f)[1:]]}

        self.lander_location = next((wp for f in static_facts if match(f, 'at_lander', '*', wp) for _, wp in [get_parts(f)[1:]]), None)

        # Communication waypoints are those visible *from* the lander location
        self.comm_waypoints = {w1 for f in static_facts if match(f, 'visible', w1, self.lander_location)}
        # Add lander location itself if visible from itself (unlikely but safe)
        if self.lander_location and f'(visible {self.lander_location} {self.lander_location})' in static_facts:
             self.comm_waypoints.add(self.lander_location)


        self.rover_navigation_graphs = {}
        self.rover_shortest_paths = {}
        all_waypoints = set(self.objects.get('waypoint', []))
        for r in self.objects.get('rover', []):
             graph = {}
             for fact in static_facts:
                  if match(fact, 'can_traverse', r_check, w1, w2) and r_check == r:
                       graph.setdefault(w1, set()).add(w2)
                       graph.setdefault(w2, set()).add(w1) # Assuming traverse is symmetric if visible is symmetric

             # Ensure all waypoints are nodes in the graph, even if isolated
             for wp in all_waypoints:
                  graph.setdefault(wp, set())

             self.rover_navigation_graphs[r] = graph
             self.rover_shortest_paths[r] = {}
             for start_wp in graph:
                  self.rover_shortest_paths[r][start_wp] = bfs(graph, start_wp)

        self.initial_soil_sample_locations = {w for f in initial_state if match(f, 'at_soil_sample', w)}
        self.initial_rock_sample_locations = {w for f in initial_state if match(f, 'at_rock_sample', w)}

        # 3. Pre-calculate potential image tasks
        self.image_tasks = {} # {(o, m): set of (r, i, p_image, p_calib)}

        on_board = {(i, r) for f in static_facts if match(f, 'on_board', i, r)}
        supports = {(i, m) for f in static_facts if match(f, 'supports', i, m)}
        calibration_target = {(i, t) for f in static_facts if match(f, 'calibration_target', i, t)}
        visible_from = {(obj_or_target, w) for f in static_facts if match(f, 'visible_from', obj_or_target, w)}

        for r in self.objects.get('rover', []):
             if f'(equipped_for_imaging {r})' not in self.rover_capabilities.get(r, set()): continue
             for i in self.objects.get('camera', []):
                  if (i, r) not in on_board: continue
                  calib_t = next((t for cam, t in calibration_target if cam == i), None)
                  if not calib_t: continue # Camera needs a calibration target

                  calib_wps = {w for (target, w) in visible_from if target == calib_t}
                  if not calib_wps: continue # Calibration target must be visible from somewhere

                  for m in self.objects.get('mode', []):
                       if (i, m) not in supports: continue
                       for o in self.objects.get('objective', []):
                            image_wps = {w for (obj, w) in visible_from if obj == o}
                            if not image_wps: continue # Objective must be visible from somewhere

                            # Add all valid combinations
                            for p_image in image_wps:
                                 for p_calib in calib_wps:
                                      self.image_tasks.setdefault((o, m), set()).add((r, i, p_image, p_calib))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0
        unreachable = False

        # 1. Extract current state information
        current_rover_locations = {}
        for r in self.objects.get('rover', []):
             loc = next((wp for f in state if match(f, 'at', r_check, wp) and r_check == r), None)
             if loc:
                  current_rover_locations[r] = loc

        held_soil = {(r, w) for f in state if match(f, 'have_soil_analysis', r, w)}
        held_rock = {(r, w) for f in state if match(f, 'have_rock_analysis', r, w)}
        held_image = {(r, o, m) for f in state if match(f, 'have_image', r, o, m)}
        calibrated_cameras = {(i, r) for f in state if match(f, 'calibrated', i, r)}
        full_stores = {s for f in state if match(f, 'full', s)}

        # 2. Identify ungratified goals
        ungratified_soil_goals = {w for f in self.goals if match(f, 'communicated_soil_data', w) and f not in state}
        ungratified_rock_goals = {w for f in self.goals if match(f, 'communicated_rock_data', w) and f not in state}
        ungratified_image_goals = {(o, m) for f in self.goals if match(f, 'communicated_image_data', o, m) and f not in state}

        # Simplified Drop Cost Logic: Add 1 if any sampling is needed and any equipped rover has a full store
        needs_soil_sample_action = any(w in ungratified_soil_goals and (r, w) not in held_soil for w in ungratified_soil_goals for r in self.objects.get('rover', []))
        needs_rock_sample_action = any(w in ungratified_rock_goals and (r, w) not in held_rock for w in ungratified_rock_goals for r in self.objects.get('rover', []))

        equipped_soil_rovers = {r for r in self.objects.get('rover', []) if f'(equipped_for_soil_analysis {r})' in self.rover_capabilities.get(r, set())}
        equipped_rock_rovers = {r for r in self.objects.get('rover', []) if f'(equipped_for_rock_analysis {r})' in self.rover_capabilities.get(r, set())}

        any_equipped_soil_rover_store_full = any(self.rover_stores.get(r) in full_stores for r in equipped_soil_rovers)
        any_equipped_rock_rover_store_full = any(self.rover_stores.get(r) in full_stores for r in equipped_rock_rovers)

        if needs_soil_sample_action and any_equipped_soil_rover_store_full:
             h += 1 # Add cost for one drop action for soil sampling needs
        if needs_rock_sample_action and any_equipped_rock_rover_store_full:
             h += 1 # Add cost for one drop action for rock sampling needs


        # 3. Calculate cost for each ungratified goal (excluding drop, handled above)
        for w in ungratified_soil_goals:
            # Is analysis held?
            rover_holding = next((r for r in self.objects.get('rover', []) if (r, w) in held_soil), None)

            if rover_holding:
                # Held, need to communicate
                r = rover_holding
                current_loc = current_rover_locations.get(r)
                if not current_loc: # Rover location unknown? Should not happen in valid state.
                     unreachable = True; break

                min_nav_to_comm = float('inf')
                if current_loc in self.rover_shortest_paths.get(r, {}):
                     for comm_wp in self.comm_waypoints:
                          if comm_wp in self.rover_shortest_paths[r][current_loc]:
                              min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[r][current_loc][comm_wp])

                if min_nav_to_comm == float('inf'):
                     unreachable = True; break
                h += min_nav_to_comm + 1 # Nav + Communicate
            else:
                # Not held, need to sample and communicate
                # Find best equipped rover
                min_cost_sample_comm = float('inf')

                # Check if sample exists initially. If not, goal is impossible.
                if w not in self.initial_soil_sample_locations:
                     unreachable = True; break

                for r in self.objects.get('rover', []):
                     if f'(equipped_for_soil_analysis {r})' in self.rover_capabilities.get(r, set()):
                          current_loc = current_rover_locations.get(r)
                          if not current_loc: continue # Rover location unknown

                          # Nav to sample location w
                          nav_to_sample = float('inf')
                          if current_loc in self.rover_shortest_paths.get(r, {}) and w in self.rover_shortest_paths[r][current_loc]:
                               nav_to_sample = self.rover_shortest_paths[r][current_loc][w]
                          if nav_to_sample == float('inf'): continue # Cannot reach sample point

                          # Nav from sample location w to comm location
                          nav_sample_to_comm = float('inf')
                          if w in self.rover_shortest_paths.get(r, {}):
                               for comm_wp in self.comm_waypoints:
                                    if comm_wp in self.rover_shortest_paths[r][w]:
                                         nav_sample_to_comm = min(nav_sample_to_comm, self.rover_shortest_paths[r][w][comm_wp])
                              if nav_sample_to_comm == float('inf'): continue # Cannot reach comm point from sample point

                          cost_this_rover = nav_to_sample + 1 + nav_sample_to_comm + 1 # Nav + Sample + Nav + Communicate

                          min_cost_sample_comm = min(min_cost_sample_comm, cost_this_rover)

                if min_cost_sample_comm == float('inf'):
                     unreachable = True; break
                h += min_cost_sample_comm

        if unreachable: return float('inf')

        for w in ungratified_rock_goals:
            # Similar logic for rock data
            rover_holding = next((r for r in self.objects.get('rover', []) if (r, w) in held_rock), None)

            if rover_holding:
                r = rover_holding
                current_loc = current_rover_locations.get(r)
                if not current_loc:
                     unreachable = True; break

                min_nav_to_comm = float('inf')
                if current_loc in self.rover_shortest_paths.get(r, {}):
                     for comm_wp in self.comm_waypoints:
                          if comm_wp in self.rover_shortest_paths[r][current_loc]:
                              min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[r][current_loc][comm_wp])

                if min_nav_to_comm == float('inf'):
                     unreachable = True; break
                h += min_nav_to_comm + 1
            else:
                min_cost_sample_comm = float('inf')

                if w not in self.initial_rock_sample_locations:
                     unreachable = True; break

                for r in self.objects.get('rover', []):
                     if f'(equipped_for_rock_analysis {r})' in self.rover_capabilities.get(r, set()):
                          current_loc = current_rover_locations.get(r)
                          if not current_loc: continue

                          nav_to_sample = float('inf')
                          if current_loc in self.rover_shortest_paths.get(r, {}) and w in self.rover_shortest_paths[r][current_loc]:
                               nav_to_sample = self.rover_shortest_paths[r][current_loc][w]
                          if nav_to_sample == float('inf'): continue

                          nav_sample_to_comm = float('inf')
                          if w in self.rover_shortest_paths.get(r, {}):
                               for comm_wp in self.comm_waypoints:
                                    if comm_wp in self.rover_shortest_paths[r][w]:
                                         nav_sample_to_comm = min(nav_sample_to_comm, self.rover_shortest_paths[r][w][comm_wp])
                              if nav_sample_to_comm == float('inf'): continue

                          cost_this_rover = nav_to_sample + 1 + nav_sample_to_comm + 1

                          min_cost_sample_comm = min(min_cost_sample_comm, cost_this_rover)

                if min_cost_sample_comm == float('inf'):
                     unreachable = True; break
                h += min_cost_sample_comm

        if unreachable: return float('inf')

        for o, m in ungratified_image_goals:
            # Is image held?
            rover_holding = next((r for r in self.objects.get('rover', []) if (r, o, m) in held_image), None)

            if rover_holding:
                r = rover_holding
                current_loc = current_rover_locations.get(r)
                if not current_loc:
                     unreachable = True; break

                min_nav_to_comm = float('inf')
                if current_loc in self.rover_shortest_paths.get(r, {}):
                     for comm_wp in self.comm_waypoints:
                          if comm_wp in self.rover_shortest_paths[r][current_loc]:
                              min_nav_to_comm = min(min_nav_to_comm, self.rover_shortest_paths[r][current_loc][comm_wp])

                if min_nav_to_comm == float('inf'):
                     unreachable = True; break
                h += min_nav_to_comm + 1
            else:
                # Not held, need to take image and communicate
                min_cost_image_comm = float('inf')

                # Iterate through pre-calculated possible combos
                for r, i, p_image, p_calib in self.image_tasks.get((o, m), []):
                     current_loc = current_rover_locations.get(r)
                     if not current_loc: continue # Rover location unknown

                     cost_this_combo = 0

                     needs_calibration = (i, r) not in calibrated_cameras

                     if needs_calibration:
                         # Nav to calibration point p_calib
                         nav_to_calib = float('inf')
                         if current_loc in self.rover_shortest_paths.get(r, {}) and p_calib in self.rover_shortest_paths[r][current_loc]:
                              nav_to_calib = self.rover_shortest_paths[r][current_loc][p_calib]
                         if nav_to_calib == float('inf'): continue # Cannot reach calib point
                         cost_this_combo += nav_to_calib + 1 # Nav + Calibrate
                         loc_after_calib = p_calib

                         # Nav from calib point to image point p_image
                         nav_calib_to_image = float('inf')
                         if loc_after_calib in self.rover_shortest_paths.get(r, {}) and p_image in self.rover_shortest_paths[r][loc_after_calib]:
                              nav_calib_to_image = min(nav_calib_to_image, self.rover_shortest_paths[r][loc_after_calib][p_image]) # Use min in case multiple paths
                         if nav_calib_to_image == float('inf'): continue # Cannot reach image point from calib point
                         cost_this_combo += nav_calib_to_image + 1 # Nav + TakeImage
                         loc_after_image = p_image

                     else: # Already calibrated
                         # Nav to image point p_image
                         nav_to_image = float('inf')
                         if current_loc in self.rover_shortest_paths.get(r, {}) and p_image in self.rover_shortest_paths[r][current_loc]:
                              nav_to_image = self.rover_shortest_paths[r][current_loc][p_image]
                         if nav_to_image == float('inf'): continue # Cannot reach image point
                         cost_this_combo += nav_to_image + 1 # Nav + TakeImage
                         loc_after_image = p_image

                     # Nav from image point to comm location
                     nav_image_to_comm = float('inf')
                     if loc_after_image in self.rover_shortest_paths.get(r, {}):
                          for comm_wp in self.comm_waypoints:
                               if comm_wp in self.rover_shortest_paths[r][loc_after_image]:
                                   nav_image_to_comm = min(nav_image_to_comm, self.rover_shortest_paths[r][loc_after_image][comm_wp])
                     if nav_image_to_comm == float('inf'): continue # Cannot reach comm point from image point
                     cost_this_combo += nav_image_to_comm + 1 # Nav + Communicate

                     min_cost_image_comm = min(min_cost_image_comm, cost_this_combo)

                if min_cost_image_comm == float('inf'):
                     unreachable = True; break
                h += min_cost_image_comm

        if unreachable:
             return float('inf')

        return h
