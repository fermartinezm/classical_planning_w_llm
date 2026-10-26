import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure we have enough parts to match the pattern args
    if len(parts) < len(args):
         return False
    # Use zip, it stops at the shortest sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Compute shortest path distances from start_node in the graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph: # Handle cases where start_node is isolated or not in graph
        return distances
    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Use graph.get(current_node, []) to handle nodes with no outgoing edges
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal conditions.
    It sums the estimated minimum cost for each unachieved goal fact independently.
    The cost for collecting data (soil, rock, image) includes navigation to the sample/image location,
    potential store dropping, sampling/imaging (including calibration if needed).
    The cost for communicating data includes navigation to a communication waypoint
    (visible from the lander) and the communication action.
    Navigation costs are estimated using precomputed shortest path distances on the waypoint graph.

    # Assumptions
    - All necessary objects (rovers, cameras, stores, objectives, targets, lander) exist and are functional as defined in the static facts.
    - Waypoints required for goals (sample locations, image locations, calibration locations, communication locations) are reachable from rover starting positions via the 'visible' graph.
    - Any rover equipped for a task can perform it.
    - Any camera on a rover supporting a mode can be used for imaging.
    - The waypoint graph built from 'visible' facts represents traversable links for rovers. The domain file suggests 'can_traverse' facts align with 'visible' links for equipped rovers. We assume any equipped rover can traverse any 'visible' link.
    - The heuristic ignores resource conflicts (e.g., multiple rovers needing the same store or camera simultaneously).
    - The heuristic assumes that if a sample/image goal requires data from a specific waypoint/objective, that sample/image source exists at that location initially if the goal is not yet met.

    # Heuristic Initialization
    - Parses static facts to identify rovers, waypoints, stores, cameras, modes, landers, objectives, their properties (equipment, store ownership, camera capabilities, calibration targets), objective/target visibility from waypoints, and lander location.
    - Builds the waypoint graph based on 'visible' facts.
    - Precomputes all-pairs shortest path distances between waypoints using BFS on the waypoint graph.
    - Identifies communication waypoints (waypoints visible from the lander location).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost to 0.
    2.  Parse the current state to get the location of each rover, the status of stores (empty/full), collected data (have_soil_analysis, have_rock_analysis, have_image), communicated data (communicated_soil_data, communicated_rock_data, communicated_image_data), and camera calibration status.
    3.  Identify all goal facts that are not yet satisfied in the current state.
    4.  For each unachieved goal fact:
        *   Estimate the minimum cost to achieve *only* this goal fact, independent of other goals.
        *   This minimum cost is calculated by considering all possible rovers/cameras that could achieve the goal and finding the minimum cost path/sequence of actions for any single suitable agent.
        *   **For a `(communicated_soil_data ?w)` goal:**
            *   If `(have_soil_analysis r w)` is false for all equipped rovers `r`:
                *   Estimate cost to sample: Find an equipped rover `r`. If its store is full, add 1 (for `drop`). Add navigation cost from the rover's current location to `w`. Add 1 (for `sample_soil`). The rover is now at `w`.
                *   Estimate cost to communicate: Add navigation cost from `w` to the nearest communication waypoint. Add 1 (for `communicate_soil_data`).
                *   The total cost for sampling and communicating is the sum of these estimates. Minimize this sum over all equipped rovers.
            *   If `(have_soil_analysis r w)` is true for some equipped rover `r`:
                *   Estimate cost to communicate: Add navigation cost from the rover's current location to the nearest communication waypoint. Add 1 (for `communicate_soil_data`). Minimize this over all equipped rovers holding the data.
            *   The estimated cost for this goal is the minimum of the above cases (if data is held, only communication is needed). If no equipped rover can achieve the goal (e.g., no path), the cost is infinite.
        *   **For a `(communicated_rock_data ?w)` goal:**
            *   Analogous calculation to the soil data goal.
        *   **For a `(communicated_image_data ?o ?m)` goal:**
            *   If `(have_image r o m)` is false for all suitable rover/camera pairs `(r, i)`:
                *   Estimate cost to take image: Find an equipped rover `r` and camera `i` on `r` supporting mode `m`. Find an imaging waypoint `img_w` for objective `o`. Add navigation cost from the rover's current location to `img_w`. If camera `i` is not calibrated, find a calibration target `t` for `i` and a calibration waypoint `cal_w` for `t`. Add navigation cost from `img_w` to `cal_w`, add 1 (for `calibrate`), add navigation cost from `cal_w` back to `img_w`. Add 1 (for `take_image`). The rover is now at `img_w`. Minimize this cost over all suitable `(r, i)` pairs and all valid `img_w`/`cal_w` pairs.
                *   Estimate cost to communicate: Add navigation cost from `img_w` to the nearest communication waypoint. Add 1 (for `communicate_image_data`).
                *   The total cost for imaging and communicating is the sum of these estimates. Minimize this sum over all suitable `(r, i)` pairs and waypoints.
            *   If `(have_image r o m)` is true for some equipped rover `r`:
                *   Estimate cost to communicate: Add navigation cost from the rover's current location to the nearest communication waypoint. Add 1 (for `communicate_image_data`). Minimize this over all equipped rovers holding the image data.
            *   The estimated cost for this goal is the minimum of the above cases. If no equipped rover/camera can achieve the goal, the cost is infinite.
        *   If the estimated cost for any unachieved goal is infinite, the total heuristic cost is infinite.
    5.  Sum the estimated costs for all unachieved goal facts to get the total heuristic value.
    6.  Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Parse Static Facts ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()

        self.rover_store = {} # rover -> store
        self.rover_cameras = {} # rover -> set(cameras)
        self.camera_supports = {} # camera -> set(modes)
        self.camera_cal_target = {} # camera -> target

        self.objective_visible_from = {} # objective -> set(waypoints)
        # Calibration targets are objectives, so they are included in objective_visible_from

        self.lander_location = None
        self.waypoint_graph = {} # waypoint -> set(neighbors)

        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'rover': self.rovers.add(parts[1])
            elif predicate == 'waypoint': self.waypoints.add(parts[1])
            elif predicate == 'store': self.stores.add(parts[1])
            elif predicate == 'camera': self.cameras.add(parts[1])
            elif predicate == 'mode': self.modes.add(parts[1])
            elif predicate == 'lander': self.landers.add(parts[1])
            elif predicate == 'objective': self.objectives.add(parts[1])
            elif predicate == 'equipped_for_soil_analysis': self.equipped_soil.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis': self.equipped_rock.add(parts[1])
            elif predicate == 'equipped_for_imaging': self.equipped_imaging.add(parts[1])
            elif predicate == 'store_of': self.rover_store[parts[2]] = parts[1] # rover -> store
            elif predicate == 'on_board': self.rover_cameras.setdefault(parts[2], set()).add(parts[1]) # rover -> camera
            elif predicate == 'supports': self.camera_supports.setdefault(parts[1], set()).add(parts[2])
            elif predicate == 'calibration_target': self.camera_cal_target[parts[1]] = parts[2]
            elif predicate == 'visible_from':
                obj_or_target, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(obj_or_target, set()).add(waypoint)
            elif predicate == 'at_lander': self.lander_location = parts[2]
            elif predicate == 'visible':
                w1, w2 = parts[1], parts[2]
                self.waypoint_graph.setdefault(w1, set()).add(w2)
                self.waypoint_graph.setdefault(w2, set()).add(w1)

        # --- Precompute Shortest Path Distances ---
        self.dist = {}
        for w in self.waypoints:
            self.dist[w] = bfs(self.waypoint_graph, w)

        # --- Identify Communication Waypoints ---
        self.comm_waypoints = set()
        if self.lander_location and self.lander_location in self.waypoint_graph:
             # A waypoint is a communication waypoint if it's visible from the lander location.
             # The waypoint_graph is built from visible facts, so neighbors of lander_location
             # in the graph are the visible waypoints.
             self.comm_waypoints = self.waypoint_graph.get(self.lander_location, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State Facts ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)
        calibrated_cameras = set() # (camera, rover)
        # soil_samples_at = set() # waypoint # Not strictly needed for heuristic logic
        # rock_samples_at = set() # waypoint # Not strictly needed for heuristic logic

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'at':
                 if parts[1] in self.rovers: # Ensure it's a rover, not lander
                    rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty': store_status[parts[1]] = 'empty'
            elif predicate == 'full': store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis': have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis': have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image': have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'communicated_soil_data': communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data': communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data': communicated_image.add((parts[1], parts[2]))
            elif predicate == 'calibrated': calibrated_cameras.add((parts[1], parts[2]))
            # elif predicate == 'at_soil_sample': soil_samples_at.add(parts[1])
            # elif predicate == 'at_rock_sample': rock_samples_at.add(parts[1])

        # --- Compute Heuristic Cost ---
        total_cost = 0

        # Goals for soil data communication
        soil_goals_todo = {w for goal in self.goals if match(goal, "communicated_soil_data", w) and w not in communicated_soil}
        for w in soil_goals_todo:
            min_goal_cost = float('inf')
            # Find if any equipped rover already has the data
            data_held_by_equipped_rovers = {r for (r_held, w_held) in have_soil if w_held == w and r_held in self.equipped_soil}

            if data_held_by_equipped_rovers:
                # Data is already collected by at least one equipped rover, just need to communicate
                for r in data_held_by_equipped_rovers:
                    current_rover_loc = rover_locations.get(r)
                    if current_rover_loc is None or current_rover_loc not in self.dist: continue # Rover location unknown or isolated

                    # Cost to communicate from current location
                    min_comm_nav_cost = float('inf')
                    for comm_w in self.comm_waypoints:
                         if current_rover_loc in self.dist and comm_w in self.dist[current_rover_loc]:
                             min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_rover_loc][comm_w])

                    if min_comm_nav_cost != float('inf'):
                         goal_cost = min_comm_nav_cost + 1 # navigate + communicate
                         min_goal_cost = min(min_goal_cost, goal_cost)
            else:
                # Data is not yet collected by any equipped rover, need to sample and communicate
                # Assume sample exists if it's a goal waypoint and not communicated
                # if w not in soil_samples_at: continue # Cannot sample if sample is gone

                for r in self.equipped_soil:
                    current_rover_loc = rover_locations.get(r)
                    if current_rover_loc is None or current_rover_loc not in self.dist: continue

                    rover_store = self.rover_store.get(r)
                    if rover_store is None: continue # Rover has no store

                    goal_cost = 0

                    # Cost to sample
                    if store_status.get(rover_store) == 'full':
                         goal_cost += 1 # drop

                    if w not in self.dist.get(current_rover_loc, {}):
                         # Cannot reach sample location
                         continue # Try next rover
                    else:
                         goal_cost += self.dist[current_rover_loc][w] # navigate to sample
                         goal_cost += 1 # sample_soil
                         current_rover_loc_after_sample = w # Rover is now at sample location

                    # Cost to communicate (from sample location)
                    min_comm_nav_cost = float('inf')
                    for comm_w in self.comm_waypoints:
                         if current_rover_loc_after_sample in self.dist and comm_w in self.dist[current_rover_loc_after_sample]:
                             min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_rover_loc_after_sample][comm_w])

                    if min_comm_nav_cost == float('inf'):
                         # Cannot reach communication point from sample location
                         continue # Try next rover
                    else:
                         goal_cost += min_comm_nav_cost
                         goal_cost += 1 # communicate_soil_data

                    min_goal_cost = min(min_goal_cost, goal_cost)

            if min_goal_cost == float('inf'):
                # If we reach here, it means for this soil goal 'w',
                # either no equipped rover has the data and no equipped rover can sample and communicate,
                # or some rover has the data but cannot reach a communication point.
                # In either case, this goal is currently impossible.
                return float('inf')
            total_cost += min_goal_cost

        # Goals for rock data communication
        rock_goals_todo = {w for goal in self.goals if match(goal, "communicated_rock_data", w) and w not in communicated_rock}
        for w in rock_goals_todo:
            min_goal_cost = float('inf')
            # Find if any equipped rover already has the data
            data_held_by_equipped_rovers = {r for (r_held, w_held) in have_rock if w_held == w and r_held in self.equipped_rock}

            if data_held_by_equipped_rovers:
                # Data is already collected by at least one equipped rover, just need to communicate
                for r in data_held_by_equipped_rovers:
                    current_rover_loc = rover_locations.get(r)
                    if current_rover_loc is None or current_rover_loc not in self.dist: continue

                    # Cost to communicate from current location
                    min_comm_nav_cost = float('inf')
                    for comm_w in self.comm_waypoints:
                         if current_rover_loc in self.dist and comm_w in self.dist[current_rover_loc]:
                             min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_rover_loc][comm_w])

                    if min_comm_nav_cost != float('inf'):
                         goal_cost = min_comm_nav_cost + 1 # navigate + communicate
                         min_goal_cost = min(min_goal_cost, goal_cost)
            else:
                # Data is not yet collected by any equipped rover, need to sample and communicate
                # Assume sample exists if it's a goal waypoint and not communicated
                # if w not in rock_samples_at: continue # Cannot sample if sample is gone

                for r in self.equipped_rock:
                    current_rover_loc = rover_locations.get(r)
                    if current_rover_loc is None or current_rover_loc not in self.dist: continue

                    rover_store = self.rover_store.get(r)
                    if rover_store is None: continue # Rover has no store

                    goal_cost = 0

                    # Cost to sample
                    if store_status.get(rover_store) == 'full':
                         goal_cost += 1 # drop

                    if w not in self.dist.get(current_rover_loc, {}):
                         # Cannot reach sample location
                         continue # Try next rover
                    else:
                         goal_cost += self.dist[current_rover_loc][w] # navigate to sample
                         goal_cost += 1 # sample_rock
                         current_rover_loc_after_sample = w # Rover is now at sample location

                    # Cost to communicate (from sample location)
                    min_comm_nav_cost = float('inf')
                    for comm_w in self.comm_waypoints:
                         if current_rover_loc_after_sample in self.dist and comm_w in self.dist[current_rover_loc_after_sample]:
                             min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_rover_loc_after_sample][comm_w])

                    if min_comm_nav_cost == float('inf'):
                         # Cannot reach communication point from sample location
                         continue # Try next rover
                    else:
                         goal_cost += min_comm_nav_cost
                         goal_cost += 1 # communicate_rock_data

                    min_goal_cost = min(min_goal_cost, goal_cost)

            if min_goal_cost == float('inf'):
                return float('inf') # This goal is impossible
            total_cost += min_goal_cost


        # Goals for image data communication
        image_goals_todo = {(o, m) for goal in self.goals if match(goal, "communicated_image_data", o, m) and (o, m) not in communicated_image}
        for (o, m) in image_goals_todo:
            min_goal_cost = float('inf')
            # Find if any equipped rover already has the image
            image_held_by_equipped_rovers = {r for (r_held, o_held, m_held) in have_image if o_held == o and m_held == m and r_held in self.equipped_imaging}

            if image_held_by_equipped_rovers:
                 # Image is already taken by at least one equipped rover, just need to communicate
                 for r in image_held_by_equipped_rovers:
                    current_rover_loc = rover_locations.get(r)
                    if current_rover_loc is None or current_rover_loc not in self.dist: continue

                    # Cost to communicate from current location
                    min_comm_nav_cost = float('inf')
                    for comm_w in self.comm_waypoints:
                         if current_rover_loc in self.dist and comm_w in self.dist[current_rover_loc]:
                             min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_rover_loc][comm_w])

                    if min_comm_nav_cost != float('inf'):
                         goal_cost = min_comm_nav_cost + 1 # navigate + communicate
                         min_goal_cost = min(min_goal_cost, goal_cost)
            else:
                # Image is not yet taken by any equipped rover/camera, need to image and communicate
                img_waypoints = self.objective_visible_from.get(o, set())
                if not img_waypoints:
                    # Cannot take image of this objective
                    # min_goal_cost remains inf, will be handled below
                    pass # Continue to next goal or return inf

                for r in self.equipped_imaging:
                    current_rover_loc = rover_locations.get(r)
                    if current_rover_loc is None or current_rover_loc not in self.dist: continue

                    # Find cameras on this rover that support the mode
                    suitable_cameras = {
                        cam for cam in self.rover_cameras.get(r, set())
                        if m in self.camera_supports.get(cam, set())
                    }

                    for i in suitable_cameras:
                        # Find calibration target t for camera i
                        cal_target = self.camera_cal_target.get(i)
                        if cal_target is None: continue # Cannot calibrate this camera

                        # Find calibration waypoint(s) for target t
                        cal_waypoints = self.objective_visible_from.get(cal_target, set()) # Calibration targets are objectives
                        if not cal_waypoints: continue # Cannot calibrate this camera

                        min_image_path_cost = float('inf')
                        best_img_w = None

                        # Iterate through possible imaging waypoints
                        for img_w in img_waypoints:
                            if current_rover_loc not in self.dist or img_w not in self.dist[current_rover_loc] or self.dist[current_rover_loc][img_w] == float('inf'):
                                continue # Cannot reach image waypoint

                            current_path_cost = self.dist[current_rover_loc][img_w] # Navigate to image location
                            current_rover_loc_at_img_w = img_w

                            if (i, r) not in calibrated_cameras:
                                # Cost to calibrate
                                min_cal_nav_cost = float('inf')
                                best_cal_w = None
                                for cal_w in cal_waypoints:
                                    if current_rover_loc_at_img_w not in self.dist or cal_w not in self.dist[current_rover_loc_at_img_w] or self.dist[current_rover_loc_at_img_w][cal_w] == float('inf'):
                                         continue # Cannot reach cal waypoint
                                    cal_nav_cost = self.dist[current_rover_loc_at_img_w][cal_w] # Navigate to calibrate
                                    min_cal_nav_cost = min(min_cal_nav_cost, cal_nav_cost)
                                    if min_cal_nav_cost == cal_nav_cost:
                                        best_cal_w = cal_w

                                if min_cal_nav_cost == float('inf'):
                                    current_path_cost = float('inf') # Cannot calibrate
                                else:
                                    current_path_cost += min_cal_nav_cost # Nav to calibrate
                                    current_path_cost += 1 # calibrate action
                                    if best_cal_w not in self.dist or img_w not in self.dist[best_cal_w] or self.dist[best_cal_w][img_w] == float('inf'):
                                         current_path_cost = float('inf') # Cannot navigate back
                                    else:
                                         current_path_cost += self.dist[best_cal_w][img_w] # Nav back to image location

                            # Cost for take_image action
                            if current_path_cost != float('inf'):
                                current_path_cost += 1

                            if current_path_cost < min_image_path_cost:
                                min_image_path_cost = current_path_cost
                                best_img_w = img_w

                        if min_image_path_cost == float('inf'):
                            # Cannot take this image with this rover/camera combination
                            continue # Try next camera or rover
                        else:
                            goal_cost = min_image_path_cost
                            current_rover_loc_after_image = best_img_w # Rover is now at image location

                            # Cost to communicate (from image location)
                            min_comm_nav_cost = float('inf')
                            for comm_w in self.comm_waypoints:
                                 if current_rover_loc_after_image in self.dist and comm_w in self.dist[current_rover_loc_after_image]:
                                     min_comm_nav_cost = min(min_comm_nav_cost, self.dist[current_rover_loc_after_image][comm_w])

                            if min_comm_nav_cost == float('inf'):
                                 # Cannot reach communication point from image location
                                 continue # Try next camera or rover
                            else:
                                 goal_cost += min_comm_nav_cost
                                 goal_cost += 1 # communicate_image_data

                            min_goal_cost = min(min_goal_cost, goal_cost)


            if min_goal_cost == float('inf'):
                return float('inf') # This goal is impossible
            total_cost += min_goal_cost

        return total_cost
