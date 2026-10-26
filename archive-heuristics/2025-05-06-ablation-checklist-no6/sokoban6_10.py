import re
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a (R, C) tuple."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen in valid sokoban instances

def format_location(row, col):
    """Formats a (R, C) tuple back into a location string 'loc_R_C'."""
    return f"loc_{row}_{col}"

def manhattan_distance(loc1_str, loc2_str, location_coords):
    """Calculates Manhattan distance between two location strings."""
    coords1 = location_coords.get(loc1_str)
    coords2 = location_coords.get(loc2_str)
    if coords1 is None or coords2 is None:
        # This case indicates an issue with location mapping, should ideally not happen
        # if all relevant locations are in static facts.
        return float('inf') # Or handle error appropriately
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

def bfs_shortest_path(start_node, goal_nodes, adj_list, obstacles):
    """
    Finds the shortest path distance from start_node to any node in goal_nodes
    avoiding obstacles.
    Returns distance or float('inf') if no path exists.
    """
    if start_node in goal_nodes:
        return 0

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        if current_node in goal_nodes:
            return dist

        # Robot can only move to adjacent locations that are NOT obstacles
        for neighbor in adj_list.get(current_node, []):
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of Manhattan distances for each box to its goal,
    plus the shortest path distance for the robot to reach a position from which
    it can push any misplaced box towards its goal.

    This heuristic is non-admissible. It provides an estimate that guides
    greedy best-first search towards states where boxes are closer to their
    goals and the robot is positioned to push a box.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds location coordinate maps and adjacency list.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        # Extract box goal locations from the goal state
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box location)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        self.location_coords = {}
        self.coords_location = {}
        self.adj_list = {} # Stores adjacency for robot movement

        # Build location maps and adjacency list from static facts
        # Assuming all relevant locations are connected and defined by adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]

                # Parse and store coordinates if not already known
                if loc1 not in self.location_coords:
                    coords1 = parse_location(loc1)
                    if coords1:
                        self.location_coords[loc1] = coords1
                        self.coords_location[coords1] = loc1
                if loc2 not in self.location_coords:
                    coords2 = parse_location(loc2)
                    if coords2:
                        self.location_coords[loc2] = coords2
                        self.coords_location[coords2] = loc2

                # Add adjacency (undirected for robot movement on the grid)
                self.adj_list.setdefault(loc1, set()).add(loc2)
                self.adj_list.setdefault(loc2, set()).add(loc1)

        # It's possible a location exists in init/goal but not in any adjacent fact
        # (e.g., a single isolated location). Ensure all are mapped.
        all_locations_in_task = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["at-robot", "at"] and len(parts) >= 2:
                 all_locations_in_task.add(parts[-1]) # last part is location
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == "at" and len(parts) == 3:
                 all_locations_in_task.add(parts[2])

        for loc_str in all_locations_in_task:
             if loc_str not in self.location_coords:
                 coords = parse_location(loc_str)
                 if coords:
                     self.location_coords[loc_str] = coords
                     self.coords_location[coords] = loc_str
                 # If parse_location fails or loc_str is not loc_R_C format,
                 # it won't be added. This heuristic assumes loc_R_C format.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        h = sum(MD(box_loc, goal_loc) for misplaced_boxes)
            + RobotDistance(robot_loc, nearest_push_location_for_any_misplaced_box)
        """
        state = node.state

        # Extract current state information
        robot_loc_str = None
        box_locations = {} # {box_name: loc_str}
        occupied_locations = set() # Locations occupied by robot or boxes

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_loc_str = parts[1]
                occupied_locations.add(robot_loc_str)
            elif parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, location = parts[1], parts[2]
                box_locations[box] = location
                occupied_locations.add(location)
            # We assume locations not in occupied_locations are clear for robot movement.


        # Ensure robot location is found (should always be the case in valid states)
        if robot_loc_str is None:
             # This state is likely invalid or unreachable in a standard problem
             return float('inf')

        total_box_md = 0
        misplaced_boxes = []
        target_robot_locs = set() # Potential locations for the robot to push from

        for box, goal_loc_str in self.goal_locations.items():
            current_loc_str = box_locations.get(box) # Use .get for safety

            # If box is not found (e.g., carried, though not possible in this domain)
            # or is at the goal, skip.
            if current_loc_str is None or current_loc_str == goal_loc_str:
                continue

            misplaced_boxes.append(box)

            # Add Manhattan distance for this box
            # Check if locations exist in our map before calculating MD
            if current_loc_str not in self.location_coords or goal_loc_str not in self.location_coords:
                 # This indicates an issue with the task definition or state
                 return float('inf') # Cannot compute heuristic

            total_box_md += manhattan_distance(current_loc_str, goal_loc_str, self.location_coords)

            # Determine potential push locations for this box
            l_b_coords = self.location_coords[current_loc_str]
            g_b_coords = self.location_coords[goal_loc_str]

            r_box, c_box = l_b_coords
            r_goal, c_goal = g_b_coords

            # Required robot location (r_req) is adjacent to box (l_b)
            # in the opposite direction of the push towards goal (g_b).

            # Check directions that reduce distance to goal
            # If row needs to decrease (push up), robot needs to be below (row + 1)
            if r_goal < r_box:
                r_req_coords = (r_box + 1, c_box)
                if r_req_coords in self.coords_location:
                    target_robot_locs.add(self.coords_location[r_req_coords])
            # If row needs to increase (push down), robot needs to be above (row - 1)
            elif r_goal > r_box:
                r_req_coords = (r_box - 1, c_box)
                if r_req_coords in self.coords_location:
                    target_robot_locs.add(self.coords_location[r_req_coords])

            # If col needs to decrease (push left), robot needs to be right (col + 1)
            if c_goal < c_box:
                r_req_coords = (r_box, c_box + 1)
                if r_req_coords in self.coords_location:
                    target_robot_locs.add(self.coords_location[r_req_coords])
            # If col needs to increase (push right), robot needs to be left (col - 1)
            elif c_goal > c_box:
                r_req_coords = (r_box, c_box - 1)
                if r_req_coords in self.coords_location:
                    target_robot_locs.add(self.coords_location[r_req_coords])

        # If no boxes are misplaced, the heuristic is 0
        if not misplaced_boxes:
            return 0

        # If there are misplaced boxes but no target push locations were identified
        # (e.g., box is in a corner with no adjacent space for robot in the required direction),
        # this might indicate a deadlock or unsolvable state.
        if not target_robot_locs:
             return float('inf') # Cannot reach a position to push any box

        # Calculate robot cost: shortest path to any target push location
        # Robot can move to any location that is NOT occupied by a box.
        # The BFS `obstacles` parameter should be the set of locations the robot cannot enter.
        # These are the locations occupied by boxes.
        robot_obstacles = set(box_locations.values())

        robot_dist = bfs_shortest_path(robot_loc_str, target_robot_locs, self.adj_list, robot_obstacles)

        # If robot cannot reach any push location avoiding boxes, return infinity
        if robot_dist == float('inf'):
            return float('inf')

        # Total heuristic is sum of box MD + robot distance to nearest push location
        # This is a non-admissible estimate.
        return total_box_md + robot_dist
