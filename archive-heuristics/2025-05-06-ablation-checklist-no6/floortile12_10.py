from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
from queue import PriorityQueue

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for weighted BFS
def bfs_distance_weighted(start_tile, target_tiles, current_clear_tiles, adjacency_map, all_tiles):
    """
    BFS with weights to find shortest distance from start_tile to any tile in target_tiles.
    Weight of edge (u, v) is 1 if v is clear, infinity otherwise.
    Movement is from u to v, possible if v is clear.
    Returns infinity if no path exists.
    """
    if not target_tiles: return float('inf')
    if start_tile in target_tiles: return 0 # Robot is already at a target paint position

    q = PriorityQueue()
    q.put((0, start_tile))
    dist = {tile: float('inf') for tile in all_tiles}
    dist[start_tile] = 0
    visited = set()

    while not q.empty():
        d, current = q.get()

        if current in visited: continue
        visited.add(current)

        # If the current node is one of the targets, we found the shortest path
        if current in target_tiles:
            return d

        # Explore neighbors
        for neighbor in adjacency_map.get(current, []):
            # Can move from current to neighbor if neighbor is clear
            move_cost = 1 if neighbor in current_clear_tiles else float('inf')

            # Only consider paths where moves are possible (finite cost)
            if move_cost != float('inf'):
                 if dist[current] + move_cost < dist[neighbor]:
                     dist[neighbor] = dist[current] + move_cost
                     q.put((dist[neighbor], neighbor))

    return float('inf') # No path found


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost by summing the minimum cost for each unsatisfied goal tile
    to be painted. The cost for a single tile includes:
    1. Cost to get the correct color to a robot (1 if the best robot doesn't have it).
    2. Cost for that robot to move to a paint position (BFS distance on clear tiles).
    3. Cost to perform the paint action (1).

    Assumes solvable problems do not start with wrongly painted goal tiles.
    Returns infinity for states where a goal tile is wrongly painted or unreachable.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Precompute adjacency graph from static facts
        # This represents the grid structure for movement.
        self.adj = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                # (adj tile1 tile2) means tile1 is adj to tile2.
                tile1, tile2 = parts[1], parts[2]
                self.adj.setdefault(tile1, set()).add(tile2)
                self.adj.setdefault(tile2, set()).add(tile1) # Assuming symmetric connections for movement

        # Precompute paint positions: {tile_to_paint: {robot_position1, robot_position2, ...}}
        # A robot at position 'x' can paint tile 'y' if (adj_dir y x) is true.
        self.paint_positions = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Action paint_dir ?r ?y ?x requires robot-at ?x, adj_dir ?y ?x.
            # So, x is a paint position for y.
            if parts[0] in ['up', 'down', 'left', 'right']:
                 tile_y, tile_x = parts[1], parts[2]
                 self.paint_positions.setdefault(tile_y, set()).add(tile_x)

        # Store goal requirements: {tile_name: color_name}
        self.goal_requirements = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_requirements[tile] = color

        # Get list of all tiles involved in the grid (from adjacency map keys)
        # This assumes all relevant tiles are part of the connected grid.
        self.all_tiles = list(self.adj.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Find robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Identify clear tiles
        clear_tiles = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'clear'}

        # Identify goal tiles that are not correctly painted
        unsatisfied_goals = {} # {tile: color}
        for goal_tile, goal_color in self.goal_requirements.items():
            # Check if the tile is already painted correctly
            if f'(painted {goal_tile} {goal_color})' not in state:
                 # Check if the tile is painted with the wrong color (unreachable state)
                 is_wrongly_painted = False
                 for fact in state:
                     parts = get_parts(fact)
                     if parts[0] == 'painted' and parts[1] == goal_tile and parts[2] != goal_color:
                         is_wrongly_painted = True
                         break
                 if is_wrongly_painted:
                     # This state is likely a dead end. Return a large value.
                     return float('inf') # Or a large constant like 1000

                 # If not painted correctly and not wrongly painted, it must be clear.
                 # We assume solvable problems don't start with wrongly painted goal tiles.
                 # In a reachable state where a goal tile needs painting, it must be clear.
                 # So, unsatisfied goals are exactly the clear goal tiles needing paint.
                 if f'(clear {goal_tile})' in state:
                     unsatisfied_goals[goal_tile] = goal_color
                 # else: # It's not clear and not painted correctly/wrongly? Invalid state assumption.
                 #    pass


        # If all goals are satisfied, heuristic is 0.
        if not unsatisfied_goals:
            return 0

        total_estimated_cost = 0

        # Calculate minimum cost for each unsatisfied clear goal tile independently
        for goal_tile, goal_color in unsatisfied_goals.items():
            min_cost_for_this_tile = float('inf')

            # Get the possible paint positions for this goal tile
            possible_paint_positions = self.paint_positions.get(goal_tile, set())
            if not possible_paint_positions:
                 # Should not happen if domain is well-formed, but handle defensively
                 # Cannot paint this tile if there are no defined paint positions for it.
                 return float('inf') # Dead end

            # Consider each robot as a potential painter for this tile
            for robot, robot_loc in robot_locations.items():
                # Cost to get color C: 1 if robot doesn't have it, 0 otherwise.
                cost_get_color = 1 if robot_colors[robot] != goal_color else 0

                # Cost to move from robot_loc to any tile in possible_paint_positions
                # Movement is restricted to clear tiles.
                dist = bfs_distance_weighted(robot_loc, possible_paint_positions, clear_tiles, self.adj, self.all_tiles)

                if dist is not float('inf'):
                    # Total cost for this robot to paint this tile:
                    # (Cost to get color) + (Cost to move) + (Cost to paint)
                    cost_this_robot = cost_get_color + dist + 1
                    min_cost_for_this_tile = min(min_cost_for_this_tile, cost_this_robot)

            # If no robot can reach a paint position, the state is unreachable.
            if min_cost_for_this_tile is float('inf'):
                 return float('inf') # Dead end

            total_estimated_cost += min_cost_for_this_tile

        return total_estimated_cost
