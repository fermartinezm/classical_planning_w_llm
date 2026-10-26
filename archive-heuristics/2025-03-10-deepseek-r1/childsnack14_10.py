from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnack14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering:
    - Required gluten-free and regular sandwiches.
    - Current locations of trays and sandwiches.
    - Steps to make, place, move, and serve sandwiches.

    # Assumptions
    - Gluten-free ingredients are available if needed (heuristic may overestimate).
    - Trays can be moved as needed, with each move costing one action.
    - One tray can carry multiple sandwiches, but each sandwich requires a put action.

    # Heuristic Initialization
    - Extracts static information: allergic children, gluten-free bread/content.

    # Step-By-Step Thinking
    1. Group unserved children by location and allergy.
    2. For each group, check existing suitable sandwiches on trays.
    3. Calculate deficit sandwiches needed.
    4. Use available kitchen sandwiches to reduce deficit.
    5. Estimate actions for making, placing, moving, and serving.
    6. Add tray movements if kitchen has no trays.
    """

    def __init__(self, task):
        self.allergic = set()  # Allergic children
        self.non_allergic = set()  # Non-allergic children
        self.gluten_free_breads = set()  # Gluten-free bread portions
        self.gluten_free_contents = set()  # Gluten-free content portions

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.gluten_free_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = set()
        waiting = {}  # child -> location
        trays_in_kitchen = 0
        at_kitchen_sandwiches = set()
        gluten_free_sandwiches = set()
        ontray = defaultdict(set)  # tray -> {sandwiches}
        tray_locations = {}  # tray -> location

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'waiting':
                waiting[parts[1]] = parts[2]
            elif parts[0] == 'at' and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]
                if parts[2] == 'kitchen':
                    trays_in_kitchen += 1
            elif parts[0] == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                gluten_free_sandwiches.add(parts[1])
            elif parts[0] == 'ontray':
                ontray[parts[2]].add(parts[1])

        # Group unserved children by (location, is_allergic)
        groups = defaultdict(lambda: {'count': 0, 'needs_gluten_free': False})
        for child in (self.allergic | self.non_allergic):
            if child in served:
                continue
            loc = waiting.get(child)
            if not loc:
                continue  # invalid state, heuristic will be high
            is_allergic = child in self.allergic
            key = (loc, is_allergic)
            groups[key]['count'] += 1
            groups[key]['needs_gluten_free'] = is_allergic

        heuristic = 0
        put_actions_needed = 0
        make_actions_needed = 0

        # Process each group
        for (loc, is_allergic), data in groups.items():
            count = data['count']
            needs_gf = data['needs_gluten_free']

            # Existing sandwiches on trays at loc
            existing = 0
            for tray, sandwiches in ontray.items():
                if tray_locations.get(tray) != loc:
                    continue
                for s in sandwiches:
                    if not needs_gf or s in gluten_free_sandwiches:
                        existing += 1
            existing = min(existing, count)
            deficit = max(0, count - existing)

            if deficit == 0:
                continue

            # Available in kitchen
            available = 0
            if needs_gf:
                available = len([s for s in at_kitchen_sandwiches if s in gluten_free_sandwiches])
            else:
                available = len(at_kitchen_sandwiches)
            use_kitchen = min(available, deficit)
            put_actions_needed += use_kitchen
            make_actions_needed += deficit - use_kitchen

            # Add costs: use_kitchen needs put+move+serve (3), make needs 4
            heuristic += use_kitchen * 3 + (deficit - use_kitchen) * 4
            # Add move tray to location (1 per group)
            heuristic += 1

        # If need to put/make sandwiches and no trays in kitchen, add move
        if (put_actions_needed + make_actions_needed) > 0 and trays_in_kitchen == 0:
            heuristic += 1

        return heuristic
