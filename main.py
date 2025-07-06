import pygame, random, sys

pygame.init()

# ── window ───────────────────────────────────────────────
W, H = 1070, 700
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Operation Sindoor: Disinformation Simulation")

# ── fonts & colors ───────────────────────────────────────
FONT = pygame.font.SysFont("arial", 18)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED = (34, 139, 34), (220, 20, 60)
GRAY, BLUE = (200, 200, 200), (70, 130, 180)

# ── influence map (baseline) ─────────────────────────────
INF = {
    "Prime Minister": 1.0,
    "Opposition": 0.8,
    "Soldier": 0.7,
    "Indian Media": 0.6,
    "Pakistan Media": 0.6,
    "Civilian": 0.4,
}

# ── trust‑update formula ─────────────────────────────────
def new_trust(t: float, I: float, r: float, s: float) -> float:
    """Return updated trust using T(new)=t−I*((100+(r−t))/100*((s−t)/4))."""
    drop = I * (((100 + (r - t)) / 100) * ((s - t) / 4))
    drop = max(0, drop)
    return max(0, t - drop)

# ── NPC object ───────────────────────────────────────────
class NPC:
    W, H = 180, 105

    def __init__(self, name: str, role: str, col: int, row: int):
        self.name, self.role = name, role
        self.power = INF[role]
        self.trust = random.randint(60, 90)
        self.x = 10 + col * (NPC.W + 20)
        self.y = 10 + row * (NPC.H + 30)
        self.initial_trust = self.trust

    def draw(self, surf):
        pygame.draw.rect(surf, GRAY, (self.x, self.y, NPC.W, NPC.H))
        pygame.draw.rect(surf, BLACK, (self.x, self.y, NPC.W, NPC.H), 2)
        surf.blit(FONT.render(self.name, True, BLACK), (self.x + 8, self.y + 6))
        surf.blit(FONT.render(self.role, True, BLACK), (self.x + 8, self.y + 26))
        bar_len = int(max(0, min(100, self.trust)))
        color = GREEN if self.trust > 50 else RED
        pygame.draw.rect(surf, color, (self.x + 8, self.y + 60, bar_len, 18))
        pygame.draw.rect(surf, BLACK, (self.x + 8, self.y + 60, 100, 18), 1)
        surf.blit(FONT.render(f"Trust: {int(self.trust)}", True, BLACK),
                  (self.x + 8, self.y + 82))
        surf.blit(FONT.render(f"Power: {self.power:.2f}", True, BLACK),
                  (self.x + 90, self.y + 82))

    def reset(self):
        self.trust = self.initial_trust

# ── build NPC roster ─────────────────────────────────────
roles_order = ["Prime Minister", "Opposition",
               "Soldier", "Indian Media", "Pakistan Media"]
npcs = []
for i in range(30):
    role = roles_order[i] if i < 5 else "Civilian"
    col, row = i % 5, i // 5
    npc = NPC(f"NPC {i + 1}", role, col, row)
    if i >= 5:
        npc.trust = random.randint(10, 100)
        npc.initial_trust = npc.trust
        npc.power = round(random.uniform(0.1, 0.5), 2)
    npcs.append(npc)

# ── disinformation events ───────────────────────────────
events = [
    {"text": "Pahalgam Attack",            "strength": 85, "realism": 75},
    {"text": "Pakistani Airspace Closure", "strength": 70, "realism": 60},
    {"text": "Operation Sindoor Claims",   "strength": 90, "realism": 30},
]

# ── scrollable NPC box ──────────────────────────────────
npc_area = pygame.Rect(40, 30, 1000, 400)
ROWS = (len(npcs) + 4) // 5
CONTENT_H = 10 + ROWS * (NPC.H + 30)
npc_surface = pygame.Surface((npc_area.width, CONTENT_H))
npc_scroll, scroll_speed = 0, 20
npc_max_scroll = max(0, CONTENT_H - npc_area.height)

# ── scrollable LOG box ──────────────────────────────────
LOG_LINE_H = 22
log_area = pygame.Rect(40, npc_area.bottom + 60, 1000, 180)
log_scroll = 0

# ── button helper ───────────────────────────────────────
class Button:
    def __init__(self, txt: str, x: int, y: int, act: str):
        self.txt, self.rect, self.act = txt, pygame.Rect(x, y, 200, 40), act
    def draw(self):
        pygame.draw.rect(screen, BLUE, self.rect)
        screen.blit(FONT.render(self.txt, True, WHITE),
                    (self.rect.x + 10, self.rect.y + 10))
    def click(self, pos):
        return self.rect.collidepoint(pos)

# buttons sit between the two boxes
BTN_Y = npc_area.bottom + 10
buttons = [
    Button("Start Simulation",  50, BTN_Y, "start"),
    Button("Reset Simulation", 300, BTN_Y, "reset"),
    Button("Scroll Up",        550, BTN_Y, "scroll_up"),
    Button("Scroll Down",      800, BTN_Y, "scroll_down"),
]

# ── main loop ────────────────────────────────────────────
clock = pygame.time.Clock()
log, running = [], True

while running:
    screen.fill(WHITE)

    # ─ event handling ───────────────────────────────────
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        # button clicks
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for b in buttons:
                if b.click(e.pos):
                    if b.act == "start":
                        ev = random.choice(events)
                        spreaders = random.sample(npcs, 2)
                        for sp in spreaders:
                            targets = random.sample([n for n in npcs if n != sp], 3)
                            for tgt in targets:
                                tgt.trust = new_trust(
                                    tgt.trust, sp.power,
                                    ev["realism"], ev["strength"]
                                )
                                log.append(
                                    f"{sp.name} ➜ {tgt.name}  [{ev['text']}]"
                                )
                    if b.act == "reset":
                        for n in npcs:
                            n.reset()
                        log.clear()
                        npc_scroll = log_scroll = 0
                    if b.act == "scroll_up":
                        # decide which box cursor is over
                        mx, my = pygame.mouse.get_pos()
                        if npc_area.collidepoint(mx, my):
                            npc_scroll = max(npc_scroll - scroll_speed, 0)
                        elif log_area.collidepoint(mx, my):
                            log_scroll = max(log_scroll - scroll_speed, 0)
                    if b.act == "scroll_down":
                        mx, my = pygame.mouse.get_pos()
                        if npc_area.collidepoint(mx, my):
                            npc_scroll = min(npc_scroll + scroll_speed,
                                             npc_max_scroll)
                        elif log_area.collidepoint(mx, my):
                            # compute log surface height each click
                            log_h = max(log_area.height,
                                        len(log) * LOG_LINE_H + 10)
                            log_max = max(0, log_h - log_area.height)
                            log_scroll = min(log_scroll + scroll_speed,
                                             log_max)

        # mouse‑wheel scrolling
        if e.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if npc_area.collidepoint(mx, my):
                npc_scroll -= e.y * scroll_speed
                npc_scroll = max(0, min(npc_scroll, npc_max_scroll))
            elif log_area.collidepoint(mx, my):
                # recalc max each wheel event (log grows)
                log_h = max(log_area.height, len(log) * LOG_LINE_H + 10)
                log_max = max(0, log_h - log_area.height)
                log_scroll -= e.y * scroll_speed
                log_scroll = max(0, min(log_scroll, log_max))

    # ─ draw NPCs box ─────────────────────────────────────
    npc_surface.fill(WHITE)
    for n in npcs:
        n.draw(npc_surface)
    view_rect = pygame.Rect(0, npc_scroll, npc_area.width, npc_area.height)
    screen.blit(npc_surface, npc_area.topleft, area=view_rect)
    pygame.draw.rect(screen, BLACK, npc_area, 2)

    # ─ draw LOG box ─────────────────────────────────────
    log_h = max(log_area.height, len(log) * LOG_LINE_H + 10)
    log_surface = pygame.Surface((log_area.width, log_h))
    log_surface.fill(WHITE)
    for idx, line in enumerate(log):
        log_surface.blit(FONT.render(line, True, BLACK),
                         (10, idx * LOG_LINE_H))
    log_view = pygame.Rect(0, log_scroll, log_area.width, log_area.height)
    screen.blit(log_surface, log_area.topleft, area=log_view)
    pygame.draw.rect(screen, BLACK, log_area, 2)

    # ─ draw buttons ─────────────────────────────────────
    for b in buttons:
        b.draw()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
