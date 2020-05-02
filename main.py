import datetime, functools, os, pygame, sys
from enum import Enum
import random as lrandom
from typing import Callable, List, Tuple, TypeVar

### Game settings

TITLE = "Maze Solver"       # Window title
WINDOW_SIZE = (800, 800)    # Window size
FPS = 60                    # Frame rate
FRAMES_PER_STEP = 20        # Duration of one maze step (currently 3/sec)
OVERLAY_ALPHA = 128         # Overlay alpha, out of 255
TOLERANCE = 50              # Color tolerance for image parsing
RANDOMNESS = 0.1            # Default randomness

### ASCII maze conversion

ASCII_Block     = "X"
ASCII_Block_Alt = "#"
ASCII_Start     = "S"
ASCII_End       = "E"
ASCII_Blank     = " "

### Colors

WHITE    = pygame.Color(255, 255, 255)
BLACK    = pygame.Color(  0,   0,   0)
GRAY     = pygame.Color(150, 150, 150)
RED      = pygame.Color(255, 100, 100)
PURE_RED = pygame.Color(255,   0,   0)
GREEN    = pygame.Color(100, 255, 100)
BLUE     = pygame.Color(100, 100, 255)
AQUA     = pygame.Color(100, 200, 200)
AQUAX    = pygame.Color(177, 150, 150) # Grayer aqua
PURPLE   = pygame.Color(255, 100, 255)
ORANGE   = pygame.Color(255, 175, 100)

### Color themes

class ColorTheme:
    def __init__(self):
        self.BLOCKS = BLACK
        self.UNVISITED = GRAY
        self.VISITED = WHITE
        self.START = AQUA
        self.START_DEAD = AQUAX
        self.GOAL = ORANGE
        self.DEAD = RED
        self.FINISHED = GREEN
        self.OVERLAY = PURE_RED
class InfectColorTheme(ColorTheme):
    def __init__(self):
        super().__init__()
        self.BLOCKS = GRAY
        self.UNVISITED = WHITE
        self.VISITED = BLACK

BLOCK_COLORS = ColorTheme()

### Helper functions

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

def load_from_txt(fname: str) -> Tuple[List[str], Tuple[int, int]]:
    """Reading a maze from a text file"""
    
    # Error values
    maze: List[str] = None
    size = (0, 0)
    
    # Try to read file
    try:
        # Read file (newline separated)
        with open(fname, "r") as f:
            maze = f.read().split("\n")
        
        # Logging
        print(f"Loaded maze from: {fname}")
        
        # Assume the text is a rectangle
        size = (len(maze[0]), len(maze))
    
    # Couldn't read file
    except IOError:
        print(f"Unable to open file: {fname}")
    
    # Return values
    return (maze, size)

def chunked(arr: List[T], n: int) -> List[List[T]]:
    """Divide array into at most n-length sub-arrays"""
    
    # Result
    res: List[T] = list()
    
    # While array has more than n elements
    while len(arr) > n:
        # Take first n elements as sub-array
        res.append(arr[:n])
        
        # Discard n elements on original array
        arr = arr[n:]
    
    # Add the last part (at most n elements)
    res.append(arr)
    
    # Return
    return res

def load_from_png(fname: str) -> Tuple[List[str], Tuple[int, int]]:
    """Read map from png file"""
    
    # TODO: Process window events, and possibly have some kind of progress bar. Freezing isn't great
    
    # Load image
    img = pygame.image.load(fname)
    w,h = img.get_size()
    
    # Logging
    print(f"Trying to parse image of size {w}x{h}, it might take a while")
    
    # Get color data as [(R00,G00,B00), (R01,G01,B01), ...]
    data = chunked([x for x in pygame.image.tostring(img, "RGB")], 3)
    
    # Process each pixel into a text equivalent
    for i in range(len(data)):
        # Get color of pixel
        c = data[i]
        
        # Check color
        if c[0] <= TOLERANCE and c[1] <= TOLERANCE and c[2] <= TOLERANCE:
            # Mostly black
            data[i] = ASCII_Block
        elif c[1] >= 255 - TOLERANCE and c[0] < TOLERANCE and c[2] < TOLERANCE:
            # Mostly green
            data[i] = ASCII_Start
        elif c[0] >= 255 - TOLERANCE and c[1] < TOLERANCE and c[2] < TOLERANCE:
            # Mostly red
            data[i] = ASCII_End
        else:
            # Everything else, probably white
            data[i] = ASCII_Blank
    
    # Divide into rows
    maze = chunked("".join(data), w)
    
    # Return maze and size
    return (maze, (w, h))

def map_2d(grid: List[List[U]], cb: Callable[[U, int, int], V], default: U = None) -> List[List[V]]:
    """The usual map function, but over a 2D grid (accepts default value)"""
    
    # Get grid dimensions
    width = len(grid[0])
    height = len(grid)
    
    # Result
    new_grid: List[List[V]] = list()
    
    # Iterate over rows
    for y in range(height):
        # Get row
        row = grid[y]
        
        # Result row
        new_row: List[V] = list()
        
        # Iterate over columns
        for x in range(width):
            # Handle staggered grid, even though it should be a rectangle
            value = row[x] if x < len(row) else default
            
            # Processes with: CALLBACK(VALUE, X, Y)
            new_row.append(cb(value, x, y))
        
        # Add result row
        new_grid.append(new_row)
    
    # Return
    return new_grid

def save_maze(maze: List[str], file_name: str, file_ext: str) -> None:
    """Save maze to text file"""
    
    # Result filename
    ext = "" if file_ext == ".txt" else file_ext
    out_file = f"{file_name}{ext}.txt"
    
    # Logging
    print(f"Saving maze to file: {out_file}")
    
    # Try to save to file
    try:
        # Save to file
        with open(out_file, "w") as f:
            f.write("\n".join(maze))
        
        # Logging
        print("Saved successfully")
    
    # Error
    except IOError as e:
        print(f"Error saving to file: {e}")

### Classes

class Square:
    class State(Enum):
        solid = -1      # Wall
        unvisited = 0   # Unvisited
        visited = 1     # Visited
        dead = 2        # Dead end
        path = 3        # Successful path
        finished = 4    # Path displayed when finished
    
    def __init__(self, world: 'World', x: int, y: int, isWall: bool):
        self.world = world                                                      # To see other squares
        self.x = x                                                              # Column
        self.y = y                                                              # Row
        self.state = Square.State.solid if isWall else Square.State.unvisited   # Assign a state based on isWall
        self.parent: Square = None                                              # For path, previous location along path
        self.neighbors: List[Square] = list()                                   # Unpopulated neighbors
        self.pcol = BLACK                                                       # Drawing optimization
    
    def is_at(self, x: int, y: int) -> bool:
        """Check position"""
        
        return self.x == x and self.y == y
    
    def is_eq(self, other: 'Square') -> bool:
        """Check square equality by position"""
        
        if not other:
            return False
        return other.is_at(self.x, self.y)
    
    def visit(self, parent: 'Square') -> None:
        """Set path from parent to current"""
        
        # Don't travel to if visited or parent already assigned
        if self.state != Square.State.unvisited or self.parent != None:
            return
        
        # Assign parent
        self.parent = parent
        
        # Can't travel back to parent, remove from neighbors list
        self.neighbors.remove(self.parent)
        
        # Visited
        self.state = Square.State.visited
        
        # Add to list of squares to update every frame
        self.world.seen.append(self)
    
    def update(self) -> None:
        """Check neighbors and handle step update"""
        
        # Bool to make sure the maze stops if no squares change
        state_changed = False
        
        # Clean up dead ends, also handles optimizations
        if (self.state == Square.State.path and (FRAMES_PER_STEP > 0 or len(self.neighbors) <= 1)) or self.state == Square.State.dead:
            try:
                self.world.seen.remove(self)
            except ValueError:
                pass
            return
        
        # Try to expand to each neighbor
        for c in self.neighbors:
            # Default change variable to true
            changed = True
            
            if c.state == Square.State.unvisited:
                # Try to visit neighbor
                c.visit(self)
            elif c.state == Square.State.dead or (c.state == Square.State.visited and not c.parent.is_eq(self)):
                # Ignore dead ends
                self.neighbors.remove(c)
            else:
                # Didn't change anything
                changed = False
            
            # Save if there was a change
            state_changed = state_changed or changed
        
        # If no more neighbors, it's a dead end (the parent was removed from the neighbors list)
        if len(self.neighbors) == 0:
            # Assign as dead
            self.state = Square.State.dead
            
            # There was a change
            self.world.changed = True
            
            # Speed optimization of dead ends
            if FRAMES_PER_STEP == 0:
                # Recurse up the parent chain assigning as dead ends until the first split
                cur = self
                while cur and cur.parent and len(cur.parent.neighbors) > 0:
                    # If only one neighbor on parent, is an only child
                    if len(cur.parent.neighbors) == 1:
                        # Clean up dead end
                        cur.parent.neighbors = list()
                        cur.parent.state = Square.State.dead
                        
                        # Recurse
                        cur = cur.parent
                    else:
                        break
            
            # Is dead, nothing else to do
            return
        
        # Optimization for path finding, if the parent is a successful path and current is an only sibling, it must also be a path
        if FRAMES_PER_STEP == 0 and self.parent and len(self.parent.neighbors) <= 1 and self.parent.state == Square.State.path:
            self.state = Square.State.path
            state_changed = True
        
        # Record if there was a change
        self.world.changed = self.world.changed or state_changed
    
    def draw(self, screen: pygame.Surface, line: bool, optimized: bool) -> None:
        """Draw depending on state, allows for optimization and path drawing"""
        
        # If not drawing the lines between squares
        if not line:
            # Debug color
            col = PURPLE
            
            # Set the draw color based on the state
            if self.state == Square.State.solid:
                col = BLOCK_COLORS.BLOCKS
            elif self.is_eq(self.world.start):
                col = BLOCK_COLORS.START
                
                # If the path was unsuccessful
                if self.state == Square.State.dead:
                    col = BLOCK_COLORS.START_DEAD
            elif self.is_eq(self.world.goal):
                col = BLOCK_COLORS.GOAL
            elif self.state == Square.State.unvisited:
                col = BLOCK_COLORS.UNVISITED
            elif self.state == Square.State.visited or self.state == Square.State.path:
                col = BLOCK_COLORS.VISITED
            elif self.state == Square.State.dead:
                col = BLOCK_COLORS.DEAD
            elif self.state == Square.State.finished:
                col = BLOCK_COLORS.FINISHED
            
            # Only draw if: unoptimized or (optimized and different color)
            if not optimized or col != self.pcol or self.state == Square.State.finished:
                # Draw the square using pygame
                pygame.draw.rect(screen, col, self.world.as_rect(self), 0)
                
                # Record current color for optimization
                self.pcol = col
        elif not optimized and self.parent and (self.state == Square.State.visited or self.state == Square.State.path):
            # Draw the line between squares, only if unoptimized and is still a valid path
            pygame.draw.line(screen, BLACK, self.world.as_point(self.parent), self.world.as_point(self), 2)

class World:
    def __init__(self, maze: List[str], random: bool = False):
        self.start: Square = None                                                       # Where the maze starts
        self.goal: Square = None                                                        # The goal
        self.random: bool = random                                                      # Allow for randomness
        self.maze: List[List[Square]] = map_2d(maze, self.create_square, ASCII_Block)   # Parse the map using the create_square method as a callback
        self.width: int = len(self.maze[0])                                             # Width of the maze in squares
        self.dx: float = 1.0 * WINDOW_SIZE[0] / self.width                              # Width of a square on the screen
        self.height: int = len(self.maze)                                               # Height of the maze in squares
        self.dy: float = 1.0 * WINDOW_SIZE[1] / self.height                             # Height of a square on the screen
        
        map_2d(self.maze, self.populate_neighbors)  # Update the neighbors of all the squares
        
        self.seen: List[Square] = [self.start]  # List of squares to expand each step
        self.changed = True                     # If the maze has changed, will stop if no change between steps
        self.logged = False                     # If the maze solution has been logged
        self.render_solution = False            # If the maze was solved, and the final path requires rendering
        
        # The overlay when paused, slightly transparent color (only shown when unoptimized)
        # pylint: disable=too-many-function-args
        self.overlay = pygame.Surface(WINDOW_SIZE)
        # pylint: enable=too-many-function-args
        self.overlay.set_alpha(OVERLAY_ALPHA)
        self.overlay.fill(BLOCK_COLORS.OVERLAY)
        
        # If step optimized, the start square is a valid path, but must update first
        if FRAMES_PER_STEP == 0:
            self.start.update()
            self.start.state = Square.State.path
    
    def create_square(self, v, x, y):
        """Callback for parsing the map from text"""
        
        # Case insensitive
        v: str = v.upper() if isinstance(v, str) else v
        
        # Check if it's a block character
        is_block = v in [ASCII_Block, ASCII_Block_Alt]
        
        # If there is randomness
        if self.random and lrandom.random() <= RANDOMNESS:
            # Assign the square a random solidity
            is_block = lrandom.random() < 0.5
        
        # Create the square object
        res = Square(self, x, y, is_block)
        
        # Check if start character
        if v == ASCII_Start:
            # Can't have multiple starting squares
            if not self.start:
                # Assign as start
                self.start = res
                res.state = Square.State.visited
            else:
                print(f"Duplicate starting location: {x},{y}")
        # Check if end character
        elif v == ASCII_End:
            # Can't have multiple ending squares
            if not self.goal:
                self.goal = res
                res.state = Square.State.unvisited
            else:
                print(f"Duplicate ending location: {x},{y}")
        
        # Return square
        return res
    
    def populate_neighbors(self, v: Square, x: int, y: int) -> Square:
        """Populate neighbors from world"""
        
        # Right, down, left, up
        offsets = [(1,0), (0,1), (-1,0), (0,-1)]
        
        # Get neighbors
        for off in offsets:
            # Get neighbor from offset
            neighbor = self.get(x + off[0], y + off[1])
            
            # If invalid square, will be None
            if not neighbor:
                continue
            
            # Only add walkable tiles as neighbors
            if neighbor.state != Square.State.solid:
                v.neighbors.append(neighbor)
        
        return v
    
    def draw_self(self, screen, line, optimized, v: Square, x: int, y: int) -> Square:
        """Passthrough for drawing each square with various options"""
        
        v.draw(screen, line, optimized)
        
        return v
    
    def get(self, x: int, y: int) -> Square:
        """Get a square"""
        
        # No squares out of the maze
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        
        # Return the square
        return self.maze[y][x]
    
    def update(self) -> None:
        """Update the maze"""
        
        # Quit if there are no more steps to perform
        if not (self.start and self.goal and self.changed and self.start.state == (Square.State.path if FRAMES_PER_STEP == 0 else Square.State.visited) and self.goal.state == Square.State.unvisited):
            self.changed = False
            return
        
        # Start with no changes
        self.changed = False
        
        # Copy the current list so it only updates one frame, not the new squares
        current: List[Square] = list(self.seen)
        
        # Update each current square
        for node in current:
            node.update()
    
    def log_path(self) -> None:
        """The maze is finished, log and update path"""
        
        # Don't execute twice
        if self.logged:
            return
        
        # The function has been called
        self.logged = True
        
        # If the path couldn't be found
        if not self.start or not self.goal or not self.goal.state == Square.State.visited:
            # Don't log if there were random changes made to the maze
            if not self.random:
                print("Unable to find path")
            
            # Nothing else to do
            return
        
        # Logging
        print("Path found!")
        
        # Recurse up the path and set as finished
        cur = self.goal
        while cur:
            cur.state = Square.State.finished
            cur = cur.parent
        
        # Request rendering
        self.render_solution = True
    
    def as_rect(self, node: Square) -> Tuple[int, int, int, int]:
        """Convert square to on screen rectangle"""
        
        return (round(node.x * self.dx), round(node.y * self.dy), round(self.dx), round(self.dy))
    
    def as_point(self, node: Square) -> Tuple[int, int]:
        """Convert square to on screen point (for drawing lines)"""
        
        return (round((node.x + 0.5) * self.dx), round((node.y + 0.5) * self.dy))
    
    def draw(self, screen: pygame.Surface, paused: bool, lines: bool, optimized: bool) -> None:
        """Draw the maze, with various options"""
        
        # Draw everything if unoptimized, or the final frame
        if not optimized:
            # Draw all squares without optimization
            map_2d(self.maze, functools.partial(self.draw_self, screen, False, False))
            
            # If drawing lines between squares
            if lines:
                # Draw all squares' lines without optimization
                map_2d(self.maze, functools.partial(self.draw_self, screen, True, False))
            
            # Draw the paused overlay
            if paused:
                screen.blit(self.overlay, (0,0))
        else:
            # Only draw the nodes that have been seen (the only ones that could've changed)
            for node in self.seen:
                node.draw(screen, False, True)
            
            # Draw the final path if optimized
            if self.render_solution:
                node = self.goal
                while node is not None:
                    node.draw(screen, False, True)
                    node = node.parent
        
        # Was handled
        self.render_solution = False

def main():
    """Main"""
    
    # Allow write access for certain global variables
    global BLOCK_COLORS
    global FPS
    global FRAMES_PER_STEP
    global RANDOMNESS
    global WINDOW_SIZE
    
    # Randomize
    lrandom.seed(datetime.datetime.now())
    
    # Try to go super fast
    if "fast" in sys.argv:
        # Solve the maze quickly
        FRAMES_PER_STEP = 1
    elif "fastest" in sys.argv:
        # Optimize solving
        FRAMES_PER_STEP = 0
    
    # No frame cap and has optimization
    if "turbo" in sys.argv:
        FPS = 10000
        FRAMES_PER_STEP = 0
    
    # Pallette
    if "infect" in sys.argv:
        BLOCK_COLORS = InfectColorTheme()
    
    # Default randomness factor
    randomized = "random" in sys.argv
    
    # Check for specific randomness factor
    if not randomized:
        for key in sys.argv:
            if key.startswith("random"):
                amount = int(key[6:])
                if amount:
                    RANDOMNESS = amount * 0.001
                    randomized = True
                break
    
    # Windows setup
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(TITLE)
    
    # Game loop setup
    running = True                # If the program is running
    paused = False                # If the maze solve is paused
    lines = False                 # If drawing lines between squares
    optimized = True              # If drawing is optimized
    clock = pygame.time.Clock()   # For limiting frame rate
    step = 0                      # For controlling frames per step
    
    # Default maze
    maze = ["SXE ",
            " XX ",
            "    "]
    fname = "default.txt"
    file_name, file_ext = os.path.splitext(fname)
    
    if len(sys.argv) >= 2:
        # File name supplied
        fname = sys.argv[1]
        file_name, file_ext = os.path.splitext(fname)
        
        # Parse the maze
        nmaze: List[str] = None
        size = WINDOW_SIZE
        if file_ext in ('.png', '.bmp', '.jpg'):
            # Input is an image
            if len(sys.argv) >= 3 and sys.argv[2] == "convert":
                print("Only processing maze, not solving")
                running = False
            
            # Processed file already exists?
            nfname = f"{file_name}{file_ext}.txt"
            if os.path.isfile(nfname) and running:
                print(f"Processed version already exists, loading from: {nfname}")
                nmaze,size = load_from_txt(nfname)
            else:
                # Requires image parsing
                if file_ext == '.png':
                    nmaze, size = load_from_png(fname)
                else:
                    # TODO: Can the other formats get loaded similarly?
                    print(f"Not yet supported: {file_ext}")
                
                # Save the maze, if parsed
                if nmaze:
                    save_maze(nmaze, file_name, file_ext)
        else:
            # Input is already parsed as text
            nmaze,size = load_from_txt(fname)
        
        if nmaze:
            print(f"Using loaded maze: {fname}")
            maze = nmaze
            
            if len(sys.argv) >= 3 and sys.argv[2].startswith("fit"):
                if len(sys.argv[2]) > 3:
                    scale = int(sys.argv[2][3:])
                    if scale:
                        if scale < 1 or scale > 64:
                            print("Can only scale 1-64 times")
                        scale = max(1, min(scale, 64))
                        size = (size[0] * scale, size[1] * scale)
                WINDOW_SIZE = size
                screen = pygame.display.set_mode(WINDOW_SIZE)
        else:
            print("Using default maze")
    
    # Create the maze
    game = World(maze, random=randomized)
    
    # Clear the screen
    screen.fill(BLACK)
    
    # Draw the screen
    game.draw(screen, False, False, False)
    
    # Program loop
    while running == True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Close button
                running = False
            elif event.type == pygame.KEYDOWN:
                # Button press
                
                if event.key == pygame.K_r:
                    # Restart the maze solver
                    
                    # Recreate the maze
                    game = World(maze, random=randomized)
                    
                    # Clear the screen
                    screen.fill(BLACK)
                    
                    # Draw the screen
                    game.draw(screen, False, False, False)
                elif event.key == pygame.K_p:
                    # Toggle paused
                    paused = not paused
                    
                    # No overlay when optimized, print(to console for clarity)
                    if optimized:
                        print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_t:
                    # Toggle lines
                    lines = not lines
                    
                    # No lines drawn when optimized, print(to console)
                    if optimized:
                        print("Showing lines" if lines else "Lines hidden")
                elif event.key == pygame.K_o:
                    # Toggle optimized
                    optimized = not optimized
                    
                    # Logging
                    print("Enabled drawing optimization" if optimized else "Disabled optimization")
                    
                    # Draw the screen once if now optimized (doesn't draw the whole screen each step)
                    if optimized:
                        game.draw(screen, False, False, False)
                elif event.key == pygame.K_s:
                    # TODO: Save map to file
                    print("Saving not implemented")
                elif event.key == pygame.K_l:
                    # TODO: Load map from file
                    print("Loading not implemented")
        
        # Clear screen when unoptimized
        if not optimized:
            screen.fill(BLACK)
        
        # Try to update the maze solver
        if not paused:
            # Iterate step
            step += 1
            
            # If enough frames have passed by, update maze solver
            if step >= FRAMES_PER_STEP:
                # Only update if there was a change last update
                if game.changed:
                    # Update maze solver
                    game.update()
                    
                    # Finished solving maze
                    if not game.changed:
                        # Final steps for path
                        game.log_path()
                        
                        # If unsuccessful and randomized
                        if game.goal.state != Square.State.finished and randomized:
                            # Recreate the maze with randomness
                            game = World(maze, random=True)
                            
                            # Clear the screen
                            screen.fill(BLACK)
                            
                            # Draw the screen
                            game.draw(screen, False, False, False)
                
                # Reset step for next update
                step = 0
        
        # Draw the maze
        game.draw(screen, paused, lines, optimized)
        
        # Display
        pygame.display.flip()
        
        # FPS cap
        clock.tick(FPS)
    
    # Quit
    pygame.quit()

# Only execute main if running this file directly
if __name__ == "__main__":
    main()
