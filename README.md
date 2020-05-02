# maze-solver

A Python maze solver inspired by Computerphile's video: [https://www.youtube.com/watch?v=rop0W4QDOUI](https://www.youtube.com/watch?v=rop0W4QDOUI)

This was originally written just after the video came out, but I upgraded the code to semi-modern Python 3 before pushing to GitHub.

## Demo:

```bash
> python main.py ./files/maze_large.png fit4 fast
```
![Demo](Demo.gif?raw=true "Demo")

## Requirements

- [Python 3.5+](https://www.python.org/downloads/) (mostly so type hints don't cause parsing errors)
- [pygame](https://www.pygame.org/news)
    ```bash
    > pip install --user -r requirements.txt
    # Or
    > pip install --user pygame
    ```

## Usage

```bash
# Image maze to ASCII maze
> python main.py <filename(image)> convert

# Or, actually solve the maze
> python main.py <filename(image/txt)> [fit<scale>] [<other args>]
```

### Command-line options

- `convert`: Converts the provided filename to `<filename>.txt` for faster startup time when solving
    - **Note**: Conversion always happens before solving, and may take some time for larger images
    - If using `convert`, the basic solve will skip conversion and use the cached version instead
        - I've included each file pre-converted so testing the project will be fast

###

- `fit<scale>`: Fit the window size to `scale` times the input maze size
    - If no scale is provided (i.e. `fit`), it will use a scale of 1
    - If `fit<scale>` is not used at all, the window size will default to 800 by 800

###

- `fast`: Will step the solver every frame, rather than the default 20
- `fastest`: Behaves the same as `fast`, but will also quickly close any dead-end paths immediately, rather than slowly back-tracking
- `infect`: Just a test alternate color scheme
- `random`: Randomly update the provided maze to switch open spaces to blocks (not recommended, was just for testing)
    - The default threshold is: `random.random() < 0.1`
- `random<threshold>`: You can also specify the threshold in the equation above, in thousandths (the value will be multiplied by 0.001)
    - Note: This can not be used in addition to `random`
- `turbo`: Steps the solver as fast as possible, not capped to 60 FPS
    - Will also handle dead-ends quickly

### Runtime options

- `o`: Enable/disable rendering optimization (enabled by default)
    - Optimized rendering will only update the squares that have changed color each step
    - Disabling this feature is **not** recommended for large mazes
- `p`: Pause/resume the solver
- `r`: Restart the solver
- `t`: Track the paths between nodes by drawing lines from each node to its parent
    - Note: This feature is only available when optimized rendering is disabled, and thus, is only recommended for demonstration on smaller images
