# YCB-Dynamic

### Installation

- Install `stillleben` and its requirements according to [this manual](https://git.ais.uni-bonn.de/schwarzm/stillleben/-/blob/master/doc/installation.rst).
- Install additional packages via pip: `pip install matplotlib scipy pandas tqdm moviepy urllib3 nimblephysics`.
- Clone this repo.
- Download the required external data from [here](https://uni-bonn.sciebo.de/s/2E7OjZtT7PvBefW) (password: `ycb-dynamic`) and place the folders into the folder "external_data" of this repo.

### Basic Usage

```python
python run.py --scenario <scenario>
```

- **Choosing a scenario** Replace `<scenario>` with the scenario you'd like to render (also try out `all`!)

- **Rendering an RGB video of the scenarios:** add `--assemble-rgb`
- **Open the stillleben viewer instead of generating data:** add `--view`
- Other options: use the argparse help (`-h`)
