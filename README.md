### Installation

- Install Stillleben and its requirements according to [this manual](https://git.ais.uni-bonn.de/schwarzm/stillleben/-/blob/master/doc/installation.rst).
- Clone this repo.
- Download the required and/or optional iBL files from [this webpage](http://www.hdrlabs.com/sibl/archive.html) and place them into the folder "external_data/light_maps"
- Download the required mesh files from [here](https://uni-bonn.sciebo.de/s/PsEd2HWUsIfvHte) and place the unpacked folders into "external_data/object_models"


### Basic Usage

```python
python run.py --scenario <scenario>
```

- **Choosing a scenario** Replace `<scenario>` with the scenario you'd like to render (also try `all`!)

- **Rendering an RGB video of the scenarios:** add `--assemble-rgb`
- **Open the stillleben viewer instead of generating data:** add `--view`
- Other options: use the argparse help (`-h`)