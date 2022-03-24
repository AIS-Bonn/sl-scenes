from .ball_box import BallBoxScenario
from .bowl import BowlScenario
from .billiards import BillardsScenario
from .bowling import BowlingScenario
from .dice_roll import DiceRollScenario
from .stack import StackScenario
from .tabletop import TabletopScenario
from .throw import ThrowScenario
from .tidy import TidyScenario
from .robopushing import RobopushingScenario

SCENARIOS = {
    "ball_box": BallBoxScenario,
    "billards": BillardsScenario,
    "bowl": BowlScenario,
    "bowling": BowlingScenario,
    "diceRoll": DiceRollScenario,
    "stack": StackScenario,
    "tabletop": TabletopScenario,
    "throw": ThrowScenario,
    "tidy": TidyScenario,
    "robopushing": RobopushingScenario,
}  #: All available scenarios and the string identifier with which they can be chosen. If the 'all' key (which maps to None) is provided as a command line argument, all scenarios are generated.
