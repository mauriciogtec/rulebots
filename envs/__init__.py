from functools import partial

from gymnasium.envs.registration import register
from weather2alert.env import HeatAlertEnv

from envs.buy_sell import BuySellSimple, BuySellSimpleLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsSimple, VitalSignsSimpleLang

from . import wrappers

Uganda = partial(VitalSignsSimpleLang, "models/uganda.npz")
MimicIII = partial(VitalSignsSimpleLang, "models/mimic-iii.npz")
MimicIV = partial(VitalSignsSimpleLang, "models/mimic-iv.npz")
HeatAlerts = partial(HeatAlertsLang)
BuySellSimple = partial(BuySellSimpleLang)

UgandaNumeric = partial(VitalSignsSimple, "models/uganda.npz")
MimicIIINumeric = partial(VitalSignsSimple, "models/mimic-iii.npz")
MimicIVNumeric = partial(VitalSignsSimple, "models/mimic-iv.npz")
HeatAlertsNumeric = partial(HeatAlertEnv)
BuySellSimpleNumeric = partial(BuySellSimple)


kwargs = {"disable_env_checker": True, "max_episode_steps": 16}
register(id="Uganda", entry_point="envs:Uganda", **kwargs)
register(id="MimicIII", entry_point="envs:MimicIII", **kwargs)
register(id="MimicIV", entry_point="envs:MimicIV", **kwargs)
register(id="BuySellSimple", entry_point="envs:BuySellSimple", **kwargs)
register(id="HeatAlerts", entry_point="envs:HeatAlerts", **kwargs)

register(id="UgandaNumeric", entry_point="envs:UgandaNumeric", **kwargs)
register(id="MimicIIINumeric", entry_point="envs:MimicIIINumeric", **kwargs)
register(id="MimicIVNumeric", entry_point="envs:MimicIVNumeric", **kwargs)
register(id="BuySellSimpleNumeric", entry_point="envs:BuySellSimpleNumeric", **kwargs)
register(id="HeatAlertsNumeric", entry_point="envs:HeatAlertsNumeric", **kwargs)
