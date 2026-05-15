"""fleet_router — Route AI queries to the cheapest model that won't break."""

from .angles import route, classify_domain, MODELS, route_with_explanation
from .providers import get_provider
from .api import app

__version__ = "0.1.0"
