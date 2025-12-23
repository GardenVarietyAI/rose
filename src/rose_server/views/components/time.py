from datetime import datetime, timezone

from htpy import (
    BaseElement,
    time as time_,
)


def render_time(value: datetime | int | str) -> BaseElement:
    if isinstance(value, datetime):
        return time_(datetime=value.isoformat())[str(value)]

    if isinstance(value, int):
        iso = datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        return time_(datetime=iso)[str(value)]

    try:
        timestamp = int(value)
    except ValueError:
        return time_()[value]
    else:
        iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        return time_(datetime=iso)[value]
