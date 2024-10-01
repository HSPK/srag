import arrow


def get_current_time_formatted(format: str | None = None, tz: str | None = None) -> str:
    if format is None:
        format = "YYYY-MM-DD HH:mm:ss"
    if tz is None:
        tz = "Asia/Shanghai"
    return arrow.now(tz).format(format)
