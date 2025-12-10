def truncate_msg(msg: str, max_length: int) -> str:
    """Truncate long messages by removing middle portion, keeping start and end with ellipsis indicator."""
    msg_len = len(msg)
    if msg_len <= max_length:
        return msg
    else:
        return (
            msg[: max_length // 2]
            + f"\n... This content has been truncated from an original {msg_len} characters to stay below {max_length} characters ...\n"
            + msg[-max_length // 2 :]
        )
