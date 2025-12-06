import redis
import json, os, time
from typing import Any, Dict, Optional, Tuple


DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
ARM_COMMAND_QUEUE = os.getenv("ARM_COMMAND_QUEUE", "device:arm:command")
ARM_STATUS_HASH = os.getenv("ARM_STATUS_HASH", "device:arm:status")
ARM_EVENT_CHANNEL = os.getenv("ARM_EVENT_CHANNEL", "device:arm:event")
ARM_DRAG_CHANNEL = os.getenv("ARM_DRAG_CHANNEL", "device:arm:drag")


class RedisUnavailableError(RuntimeError):    
    """Raised when the redis client cannot be created."""  


def create_redis_client(redis_url: Optional[str] = None, *, decode_responses: bool = True):
    if redis is None:
        raise RedisUnavailableError(
            "redis package is not installed. Please install `redis` to use Redis-based arm control."
        )
    url = redis_url or os.getenv("REDIS_URL", DEFAULT_REDIS_URL)
    return redis.Redis.from_url(url, decode_responses=decode_responses, socket_keepalive=True)


def main():
    r = create_redis_client(decode_responses=True)
    ps = r.pubsub()
    ps.subscribe(ARM_DRAG_CHANNEL)
    print(f"[drag-sub] subscribed {ARM_DRAG_CHANNEL}")
    for msg in ps.listen():
        if msg and msg.get("type") == "message":
            raw = msg.get("data")
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if data.get("command") == "drag":
                flag = (data.get("payload") or {}).get("drag_flag")
                print(f"[drag-sub] drag_flag={flag} id={data.get('id')}")
                # TODO: 在这里调用你的 ArmController 拖拽动作

if __name__ == "__main__":
    main()