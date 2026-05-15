import json
import sys
from unittest.mock import MagicMock

# Mock rich before importing agent.tools
sys.modules["rich"] = MagicMock()
sys.modules["rich.console"] = MagicMock()

from datetime import datetime
from agent.tools import compact_json

def test_compact_json_basic_dict():
    data = {"name": "Alice", "age": 30}
    result = compact_json(data)
    assert result == '{"name": "Alice", "age": 30}'

def test_compact_json_unicode():
    data = {"emoji": "🔥", "greeting": "こんにちは"}
    result = compact_json(data)
    # ensure_ascii=False means it should keep unicode characters
    assert "🔥" in result
    assert "こんにちは" in result
    assert result == '{"emoji": "🔥", "greeting": "こんにちは"}'

def test_compact_json_default_str():
    # Test with a datetime object which is not JSON serializable by default
    dt = datetime(2023, 1, 1, 12, 0, 0)
    data = {"timestamp": dt}
    result = compact_json(data)
    # default=str should convert datetime to its string representation
    assert str(dt) in result
    assert result == f'{{"timestamp": "{str(dt)}"}}'

def test_compact_json_custom_object():
    class Custom:
        def __str__(self):
            return "CustomObject"

    data = {"obj": Custom()}
    result = compact_json(data)
    assert result == '{"obj": "CustomObject"}'
