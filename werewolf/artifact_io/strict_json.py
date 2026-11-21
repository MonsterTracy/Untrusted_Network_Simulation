"""供 ITT 读取器复用的 JSON 解析，拒绝重复键和非标准数值常量。"""

import json


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
