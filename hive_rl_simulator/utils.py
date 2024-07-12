from functools import lru_cache, wraps


def cached(cache, key_func):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            key = key_func(self, *args, **kwargs)
            if key in cache:
                return cache[key]
            result = method(self, *args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return decorator
