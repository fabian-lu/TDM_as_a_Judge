"""A chat-completion client with the same machinery as DecisionClient.

    from tdm import ChatClient
    gpt = ChatClient("openai/gpt-4-turbo", temperature=0.7, seed=1)
    r = gpt.ask([{"role": "user", "content": "..."}])
    r["text"]                                   # the reply
    results = gpt.ask_many([{"messages": [...]}, ...])

Everything else works as in DecisionClient: concurrent requests, retries in rounds,
the SQLite cache (the key includes model, messages and sampling settings), the budget
guard, and show()/show_last()/preview() for the exact request and response.

`accept` is an optional function text -> None or a problem string. A reply it rejects
counts as invalid and is retried, like a malformed decision.
"""
from .cache import request_key
from .client import DecisionClient
from .config import OPENROUTER_CHAT_URL
from .show import redact, show_exchange


class ChatClient(DecisionClient):
    def __init__(self, model, temperature=None, seed=None, max_tokens=None, accept=None, **kwargs):
        super().__init__(model, backend="openrouter", **kwargs)
        self.url = OPENROUTER_CHAT_URL
        self.params = {k: v for k, v in {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}.items()
                       if v is not None}
        self.accept = accept

    def ask(self, messages, repeat=0):
        """One request; returns a result dict with the reply in result["text"]."""
        return self.ask_many([{"messages": messages, "repeat": repeat}], progress=False)[0]

    def preview(self, messages, raw=False):
        """Print the exact request that ask() would send, without sending it."""
        show_exchange({"method": "POST", "url": self.url, "headers": redact(self.headers),
                       "request_raw": self._serialize(self._body({"messages": messages})), "sent": False},
                      raw=raw, title="REQUEST (preview, not sent)")

    # -- the four places where a chat request differs from a decision request -----------
    def _body(self, it):
        return {"model": self.model, "messages": it["messages"], **self.params}

    def _check(self, it, response):
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return "no reply text in the response"
        if not text or not text.strip():
            return "empty reply"
        return self.accept(text) if self.accept else None

    def _key(self, it):
        return request_key("openrouter-chat", self.model, it["messages"], self.params, it.get("repeat", 0))

    @staticmethod
    def _result(status, response, latency, attempts, cached=False, error=None, key=None, exchanges=None):
        r = DecisionClient._result(status, response, latency, attempts, cached, error, key, exchanges)
        r["answers"] = None
        r["text"] = response["choices"][0]["message"]["content"] if r["ok"] else None
        return r
