"""Exercise documentation retrieval through the real MCP SDK over mock HTTP."""
import asyncio
import json
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from uuid import UUID

import httpx

from vulnhuntr import documentation_search as documentation


class SearchServer:
    """JSON-RPC fixture; only the HTTP boundary is replaced."""

    def __init__(self, *, results=None, error=None, delay=0, structured=False):
        self.requests = []
        self.calls = []
        self.error = error
        self.delay = delay
        self.structured = structured
        self.results = results if results is not None else [
            {
                "url": "https://docs.example.org/api#redirects",
                "title": "HTTP redirects",
                "excerpts": ["Redirects are followed unless explicitly disabled."],
            }
        ]
        self._client = httpx.AsyncClient

    async def handle(self, request):
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(200)
        message = json.loads(request.content)
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "documentation-test", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Find public API documentation",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "objective": {"type": "string"},
                                "search_queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "session_id": {"type": "string"},
                            },
                            "required": ["objective", "search_queries"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            self.calls.append(message["params"])
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error == "http_quota":
                return httpx.Response(
                    429, json={"error": "quota exceeded"}, headers={"Retry-After": "60"}
                )
            if self.error == "rpc":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32603, "message": "private provider detail"},
                    },
                )
            data = {"results": self.results}
            result = {"content": [{"type": "text", "text": json.dumps(data)}]}
            if self.structured:
                result["structuredContent"] = data
            if self.error == "tool":
                result = {
                    "isError": True,
                    "content": [{"type": "text", "text": "private provider detail"}],
                }
        else:
            raise AssertionError("Unexpected MCP method: " + method)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": message["id"], "result": result},
            headers={"Mcp-Session-Id": "fixture-session"},
        )

    def client(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(self.handle)
        return self._client(*args, **kwargs)

    def patch(self):
        return mock.patch.object(documentation.httpx, "AsyncClient", self.client)


class DocumentationSearchTests(unittest.TestCase):
    def test_sdk_search_returns_cited_excerpt_and_project_headers(self):
        server = SearchServer()
        research = documentation.DocumentationSearch()
        with server.patch(), mock.patch.dict(os.environ, {
            "PARALLEL_API_KEY": "fixture-key-never-forward",
            "OPENAI_API_KEY": "fixture-llm-key-never-forward",
        }):
            self.assertTrue(research.collect(["requests_redirects"]))
        rendered = research.render()
        self.assertIn("https://docs.example.org/api#redirects", rendered)
        self.assertIn("Redirects are followed unless explicitly disabled.", rendered)
        self.assertEqual(server.calls[0]["name"], "web_search")
        self.assertEqual(server.calls[0]["arguments"]["session_id"], research.session_id)
        self.assertEqual(
            server.calls[0]["arguments"]["search_queries"],
            [documentation.TOPICS["requests_redirects"]],
        )
        UUID(research.session_id)
        for request in server.requests:
            self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            self.assertEqual(request.headers["user-agent"], documentation.USER_AGENT)
            self.assertIn("vulnhuntr", request.headers["user-agent"].lower())
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("x-api-key", request.headers)
            self.assertNotIn("fixture-key-never-forward", request.content.decode())
            self.assertNotIn("fixture-llm-key-never-forward", request.content.decode())
        methods = [
            json.loads(request.content)["method"]
            for request in server.requests
            if request.method == "POST"
        ]
        self.assertIn("initialize", methods)
        self.assertIn("tools/list", methods)
        self.assertIn("tools/call", methods)

    def test_structured_tool_results_are_consumed(self):
        server = SearchServer(structured=True)
        with server.patch():
            research = documentation.DocumentationSearch()
            research.collect(["requests_redirects"])
        self.assertIn("https://docs.example.org/api#redirects", research.render())

    def test_invalid_topics_never_enter_queries(self):
        server = SearchServer()
        private_query = "internal/service.py customer-secret-token"
        research = documentation.DocumentationSearch()
        with server.patch():
            self.assertFalse(research.can_collect([private_query]))
            self.assertFalse(research.collect([private_query]))
            research.collect(["requests_redirects", private_query])
        self.assertEqual(len(server.calls), 1)
        self.assertNotIn(private_query, json.dumps(server.calls))
        self.assertNotIn(private_query, research.render())

    def test_disabled_search_never_creates_http_client(self):
        research = documentation.DocumentationSearch(enabled=False)
        with mock.patch.object(documentation.httpx, "AsyncClient") as client:
            self.assertFalse(research.can_collect(["requests_redirects"]))
            self.assertFalse(research.collect(["requests_redirects"]))
        client.assert_not_called()

    def test_budget_and_dedup_cover_follow_up_calls(self):
        server = SearchServer()
        research = documentation.DocumentationSearch(max_searches=2)
        with server.patch():
            self.assertTrue(research.collect(["requests_redirects", "requests_redirects"]))
            self.assertFalse(research.collect(["requests_redirects"]))
            self.assertTrue(research.can_collect(["urllib_urljoin"]))
            self.assertTrue(research.collect(["urllib_urljoin", "html_escaping"]))
            self.assertFalse(research.can_collect(["html_escaping"]))
            self.assertFalse(research.collect(["html_escaping"]))
        self.assertEqual(len(server.calls), 2)
        self.assertEqual(len(research.attempted), 2)

    def test_result_urls_reject_credentials_and_non_http_schemes(self):
        urls = [
            "https://docs.example.org/safe",
            "http://docs.example.org/also-safe",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "https://alice:secret@example.org/private",
            "//example.org/relative",
        ]
        server = SearchServer(
            results=[{"url": url, "title": "Documentation", "excerpts": ["API behavior"]} for url in urls]
        )
        with server.patch():
            research = documentation.DocumentationSearch()
            research.collect(["requests_redirects"])
        rendered = research.render()
        for url in urls[:2]:
            self.assertIn(url, rendered)
        for url in urls[2:]:
            self.assertNotIn(url, rendered)
        self.assertNotIn("alice:secret", rendered)

    def test_failure_is_bounded_sanitized_and_deduplicated(self):
        for error in ("http_quota", "rpc", "tool"):
            with self.subTest(error=error):
                server = SearchServer(error=error)
                research = documentation.DocumentationSearch(max_searches=1)
                with server.patch():
                    self.assertTrue(research.collect(["requests_redirects"]))
                    self.assertFalse(research.collect(["requests_redirects"]))
                    self.assertFalse(research.can_collect(["urllib_urljoin"]))
                self.assertIn("Unavailable:", research.render())
                self.assertNotIn("private provider detail", research.render())
                self.assertEqual(len(server.calls), 1)

    def test_transport_timeout_returns_unavailable_prompt_context(self):
        server = SearchServer(delay=1)
        research = documentation.DocumentationSearch(timeout=0.03)
        started = time.monotonic()
        with server.patch():
            self.assertTrue(research.collect(["requests_redirects"]))
        self.assertLess(time.monotonic() - started, 2)
        self.assertIn("Unavailable:", research.render())

    def test_malformed_results_are_a_sanitized_failure(self):
        server = SearchServer(results="private malformed provider response")
        with server.patch():
            research = documentation.DocumentationSearch()
            research.collect(["requests_redirects"])
        self.assertIn("Unavailable:", research.render())
        self.assertNotIn("private malformed provider response", research.render())

    def test_reference_count_and_large_excerpts_are_bounded(self):
        server = SearchServer(results=[
            {"url": f"https://docs.example.org/{index}", "title": "API", "excerpts": ["x" * 20000]}
            for index in range(20)
        ])
        with server.patch():
            research = documentation.DocumentationSearch()
            research.collect(["requests_redirects"])
        references = json.loads(research.results["requests_redirects"])
        self.assertGreater(len(references), 0)
        self.assertLessEqual(len(references), 3)
        self.assertLess(len(research.results["requests_redirects"]), 6500)

    def test_cancellation_and_keyboard_interrupt_propagate(self):
        research = documentation.DocumentationSearch()
        with mock.patch.object(research, "_search", new=mock.AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                research.collect(["requests_redirects"])

        def interrupt(coroutine):
            # Model Ctrl-C at the synchronous runner boundary. Raising it inside
            # a mocked child task leaks an asyncio task warning on Python 3.10.
            coroutine.close()
            raise KeyboardInterrupt

        research = documentation.DocumentationSearch()
        with mock.patch.object(documentation.asyncio, "run", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                research.collect(["requests_redirects"])

    def test_search_follow_ups_keep_one_session_and_new_runs_get_new_ids(self):
        server = SearchServer()
        first = documentation.DocumentationSearch()
        second = documentation.DocumentationSearch()
        with server.patch():
            first.collect(["requests_redirects"])
            second.collect(["requests_redirects"])
            first.collect(["urllib_urljoin"])
        ids = [call["arguments"]["session_id"] for call in server.calls]
        self.assertEqual(ids, [first.session_id, second.session_id, first.session_id])
        self.assertNotEqual(first.session_id, second.session_id)

    def test_quota_response_does_not_replace_the_analysis_session_id(self):
        server = SearchServer(error="http_quota")
        research = documentation.DocumentationSearch()
        original_id = research.session_id
        with server.patch():
            research.collect(["requests_redirects"])
            server.error = None
            research.collect(["urllib_urljoin"])
        self.assertEqual(research.session_id, original_id)
        self.assertEqual(
            [call["arguments"]["session_id"] for call in server.calls],
            [original_id, original_id],
        )

    def test_concurrent_runs_keep_separate_session_ids(self):
        server = SearchServer(delay=0.01)
        runs = [documentation.DocumentationSearch(), documentation.DocumentationSearch()]
        with server.patch(), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda run: run.collect(["requests_redirects", "urllib_urljoin"]), runs))
        self.assertEqual(results, [True, True])
        ids = [call["arguments"]["session_id"] for call in server.calls]
        self.assertEqual(ids.count(runs[0].session_id), 2)
        self.assertEqual(ids.count(runs[1].session_id), 2)
        self.assertNotEqual(runs[0].session_id, runs[1].session_id)


if __name__ == "__main__":
    unittest.main()
