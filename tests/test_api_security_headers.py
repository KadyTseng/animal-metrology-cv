import asyncio
import importlib
import unittest
from unittest.mock import patch

from starlette.requests import Request
from starlette.responses import Response


with patch("src.metrology.AnimalMetrologyPipeline"):
    api = importlib.import_module("src.api")


class SecurityHeadersTests(unittest.TestCase):
    def test_ui_policy_allows_uploaded_image_previews(self):
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/ui/",
                "headers": [],
                "scheme": "http",
                "server": ("test", 80),
                "client": ("test", 1),
                "query_string": b"",
                "root_path": "",
                "http_version": "1.1",
            }
        )

        async def call_next(_request):
            return Response()

        response = asyncio.run(api.add_security_headers(request, call_next))

        self.assertIn(
            "img-src 'self' data: blob:",
            response.headers["Content-Security-Policy"],
        )
