"""OpenTelemetry tracing: one trace per query, one span per pipeline stage.

Design rules:
  - GRACEFUL NO-OP. If the OpenTelemetry libraries are missing, or the OTLP
    (OpenTelemetry Protocol) endpoint is not configured, every helper here
    degrades to doing nothing. Tracing must never be able to break a query.
  - ONE integration point. The pipeline already wraps every stage in
    `timings.stage(...)` for Prometheus; that same context manager now also
    opens a span, so stages can never drift between metrics and traces.

Where traces go locally: the Jaeger all-in-one container from
observability/docker-compose.yml, which listens for OTLP on :4318 and serves
the trace UI on http://localhost:16686.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

_tracer = None          # stays None unless init_tracing() succeeds


def init_tracing() -> bool:
    """Set up the tracer once at startup. Returns True if tracing is live."""
    global _tracer
    # settings, not os.getenv: .env values are loaded by pydantic-settings into
    # the Settings object and are NOT exported into the process environment
    from app.config import get_settings
    s = get_settings()
    endpoint = (getattr(s, "otel_exporter_otlp_endpoint", "")
                or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip()
    service = (getattr(s, "otel_service_name", "")
               or os.getenv("OTEL_SERVICE_NAME", "advanced-rag-system"))
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({
            "service.name": service,
        }))
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("rag")
        return True
    except Exception:                       # noqa: BLE001 - never break the app
        _tracer = None
        return False


@contextmanager
def span(name: str, **attributes):
    """Open a child span, or do nothing when tracing is off."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as sp:
        for key, value in attributes.items():
            if value is not None:
                sp.set_attribute(key, value)
        yield sp
