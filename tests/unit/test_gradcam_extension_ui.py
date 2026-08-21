from __future__ import annotations

from pathlib import Path

from trustcxr.serving.api import create_app

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = ROOT / "src/trustcxr/serving/static/extensions/explainability"


def test_explainability_extension_route_is_registered() -> None:
    routes = {
        (method, route.path)
        for route in create_app().routes
        if route.path
        for method in (route.methods or set())
    }
    assert ("GET", "/research/explainability/ui") in routes
    assert ("GET", "/research/explainability/ui/app.css") in routes
    assert ("GET", "/research/explainability/ui/app.js") in routes


def test_extension_ui_is_separate_and_governed() -> None:
    html = (EXTENSION_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (EXTENSION_ROOT / "app.js").read_text(encoding="utf-8")
    assert "Model Attribution" in html
    assert 'id="image-input"' in html
    assert 'id="attribution-label"' in html
    assert 'id="generate-attribution"' in html
    assert 'id="original"' in html
    assert 'id="heatmap"' in html
    assert 'id="overlay"' in html
    assert "Research attribution only — not lesion localization or diagnostic evidence." in html
    assert "Color intensity reflects relative attribution strength within this map." in html
    assert "features.norm5" in javascript or "features.norm5" in html
    assert "7 × 7" in html
    assert 'fetch("/research/explainability/gradcam"' in javascript
    assert '"X-TrustCXR-Attribution-Label"' in javascript
    assert "No non-degenerate Grad-CAM attribution was available" in javascript
    assert "This does not imply absence of pathology." in javascript
    assert "lesion localization" in html
    assert "diagnostic" in html
    assert "http://" not in html + javascript
    assert "https://" not in html + javascript
    assert "localStorage" not in html + javascript
    assert "sessionStorage" not in html + javascript
    assert "indexedDB" not in html + javascript


def test_core_ui_assets_do_not_contain_extension_surface() -> None:
    static_root = ROOT / "src/trustcxr/serving/static"
    for filename in ("index.html", "app.js", "app.css"):
        content = (static_root / filename).read_text(encoding="utf-8")
        assert "OPTIONAL_GRADCAM_ATTRIBUTION_PANEL" not in content
        assert "/research/explainability/gradcam" not in content
