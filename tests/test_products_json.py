from pathlib import Path

def test_products_json_exists():
    p = Path('results/website_dressa_inspired/data/products.json')
    assert p.exists(), 'products.json must exist'
    data = p.read_text(encoding='utf-8')
    assert '"id":1' in data


def test_js_has_fetch_fallback():
    p = Path('results/website_dressa_inspired/js/main.js')
    s = p.read_text(encoding='utf-8')
    assert 'fetch(\'data/products.json\')' in s or 'PRODUCTS = [' in s
