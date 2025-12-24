from pathlib import Path
from bs4 import BeautifulSoup

def _load_index():
    p = Path('results/website_dressa_inspired/index.html')
    assert p.exists(), 'index.html must exist'
    return BeautifulSoup(p.read_text(encoding='utf-8'), 'html.parser')


def test_nav_links_and_sections():
    soup = _load_index()
    # nav links exist
    nav = soup.find('nav')
    assert nav is not None
    links = [a.get('href') for a in nav.find_all('a')]
    assert 'catalog.html' in links and 'contacts.html' in links
    # hero checks
    assert soup.find('section', {'class':'hero'}) is not None
    # reserve button
    assert soup.select_one('#reserveBtn') is not None
    # footer
    assert soup.find('footer') is not None
