import pytest
from unittest.mock import patch, MagicMock
from agent.browser import Browser

@pytest.fixture
def browser():
    b = Browser()
    # Mock playwright to avoid actual browser launching in these unit tests
    b.page = MagicMock()
    b._page = MagicMock(return_value=b.page)
    b._settle = MagicMock()
    return b

def test_goto_allowed_url(browser):
    with patch('socket.gethostbyname', return_value='8.8.8.8'):
        result = browser.goto('https://www.google.com')
        assert result['ok'] is True
        assert result['message'] == 'successfully navigated'
        browser._page().goto.assert_called_once_with('https://www.google.com', wait_until='domcontentloaded', timeout=15000)

def test_goto_disallowed_scheme(browser):
    result = browser.goto('file:///etc/passwd')
    assert result['ok'] is False
    assert result['message'] == 'navigation failed: blocked by security policy'
    assert 'scheme must be http or https' in str(result['data']['error'])
    browser._page().goto.assert_not_called()

def test_goto_missing_hostname(browser):
    result = browser.goto('http://')
    assert result['ok'] is False
    assert result['message'] == 'navigation failed: blocked by security policy'
    assert 'missing hostname' in str(result['data']['error'])
    browser._page().goto.assert_not_called()

def test_goto_localhost(browser):
    with patch('socket.gethostbyname', return_value='127.0.0.1'):
        result = browser.goto('http://localhost:8080')
        assert result['ok'] is False
        assert result['message'] == 'navigation failed: blocked by security policy'
        assert 'resolved to an internal/private IP' in str(result['data']['error'])
        browser._page().goto.assert_not_called()

def test_goto_private_ip(browser):
    with patch('socket.gethostbyname', return_value='192.168.1.100'):
        result = browser.goto('http://192.168.1.100/admin')
        assert result['ok'] is False
        assert result['message'] == 'navigation failed: blocked by security policy'
        assert 'resolved to an internal/private IP' in str(result['data']['error'])
        browser._page().goto.assert_not_called()

def test_goto_dns_rebinding_simulation(browser):
    # Simulates an attacker using a public domain that resolves to an internal IP (like 127.0.0.1.nip.io)
    with patch('socket.gethostbyname', return_value='127.0.0.1'):
        result = browser.goto('http://127.0.0.1.nip.io')
        assert result['ok'] is False
        assert result['message'] == 'navigation failed: blocked by security policy'
        assert 'resolved to an internal/private IP' in str(result['data']['error'])
        browser._page().goto.assert_not_called()

def test_goto_invalid_hostname(browser):
    with patch('socket.gethostbyname', side_effect=Exception("Name or service not known")):
        result = browser.goto('http://invalid-hostname-that-doesnt-exist')
        assert result['ok'] is False
        assert result['message'] == 'navigation failed'
        assert 'hostname resolution failed: Name or service not known' in str(result['data']['error'])
        browser._page().goto.assert_not_called()
