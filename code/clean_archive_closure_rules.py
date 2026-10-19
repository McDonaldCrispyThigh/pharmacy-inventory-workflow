"""Narrow offline URL-binding correction; does not change the frozen crawler.

Rules are restricted to the observed site's complete route. No path component,
store ID, nontracking query, subdomain, or capture timestamp may be discarded.
This is a research binding rule, not a claim that every HTTP path ignores case.
"""
import json
import re
from urllib.parse import unquote

from wayback_snapshot_fetcher import website_key, parse_replay

VERSION = 'clean-archive-closure-binding-v1'
ROUTES = {
    'walgreens.com': r'/locator/walgreens-[a-z0-9+.-]+/id=[0-9]+',
    'costco.com': r'/warehouse-locations/[a-z0-9-]+-[0-9]+\.html',
    'locations.michaels.com': r'/mn/[a-z-]+/[0-9]+',
    'mhealthfairview.org': r'/locations/m-health-fairview-[a-z-]+',
    'regencyhospital.com': r'/locations-and-tours/mn/golden-valley/minneapolis',
    'fairview.org': r'/pharmacy(?:/fairview-home-infusion)?',
}
TRACKING = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term'}
CATEGORIES = {
    'A': 'harmless URL normalization',
    'B': 'historical path variation on the same bound site',
    'C': 'query normalization / tracking difference',
    'D': 'ambiguous same-domain change',
    'E': 'cross-domain or cross-entity mismatch',
    'F': 'anomalous snapshot/status response',
    'G': 'other technical validation failure',
}


def registered_domain(host):
    # Reporting only, never an acceptance test. All observed audit hosts end in
    # the single-label public suffix .com or .org; other suffixes fail closed.
    labels = host.split('.')
    return '.'.join(labels[-2:]) if len(labels) >= 2 and labels[-1] in {'com', 'org'} else None


def query_key(query):
    return tuple((k, sep, unquote(v, encoding='utf-8', errors='strict') if k in TRACKING else v)
                 for k, sep, v in query)


def compare_urls(website, original):
    try:
        a, b = website_key(website), website_key(original)
        aq, bq = query_key(a[3]), query_key(b[3])
    except (ValueError, TypeError, UnicodeError):
        return {'accepted': False, 'category': 'G', 'rule': 'invalid_url'}
    result = {
        'normalized_hostname_comparison': {'frozen': a[0], 'returned': b[0], 'equal': a[:2] == b[:2],
                                           'frozen_port': a[1], 'returned_port': b[1]},
        'normalized_path_comparison': {'frozen': a[2], 'returned': b[2], 'equal': a[2] == b[2],
                                       'ascii_case_only': a[2].isascii() and b[2].isascii() and a[2].lower() == b[2].lower()},
        'query_parameter_comparison': {'frozen': a[3], 'returned': b[3], 'equal': a[3] == b[3],
                                       'known_tracking_percent_equivalent': aq == bq},
        'registered_domain_comparison': {'frozen': registered_domain(a[0]), 'returned': registered_domain(b[0]),
                                         'equal': registered_domain(a[0]) is not None and registered_domain(a[0]) == registered_domain(b[0])},
    }
    def done(accepted, category, rule):
        return {**result, 'accepted': accepted, 'category': category, 'rule': rule}
    if a[:2] != b[:2]:
        return done(False, 'D' if result['registered_domain_comparison']['equal'] else 'E', 'authority_mismatch')
    if a == b:
        return done(True, 'A', 'frozen_normalization_equal')
    if aq != bq:
        return done(False, 'D', 'query_values_not_equivalent')
    if a[2] == b[2]:
        return done(True, 'C', 'known_tracking_percent_encoding_only')
    route = ROUTES.get(a[0])
    if (route and result['normalized_path_comparison']['ascii_case_only']
            and re.fullmatch(route, a[2].lower())):
        return done(True, 'A', 'same_complete_site_route_ascii_case_only')
    # A stable numeric route ID can describe a historical slug variation, but
    # this closure does NOT accept it. Changed IDs are a genuine entity mismatch.
    if a[0] in {'walgreens.com', 'costco.com', 'locations.michaels.com'}:
        ids = [re.findall(r'\d+', p) for p in (a[2], b[2])]
        if ids[0] and ids[1] and ids[0] != ids[1]:
            return done(False, 'E', 'location_numeric_identifier_changed')
        if ids[0] and ids[0] == ids[1]:
            return done(False, 'B', 'historical_slug_variation_not_accepted')
    return done(False, 'D', 'unapproved_path_change')


def revalidate(body, website, requested, cutoff):
    """Validate original official bytes, without editing or laundering a URL."""
    def unique(pairs):
        value = {}
        for k, v in pairs:
            if k in value:
                raise ValueError('duplicate JSON key')
            value[k] = v
        return value
    try:
        payload = json.loads(body, object_pairs_hook=unique)
        if not isinstance(payload, dict) or not isinstance(payload.get('archived_snapshots'), dict):
            raise ValueError('schema')
        if 'url' in payload and not compare_urls(website, payload['url'])['accepted']:
            return {'state': 'failure', 'category': 'G', 'reason': 'response_website_binding_failed'}
        if 'timestamp' in payload and str(payload['timestamp']) != requested:
            return {'state': 'failure', 'category': 'F', 'reason': 'requested_timestamp_mismatch'}
        closest = payload['archived_snapshots'].get('closest')
        if not isinstance(closest, dict) or type(closest.get('available')) is not bool:
            raise ValueError('closest schema')
        replay = parse_replay(closest['url'])
        comparison = compare_urls(website, replay['original_url'])
        metadata = {'returned_archived_url': closest['url'], 'returned_original_url': replay['original_url'],
                    'returned_timestamp': closest.get('timestamp'), 'returned_status': closest.get('status'),
                    'comparison': comparison}
        if (not closest['available'] or str(closest.get('status')) != '200'
                or replay['timestamp'] != str(closest.get('timestamp')) or replay['timestamp'] > cutoff):
            return {**metadata, 'state': 'failure', 'category': 'F', 'reason': 'anomalous_snapshot_or_status'}
        if not comparison['accepted']:
            return {**metadata, 'state': 'failure', 'category': comparison['category'], 'reason': comparison['rule']}
        return {**metadata, 'state': 'confirmed', 'category': comparison['category'], 'reason': comparison['rule'],
                'confirmed_snapshot_timestamp': replay['timestamp'], 'original_target_url': replay['original_url'],
                'exact_wayback_snapshot_url': f'https://web.archive.org/web/{replay["timestamp"]}id_/{replay["original_url"]}'}
    except (ValueError, TypeError, KeyError, UnicodeError):
        return {'state': 'failure', 'category': 'G', 'reason': 'invalid_official_response'}
