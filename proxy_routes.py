from flask import Blueprint, request, jsonify
import requests
from flask_cors import CORS

proxy_bp = Blueprint('proxy', __name__)

# NASA Exoplanet Archive API
NASA_EXOPLANET_TAP_API = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'

@proxy_bp.route('/api/exoplanetarchive/exoplanets', methods=['GET'])
def proxy_exoplanets():
    """
    Proxy for NASA Exoplanet Archive TAP API to avoid CORS issues
    """
    try:
        # Get query parameters
        table = request.args.get('table', 'ps')
        format_type = request.args.get('format', 'json')

        # Build the query
        select_fields = 'pl_name,hostname,pl_orbper,pl_rade,pl_masse,pl_eqt,st_dist,pl_status,pl_discmethod,pl_radj,pl_massj,st_teff,st_rad,st_mass,ra,dec,pl_orbeccen,pl_orbsmax'
        query = f'SELECT {select_fields} FROM {table} WHERE pl_name IS NOT NULL'

        params = {
            'query': query,
            'format': format_type,
        }

        # Make request to NASA API
        response = requests.get(NASA_EXOPLANET_TAP_API, params=params)

        if not response.ok:
            return jsonify({
                'error': 'NASA API error',
                'message': f'Failed to fetch from NASA API: {response.status_code}',
                'details': response.text
            }), response.status_code

        # Return the response as JSON
        return jsonify(response.json())

    except Exception as e:
        return jsonify({
            'error': 'Proxy error',
            'message': str(e)
        }), 500

@proxy_bp.route('/api/exoplanetarchive/confirmed', methods=['GET'])
def proxy_confirmed_exoplanets():
    """
    Proxy for confirmed exoplanets only
    """
    try:
        # Get all exoplanets
        table = request.args.get('table', 'ps')
        format_type = request.args.get('format', 'json')

        select_fields = 'pl_name,hostname,pl_orbper,pl_rade,pl_masse,pl_eqt,st_dist,pl_status,pl_discmethod,pl_radj,pl_massj,st_teff,st_rad,st_mass,ra,dec,pl_orbeccen,pl_orbsmax'
        query = f'SELECT {select_fields} FROM {table} WHERE pl_name IS NOT NULL'

        params = {
            'query': query,
            'format': format_type,
        }

        response = requests.get(NASA_EXOPLANET_TAP_API, params=params)

        if not response.ok:
            return jsonify({
                'error': 'NASA API error',
                'message': f'Failed to fetch from NASA API: {response.status_code}'
            }), response.status_code

        data = response.json()

        # Filter for confirmed exoplanets only
        if 'data' in data and data['data']:
            confirmed = [planet for planet in data['data'] if planet.get('pl_status') == 'Confirmed']
            data['data'] = confirmed
            data['metadata'] = data.get('metadata', [])

        return jsonify(data)

    except Exception as e:
        return jsonify({
            'error': 'Proxy error',
            'message': str(e)
        }), 500

@proxy_bp.route('/api/exoplanetarchive/count', methods=['GET'])
def proxy_exoplanet_count():
    """
    Get total count of exoplanets
    """
    try:
        table = request.args.get('table', 'ps')
        query = f'SELECT COUNT(*) as total FROM {table} WHERE pl_name IS NOT NULL'

        params = {
            'query': query,
            'format': 'json',
        }

        response = requests.get(NASA_EXOPLANET_TAP_API, params=params)

        if response.ok:
            data = response.json()
            return jsonify(data)
        else:
            return jsonify({
                'error': 'NASA API error',
                'message': f'Failed to fetch count: {response.status_code}'
            }), response.status_code

    except Exception as e:
        return jsonify({
            'error': 'Proxy error',
            'message': str(e)
        }), 500
