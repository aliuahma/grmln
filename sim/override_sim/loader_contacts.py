"""CAD-derived planar shell contacts preserving the loader's lower cutouts."""
import json
import xml.etree.ElementTree as ET

import numpy as np

from .field import ASSETS, numbers


def replace_loader_contacts(root):
    """Replace the four-box sleeve approximation with convex CAD wall panels.

    The source faces are the two main side walls and the front wall, extruded
    inward by their measured 3 mm thickness. Planar boundaries retain the two
    holes in each side wall and the diagonal bottom-front cutouts. Handles and
    narrow molded transition fillets remain visual only, as in the old proxy.
    """
    profile = json.loads((ASSETS / 'loader_contacts.json').read_text())
    asset = root.find('asset')
    for index, vertices in enumerate(profile['panels']):
        if asset.find(f"mesh[@name='loader_shell_panel_{index}']") is None:
            ET.SubElement(asset, 'mesh', name=f'loader_shell_panel_{index}', vertex=numbers(np.asarray(vertices).ravel()))
    # The slide joint owns the sleeve/guide bearing constraint. Guide boxes
    # overlap the real wall thickness, so exclude that bearing pair only.
    for geom in root.findall('worldbody/geom'):
        if geom.get('name', '').startswith('loader_guide_'):
            geom.set('contype', '32')
            geom.set('conaffinity', '11')
    for body in root.findall('worldbody/body'):
        if not body.get('name', '').startswith('loader_'):
            continue
        for geom in list(body.findall('geom')):
            if geom.get('class') == 'collision':
                body.remove(geom)
        for index in range(len(profile['panels'])):
            ET.SubElement(body, 'geom', name=f'{body.get("name")}_shell_{index}',
                          **{'class': 'collision'}, type='mesh', mesh=f'loader_shell_panel_{index}',
                          contype='16', conaffinity='11')
    return {key: value for key, value in profile.items() if key != 'panels'}
