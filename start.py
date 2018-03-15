import ipywidgets as ipw


def get_start_widget(appbase, jupbase):
    template = """
    <table>
    <tr>
        <th style="text-align:center">NEB</th>
    </tr>
    <tr>
    <td valign="top"><ul>
    <li><a href="{appbase}/upload_structure.ipynb" target="_blank">Upload structures</a>
    <li><a href="{appbase}/replicas.ipynb" target="_blank">Generate replicas</a>
    <li><a href="{appbase}/NEB.ipynb" target="_blank">NEB</a>
    <li><a href="{appbase}/Search.ipynb" target="_blank">Search</a>
    <li><a href="{appbase}/../../../tree/apps/reactions" target="_blank">File Tree</li>
    </ul></td>
    </tr>
"""

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
