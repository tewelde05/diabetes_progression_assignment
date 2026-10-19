"""Execute this report's explicit Python cells and embed tables and figures.

This lightweight batch executor uses one shared Python namespace. Interactive
use remains available through any Jupyter frontend with the project kernel.
"""

import base64
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def execute_notebook(path: Path) -> dict:
    """Run explicit Python cells sequentially; raise immediately on any failure."""
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    outputs, stream, offset = [], io.StringIO(), 0
    figure_count = 0

    def flush_text():
        nonlocal offset
        text = stream.getvalue()[offset:]
        if text:
            outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': text})
            offset = stream.tell()

    def display(value):
        flush_text()
        data = {'text/plain': repr(value)}
        if hasattr(value, '_repr_html_'):
            html = value._repr_html_()
            if html is not None:
                data['text/html'] = html
        outputs.append({'output_type': 'display_data', 'data': data, 'metadata': {}})

    def show(*args, **kwargs):
        nonlocal figure_count
        flush_text()
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight', facecolor='white')
            outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {
                'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                'text/plain': '<Matplotlib figure>',
            }})
            figure_count += 1
        plt.close('all')

    namespace['display'] = display
    original_show, original_cwd = plt.show, Path.cwd()
    plt.show = show
    os.chdir(path.parent)
    try:
        count = 0
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            count += 1
            outputs, stream, offset = [], io.StringIO(), 0
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell['source'], f'report_cell_{count}', 'exec'), namespace)
                flush_text()
            cell.update(execution_count=count, outputs=outputs)
        notebook['metadata']['language_info']['version'] = __import__('sys').version.split()[0]
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
        return {'code_cells': count, 'figures': figure_count, 'namespace': namespace}
    finally:
        plt.show = original_show
        os.chdir(original_cwd)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = execute_notebook(root / 'notebooks' / 'Diabetes_Progression_Report.ipynb')
    print(f'Executed {result["code_cells"]} code cells and embedded {result["figures"]} figures.')
