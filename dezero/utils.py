import os
import subprocess

from . import Function, Variable


def _dot_var(v: Variable, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f: Function):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))

    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    functions = []
    seen = set()

    def add_func(f):
        if f not in seen:
            functions.append(f)
            seen.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while functions:
        f = functions.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    import numpy as np
    from . import train
    from .functions import goldstein

    x = Variable(np.array(1.0), name='x')
    y = Variable(np.array(1.0), name='y')
    with train():
        z = goldstein(x, y)
        z.name = 'z'
        z.backward()

    plot_dot_graph(z, verbose=False, to_file='goldstein.png')
