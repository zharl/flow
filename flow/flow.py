# This code was written by Marco Renedo prior to joining OnTrade.
# Original creation date: 2021-06-01
# This code is being reused with permission from the original author and it is not part of the OnTrade codebase.

import inspect, pydot
from IPython.display import SVG, display, Markdown, HTML, clear_output
import os, pandas as pd, matplotlib as mpl, datetime as dt, numpy as np
import ipywidgets as widgets
import copy
import autopep8, re, inspect
import pickle

def get_gen(pars):
    updated = True
    gen = {k: 0 for k in pars.keys()}
    while updated:
        updated = False
        for k,ps in pars.items():
            for p in ps:
                if gen[k] < gen[p]+1:
                    gen[k] = gen[p]+1
                    updated = True
    g = {}
    for k,v in gen.items():
        if v not in g: g[v] = [k]
        else: g[v].append(k)
    return g
def get_order(pars): # Kahn's topological sort algorithm
    childcount = {k: 0 for k in pars}
    for x in pars:
        for y in pars[x]:
            if y not in childcount:
                raise Exception("Parent node '{}' of '{}' doesn't exist.".format(y,x))
            childcount[y] += 1
    order = [x for x in childcount if childcount[x] == 0]
    for y in order:
        for x in pars[y]:
            childcount[x] -= 1
            if childcount[x] == 0: order.append(x)
    if len(order) < len(pars):
        raise Exception('There is a cycle: '+', '.join({k for k in pars if k not in order})+'.')
    return order
def val(v): # displays the value if it's not a large object
    if type(v) in [pd.DataFrame, pd.Series, np.ndarray]: label = v.__class__.__name__ + str(v.shape)
    elif type(v) in [dt.datetime, dt.date, pd.Timestamp]: label = v.strftime('%Y-%m-%d')
    elif type(v) in [list,dict,set]:label = v.__class__.__name__ + str(len(v))
    elif type(v) in [int, float, np.float64,bool]: label = v.__repr__()
    elif type(v) in [slice]: label = str(v.start) + ', ' + str(v.stop)
    elif type(v) in [str]: label=v[:10]
    elif v is None: label = 'None'
    else: label = v.__class__.__name__
    return label
class flow:
    def __init__(self, *args, **kwargs):
        self._values = {}
        self._fixed = set()
        self._defined = list(kwargs.keys())
        if len(kwargs) == 0 and len(args)==1:
            self._functions = args[0]._functions
            self._parents = args[0]._parents
            self._order = args[0]._order
        else:
            self._functions = {}
            for x in args:
                if type(x) == flow: self._functions.update(x._functions)
                elif type(x) == dict: self._functions.update(x)
            for k,f in kwargs.items():
                if k == 'doc' and k in self._functions: self._functions[k] += " "+kwargs['doc']
                elif f == None: del self._functions[k]
                else: self._functions[k] = f
            self._parents = {}
            for k,f in self._functions.items():
                if k == 'doc' or k == 'name': continue
                self._parents[k] = list(inspect.signature(f)._parameters)
            self._order = get_order(self._parents)
    def _name(self):
        return self._functions['name'] if 'name' in self._functions else ''
    def _doc(self):
        return self._functions['doc']
    def __getattr__(self, k):
        name = self._name()
        if k[0] != '_' and k[-1] == '_':
            k = k[:-1]
            self._touch(k)
        if k not in self._parents:
            s = "{} has no node {}. Nodes available: {}."
            s = s.format(name,k,', '.join(self._parents.keys()))
            raise Exception(s)
        if k in self._values: return self._values[k]
        tovalue = {i: False for i in self._parents.keys()}
        tovalue[k] = True
        for i in self._order:
            if not tovalue[i]: continue
            for j in self._parents[i]:
                if j not in self._values:
                    tovalue[j] = True
        for i in reversed(self._order):
            if tovalue[i]:
                pars = {j: self._values[j] for j in self._parents[i]}
                try: self._values[i] = self._functions[i](**pars)
                except Exception:
                    print(name+"."+i)
                    raise
        return self._values[k]
    def _touch(self, k):
        if k not in self._parents:
            raise Exception('`'+k+'` is not a node of `'+ self._name()+ '`')
        toclear = set([k])
        for i in reversed(self._order):
            for j in self._parents[i]:
                    if j in toclear:
                        toclear.add(i)
                        break
        toclear.remove(k)
        for i in toclear:
            if i in self._values:
                del self._values[i]
            if i in self._fixed:
                self._fixed.remove(i)
    def __repr__(self):
        import numpy as np
        classDoc = self._doc()
        G = pydot.Dot(rankdir='LR',pad='.05',nodesep='.03',outputorder='edgesfirst',
                      tooltip=classDoc,bgcolor="transparent",ranksep='.2',mincross='2.0',
                      splines='spline')
        for a in reversed(self._order):
            label = '<FONT face="verdana" POINT-SIZE="9" COLOR="BLUE">' + a.replace('_',' ') + '</FONT>'
            u  = (self._functions[a].__doc__ or '').split('\n')
            if u[0] != '':
                label += '<BR></BR><FONT POINT-SIZE="6" COLOR="RED">' + u[0] + '</FONT>'
            if len(u) > 1:
                classDoc += ' **'+a.replace('_',' ')+'**:'+', '.join(u[1:]) + ';'
            if a in self._values:
                label += '<BR></BR><FONT POINT-SIZE="6">' + val(self._values[a]) + '</FONT>'
            nodeA = pydot.Node(
                a, fillcolor='#EEEEEE' if a not in self._fixed else '#FFDDDD',
                color='#000000' if a not in self._defined else '#00AA00', penwidth='1', label='<'+label+'>',
                shape = 'box',style='filled', fontcolor='#888888',
                minsize='0', height='.1',width='.1', margin='0.03,0.03',
                tooltip=inspect.getsource(self._functions[a]).replace('\n', '&#10;')
            )
            G.add_node(nodeA)
        for b in self._order:
            for a in self._parents[b]:
                G.add_edge(pydot.Edge(a,b,tailport='e', color='#feb20930',
                                      penwidth=5, style='line',arrowhead='none'))
        try: display(Markdown(classDoc),HTML(G.create_svg().decode()))
        except: print(G.to_string())
        return self._name()
    def __call__(self, **kwargs):
        for k in kwargs:
            if k not in self._parents:
                raise Exception('`'+k+'` is not a node of `'+ self._name()+ '`')
        toclear = set(kwargs.keys())
        for i in reversed(self._order):
            for j in self._parents[i]:
                    if j in toclear:
                        toclear.add(i)
                        break
        for i in toclear:
            if i in self._values:
                del self._values[i]
            if i in self._fixed:
                self._fixed.remove(i)
        self._values.update(**kwargs)
        for i in kwargs:
            self._fixed.add(i)
        return self
def show_flows(d):
    for k,v in d.items():
        display(HTML(f'<FONT face="verdana" POINT-SIZE="14" COLOR="BLUE">{k}</FONT'))
        v.__repr__()
def print_flow(f):
    s = 'flow(\n'
    for k,v in f._functions.items():
        s += '\t'+k+'='
        if callable(v): s+= v.__name__
        else: s+= 'r'+f'"""{v}"""'
        s += ',\n'
    s = s[:-2] + '\n)'
    print(s)

def gui(u,k0=''):
    if hasattr(u,'_parents'): display(gui_flow(u,k0))
    elif type(u) == dict: display(gui_dict(u,k0))
    else: display(widgets.Label(k0+' : '+str(type(u))),u)
def gui_flow(a,k0=''):
    button_dict = {}
    output = widgets.Output()
    def f(x):
        with output:
            clear_output()
            u = getattr(a,x.description)
            gui(u,x.description)
        for k in button_dict:
                button_dict[k].style.button_color='orange' if k in a._fixed else 'lightgreen' if k in a._values else None
    blayout = widgets.Layout(display='flex',justify_content='flex-start',align_items='center',
                             width='auto',height='20px',padding='7px',margin='2px')
    for k in a._parents:
        button_dict[k] = widgets.Button(description=k, 
            layout=blayout, 
            tooltip=inspect.getsource(a._functions[k]))
        button_dict[k].on_click(f)
    gen = get_gen(a._parents)
    l = []
    i = 0
    layout = widgets.Layout(display='flex',justify_content='flex-start',align_content='flex-start')
    while i in gen:
        bs = [button_dict[k] for k in sorted(gen[i])]
        l.append(widgets.VBox(bs,layout=layout))
        i += 1
    for k in button_dict:
            button_dict[k].style.button_color='orange' if k in a._fixed else 'lightgreen' if k in a._values else None
    return widgets.VBox([widgets.Label(k0+' : '+a._name()),widgets.HBox(l),output])
def gui_dict(a,k0):
    button_dict = {}
    output = widgets.Output()
    def f(x):
        with output:
            clear_output()
            u = a[x.description]
            gui(u,x.description)
    for k in a.keys():
        button_dict[k] = widgets.Button(description=k, 
            layout=widgets.Layout(width='auto'))
        button_dict[k].on_click(f)
    layout = widgets.Layout(display='flex',justify_content='flex-start',align_content='flex-start',flex_flow='row wrap')
    l = widgets.VBox([button_dict[k] for k in button_dict],layout=layout)
    return widgets.VBox([widgets.Label(k0+' : dict'),l,output])


def save_flow(f,inputs, outputs):
    inputs = {k:getattr(f,k) for k in inputs}
    outputs = {k:getattr(f,k) for k in outputs}
    store = {'name':f._name(),'inputs':inputs,'outputs':outputs}
    filename = f._name().replace(' ','_')+pd.Timestamp.now().strftime('%Y%m%d-%H%M')+'.pk'
    with open(filename,'wb') as file:
        pickle.dump(store,file)
    return filename
    
def check_flow(f,filename):
    print('Checking '+filename+':')
    with open(filename,'rb') as file:
        f1 = pickle.load(file)
        assert f1['name'] == f._name()
        f(**f1['inputs'])
        outputs = f1['outputs']
        for k in outputs:
            print(f'- {k}: ',end='')
            try:
                np.testing.assert_array_almost_equal(outputs[k],getattr(f,k))
                print('Passed')
            except:
                print('Failed')
                display(Markdown('**Saved**'),outputs[k],Markdown('**Current**'),getattr(f,k))
def copy_flow(a):
    b = flow(a)
    b._values = copy.copy(a._values)
    return b

def flow_flatten(x):
    s = 'flow('
    s += f'name=\'{x._name()}\','
    s += f'doc=\'{x._doc()}\','
    for k,v in x._functions.items():
        if k=='name' or k=='doc': continue
        code = inspect.getsource(v)
        name = re.search(r'def (\w+)\(',code).groups()[0]
        if name=='': name = code
        s += f'{k}={name},'
    s += ')'
    return autopep8.fix_code(s,options={'aggressive': 5})
