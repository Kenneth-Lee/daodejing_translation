from docutils.nodes import NodeVisitor, Text, TextElement

def setup(app):
    app.connect('doctree-resolved', process_chinese_para)

def process_chinese_para(app, doctree, docname):
    doctree.walk(ParaVisitor(doctree))

class ParaVisitor(NodeVisitor):
    def dispatch_visit(self, node):
        if isinstance(node, TextElement):
            for i in range(len(node.children)):
                if type(node[i]) == Text:
                    node[i] = Text(node[i].astext().replace('\r', '').replace('\n', ''))
