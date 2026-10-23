"""Auditable minimal AST derivation; original source files remain immutable."""
import ast
import inspect
import textwrap

def derive(function,kind,overrides=None):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)));changes=[]
    class Cadence(ast.NodeTransformer):
        def visit_Tuple(self,node):
            self.generic_visit(node)
            if kind=='contract' and [getattr(v,'value',None) for v in node.elts]==[50,10,17]:
                node.elts[1]=ast.copy_location(ast.Constant(50),node.elts[1]);changes.append('declared_generation50_execution50_motor17')
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            if kind=='init' and len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple) and isinstance(node.value,ast.Tuple):
                targets=node.targets[0].elts
                if [getattr(v,'attr',None) for v in targets]==['act_step','state_obs_path','strict_action_dim']:
                    assert node.value.elts[0].value==10;node.value.elts[0]=ast.copy_location(ast.Constant(50),node.value.elts[0]);changes.append('model_act_step50')
            return node
        def visit_keyword(self,node):
            self.generic_visit(node)
            if kind in ('init','action') and node.arg=='native_executed' and isinstance(node.value,ast.Constant):
                assert node.value.value==10;node.value=ast.copy_location(ast.Constant(50),node.value);changes.append('logged_native_executed50')
            return node
        def visit_Slice(self,node):
            self.generic_visit(node)
            if kind=='action' and node.lower is None and isinstance(node.upper,ast.Constant) and node.upper.value==10:
                node.upper=ast.copy_location(ast.Constant(50),node.upper);changes.append('queue_first50')
            return node
        def visit_Constant(self,node):
            if kind=='action' and node.value=='[current student prefix10 replan] ':
                changes.append('replan_tag_prefix50');return ast.copy_location(ast.Constant('[current student prefix50 replan] '),node)
            return node
    expected={'contract':['declared_generation50_execution50_motor17'],
              'init':['model_act_step50','logged_native_executed50'],
              'action':['queue_first50','logged_native_executed50','replan_tag_prefix50']}[kind]
    tree=Cadence().visit(tree);assert sorted(changes)==sorted(expected),(kind,changes)
    ast.fix_missing_locations(tree);scope=dict(function.__globals__,**(overrides or {}))
    exec(compile(tree,str(__file__)+'#'+kind,'exec'),scope)
    result=scope[function.__name__];result.cadence_only_AST_changes=changes
    return result
