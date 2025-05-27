from ply import lex, yacc
import graphviz
from typing import List, Dict, Any

# Lexer tokens
tokens = (
    'ID',
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'EQUALS',
    'SEMICOLON',
    'LPAREN',
    'RPAREN',
    'LBRACE',
    'RBRACE',
    'LBRACKET',
    'RBRACKET',
    'COMMA',
    'STRING',
    'MODULO',
    'INCREMENT',
    'DECREMENT',
    'AND',
    'OR',
    'NOT',
    'EQUALS_EQUALS',
    'NOT_EQUALS',
    'LESS_THAN',
    'GREATER_THAN',
    'LESS_EQUALS',
    'GREATER_EQUALS',
)

# Reserved keywords
reserved = {
    'int': 'INT',
    'float': 'FLOAT',
    'string': 'STRING_TYPE',
    'bool': 'BOOL',
    'void': 'VOID',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'for': 'FOR',
    'return': 'RETURN',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'true': 'TRUE',
    'false': 'FALSE',
    'function': 'FUNCTION',
    'array': 'ARRAY',
}

# Add reserved tokens to the token list
tokens = tokens + tuple(reserved.values())

# Token rules
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_MODULO = r'%'
t_EQUALS = r'='
t_SEMICOLON = r';'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_INCREMENT = r'\+\+'
t_DECREMENT = r'--'
t_AND = r'&&'
t_OR = r'\|\|'
t_NOT = r'!'
t_EQUALS_EQUALS = r'=='
t_NOT_EQUALS = r'!='
t_LESS_THAN = r'<'
t_GREATER_THAN = r'>'
t_LESS_EQUALS = r'<='
t_GREATER_EQUALS = r'>='

t_ignore = ' \t'

def t_STRING(t):
    r'"[^"]*"'
    t.value = t.value[1:-1]  # Remove quotes
    return t

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_NUMBER(t):
    r'\d+(\.\d+)?'
    t.value = float(t.value) if '.' in t.value else int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# Parser rules
def p_program(p):
    '''program : statement
               | program statement'''
    if len(p) == 2:
        p[0] = ['program', p[1]]
    else:
        p[0] = ['program', p[1], p[2]]

def p_statement(p):
    '''statement : declaration
                 | assignment
                 | expression
                 | if_statement
                 | while_statement
                 | for_statement
                 | function_declaration
                 | return_statement
                 | break_statement
                 | continue_statement
                 | block'''
    p[0] = p[1]

def p_block(p):
    '''block : LBRACE statement_list RBRACE'''
    p[0] = ['block', p[2]]

def p_statement_list(p):
    '''statement_list : statement
                     | statement_list statement'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_declaration(p):
    '''declaration : type ID EQUALS expression SEMICOLON
                  | type ID LBRACKET NUMBER RBRACKET SEMICOLON
                  | type ID SEMICOLON'''
    if len(p) == 6:
        p[0] = ['declaration', p[1], p[2], p[4]]
    elif len(p) == 7:
        p[0] = ['array_declaration', p[1], p[2], p[4]]
    else:
        p[0] = ['declaration', p[1], p[2], None]

def p_assignment(p):
    '''assignment : ID EQUALS expression SEMICOLON
                 | ID LBRACKET expression RBRACKET EQUALS expression SEMICOLON
                 | ID INCREMENT SEMICOLON
                 | ID DECREMENT SEMICOLON'''
    if len(p) == 5:
        p[0] = ['assignment', p[1], p[3]]
    elif len(p) == 8:
        p[0] = ['array_assignment', p[1], p[3], p[6]]
    else:
        p[0] = ['increment', p[1], p[2]]

def p_expression(p):
    '''expression : term
                  | expression PLUS term
                  | expression MINUS term
                  | expression AND term
                  | expression OR term'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ['binary', p[2], p[1], p[3]]

def p_term(p):
    '''term : factor
            | term TIMES factor
            | term DIVIDE factor
            | term MODULO factor
            | term EQUALS_EQUALS factor
            | term NOT_EQUALS factor
            | term LESS_THAN factor
            | term GREATER_THAN factor
            | term LESS_EQUALS factor
            | term GREATER_EQUALS factor'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ['binary', p[2], p[1], p[3]]

def p_factor(p):
    '''factor : NUMBER
              | STRING
              | ID
              | TRUE
              | FALSE
              | LPAREN expression RPAREN
              | ID LBRACKET expression RBRACKET
              | function_call
              | NOT factor'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = p[2]
    elif len(p) == 5:
        p[0] = ['array_access', p[1], p[3]]
    elif len(p) == 3:
        p[0] = ['unary', p[1], p[2]]

def p_function_declaration(p):
    '''function_declaration : FUNCTION type ID LPAREN parameter_list RPAREN block'''
    p[0] = ['function_declaration', p[2], p[3], p[5], p[7]]

def p_parameter_list(p):
    '''parameter_list : type ID
                     | parameter_list COMMA type ID
                     | empty'''
    if len(p) == 2:
        p[0] = []
    elif len(p) == 3:
        p[0] = [(p[1], p[2])]
    else:
        p[0] = p[1] + [(p[3], p[4])]

def p_function_call(p):
    '''function_call : ID LPAREN argument_list RPAREN'''
    p[0] = ['function_call', p[1], p[3]]

def p_argument_list(p):
    '''argument_list : expression
                    | argument_list COMMA expression
                    | empty'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = []

def p_if_statement(p):
    '''if_statement : IF LPAREN expression RPAREN block
                   | IF LPAREN expression RPAREN block ELSE block'''
    if len(p) == 6:
        p[0] = ['if', p[3], p[5]]
    else:
        p[0] = ['if_else', p[3], p[5], p[7]]

def p_while_statement(p):
    '''while_statement : WHILE LPAREN expression RPAREN block'''
    p[0] = ['while', p[3], p[5]]

def p_for_statement(p):
    '''for_statement : FOR LPAREN declaration expression SEMICOLON expression RPAREN block'''
    p[0] = ['for', p[3], p[4], p[6], p[8]]

def p_return_statement(p):
    '''return_statement : RETURN expression SEMICOLON
                       | RETURN SEMICOLON'''
    if len(p) == 4:
        p[0] = ['return', p[2]]
    else:
        p[0] = ['return', None]

def p_break_statement(p):
    '''break_statement : BREAK SEMICOLON'''
    p[0] = ['break']

def p_continue_statement(p):
    '''continue_statement : CONTINUE SEMICOLON'''
    p[0] = ['continue']

def p_type(p):
    '''type : INT
            | FLOAT
            | STRING_TYPE
            | BOOL
            | VOID
            | ARRAY'''
    p[0] = p[1]

def p_empty(p):
    'empty :'
    pass

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}'")
    else:
        print("Syntax error at EOF")

class Compiler:
    def __init__(self):
        self.lexer = lex.lex()
        self.parser = yacc.yacc()
        self.symbol_table = {}
        self.function_table = {}
        self.current_scope = "global"
        self.errors = []
        self.temp_counter = 0
        self.label_counter = 0
        self.ir_code = []
        self.machine_code = []
        self.linked_code = []

    def get_temp(self):
        """Generate a new temporary variable."""
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def get_label(self):
        """Generate a new label."""
        self.label_counter += 1
        return f"L{self.label_counter}"

    def generate_ir(self, ast: Any) -> List[Dict[str, Any]]:
        """Generate Three-Address Code (TAC) from AST."""
        self.ir_code = []
        self._generate_ir_node(ast)
        return self.ir_code

    def _generate_ir_node(self, node):
        """Recursively generate IR code from AST nodes."""
        if not isinstance(node, list):
            return str(node)  # Convert non-list nodes to strings

        if node[0] == 'program':
            for child in node[1:]:
                self._generate_ir_node(child)
            return None

        elif node[0] == 'declaration':
            var_type, var_name, expr = node[1], node[2], node[3]
            if expr is not None:
                result = self._generate_ir_node(expr)
                self.ir_code.append({
                    'op': '=',
                    'arg1': result,
                    'result': var_name
                })
            return var_name

        elif node[0] == 'binary':
            op = node[1]
            left = self._generate_ir_node(node[2])
            right = self._generate_ir_node(node[3])
            temp = self.get_temp()
            
            self.ir_code.append({
                'op': op,
                'arg1': left,
                'arg2': right,
                'result': temp
            })
            return temp

        elif node[0] == 'if':
            condition = self._generate_ir_node(node[1])
            false_label = self.get_label()
            end_label = self.get_label()
            
            self.ir_code.append({
                'op': 'if_false',
                'arg1': condition,
                'label': false_label
            })
            
            self._generate_ir_node(node[2])
            self.ir_code.append({
                'op': 'goto',
                'label': end_label
            })
            
            self.ir_code.append({
                'op': 'label',
                'label': false_label
            })
            
            if len(node) > 3:  # if-else
                self._generate_ir_node(node[3])
            
            self.ir_code.append({
                'op': 'label',
                'label': end_label
            })

        elif node[0] == 'while':
            start_label = self.get_label()
            end_label = self.get_label()
            
            self.ir_code.append({
                'op': 'label',
                'label': start_label
            })
            
            condition = self._generate_ir_node(node[1])
            self.ir_code.append({
                'op': 'if_false',
                'arg1': condition,
                'label': end_label
            })
            
            self._generate_ir_node(node[2])
            self.ir_code.append({
                'op': 'goto',
                'label': start_label
            })
            
            self.ir_code.append({
                'op': 'label',
                'label': end_label
            })

        elif node[0] == 'function_declaration':
            func_name = node[2]
            self.ir_code.append({
                'op': 'function',
                'name': func_name
            })
            
            # Generate IR for function body
            self._generate_ir_node(node[4])
            
            self.ir_code.append({
                'op': 'end_function',
                'name': func_name
            })

        return str(node[0])  # Return the node type as a string

    def generate_machine_code(self, ir_code: List[Dict[str, Any]]) -> List[str]:
        """Generate x86 assembly code from IR."""
        self.machine_code = []
        
        for instr in ir_code:
            if instr['op'] == '=':
                self.machine_code.append(f"mov eax, {instr['arg1']}")
                self.machine_code.append(f"mov [{instr['result']}], eax")
            
            elif instr['op'] in ['+', '-', '*', '/']:
                self.machine_code.append(f"mov eax, {instr['arg1']}")
                self.machine_code.append(f"mov ebx, {instr['arg2']}")
                
                if instr['op'] == '+':
                    self.machine_code.append("add eax, ebx")
                elif instr['op'] == '-':
                    self.machine_code.append("sub eax, ebx")
                elif instr['op'] == '*':
                    self.machine_code.append("mul ebx")
                elif instr['op'] == '/':
                    self.machine_code.append("div ebx")
                
                self.machine_code.append(f"mov [{instr['result']}], eax")
            
            elif instr['op'] == 'if_false':
                self.machine_code.append(f"cmp {instr['arg1']}, 0")
                self.machine_code.append(f"je {instr['label']}")
            
            elif instr['op'] == 'goto':
                self.machine_code.append(f"jmp {instr['label']}")
            
            elif instr['op'] == 'label':
                self.machine_code.append(f"{instr['label']}:")
            
            elif instr['op'] == 'function':
                self.machine_code.append(f"{instr['name']}:")
                self.machine_code.append("push ebp")
                self.machine_code.append("mov ebp, esp")
            
            elif instr['op'] == 'end_function':
                self.machine_code.append("mov esp, ebp")
                self.machine_code.append("pop ebp")
                self.machine_code.append("ret")
        
        return self.machine_code

    def link_code(self, machine_code: List[str]) -> str:
        """Generate a complete executable with necessary sections and linking."""
        self.linked_code = []
        
        # Add data section
        self.linked_code.append("section .data")
        self.linked_code.append("    ; Global variables")
        
        # Add bss section for uninitialized data
        self.linked_code.append("section .bss")
        self.linked_code.append("    ; Uninitialized variables")
        
        # Add text section with code
        self.linked_code.append("section .text")
        self.linked_code.append("global _start")
        self.linked_code.append("_start:")
        
        # Add the machine code
        self.linked_code.extend(machine_code)
        
        # Add exit code
        self.linked_code.append("    mov eax, 1")  # sys_exit
        self.linked_code.append("    xor ebx, ebx")  # exit code 0
        self.linked_code.append("    int 0x80")
        
        return "\n".join(self.linked_code)

    def lexical_analysis(self, code: str) -> List[Dict[str, Any]]:
        """Perform lexical analysis and return tokens."""
        self.lexer.input(code)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append({
                'type': tok.type,
                'value': tok.value,
                'line': tok.lineno,
                'lexpos': tok.lexpos
            })
        return tokens

    def syntax_analysis(self, code: str) -> Any:
        """Perform syntax analysis and return AST."""
        return self.parser.parse(code)

    def semantic_analysis(self, ast: Any) -> Dict[str, Any]:
        """Perform semantic analysis and return symbol table."""
        self.symbol_table = {}
        self.function_table = {}
        self.errors = []
        self._analyze_node(ast)
        return {
            'symbol_table': self.symbol_table,
            'function_table': self.function_table,
            'errors': self.errors
        }

    def _analyze_node(self, node):
        if not isinstance(node, list):
            return

        if node[0] == 'declaration':
            var_type, var_name, expr = node[1], node[2], node[3]
            if var_name in self.symbol_table:
                self.errors.append(f"Variable '{var_name}' already declared")
            else:
                self.symbol_table[var_name] = {
                    'type': var_type,
                    'scope': self.current_scope
                }
        elif node[0] == 'function_declaration':
            return_type, func_name, params, body = node[1], node[2], node[3], node[4]
            if func_name in self.function_table:
                self.errors.append(f"Function '{func_name}' already declared")
            else:
                self.function_table[func_name] = {
                    'return_type': return_type,
                    'parameters': params,
                    'scope': self.current_scope
                }
                # Analyze function body
                old_scope = self.current_scope
                self.current_scope = func_name
                self._analyze_node(body)
                self.current_scope = old_scope

    def generate_ast_graph(self, ast: Any) -> str:
        """Generate GraphViz DOT representation of AST."""
        dot = graphviz.Digraph(comment='AST')
        dot.attr(rankdir='TB')
        
        def add_node(node, parent=None):
            if not isinstance(node, list):
                node_id = str(id(node))
                dot.node(node_id, str(node))
                if parent:
                    dot.edge(parent, node_id)
                return node_id
            
            node_id = str(id(node))
            dot.node(node_id, str(node[0]))
            if parent:
                dot.edge(parent, node_id)
            
            for child in node[1:]:
                add_node(child, node_id)
            return node_id

        add_node(ast)
        return dot.source

    def optimize(self, ast: Any) -> Any:
        """Perform basic optimizations on the AST."""
        if not isinstance(ast, list):
            return ast

        # Constant folding
        if ast[0] == 'binary':
            left = self.optimize(ast[2])
            right = self.optimize(ast[3])
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if ast[1] == '+':
                    return left + right
                elif ast[1] == '-':
                    return left - right
                elif ast[1] == '*':
                    return left * right
                elif ast[1] == '/':
                    return left / right if right != 0 else None
                elif ast[1] == '%':
                    return left % right if right != 0 else None
                elif ast[1] == '==':
                    return left == right
                elif ast[1] == '!=':
                    return left != right
                elif ast[1] == '<':
                    return left < right
                elif ast[1] == '>':
                    return left > right
                elif ast[1] == '<=':
                    return left <= right
                elif ast[1] == '>=':
                    return left >= right

        # Dead code elimination
        if ast[0] == 'if':
            condition = self.optimize(ast[1])
            if isinstance(condition, bool):
                return ast[2] if condition else None

        # Loop optimization
        if ast[0] == 'while':
            condition = self.optimize(ast[1])
            if isinstance(condition, bool) and not condition:
                return None

        return [ast[0]] + [self.optimize(x) for x in ast[1:]] 