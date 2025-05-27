import streamlit as st
import graphviz
from compiler import Compiler

st.set_page_config(
    page_title="Compiler Phase Visualizer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Compiler Phase Visualizer")

# Initialize session state
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 0
if 'compiler' not in st.session_state:
    st.session_state.compiler = Compiler()

# Example code snippets
example_code = {
    "Basic Arithmetic": "int x = 5 + 2 * 3;",
    "Function Definition": """function int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}""",
    "Array Operations": """int arr[5];
arr[0] = 1;
arr[1] = arr[0] * 2;
int sum = arr[0] + arr[1];""",
    "Control Flow": """int x = 10;
while (x > 0) {
    if (x % 2 == 0) {
        x = x / 2;
    } else {
        x = x - 1;
    }
}""",
    "Complex Expression": """bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}"""
}

# Sidebar for example selection
st.sidebar.markdown("### Example Code")
selected_example = st.sidebar.selectbox(
    "Choose an example:",
    list(example_code.keys())
)

# Input area
st.subheader("Input Code")
code = st.text_area(
    "Enter your code:",
    value=example_code[selected_example],
    height=200
)

# Compilation phases
phases = [
    "Lexical Analysis",
    "Syntax Analysis",
    "Semantic Analysis",
    "Intermediate Code Generation",
    "Code Optimization",
    "Code Generation",
    "Linking & Loading"
]

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Compile & Visualize", key="compile_button_top"):
        st.session_state.current_phase = 0

# Phase navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Previous", key="prev_button") and st.session_state.current_phase > 0:
        st.session_state.current_phase -= 1
with col3:
    if st.button("Next", key="next_button") and st.session_state.current_phase < len(phases) - 1:
        st.session_state.current_phase += 1

# Display current phase
st.subheader(f"Phase: {phases[st.session_state.current_phase]}")

# Process and display results
compile_clicked = st.button("Compile & Visualize", key="compile_button_bottom")
if compile_clicked or st.session_state.current_phase > 0:
    try:
        # Lexical Analysis
        if st.session_state.current_phase >= 0:
            with st.expander("✅ Lexical Analysis", expanded=st.session_state.current_phase == 0):
                tokens = st.session_state.compiler.lexical_analysis(code)
                st.json(tokens)

        # Syntax Analysis
        if st.session_state.current_phase >= 1:
            with st.expander("🌳 Syntax Analysis", expanded=st.session_state.current_phase == 1):
                ast = st.session_state.compiler.syntax_analysis(code)
                dot_source = st.session_state.compiler.generate_ast_graph(ast)
                st.graphviz_chart(dot_source)

        # Semantic Analysis
        if st.session_state.current_phase >= 2:
            with st.expander("📊 Semantic Analysis", expanded=st.session_state.current_phase == 2):
                ast = st.session_state.compiler.syntax_analysis(code)
                semantic_info = st.session_state.compiler.semantic_analysis(ast)
                
                # Display symbol table
                if semantic_info['symbol_table']:
                    st.write("Symbol Table:")
                    st.table(semantic_info['symbol_table'])
                
                # Display function table
                if semantic_info['function_table']:
                    st.write("Function Table:")
                    st.table(semantic_info['function_table'])
                
                # Display errors if any
                if semantic_info['errors']:
                    for error in semantic_info['errors']:
                        st.error(error)

        # Intermediate Code Generation
        if st.session_state.current_phase >= 3:
            with st.expander("🔧 Intermediate Code Generation", expanded=st.session_state.current_phase == 3):
                ast = st.session_state.compiler.syntax_analysis(code)
                ir_code = st.session_state.compiler.generate_ir(ast)
                
                st.write("Three-Address Code (TAC):")
                for i, instr in enumerate(ir_code):
                    if 'label' in instr:
                        st.code(f"{instr['label']}:")
                    elif 'result' in instr:
                        st.code(f"{instr['result']} = {instr['arg1']} {instr['op']} {instr.get('arg2', '')}")
                    else:
                        st.code(str(instr))

        # Code Optimization
        if st.session_state.current_phase >= 4:
            with st.expander("⚡ Code Optimization", expanded=st.session_state.current_phase == 4):
                ast = st.session_state.compiler.syntax_analysis(code)
                optimized_ast = st.session_state.compiler.optimize(ast)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original AST:")
                    st.graphviz_chart(st.session_state.compiler.generate_ast_graph(ast))
                with col2:
                    st.write("Optimized AST:")
                    st.graphviz_chart(st.session_state.compiler.generate_ast_graph(optimized_ast))

        # Code Generation
        if st.session_state.current_phase >= 5:
            with st.expander("💻 Code Generation", expanded=st.session_state.current_phase == 5):
                ast = st.session_state.compiler.syntax_analysis(code)
                ir_code = st.session_state.compiler.generate_ir(ast)
                machine_code = st.session_state.compiler.generate_machine_code(ir_code)
                
                st.write("Generated x86 Assembly:")
                st.code("\n".join(machine_code), language="nasm")

        # Linking & Loading
        if st.session_state.current_phase >= 6:
            with st.expander("🔗 Linking & Loading", expanded=st.session_state.current_phase == 6):
                ast = st.session_state.compiler.syntax_analysis(code)
                ir_code = st.session_state.compiler.generate_ir(ast)
                machine_code = st.session_state.compiler.generate_machine_code(ir_code)
                linked_code = st.session_state.compiler.link_code(machine_code)
                
                st.write("Complete Executable:")
                st.code(linked_code, language="nasm")

    except Exception as e:
        st.error(f"Error during compilation: {str(e)}")

# Add helpful information
st.sidebar.markdown("""
### Compilation Phases
1. Lexical Analysis: Converts source code to tokens
2. Syntax Analysis: Builds AST from tokens
3. Semantic Analysis: Type checking and symbol table
4. Intermediate Code: Generates Three-Address Code
5. Code Optimization: Optimizes the IR
6. Code Generation: Generates machine code
7. Linking & Loading: Creates executable

### Supported Features
- Variable declarations and assignments
- Basic arithmetic expressions
- Type checking
- Constant folding optimization
- Function definitions and calls
- Array operations
- Control flow (if, while, for)
- Boolean expressions
- String literals
- Increment/Decrement operators
- Logical operators (&&, ||, !)
- Comparison operators
- Dead code elimination
- Loop optimization

### Navigation
Use the Previous/Next buttons to move between compilation phases.
""") 