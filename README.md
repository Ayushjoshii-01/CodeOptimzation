# Compiler Phase Visualizer

An interactive web application that visualizes different phases of the compilation process. Built with Python, Streamlit, and PLY (Python Lex-Yacc).

## Features

### Compilation Phases
1. **Lexical Analysis**
   - Token stream generation
   - Keyword and operator recognition
   - Error handling for invalid tokens

2. **Syntax Analysis**
   - Abstract Syntax Tree (AST) generation
   - Grammar rule implementation
   - Parse error handling

3. **Semantic Analysis**
   - Type checking
   - Symbol table management
   - Scope analysis
   - Error reporting

4. **Intermediate Code Generation**
   - Three-Address Code (TAC) generation
   - Control flow representation
   - Function call handling

5. **Code Optimization**
   - Constant folding
   - Dead code elimination
   - Loop optimization

6. **Code Generation**
   - x86 assembly code generation
   - Register allocation
   - Instruction selection

7. **Linking & Loading**
   - Section generation
   - Symbol resolution
   - Executable creation

### Language Features
- Variable declarations and assignments
- Basic arithmetic expressions
- Type checking
- Function definitions and calls
- Array operations
- Control flow (if, while, for)
- Boolean expressions
- String literals
- Increment/Decrement operators
- Logical operators (&&, ||, !)
- Comparison operators

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yashu22Pawar/compiler-design.git
cd compiler-design
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
compiler-design/
├── app.py              # Streamlit application
├── compiler.py         # Compiler implementation
├── requirements.txt    # Project dependencies
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## Usage

1. Open the application in your web browser
2. Select an example from the sidebar or write your own code
3. Click "Compile & Visualize" to start the compilation process
4. Use the Previous/Next buttons to navigate through phases
5. Explore the visualization of each compilation phase

## Example Code

```c
// Basic arithmetic
int x = 5 + 2 * 3;

// Function definition
function int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

// Array operations
int arr[5];
arr[0] = 1;
arr[1] = arr[0] * 2;
int sum = arr[0] + arr[1];
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PLY (Python Lex-Yacc) for parsing
- Streamlit for the web interface
- Graphviz for AST visualization 